"""
AlphaTerminal Pro - Telegram Notifications
==========================================

Send trading signals and alerts via Telegram.

Author: AlphaTerminal Team
Version: 1.0.0
"""

import logging
import asyncio
from datetime import datetime
from typing import Optional, Dict, List, Any, Union
from dataclasses import dataclass, field
from abc import ABC, abstractmethod
import json

# Optional aiohttp import
try:
    import aiohttp
    AIOHTTP_AVAILABLE = True
except ImportError:
    AIOHTTP_AVAILABLE = False
    aiohttp = None

from app.reporting.types import (
    NotificationChannel, NotificationPriority, SignalType,
    TradingSignal, NotificationMessage
)


logger = logging.getLogger(__name__)


# =============================================================================
# MESSAGE FORMATTERS
# =============================================================================

class MessageFormatter:
    """Format messages for different notification types."""
    
    @staticmethod
    def format_signal(signal: TradingSignal) -> str:
        """Format trading signal for Telegram."""
        
        # Emoji mapping
        emoji = {
            SignalType.BUY: "🟢",
            SignalType.SELL: "🔴",
            SignalType.HOLD: "🟡",
            SignalType.CLOSE: "⚪",
            SignalType.STOP_LOSS: "🛑",
            SignalType.TAKE_PROFIT: "✅",
        }
        
        signal_emoji = emoji.get(signal.signal_type, "📊")
        
        message = f"""
{signal_emoji} *{signal.signal_type.value.upper()} SIGNAL*

*Symbol:* `{signal.symbol}`
*Price:* {signal.price:.2f}
"""
        
        if signal.target_price:
            message += f"*Target:* {signal.target_price:.2f}\n"
        
        if signal.stop_loss:
            message += f"*Stop Loss:* {signal.stop_loss:.2f}\n"
        
        if signal.target_price and signal.stop_loss:
            risk = abs(signal.price - signal.stop_loss)
            reward = abs(signal.target_price - signal.price)
            rr_ratio = reward / risk if risk > 0 else 0
            message += f"*R:R Ratio:* {rr_ratio:.2f}\n"
        
        message += f"\n*Confidence:* {signal.confidence*100:.0f}%"
        message += f"\n*Strategy:* {signal.strategy_name}"
        message += f"\n*Timeframe:* {signal.timeframe}"
        
        if signal.indicators:
            message += "\n\n*Indicators:*"
            for name, value in signal.indicators.items():
                message += f"\n• {name}: {value:.2f}"
        
        if signal.notes:
            message += f"\n\n📝 _{signal.notes}_"
        
        message += f"\n\n⏰ {signal.created_at.strftime('%Y-%m-%d %H:%M:%S')}"
        
        return message
    
    @staticmethod
    def format_alert(
        title: str,
        message: str,
        priority: NotificationPriority = NotificationPriority.NORMAL
    ) -> str:
        """Format alert message for Telegram."""
        
        priority_emoji = {
            NotificationPriority.LOW: "ℹ️",
            NotificationPriority.NORMAL: "📢",
            NotificationPriority.HIGH: "⚠️",
            NotificationPriority.CRITICAL: "🚨",
        }
        
        emoji = priority_emoji.get(priority, "📢")
        
        return f"""
{emoji} *{title}*

{message}

⏰ {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
"""
    
    @staticmethod
    def format_backtest_summary(result: Dict[str, Any]) -> str:
        """Format backtest result summary for Telegram."""
        
        total_return = result.get('total_return_pct', 0)
        emoji = "📈" if total_return >= 0 else "📉"
        
        return f"""
{emoji} *Backtest Completed*

*Symbol:* `{result.get('symbol', 'N/A')}`
*Strategy:* {result.get('strategy_name', 'N/A')}

*Performance:*
• Total Return: {total_return:.2f}%
• Sharpe Ratio: {result.get('sharpe_ratio', 0):.2f}
• Max Drawdown: {result.get('max_drawdown', 0)*100:.2f}%
• Win Rate: {result.get('win_rate', 0)*100:.1f}%
• Profit Factor: {result.get('profit_factor', 0):.2f}

*Trades:* {result.get('total_trades', 0)}
• Winners: {result.get('winning_trades', 0)}
• Losers: {result.get('losing_trades', 0)}

⏰ {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
"""
    
    @staticmethod
    def format_daily_summary(data: Dict[str, Any]) -> str:
        """Format daily trading summary for Telegram."""
        
        pnl = data.get('daily_pnl', 0)
        emoji = "📈" if pnl >= 0 else "📉"
        
        return f"""
{emoji} *Daily Summary - {datetime.now().strftime('%Y-%m-%d')}*

*Portfolio Value:* ${data.get('portfolio_value', 0):,.2f}
*Daily P&L:* ${pnl:,.2f} ({data.get('daily_pnl_pct', 0):.2f}%)

*Today's Activity:*
• Trades Executed: {data.get('trades_today', 0)}
• Winners: {data.get('winners_today', 0)}
• Losers: {data.get('losers_today', 0)}

*Top Performers:*
{data.get('top_performers', 'N/A')}

*Worst Performers:*
{data.get('worst_performers', 'N/A')}

*Active Signals:* {data.get('active_signals', 0)}
"""


# =============================================================================
# TELEGRAM BOT
# =============================================================================

class TelegramBot:
    """
    Telegram bot for sending trading notifications.
    
    Supports:
    - Text messages with Markdown formatting
    - Images (charts)
    - Documents (reports)
    - Inline keyboards
    
    Requires aiohttp for async operations.
    """
    
    def __init__(
        self,
        bot_token: str,
        chat_id: Union[str, int],
        parse_mode: str = "Markdown"
    ):
        """
        Initialize Telegram bot.
        
        Args:
            bot_token: Telegram bot token
            chat_id: Chat ID to send messages to
            parse_mode: Message parse mode (Markdown, HTML)
        """
        if not AIOHTTP_AVAILABLE:
            logger.warning("aiohttp not available - Telegram bot will be limited")
        
        self.bot_token = bot_token
        self.chat_id = str(chat_id)
        self.parse_mode = parse_mode
        
        self.base_url = f"https://api.telegram.org/bot{bot_token}"
        self.formatter = MessageFormatter()
        
        self._session = None
    
    async def _get_session(self):
        """Get or create aiohttp session."""
        if not AIOHTTP_AVAILABLE:
            raise RuntimeError("aiohttp is required for Telegram operations")
        
        if self._session is None or self._session.closed:
            self._session = aiohttp.ClientSession()
        return self._session
    
    async def close(self):
        """Close the session."""
        if self._session and not self._session.closed:
            await self._session.close()
    
    async def send_message(
        self,
        text: str,
        chat_id: Optional[str] = None,
        parse_mode: Optional[str] = None,
        disable_notification: bool = False,
        reply_markup: Optional[Dict] = None
    ) -> Dict[str, Any]:
        """
        Send text message.
        
        Args:
            text: Message text
            chat_id: Override chat ID
            parse_mode: Override parse mode
            disable_notification: Send silently
            reply_markup: Inline keyboard markup
            
        Returns:
            Telegram API response
        """
        session = await self._get_session()
        
        payload = {
            "chat_id": chat_id or self.chat_id,
            "text": text,
            "parse_mode": parse_mode or self.parse_mode,
            "disable_notification": disable_notification,
        }
        
        if reply_markup:
            payload["reply_markup"] = json.dumps(reply_markup)
        
        try:
            async with session.post(
                f"{self.base_url}/sendMessage",
                data=payload
            ) as response:
                result = await response.json()
                
                if not result.get("ok"):
                    logger.error(f"Telegram error: {result}")
                
                return result
                
        except Exception as e:
            logger.error(f"Failed to send Telegram message: {e}")
            return {"ok": False, "error": str(e)}
    
    async def send_photo(
        self,
        photo: Union[bytes, str],
        caption: Optional[str] = None,
        chat_id: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Send photo.
        
        Args:
            photo: Photo bytes or file path
            caption: Photo caption
            chat_id: Override chat ID
            
        Returns:
            Telegram API response
        """
        session = await self._get_session()
        
        data = aiohttp.FormData()
        data.add_field("chat_id", chat_id or self.chat_id)
        
        if isinstance(photo, bytes):
            data.add_field("photo", photo, filename="chart.png")
        else:
            data.add_field("photo", open(photo, "rb"), filename="chart.png")
        
        if caption:
            data.add_field("caption", caption)
            data.add_field("parse_mode", self.parse_mode)
        
        try:
            async with session.post(
                f"{self.base_url}/sendPhoto",
                data=data
            ) as response:
                return await response.json()
                
        except Exception as e:
            logger.error(f"Failed to send Telegram photo: {e}")
            return {"ok": False, "error": str(e)}
    
    async def send_document(
        self,
        document: Union[bytes, str],
        filename: str,
        caption: Optional[str] = None,
        chat_id: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Send document.
        
        Args:
            document: Document bytes or file path
            filename: Document filename
            caption: Document caption
            chat_id: Override chat ID
            
        Returns:
            Telegram API response
        """
        session = await self._get_session()
        
        data = aiohttp.FormData()
        data.add_field("chat_id", chat_id or self.chat_id)
        
        if isinstance(document, bytes):
            data.add_field("document", document, filename=filename)
        else:
            data.add_field("document", open(document, "rb"), filename=filename)
        
        if caption:
            data.add_field("caption", caption)
            data.add_field("parse_mode", self.parse_mode)
        
        try:
            async with session.post(
                f"{self.base_url}/sendDocument",
                data=data
            ) as response:
                return await response.json()
                
        except Exception as e:
            logger.error(f"Failed to send Telegram document: {e}")
            return {"ok": False, "error": str(e)}
    
    # =========================================================================
    # HIGH-LEVEL METHODS
    # =========================================================================
    
    async def send_signal(
        self,
        signal: TradingSignal,
        chart: Optional[bytes] = None
    ) -> Dict[str, Any]:
        """
        Send trading signal notification.
        
        Args:
            signal: Trading signal
            chart: Optional chart image
            
        Returns:
            Telegram API response
        """
        message = self.formatter.format_signal(signal)
        
        if chart:
            return await self.send_photo(chart, caption=message)
        else:
            return await self.send_message(message)
    
    async def send_alert(
        self,
        title: str,
        message: str,
        priority: NotificationPriority = NotificationPriority.NORMAL
    ) -> Dict[str, Any]:
        """
        Send alert notification.
        
        Args:
            title: Alert title
            message: Alert message
            priority: Alert priority
            
        Returns:
            Telegram API response
        """
        formatted = self.formatter.format_alert(title, message, priority)
        
        disable_notification = priority == NotificationPriority.LOW
        
        return await self.send_message(
            formatted,
            disable_notification=disable_notification
        )
    
    async def send_backtest_report(
        self,
        result: Dict[str, Any],
        report_file: Optional[bytes] = None,
        chart: Optional[bytes] = None
    ) -> Dict[str, Any]:
        """
        Send backtest result notification.
        
        Args:
            result: Backtest result dict
            report_file: Optional PDF report
            chart: Optional equity curve chart
            
        Returns:
            Telegram API response
        """
        message = self.formatter.format_backtest_summary(result)
        
        # Send summary message
        await self.send_message(message)
        
        # Send chart if available
        if chart:
            await self.send_photo(chart, caption="📊 Equity Curve")
        
        # Send report if available
        if report_file:
            symbol = result.get('symbol', 'backtest')
            filename = f"{symbol}_report_{datetime.now().strftime('%Y%m%d')}.html"
            await self.send_document(report_file, filename, caption="📄 Full Report")
        
        return {"ok": True}
    
    async def send_daily_summary(
        self,
        data: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Send daily trading summary.
        
        Args:
            data: Daily summary data
            
        Returns:
            Telegram API response
        """
        message = self.formatter.format_daily_summary(data)
        return await self.send_message(message)


# =============================================================================
# NOTIFICATION SERVICE
# =============================================================================

class NotificationService:
    """
    Unified notification service.
    
    Manages multiple notification channels and provides
    a simple interface for sending notifications.
    """
    
    def __init__(self):
        """Initialize notification service."""
        self._channels: Dict[NotificationChannel, Any] = {}
        self._queue: List[NotificationMessage] = []
    
    def register_telegram(
        self,
        bot_token: str,
        chat_id: Union[str, int]
    ):
        """Register Telegram channel."""
        self._channels[NotificationChannel.TELEGRAM] = TelegramBot(
            bot_token=bot_token,
            chat_id=chat_id
        )
        logger.info("Telegram channel registered")
    
    def get_telegram(self) -> Optional[TelegramBot]:
        """Get Telegram bot instance."""
        return self._channels.get(NotificationChannel.TELEGRAM)
    
    async def send_signal(
        self,
        signal: TradingSignal,
        channels: Optional[List[NotificationChannel]] = None,
        chart: Optional[bytes] = None
    ) -> Dict[str, bool]:
        """
        Send trading signal to specified channels.
        
        Args:
            signal: Trading signal
            channels: Target channels (default: all)
            chart: Optional chart image
            
        Returns:
            Dict of channel to success status
        """
        channels = channels or list(self._channels.keys())
        results = {}
        
        for channel in channels:
            if channel == NotificationChannel.TELEGRAM:
                bot = self._channels.get(channel)
                if bot:
                    result = await bot.send_signal(signal, chart)
                    results[channel.value] = result.get("ok", False)
        
        return results
    
    async def send_alert(
        self,
        title: str,
        message: str,
        priority: NotificationPriority = NotificationPriority.NORMAL,
        channels: Optional[List[NotificationChannel]] = None
    ) -> Dict[str, bool]:
        """
        Send alert to specified channels.
        
        Args:
            title: Alert title
            message: Alert message
            priority: Alert priority
            channels: Target channels (default: all)
            
        Returns:
            Dict of channel to success status
        """
        channels = channels or list(self._channels.keys())
        results = {}
        
        for channel in channels:
            if channel == NotificationChannel.TELEGRAM:
                bot = self._channels.get(channel)
                if bot:
                    result = await bot.send_alert(title, message, priority)
                    results[channel.value] = result.get("ok", False)
        
        return results
    
    async def close(self):
        """Close all channels."""
        for channel in self._channels.values():
            if hasattr(channel, 'close'):
                await channel.close()


# =============================================================================
# ALL EXPORTS
# =============================================================================

__all__ = [
    "MessageFormatter",
    "TelegramBot",
    "NotificationService",
]
