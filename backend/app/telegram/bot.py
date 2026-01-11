"""
AlphaTerminal Pro - Telegram Bot v4.2
=====================================

Trading sinyal ve bildirim botu

Özellikler:
- Sinyal gönderimi
- Portföy bildirimleri
- Risk uyarıları
- Market özeti
- Interactive komutlar

Author: AlphaTerminal Team
Version: 4.2.0
"""

import asyncio
import aiohttp
from datetime import datetime
from typing import Dict, List, Optional, Any
from dataclasses import dataclass
from enum import Enum

from app.core.config import logger, TELEGRAM_CONFIG


# ═══════════════════════════════════════════════════════════════════════════════
# ENUMS & CONSTANTS
# ═══════════════════════════════════════════════════════════════════════════════

class MessageType(Enum):
    """Mesaj türü"""
    SIGNAL = "SIGNAL"
    ALERT = "ALERT"
    REPORT = "REPORT"
    WARNING = "WARNING"
    INFO = "INFO"


# Emoji mappings
EMOJI = {
    'LONG': '🟢',
    'SHORT': '🔴',
    'NEUTRAL': '⚪',
    'STRONG': '💪',
    'MODERATE': '📊',
    'WEAK': '⚠️',
    'STOP_LOSS': '🛑',
    'TAKE_PROFIT': '🎯',
    'WARNING': '⚠️',
    'SUCCESS': '✅',
    'ERROR': '❌',
    'INFO': 'ℹ️',
    'CHART': '📈',
    'MONEY': '💰',
    'FIRE': '🔥',
    'STAR': '⭐',
    'ROCKET': '🚀',
    'CLOCK': '🕐',
}


# ═══════════════════════════════════════════════════════════════════════════════
# DATA CLASSES
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass
class TelegramMessage:
    """Telegram mesajı"""
    chat_id: str
    text: str
    parse_mode: str = "HTML"
    disable_notification: bool = False
    reply_markup: Optional[Dict] = None


@dataclass
class SignalMessage:
    """Sinyal mesajı"""
    symbol: str
    direction: str
    strength: str
    entry_price: float
    stop_loss: float
    take_profit_1: float
    take_profit_2: float
    take_profit_3: float
    risk_reward: float
    confidence: float
    smc_context: str
    orderflow_context: str
    alpha_context: str
    timeframe: str
    signal_id: str


# ═══════════════════════════════════════════════════════════════════════════════
# TELEGRAM BOT CLASS
# ═══════════════════════════════════════════════════════════════════════════════

class TelegramBot:
    """
    AlphaTerminal Telegram Bot v4.2
    
    Özellikler:
    - Async mesaj gönderimi
    - Rate limiting
    - Retry mekanizması
    - Formatted messages
    - Interactive keyboards
    """
    
    def __init__(self, config=None):
        self.config = config or TELEGRAM_CONFIG
        self.bot_token = self.config.bot_token
        self.chat_id = self.config.chat_id
        self.base_url = f"https://api.telegram.org/bot{self.bot_token}"
        
        self._session: Optional[aiohttp.ClientSession] = None
        self._last_message_time = 0
        self._message_count = 0
    
    async def _get_session(self) -> aiohttp.ClientSession:
        """HTTP session al veya oluştur"""
        if self._session is None or self._session.closed:
            self._session = aiohttp.ClientSession()
        return self._session
    
    async def close(self):
        """Session'ı kapat"""
        if self._session and not self._session.closed:
            await self._session.close()
    
    # ═══════════════════════════════════════════════════════════════════════════
    # MESSAGE SENDING
    # ═══════════════════════════════════════════════════════════════════════════
    
    async def send_message(
        self,
        text: str,
        chat_id: str = None,
        parse_mode: str = "HTML",
        disable_notification: bool = False,
        reply_markup: Dict = None
    ) -> bool:
        """
        Mesaj gönder
        
        Args:
            text: Mesaj metni
            chat_id: Hedef chat ID
            parse_mode: Parse modu (HTML/Markdown)
            disable_notification: Sessiz bildirim
            reply_markup: Klavye düzeni
            
        Returns:
            Başarılı mı
        """
        if not self.bot_token:
            logger.warning("Telegram bot token not configured")
            return False
        
        chat_id = chat_id or self.chat_id
        
        if not chat_id:
            logger.warning("Telegram chat ID not configured")
            return False
        
        # Rate limiting
        await self._rate_limit()
        
        try:
            session = await self._get_session()
            
            payload = {
                'chat_id': chat_id,
                'text': text,
                'parse_mode': parse_mode,
                'disable_notification': disable_notification
            }
            
            if reply_markup:
                payload['reply_markup'] = reply_markup
            
            url = f"{self.base_url}/sendMessage"
            
            async with session.post(url, json=payload) as response:
                if response.status == 200:
                    self._message_count += 1
                    return True
                else:
                    error = await response.text()
                    logger.error(f"Telegram API error: {response.status} - {error}")
                    return False
        
        except Exception as e:
            logger.error(f"Telegram send error: {e}")
            return False
    
    async def _rate_limit(self):
        """Rate limiting"""
        now = datetime.now().timestamp()
        
        if now - self._last_message_time < self.config.rate_limit_seconds:
            await asyncio.sleep(self.config.rate_limit_seconds)
        
        self._last_message_time = now
    
    # ═══════════════════════════════════════════════════════════════════════════
    # SIGNAL MESSAGES
    # ═══════════════════════════════════════════════════════════════════════════
    
    def format_signal_message(self, signal: SignalMessage) -> str:
        """
        Sinyal mesajı formatla
        
        Args:
            signal: SignalMessage
            
        Returns:
            Formatted HTML string
        """
        direction_emoji = EMOJI.get(signal.direction, '⚪')
        strength_emoji = EMOJI.get(signal.strength, '📊')
        
        # Risk/Reward visualization
        rr_stars = '⭐' * min(int(signal.risk_reward), 5)
        
        # Confidence bar
        conf_filled = int(signal.confidence / 10)
        conf_bar = '█' * conf_filled + '░' * (10 - conf_filled)
        
        message = f"""
{direction_emoji} <b>YENİ SİNYAL: {signal.symbol}</b> {direction_emoji}

<b>📍 Yön:</b> {signal.direction} {strength_emoji}
<b>🎯 Güven:</b> [{conf_bar}] {signal.confidence:.0f}%

<b>💰 SEVİYELER</b>
├ Entry: <code>{signal.entry_price:.2f}</code> TRY
├ 🛑 Stop: <code>{signal.stop_loss:.2f}</code> TRY
├ 🎯 TP1: <code>{signal.take_profit_1:.2f}</code> TRY
├ 🎯 TP2: <code>{signal.take_profit_2:.2f}</code> TRY
└ 🎯 TP3: <code>{signal.take_profit_3:.2f}</code> TRY

<b>📊 R:R:</b> 1:{signal.risk_reward:.1f} {rr_stars}

<b>🔍 ANALİZ</b>
├ SMC: {signal.smc_context[:50]}...
├ OF: {signal.orderflow_context[:50]}...
└ Alpha: {signal.alpha_context[:50]}...

<b>⏰ Timeframe:</b> {signal.timeframe}
<b>🆔 ID:</b> <code>{signal.signal_id}</code>

<i>⚠️ Bu finansal tavsiye değildir. Risk yönetimi yapın.</i>
"""
        return message.strip()
    
    async def send_signal(self, signal: SignalMessage) -> bool:
        """
        Sinyal gönder
        
        Args:
            signal: SignalMessage
            
        Returns:
            Başarılı mı
        """
        text = self.format_signal_message(signal)
        
        # Inline keyboard
        keyboard = {
            'inline_keyboard': [
                [
                    {'text': '📈 Chart', 'callback_data': f'chart_{signal.symbol}'},
                    {'text': '📊 Detay', 'callback_data': f'detail_{signal.signal_id}'}
                ],
                [
                    {'text': '✅ Onayla', 'callback_data': f'approve_{signal.signal_id}'},
                    {'text': '❌ Reddet', 'callback_data': f'reject_{signal.signal_id}'}
                ]
            ]
        }
        
        return await self.send_message(text, reply_markup=keyboard)
    
    # ═══════════════════════════════════════════════════════════════════════════
    # ALERT MESSAGES
    # ═══════════════════════════════════════════════════════════════════════════
    
    async def send_alert(
        self,
        title: str,
        message: str,
        alert_type: str = "INFO",
        symbol: str = None
    ) -> bool:
        """
        Alert gönder
        
        Args:
            title: Başlık
            message: Mesaj
            alert_type: Tür (INFO, WARNING, ERROR)
            symbol: İlgili hisse
            
        Returns:
            Başarılı mı
        """
        emoji = EMOJI.get(alert_type, 'ℹ️')
        
        text = f"""
{emoji} <b>{title}</b>

{message}
"""
        
        if symbol:
            text += f"\n<b>📌 Symbol:</b> {symbol}"
        
        text += f"\n\n<i>🕐 {datetime.now().strftime('%H:%M:%S')}</i>"
        
        return await self.send_message(text.strip())
    
    async def send_risk_warning(
        self,
        warning_type: str,
        current_value: float,
        threshold: float,
        message: str
    ) -> bool:
        """
        Risk uyarısı gönder
        
        Args:
            warning_type: Uyarı türü
            current_value: Güncel değer
            threshold: Eşik
            message: Mesaj
            
        Returns:
            Başarılı mı
        """
        text = f"""
🚨 <b>RİSK UYARISI</b> 🚨

<b>Tür:</b> {warning_type}
<b>Değer:</b> {current_value:.2f}%
<b>Limit:</b> {threshold:.2f}%

{message}

<i>⚠️ Risk yönetimi kurallarını gözden geçirin.</i>
"""
        return await self.send_message(text.strip())
    
    # ═══════════════════════════════════════════════════════════════════════════
    # REPORT MESSAGES
    # ═══════════════════════════════════════════════════════════════════════════
    
    async def send_daily_summary(
        self,
        date: str,
        total_signals: int,
        executed_signals: int,
        winning_trades: int,
        losing_trades: int,
        total_pnl: float,
        win_rate: float,
        top_performers: List[tuple]
    ) -> bool:
        """
        Günlük özet gönder
        
        Args:
            Özet metrikleri
            
        Returns:
            Başarılı mı
        """
        pnl_emoji = '📈' if total_pnl >= 0 else '📉'
        
        # Top performers string
        top_str = ""
        for symbol, pnl in top_performers[:3]:
            emoji = '🟢' if pnl >= 0 else '🔴'
            top_str += f"\n  {emoji} {symbol}: {pnl:+.2f} TRY"
        
        text = f"""
📊 <b>GÜNLÜK ÖZET - {date}</b>

<b>📡 SİNYALLER</b>
├ Toplam: {total_signals}
└ Execute: {executed_signals}

<b>💼 TRADE'LER</b>
├ Kazanan: {winning_trades} ✅
├ Kaybeden: {losing_trades} ❌
└ Win Rate: {win_rate:.1f}%

<b>{pnl_emoji} PERFORMANS</b>
└ Günlük PnL: <code>{total_pnl:+.2f}</code> TRY

<b>🏆 EN İYİLER</b>{top_str}

<i>Detaylı rapor için /report yazın</i>
"""
        return await self.send_message(text.strip())
    
    async def send_portfolio_update(
        self,
        capital: float,
        pnl: float,
        pnl_pct: float,
        open_positions: int,
        exposure: float
    ) -> bool:
        """
        Portföy güncellemesi gönder
        
        Args:
            Portföy metrikleri
            
        Returns:
            Başarılı mı
        """
        pnl_emoji = '📈' if pnl >= 0 else '📉'
        
        text = f"""
💼 <b>PORTFÖY DURUMU</b>

<b>💰 Sermaye:</b> <code>{capital:,.0f}</code> TRY
<b>{pnl_emoji} PnL:</b> <code>{pnl:+,.2f}</code> TRY ({pnl_pct:+.2f}%)

<b>📊 POZİSYONLAR</b>
├ Açık: {open_positions}
└ Exposure: {exposure:.1f}%

<i>🕐 {datetime.now().strftime('%H:%M:%S')}</i>
"""
        return await self.send_message(text.strip())
    
    # ═══════════════════════════════════════════════════════════════════════════
    # TRADE NOTIFICATIONS
    # ═══════════════════════════════════════════════════════════════════════════
    
    async def send_trade_opened(
        self,
        symbol: str,
        direction: str,
        entry_price: float,
        position_size: int,
        stop_loss: float,
        take_profits: List[float]
    ) -> bool:
        """Trade açılış bildirimi"""
        direction_emoji = EMOJI.get(direction, '⚪')
        
        tp_str = ""
        for i, tp in enumerate(take_profits[:3], 1):
            tp_str += f"\n├ TP{i}: <code>{tp:.2f}</code>"
        
        text = f"""
{direction_emoji} <b>TRADE AÇILDI</b>

<b>📌 Symbol:</b> {symbol}
<b>📍 Yön:</b> {direction}
<b>💰 Entry:</b> <code>{entry_price:.2f}</code> TRY
<b>📦 Lot:</b> {position_size}

<b>🎯 HEDEFLER</b>{tp_str}
└ 🛑 SL: <code>{stop_loss:.2f}</code>

<i>🕐 {datetime.now().strftime('%H:%M:%S')}</i>
"""
        return await self.send_message(text.strip())
    
    async def send_trade_closed(
        self,
        symbol: str,
        direction: str,
        entry_price: float,
        exit_price: float,
        pnl: float,
        pnl_pct: float,
        exit_reason: str
    ) -> bool:
        """Trade kapanış bildirimi"""
        result_emoji = '✅' if pnl >= 0 else '❌'
        
        text = f"""
{result_emoji} <b>TRADE KAPANDI</b>

<b>📌 Symbol:</b> {symbol}
<b>📍 Yön:</b> {direction}

<b>💰 FİYATLAR</b>
├ Entry: <code>{entry_price:.2f}</code>
└ Exit: <code>{exit_price:.2f}</code>

<b>📊 SONUÇ</b>
├ PnL: <code>{pnl:+.2f}</code> TRY
├ %: <code>{pnl_pct:+.2f}</code>%
└ Neden: {exit_reason}

<i>🕐 {datetime.now().strftime('%H:%M:%S')}</i>
"""
        return await self.send_message(text.strip())
    
    async def send_target_hit(
        self,
        symbol: str,
        target_level: int,
        price: float,
        partial_pnl: float
    ) -> bool:
        """Target hit bildirimi"""
        text = f"""
🎯 <b>HEDEF ALINDI!</b>

<b>📌 Symbol:</b> {symbol}
<b>🎯 Level:</b> TP{target_level}
<b>💰 Fiyat:</b> <code>{price:.2f}</code> TRY
<b>📈 Kısmi PnL:</b> <code>{partial_pnl:+.2f}</code> TRY

<i>Kalan pozisyon devam ediyor...</i>
"""
        return await self.send_message(text.strip())


# ═══════════════════════════════════════════════════════════════════════════════
# SINGLETON INSTANCE
# ═══════════════════════════════════════════════════════════════════════════════

_telegram_bot: Optional[TelegramBot] = None


def get_telegram_bot() -> TelegramBot:
    """Global telegram bot instance"""
    global _telegram_bot
    if _telegram_bot is None:
        _telegram_bot = TelegramBot()
    return _telegram_bot


# ═══════════════════════════════════════════════════════════════════════════════
# CONVENIENCE FUNCTIONS
# ═══════════════════════════════════════════════════════════════════════════════

async def send_signal_notification(signal_data: Dict) -> bool:
    """Sinyal bildirimi gönder (convenience function)"""
    bot = get_telegram_bot()
    
    signal = SignalMessage(
        symbol=signal_data.get('symbol', ''),
        direction=signal_data.get('direction', 'NEUTRAL'),
        strength=signal_data.get('strength', 'MODERATE'),
        entry_price=signal_data.get('entry_price', 0),
        stop_loss=signal_data.get('stop_loss', 0),
        take_profit_1=signal_data.get('take_profit_1', 0),
        take_profit_2=signal_data.get('take_profit_2', 0),
        take_profit_3=signal_data.get('take_profit_3', 0),
        risk_reward=signal_data.get('risk_reward', 0),
        confidence=signal_data.get('confidence', 0),
        smc_context=signal_data.get('smc_context', ''),
        orderflow_context=signal_data.get('orderflow_context', ''),
        alpha_context=signal_data.get('alpha_context', ''),
        timeframe=signal_data.get('timeframe', '4h'),
        signal_id=signal_data.get('signal_id', '')
    )
    
    return await bot.send_signal(signal)


# ═══════════════════════════════════════════════════════════════════════════════
# TEST
# ═══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    print("Telegram Bot v4.2 - Test")
    print("=" * 60)
    
    bot = TelegramBot()
    
    # Test signal message format
    signal = SignalMessage(
        symbol="THYAO",
        direction="LONG",
        strength="STRONG",
        entry_price=145.50,
        stop_loss=140.00,
        take_profit_1=155.00,
        take_profit_2=165.00,
        take_profit_3=175.00,
        risk_reward=2.73,
        confidence=78.5,
        smc_context="Bullish BOS | OB test | Premium zone",
        orderflow_context="Delta positive | Institutional buying",
        alpha_context="Outperformer | RS positive",
        timeframe="4h",
        signal_id="SIG_20240109_001"
    )
    
    formatted = bot.format_signal_message(signal)
    print("\n📱 FORMATTED MESSAGE:")
    print("-" * 60)
    print(formatted)
    print("-" * 60)
    
    print("\n✅ Bot hazır (token gerekli)")
