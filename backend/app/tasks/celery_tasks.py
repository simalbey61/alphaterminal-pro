"""
AlphaTerminal Pro - Celery Tasks v4.2
=====================================

Background task processing ve scheduled jobs

Tasks:
- Sinyal taraması
- Fiyat güncelleme
- Portföy senkronizasyonu
- Günlük rapor
- Risk kontrol

Author: AlphaTerminal Team
Version: 4.2.0
"""

import asyncio
from datetime import datetime, timedelta
from typing import List, Dict, Any, Optional
import logging

from celery import Celery, Task
from celery.schedules import crontab

from app.core.config import settings

logger = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════════════════════════════════
# CELERY APP
# ═══════════════════════════════════════════════════════════════════════════════

celery_app = Celery(
    "alpha_terminal",
    broker=settings.REDIS_URL,
    backend=settings.REDIS_URL,
    include=[
        "app.tasks.celery_tasks",
    ]
)

# Celery configuration
celery_app.conf.update(
    task_serializer="json",
    accept_content=["json"],
    result_serializer="json",
    timezone="Europe/Istanbul",
    enable_utc=True,
    task_track_started=True,
    task_time_limit=3600,  # 1 saat
    task_soft_time_limit=3300,  # 55 dakika
    worker_prefetch_multiplier=1,
    worker_concurrency=4,
    result_expires=86400,  # 24 saat
)

# Beat schedule - Scheduled tasks
celery_app.conf.beat_schedule = {
    # Market açıkken her 5 dakikada sinyal taraması
    "scan-signals-every-5-min": {
        "task": "app.tasks.celery_tasks.scan_signals",
        "schedule": crontab(minute="*/5", hour="10-18", day_of_week="1-5"),
    },
    
    # Her dakika fiyat güncelleme
    "update-prices-every-minute": {
        "task": "app.tasks.celery_tasks.update_prices",
        "schedule": crontab(minute="*", hour="10-18", day_of_week="1-5"),
    },
    
    # Her 15 dakikada portföy senkronizasyonu
    "sync-portfolio-every-15-min": {
        "task": "app.tasks.celery_tasks.sync_portfolio",
        "schedule": crontab(minute="*/15"),
    },
    
    # Market kapanışında günlük rapor
    "daily-report": {
        "task": "app.tasks.celery_tasks.generate_daily_report",
        "schedule": crontab(hour=18, minute=30, day_of_week="1-5"),
    },
    
    # Her saat risk kontrolü
    "risk-check-hourly": {
        "task": "app.tasks.celery_tasks.check_risk_limits",
        "schedule": crontab(minute=0, hour="10-18", day_of_week="1-5"),
    },
    
    # Her gece temizlik
    "nightly-cleanup": {
        "task": "app.tasks.celery_tasks.cleanup_old_data",
        "schedule": crontab(hour=3, minute=0),
    },
    
    # Haftalık performans raporu
    "weekly-performance-report": {
        "task": "app.tasks.celery_tasks.generate_weekly_report",
        "schedule": crontab(hour=19, minute=0, day_of_week="5"),
    },
}


# ═══════════════════════════════════════════════════════════════════════════════
# BASE TASK
# ═══════════════════════════════════════════════════════════════════════════════

class BaseTask(Task):
    """Ortak task özellikleri"""
    
    autoretry_for = (Exception,)
    retry_kwargs = {"max_retries": 3}
    retry_backoff = True
    retry_backoff_max = 600
    retry_jitter = True
    
    def on_failure(self, exc, task_id, args, kwargs, einfo):
        """Task başarısız olduğunda"""
        logger.error(f"Task {self.name}[{task_id}] failed: {exc}")
        # Telegram notification gönderilebilir
    
    def on_success(self, retval, task_id, args, kwargs):
        """Task başarılı olduğunda"""
        logger.info(f"Task {self.name}[{task_id}] completed successfully")


# ═══════════════════════════════════════════════════════════════════════════════
# SIGNAL TASKS
# ═══════════════════════════════════════════════════════════════════════════════

@celery_app.task(base=BaseTask, bind=True, name="app.tasks.celery_tasks.scan_signals")
def scan_signals(self, symbols: List[str] = None, timeframe: str = "4h"):
    """
    Sinyal taraması yap
    
    Args:
        symbols: Taranacak semboller (None = tümü)
        timeframe: Zaman dilimi
    """
    from app.services.signal_generator import SignalGenerator
    from app.core.bist_data_fetcher import BISTDataFetcher
    
    logger.info(f"Starting signal scan: timeframe={timeframe}")
    
    try:
        # Get symbols
        if not symbols:
            fetcher = BISTDataFetcher()
            symbols = fetcher.get_bist100_symbols()[:50]  # İlk 50
        
        generator = SignalGenerator()
        signals = []
        
        for symbol in symbols:
            try:
                signal = generator.generate_signal(symbol, timeframe)
                if signal and signal.get("confidence", 0) >= 60:
                    signals.append(signal)
                    logger.info(f"Signal found: {symbol} - {signal.get('direction')}")
            except Exception as e:
                logger.warning(f"Failed to analyze {symbol}: {e}")
        
        # Broadcast signals via WebSocket
        if signals:
            broadcast_signals.delay(signals)
        
        return {
            "scanned": len(symbols),
            "signals_found": len(signals),
            "timestamp": datetime.now().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Signal scan failed: {e}")
        raise


@celery_app.task(base=BaseTask, name="app.tasks.celery_tasks.broadcast_signals")
def broadcast_signals(signals: List[Dict]):
    """
    Sinyalleri WebSocket ve Telegram ile broadcast et
    
    Args:
        signals: Sinyal listesi
    """
    from app.telegram.bot import get_telegram_bot, SignalMessage
    
    logger.info(f"Broadcasting {len(signals)} signals")
    
    try:
        bot = get_telegram_bot()
        
        for signal in signals:
            # Telegram bildirimi
            asyncio.run(bot.send_signal(SignalMessage(
                symbol=signal.get("symbol"),
                direction=signal.get("direction"),
                strength=signal.get("strength", "MODERATE"),
                entry_price=signal.get("entry_price"),
                stop_loss=signal.get("stop_loss"),
                take_profit_1=signal.get("take_profit_1"),
                take_profit_2=signal.get("take_profit_2"),
                take_profit_3=signal.get("take_profit_3"),
                risk_reward=signal.get("risk_reward", 0),
                confidence=signal.get("confidence", 0),
                smc_context=signal.get("smc_context", ""),
                orderflow_context=signal.get("orderflow_context", ""),
                alpha_context=signal.get("alpha_context", ""),
                timeframe=signal.get("timeframe", "4h"),
                signal_id=signal.get("signal_id", "")
            )))
        
        return {"broadcasted": len(signals)}
        
    except Exception as e:
        logger.error(f"Broadcast failed: {e}")
        raise


# ═══════════════════════════════════════════════════════════════════════════════
# PRICE TASKS
# ═══════════════════════════════════════════════════════════════════════════════

@celery_app.task(base=BaseTask, name="app.tasks.celery_tasks.update_prices")
def update_prices(symbols: List[str] = None):
    """
    Fiyatları güncelle ve Shadow Mode'a bildir
    
    Args:
        symbols: Güncellenecek semboller
    """
    from app.core.bist_data_fetcher import BISTDataFetcher
    from app.core.shadow_mode import get_shadow_system
    from app.websocket.manager import get_connection_manager
    
    logger.debug("Updating prices...")
    
    try:
        fetcher = BISTDataFetcher()
        
        if not symbols:
            # Watchlist + open positions
            symbols = ["THYAO", "GARAN", "AKBNK", "TCELL", "EREGL"]  # Default
        
        price_data = {}
        
        for symbol in symbols:
            try:
                quote = fetcher.get_stock_quote(symbol)
                if quote:
                    price_data[symbol] = quote.get("price", 0)
            except Exception as e:
                logger.warning(f"Failed to get price for {symbol}: {e}")
        
        # Update Shadow Mode
        shadow = get_shadow_system()
        if shadow.is_active:
            closed_trades = shadow.update_prices(price_data)
            if closed_trades:
                logger.info(f"Shadow trades closed: {len(closed_trades)}")
        
        # Broadcast via WebSocket
        manager = get_connection_manager()
        for symbol, price in price_data.items():
            asyncio.run(manager.broadcast_price_update(
                symbol=symbol,
                price=price,
                change=0,  # Calculate from previous
                change_pct=0,
                volume=0
            ))
        
        return {"updated": len(price_data)}
        
    except Exception as e:
        logger.error(f"Price update failed: {e}")
        raise


# ═══════════════════════════════════════════════════════════════════════════════
# PORTFOLIO TASKS
# ═══════════════════════════════════════════════════════════════════════════════

@celery_app.task(base=BaseTask, name="app.tasks.celery_tasks.sync_portfolio")
def sync_portfolio():
    """Portföy senkronizasyonu"""
    from app.core.shadow_mode import get_shadow_system
    from app.websocket.manager import get_connection_manager
    
    logger.debug("Syncing portfolio...")
    
    try:
        shadow = get_shadow_system()
        
        if not shadow.is_active:
            return {"status": "shadow_mode_inactive"}
        
        portfolio = shadow.get_portfolio()
        
        if portfolio:
            # Broadcast update
            manager = get_connection_manager()
            asyncio.run(manager.broadcast_portfolio_update(
                total_value=portfolio.current_capital,
                daily_pnl=portfolio.unrealized_pnl,
                daily_pnl_pct=portfolio.total_pnl_pct,
                open_positions=portfolio.open_positions
            ))
            
            return {
                "capital": portfolio.current_capital,
                "pnl": portfolio.total_pnl,
                "positions": portfolio.open_positions
            }
        
        return {"status": "no_portfolio"}
        
    except Exception as e:
        logger.error(f"Portfolio sync failed: {e}")
        raise


# ═══════════════════════════════════════════════════════════════════════════════
# RISK TASKS
# ═══════════════════════════════════════════════════════════════════════════════

@celery_app.task(base=BaseTask, name="app.tasks.celery_tasks.check_risk_limits")
def check_risk_limits():
    """Risk limitlerini kontrol et"""
    from app.core.shadow_mode import get_shadow_system
    from app.telegram.bot import get_telegram_bot
    from app.websocket.manager import get_connection_manager
    
    logger.info("Checking risk limits...")
    
    try:
        shadow = get_shadow_system()
        
        if not shadow.is_active:
            return {"status": "shadow_mode_inactive"}
        
        portfolio = shadow.get_portfolio()
        alerts = []
        
        if portfolio:
            # Drawdown check
            if portfolio.max_drawdown > 10:
                alerts.append({
                    "type": "DRAWDOWN",
                    "message": f"Drawdown kritik: {portfolio.max_drawdown:.1f}%",
                    "severity": "high"
                })
            elif portfolio.max_drawdown > 7:
                alerts.append({
                    "type": "DRAWDOWN",
                    "message": f"Drawdown uyarısı: {portfolio.max_drawdown:.1f}%",
                    "severity": "medium"
                })
            
            # Exposure check
            exposure_pct = (portfolio.total_exposure / portfolio.current_capital * 100) if portfolio.current_capital > 0 else 0
            if exposure_pct > 80:
                alerts.append({
                    "type": "EXPOSURE",
                    "message": f"Yüksek exposure: {exposure_pct:.1f}%",
                    "severity": "high"
                })
            
            # Win rate check
            if portfolio.total_trades >= 20 and portfolio.win_rate < 40:
                alerts.append({
                    "type": "PERFORMANCE",
                    "message": f"Düşük win rate: {portfolio.win_rate:.1f}%",
                    "severity": "medium"
                })
        
        # Send alerts
        if alerts:
            bot = get_telegram_bot()
            manager = get_connection_manager()
            
            for alert in alerts:
                asyncio.run(bot.send_risk_warning(
                    alert["type"],
                    0,  # current value
                    0,  # threshold
                    alert["message"]
                ))
                
                asyncio.run(manager.broadcast_alert(
                    alert["type"],
                    "Risk Uyarısı",
                    alert["message"],
                    alert["severity"]
                ))
        
        return {
            "checked": True,
            "alerts": len(alerts),
            "timestamp": datetime.now().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Risk check failed: {e}")
        raise


# ═══════════════════════════════════════════════════════════════════════════════
# REPORT TASKS
# ═══════════════════════════════════════════════════════════════════════════════

@celery_app.task(base=BaseTask, name="app.tasks.celery_tasks.generate_daily_report")
def generate_daily_report():
    """Günlük rapor oluştur ve gönder"""
    from app.core.shadow_mode import get_shadow_system
    from app.telegram.bot import get_telegram_bot
    
    logger.info("Generating daily report...")
    
    try:
        shadow = get_shadow_system()
        bot = get_telegram_bot()
        
        if shadow.is_active:
            report = shadow.generate_report()
            
            if report:
                asyncio.run(bot.send_daily_summary(
                    date=datetime.now().strftime("%Y-%m-%d"),
                    total_signals=report.total_signals,
                    executed_signals=report.executed_signals,
                    winning_trades=report.winning_trades,
                    losing_trades=report.losing_trades,
                    total_pnl=report.total_return,
                    win_rate=report.win_rate,
                    top_performers=report.top_symbols
                ))
                
                return {
                    "report_generated": True,
                    "pnl": report.total_return,
                    "trades": report.total_trades
                }
        
        return {"status": "no_active_session"}
        
    except Exception as e:
        logger.error(f"Daily report failed: {e}")
        raise


@celery_app.task(base=BaseTask, name="app.tasks.celery_tasks.generate_weekly_report")
def generate_weekly_report():
    """Haftalık performans raporu"""
    logger.info("Generating weekly report...")
    
    # Implement weekly report logic
    return {"status": "completed"}


# ═══════════════════════════════════════════════════════════════════════════════
# MAINTENANCE TASKS
# ═══════════════════════════════════════════════════════════════════════════════

@celery_app.task(base=BaseTask, name="app.tasks.celery_tasks.cleanup_old_data")
def cleanup_old_data(days: int = 30):
    """
    Eski verileri temizle
    
    Args:
        days: Kaç günden eski veriler silinecek
    """
    logger.info(f"Cleaning up data older than {days} days...")
    
    try:
        # Implement cleanup logic
        # - Old signals
        # - Old audit logs
        # - Cache cleanup
        
        return {
            "cleaned": True,
            "days": days,
            "timestamp": datetime.now().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Cleanup failed: {e}")
        raise


# ═══════════════════════════════════════════════════════════════════════════════
# UTILITY TASKS
# ═══════════════════════════════════════════════════════════════════════════════

@celery_app.task(name="app.tasks.celery_tasks.health_check")
def health_check():
    """Celery health check"""
    return {
        "status": "healthy",
        "timestamp": datetime.now().isoformat()
    }


@celery_app.task(name="app.tasks.celery_tasks.test_telegram")
def test_telegram():
    """Telegram bağlantı testi"""
    from app.telegram.bot import get_telegram_bot
    
    bot = get_telegram_bot()
    result = asyncio.run(bot.send_alert(
        "Test",
        "AlphaTerminal Pro Celery test mesajı",
        "INFO"
    ))
    
    return {"sent": result}
