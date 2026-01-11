"""
AlphaTerminal Pro - Audit Engine v4.2
=====================================

Kurumsal Seviye Audit & Logging Motoru

Özellikler:
- Trade Audit Trail
- Signal History Tracking
- Performance Attribution
- System Event Logging
- Compliance Reporting
- Error Tracking
- User Activity Monitoring

Author: AlphaTerminal Team
Version: 4.2.0
"""

import json
import hashlib
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Union
from dataclasses import dataclass, field, asdict
from enum import Enum
from pathlib import Path
import pandas as pd

from app.core.config import logger, LOGS_DIR


# ═══════════════════════════════════════════════════════════════════════════════
# ENUMS
# ═══════════════════════════════════════════════════════════════════════════════

class AuditEventType(Enum):
    """Audit olay türleri"""
    # Trade Events
    SIGNAL_GENERATED = "SIGNAL_GENERATED"
    SIGNAL_APPROVED = "SIGNAL_APPROVED"
    SIGNAL_REJECTED = "SIGNAL_REJECTED"
    SIGNAL_EXPIRED = "SIGNAL_EXPIRED"
    
    TRADE_OPENED = "TRADE_OPENED"
    TRADE_CLOSED = "TRADE_CLOSED"
    TRADE_MODIFIED = "TRADE_MODIFIED"
    STOP_LOSS_HIT = "STOP_LOSS_HIT"
    TAKE_PROFIT_HIT = "TAKE_PROFIT_HIT"
    
    # System Events
    SYSTEM_START = "SYSTEM_START"
    SYSTEM_STOP = "SYSTEM_STOP"
    ENGINE_INITIALIZED = "ENGINE_INITIALIZED"
    ENGINE_ERROR = "ENGINE_ERROR"
    
    # Data Events
    DATA_FETCH = "DATA_FETCH"
    DATA_ERROR = "DATA_ERROR"
    CACHE_HIT = "CACHE_HIT"
    CACHE_MISS = "CACHE_MISS"
    
    # User Events
    USER_LOGIN = "USER_LOGIN"
    USER_LOGOUT = "USER_LOGOUT"
    SETTINGS_CHANGED = "SETTINGS_CHANGED"
    
    # Risk Events
    RISK_LIMIT_WARNING = "RISK_LIMIT_WARNING"
    RISK_LIMIT_BREACH = "RISK_LIMIT_BREACH"
    PORTFOLIO_HEAT_WARNING = "PORTFOLIO_HEAT_WARNING"
    
    # AI Events
    STRATEGY_GENERATED = "STRATEGY_GENERATED"
    STRATEGY_VALIDATED = "STRATEGY_VALIDATED"
    STRATEGY_DEPLOYED = "STRATEGY_DEPLOYED"


class AuditSeverity(Enum):
    """Audit olay ciddiyeti"""
    DEBUG = "DEBUG"
    INFO = "INFO"
    WARNING = "WARNING"
    ERROR = "ERROR"
    CRITICAL = "CRITICAL"


class AuditCategory(Enum):
    """Audit kategorisi"""
    TRADE = "TRADE"
    SIGNAL = "SIGNAL"
    SYSTEM = "SYSTEM"
    DATA = "DATA"
    USER = "USER"
    RISK = "RISK"
    AI = "AI"
    SECURITY = "SECURITY"


# ═══════════════════════════════════════════════════════════════════════════════
# DATA CLASSES
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass
class AuditEvent:
    """Audit olayı"""
    event_id: str
    event_type: AuditEventType
    category: AuditCategory
    severity: AuditSeverity
    timestamp: datetime
    
    # Context
    symbol: Optional[str] = None
    user_id: Optional[str] = None
    session_id: Optional[str] = None
    
    # Details
    message: str = ""
    data: Dict[str, Any] = field(default_factory=dict)
    
    # Tracing
    correlation_id: Optional[str] = None
    parent_event_id: Optional[str] = None
    
    # Meta
    source: str = "SYSTEM"
    version: str = "4.2.0"
    
    def to_dict(self) -> Dict:
        """Dict'e çevir"""
        return {
            'event_id': self.event_id,
            'event_type': self.event_type.value,
            'category': self.category.value,
            'severity': self.severity.value,
            'timestamp': self.timestamp.isoformat(),
            'symbol': self.symbol,
            'user_id': self.user_id,
            'session_id': self.session_id,
            'message': self.message,
            'data': self.data,
            'correlation_id': self.correlation_id,
            'parent_event_id': self.parent_event_id,
            'source': self.source,
            'version': self.version
        }
    
    def to_json(self) -> str:
        """JSON'a çevir"""
        return json.dumps(self.to_dict(), ensure_ascii=False, default=str)


@dataclass
class TradeAudit:
    """Trade audit kaydı"""
    trade_id: str
    symbol: str
    direction: str
    
    # Entry
    entry_price: float
    entry_time: datetime
    entry_signal_id: str
    entry_reason: str
    
    # Exit (opsiyonel)
    exit_price: Optional[float] = None
    exit_time: Optional[datetime] = None
    exit_reason: Optional[str] = None
    
    # Position
    position_size: int = 0
    risk_amount: float = 0
    stop_loss: float = 0
    take_profits: List[float] = field(default_factory=list)
    
    # Result
    pnl: float = 0
    pnl_percent: float = 0
    mae: float = 0  # Maximum Adverse Excursion
    mfe: float = 0  # Maximum Favorable Excursion
    
    # Scores at entry
    smc_score: float = 0
    orderflow_score: float = 0
    alpha_score: float = 0
    confidence_score: float = 0
    
    # Events
    events: List[str] = field(default_factory=list)
    
    def to_dict(self) -> Dict:
        return asdict(self)


@dataclass
class SignalAudit:
    """Sinyal audit kaydı"""
    signal_id: str
    symbol: str
    direction: str
    
    # Generation
    generated_at: datetime
    generated_by: str  # Engine adı
    
    # Scores
    smc_score: float
    orderflow_score: float
    alpha_score: float
    confluence_score: float
    confidence: float
    
    # Status
    status: str  # "PENDING", "APPROVED", "REJECTED", "EXPIRED", "EXECUTED"
    status_reason: Optional[str] = None
    
    # Execution
    executed: bool = False
    execution_time: Optional[datetime] = None
    trade_id: Optional[str] = None
    
    # Outcome (if executed)
    outcome_pnl: Optional[float] = None
    outcome_hit_target: Optional[bool] = None
    
    def to_dict(self) -> Dict:
        result = asdict(self)
        result['generated_at'] = self.generated_at.isoformat()
        if self.execution_time:
            result['execution_time'] = self.execution_time.isoformat()
        return result


@dataclass
class PerformanceAudit:
    """Performans audit kaydı"""
    period_start: datetime
    period_end: datetime
    
    # Counts
    total_signals: int
    executed_signals: int
    winning_trades: int
    losing_trades: int
    
    # Returns
    total_pnl: float
    gross_profit: float
    gross_loss: float
    
    # Ratios
    win_rate: float
    profit_factor: float
    avg_win: float
    avg_loss: float
    expectancy: float
    
    # Risk
    max_drawdown: float
    avg_risk_per_trade: float
    sharpe_ratio: float
    
    # Attribution
    pnl_by_symbol: Dict[str, float] = field(default_factory=dict)
    pnl_by_direction: Dict[str, float] = field(default_factory=dict)
    pnl_by_signal_type: Dict[str, float] = field(default_factory=dict)


@dataclass
class AuditSummary:
    """Audit özeti"""
    period: str
    start_date: datetime
    end_date: datetime
    
    # Event counts
    total_events: int
    events_by_type: Dict[str, int]
    events_by_severity: Dict[str, int]
    events_by_category: Dict[str, int]
    
    # Trade summary
    total_trades: int
    total_signals: int
    signal_to_trade_ratio: float
    
    # Errors
    error_count: int
    critical_count: int
    
    # Highlights
    notable_events: List[AuditEvent]


# ═══════════════════════════════════════════════════════════════════════════════
# AUDIT ENGINE CLASS
# ═══════════════════════════════════════════════════════════════════════════════

class AuditEngine:
    """
    Kurumsal Seviye Audit & Logging Motoru v4.2
    
    Özellikler:
    - Trade audit trail
    - Signal tracking
    - Event logging
    - Performance attribution
    - Compliance reporting
    - Error analysis
    """
    
    def __init__(
        self,
        log_dir: Path = None,
        max_events_in_memory: int = 10000,
        auto_persist: bool = True
    ):
        self.log_dir = log_dir or LOGS_DIR
        self.max_events = max_events_in_memory
        self.auto_persist = auto_persist
        
        # In-memory storage
        self._events: List[AuditEvent] = []
        self._trades: Dict[str, TradeAudit] = {}
        self._signals: Dict[str, SignalAudit] = {}
        
        # Counters
        self._event_counter = 0
        
        # Ensure log directory exists
        self.log_dir.mkdir(parents=True, exist_ok=True)
        
        # Log initialization
        self.log_event(
            AuditEventType.SYSTEM_START,
            AuditCategory.SYSTEM,
            "Audit Engine initialized",
            severity=AuditSeverity.INFO
        )
    
    def _generate_event_id(self) -> str:
        """Unique event ID üret"""
        self._event_counter += 1
        timestamp = datetime.now().strftime("%Y%m%d%H%M%S%f")
        return f"EVT_{timestamp}_{self._event_counter:06d}"
    
    def _generate_hash(self, data: str) -> str:
        """Hash üret (immutability için)"""
        return hashlib.sha256(data.encode()).hexdigest()[:16]
    
    # ═══════════════════════════════════════════════════════════════════════════
    # EVENT LOGGING
    # ═══════════════════════════════════════════════════════════════════════════
    
    def log_event(
        self,
        event_type: AuditEventType,
        category: AuditCategory,
        message: str,
        severity: AuditSeverity = AuditSeverity.INFO,
        symbol: str = None,
        user_id: str = None,
        data: Dict = None,
        correlation_id: str = None,
        parent_event_id: str = None,
        source: str = "SYSTEM"
    ) -> AuditEvent:
        """
        Audit olayı logla
        
        Args:
            event_type: Olay türü
            category: Kategori
            message: Açıklama
            severity: Ciddiyet
            symbol: İlgili hisse
            user_id: Kullanıcı ID
            data: Ek veri
            correlation_id: İlişkili işlem ID
            parent_event_id: Üst olay ID
            source: Kaynak
            
        Returns:
            AuditEvent
        """
        event = AuditEvent(
            event_id=self._generate_event_id(),
            event_type=event_type,
            category=category,
            severity=severity,
            timestamp=datetime.now(),
            symbol=symbol,
            user_id=user_id,
            message=message,
            data=data or {},
            correlation_id=correlation_id,
            parent_event_id=parent_event_id,
            source=source
        )
        
        # In-memory storage
        self._events.append(event)
        
        # Memory limit check
        if len(self._events) > self.max_events:
            self._persist_old_events()
        
        # Auto-persist critical events
        if self.auto_persist and severity in [AuditSeverity.ERROR, AuditSeverity.CRITICAL]:
            self._persist_event(event)
        
        # Also log to standard logger
        log_method = getattr(logger, severity.value.lower(), logger.info)
        log_method(f"[AUDIT] {event_type.value}: {message}")
        
        return event
    
    # ═══════════════════════════════════════════════════════════════════════════
    # TRADE AUDIT
    # ═══════════════════════════════════════════════════════════════════════════
    
    def log_trade_opened(
        self,
        trade_id: str,
        symbol: str,
        direction: str,
        entry_price: float,
        position_size: int,
        stop_loss: float,
        take_profits: List[float],
        signal_id: str,
        entry_reason: str,
        scores: Dict[str, float] = None
    ) -> TradeAudit:
        """
        Trade açılışını logla
        
        Args:
            trade_id: Trade ID
            symbol: Hisse kodu
            direction: Yön
            entry_price: Giriş fiyatı
            position_size: Pozisyon büyüklüğü
            stop_loss: Stop loss
            take_profits: Take profit seviyeleri
            signal_id: Sinyal ID
            entry_reason: Giriş nedeni
            scores: Motor skorları
            
        Returns:
            TradeAudit
        """
        scores = scores or {}
        
        trade = TradeAudit(
            trade_id=trade_id,
            symbol=symbol,
            direction=direction,
            entry_price=entry_price,
            entry_time=datetime.now(),
            entry_signal_id=signal_id,
            entry_reason=entry_reason,
            position_size=position_size,
            risk_amount=abs(entry_price - stop_loss) * position_size,
            stop_loss=stop_loss,
            take_profits=take_profits,
            smc_score=scores.get('smc', 0),
            orderflow_score=scores.get('orderflow', 0),
            alpha_score=scores.get('alpha', 0),
            confidence_score=scores.get('confidence', 0)
        )
        
        self._trades[trade_id] = trade
        
        # Log event
        self.log_event(
            AuditEventType.TRADE_OPENED,
            AuditCategory.TRADE,
            f"Trade opened: {symbol} {direction} @ {entry_price}",
            symbol=symbol,
            data={
                'trade_id': trade_id,
                'entry_price': entry_price,
                'position_size': position_size,
                'stop_loss': stop_loss,
                'signal_id': signal_id
            },
            correlation_id=signal_id
        )
        
        return trade
    
    def log_trade_closed(
        self,
        trade_id: str,
        exit_price: float,
        exit_reason: str,
        pnl: float = None
    ) -> Optional[TradeAudit]:
        """
        Trade kapanışını logla
        
        Args:
            trade_id: Trade ID
            exit_price: Çıkış fiyatı
            exit_reason: Çıkış nedeni
            pnl: PnL (None ise hesaplanır)
            
        Returns:
            TradeAudit veya None
        """
        if trade_id not in self._trades:
            logger.warning(f"Trade not found: {trade_id}")
            return None
        
        trade = self._trades[trade_id]
        trade.exit_price = exit_price
        trade.exit_time = datetime.now()
        trade.exit_reason = exit_reason
        
        # PnL hesapla
        if pnl is None:
            if trade.direction == "LONG":
                pnl = (exit_price - trade.entry_price) * trade.position_size
            else:
                pnl = (trade.entry_price - exit_price) * trade.position_size
        
        trade.pnl = pnl
        trade.pnl_percent = (pnl / (trade.entry_price * trade.position_size)) * 100
        
        # Determine event type
        if "STOP" in exit_reason.upper():
            event_type = AuditEventType.STOP_LOSS_HIT
        elif "TARGET" in exit_reason.upper() or "TP" in exit_reason.upper():
            event_type = AuditEventType.TAKE_PROFIT_HIT
        else:
            event_type = AuditEventType.TRADE_CLOSED
        
        # Log event
        self.log_event(
            event_type,
            AuditCategory.TRADE,
            f"Trade closed: {trade.symbol} @ {exit_price} | PnL: {pnl:.2f}",
            symbol=trade.symbol,
            data={
                'trade_id': trade_id,
                'exit_price': exit_price,
                'exit_reason': exit_reason,
                'pnl': pnl,
                'pnl_percent': trade.pnl_percent
            },
            correlation_id=trade.entry_signal_id
        )
        
        return trade
    
    def update_trade_mae_mfe(
        self,
        trade_id: str,
        current_price: float
    ) -> None:
        """
        MAE/MFE güncelle
        
        Args:
            trade_id: Trade ID
            current_price: Güncel fiyat
        """
        if trade_id not in self._trades:
            return
        
        trade = self._trades[trade_id]
        
        if trade.direction == "LONG":
            excursion = current_price - trade.entry_price
        else:
            excursion = trade.entry_price - current_price
        
        if excursion > 0:
            trade.mfe = max(trade.mfe, excursion)
        else:
            trade.mae = max(trade.mae, abs(excursion))
    
    # ═══════════════════════════════════════════════════════════════════════════
    # SIGNAL AUDIT
    # ═══════════════════════════════════════════════════════════════════════════
    
    def log_signal_generated(
        self,
        signal_id: str,
        symbol: str,
        direction: str,
        scores: Dict[str, float],
        confidence: float,
        generated_by: str = "SignalGenerator"
    ) -> SignalAudit:
        """
        Sinyal üretimini logla
        
        Args:
            signal_id: Sinyal ID
            symbol: Hisse kodu
            direction: Yön
            scores: Motor skorları
            confidence: Güven skoru
            generated_by: Üreten engine
            
        Returns:
            SignalAudit
        """
        signal = SignalAudit(
            signal_id=signal_id,
            symbol=symbol,
            direction=direction,
            generated_at=datetime.now(),
            generated_by=generated_by,
            smc_score=scores.get('smc', 0),
            orderflow_score=scores.get('orderflow', 0),
            alpha_score=scores.get('alpha', 0),
            confluence_score=scores.get('confluence', 0),
            confidence=confidence,
            status="PENDING"
        )
        
        self._signals[signal_id] = signal
        
        self.log_event(
            AuditEventType.SIGNAL_GENERATED,
            AuditCategory.SIGNAL,
            f"Signal generated: {symbol} {direction} (confidence: {confidence:.1f}%)",
            symbol=symbol,
            data={
                'signal_id': signal_id,
                'scores': scores,
                'confidence': confidence
            }
        )
        
        return signal
    
    def log_signal_status_change(
        self,
        signal_id: str,
        new_status: str,
        reason: str = None,
        trade_id: str = None
    ) -> None:
        """
        Sinyal durumu değişikliğini logla
        
        Args:
            signal_id: Sinyal ID
            new_status: Yeni durum
            reason: Neden
            trade_id: İlişkili trade ID
        """
        if signal_id not in self._signals:
            logger.warning(f"Signal not found: {signal_id}")
            return
        
        signal = self._signals[signal_id]
        old_status = signal.status
        signal.status = new_status
        signal.status_reason = reason
        
        if trade_id:
            signal.trade_id = trade_id
            signal.executed = True
            signal.execution_time = datetime.now()
        
        # Determine event type
        if new_status == "APPROVED":
            event_type = AuditEventType.SIGNAL_APPROVED
        elif new_status == "REJECTED":
            event_type = AuditEventType.SIGNAL_REJECTED
        elif new_status == "EXPIRED":
            event_type = AuditEventType.SIGNAL_EXPIRED
        else:
            event_type = AuditEventType.SIGNAL_GENERATED
        
        self.log_event(
            event_type,
            AuditCategory.SIGNAL,
            f"Signal status changed: {signal.symbol} {old_status} -> {new_status}",
            symbol=signal.symbol,
            data={
                'signal_id': signal_id,
                'old_status': old_status,
                'new_status': new_status,
                'reason': reason,
                'trade_id': trade_id
            }
        )
    
    # ═══════════════════════════════════════════════════════════════════════════
    # RISK EVENTS
    # ═══════════════════════════════════════════════════════════════════════════
    
    def log_risk_warning(
        self,
        warning_type: str,
        current_value: float,
        threshold: float,
        message: str,
        symbol: str = None
    ) -> AuditEvent:
        """
        Risk uyarısı logla
        
        Args:
            warning_type: Uyarı türü
            current_value: Güncel değer
            threshold: Eşik değer
            message: Mesaj
            symbol: İlgili hisse
            
        Returns:
            AuditEvent
        """
        severity = AuditSeverity.CRITICAL if current_value > threshold * 1.2 else AuditSeverity.WARNING
        event_type = AuditEventType.RISK_LIMIT_BREACH if severity == AuditSeverity.CRITICAL else AuditEventType.RISK_LIMIT_WARNING
        
        return self.log_event(
            event_type,
            AuditCategory.RISK,
            message,
            severity=severity,
            symbol=symbol,
            data={
                'warning_type': warning_type,
                'current_value': current_value,
                'threshold': threshold
            }
        )
    
    # ═══════════════════════════════════════════════════════════════════════════
    # QUERIES
    # ═══════════════════════════════════════════════════════════════════════════
    
    def get_events(
        self,
        event_type: AuditEventType = None,
        category: AuditCategory = None,
        severity: AuditSeverity = None,
        symbol: str = None,
        start_time: datetime = None,
        end_time: datetime = None,
        limit: int = 100
    ) -> List[AuditEvent]:
        """
        Olayları filtrele
        
        Args:
            Filtre parametreleri
            
        Returns:
            AuditEvent listesi
        """
        events = self._events.copy()
        
        if event_type:
            events = [e for e in events if e.event_type == event_type]
        
        if category:
            events = [e for e in events if e.category == category]
        
        if severity:
            events = [e for e in events if e.severity == severity]
        
        if symbol:
            events = [e for e in events if e.symbol == symbol]
        
        if start_time:
            events = [e for e in events if e.timestamp >= start_time]
        
        if end_time:
            events = [e for e in events if e.timestamp <= end_time]
        
        return events[-limit:]
    
    def get_trade_history(
        self,
        symbol: str = None,
        direction: str = None,
        start_time: datetime = None,
        end_time: datetime = None
    ) -> List[TradeAudit]:
        """
        Trade geçmişini getir
        
        Args:
            Filtre parametreleri
            
        Returns:
            TradeAudit listesi
        """
        trades = list(self._trades.values())
        
        if symbol:
            trades = [t for t in trades if t.symbol == symbol]
        
        if direction:
            trades = [t for t in trades if t.direction == direction]
        
        if start_time:
            trades = [t for t in trades if t.entry_time >= start_time]
        
        if end_time:
            trades = [t for t in trades if t.entry_time <= end_time]
        
        return sorted(trades, key=lambda t: t.entry_time, reverse=True)
    
    def get_signal_history(
        self,
        symbol: str = None,
        status: str = None,
        limit: int = 100
    ) -> List[SignalAudit]:
        """
        Sinyal geçmişini getir
        
        Args:
            Filtre parametreleri
            
        Returns:
            SignalAudit listesi
        """
        signals = list(self._signals.values())
        
        if symbol:
            signals = [s for s in signals if s.symbol == symbol]
        
        if status:
            signals = [s for s in signals if s.status == status]
        
        signals = sorted(signals, key=lambda s: s.generated_at, reverse=True)
        
        return signals[:limit]
    
    # ═══════════════════════════════════════════════════════════════════════════
    # PERFORMANCE ATTRIBUTION
    # ═══════════════════════════════════════════════════════════════════════════
    
    def calculate_performance_audit(
        self,
        start_date: datetime = None,
        end_date: datetime = None
    ) -> PerformanceAudit:
        """
        Performans audit raporu
        
        Args:
            start_date: Başlangıç tarihi
            end_date: Bitiş tarihi
            
        Returns:
            PerformanceAudit
        """
        if start_date is None:
            start_date = datetime.now() - timedelta(days=30)
        if end_date is None:
            end_date = datetime.now()
        
        # Filter trades
        trades = [
            t for t in self._trades.values()
            if t.entry_time >= start_date and t.entry_time <= end_date and t.exit_time is not None
        ]
        
        signals = [
            s for s in self._signals.values()
            if s.generated_at >= start_date and s.generated_at <= end_date
        ]
        
        if not trades:
            return PerformanceAudit(
                period_start=start_date,
                period_end=end_date,
                total_signals=len(signals),
                executed_signals=0,
                winning_trades=0,
                losing_trades=0,
                total_pnl=0,
                gross_profit=0,
                gross_loss=0,
                win_rate=0,
                profit_factor=0,
                avg_win=0,
                avg_loss=0,
                expectancy=0,
                max_drawdown=0,
                avg_risk_per_trade=0,
                sharpe_ratio=0
            )
        
        # Calculate metrics
        winning = [t for t in trades if t.pnl > 0]
        losing = [t for t in trades if t.pnl <= 0]
        
        total_pnl = sum(t.pnl for t in trades)
        gross_profit = sum(t.pnl for t in winning)
        gross_loss = abs(sum(t.pnl for t in losing))
        
        win_rate = len(winning) / len(trades) * 100 if trades else 0
        profit_factor = gross_profit / gross_loss if gross_loss > 0 else 0
        
        avg_win = gross_profit / len(winning) if winning else 0
        avg_loss = gross_loss / len(losing) if losing else 0
        
        expectancy = (win_rate / 100 * avg_win) - ((1 - win_rate / 100) * avg_loss)
        
        avg_risk = sum(t.risk_amount for t in trades) / len(trades) if trades else 0
        
        # PnL by symbol
        pnl_by_symbol: Dict[str, float] = {}
        for t in trades:
            pnl_by_symbol[t.symbol] = pnl_by_symbol.get(t.symbol, 0) + t.pnl
        
        # PnL by direction
        pnl_by_direction = {
            'LONG': sum(t.pnl for t in trades if t.direction == 'LONG'),
            'SHORT': sum(t.pnl for t in trades if t.direction == 'SHORT')
        }
        
        return PerformanceAudit(
            period_start=start_date,
            period_end=end_date,
            total_signals=len(signals),
            executed_signals=len([s for s in signals if s.executed]),
            winning_trades=len(winning),
            losing_trades=len(losing),
            total_pnl=total_pnl,
            gross_profit=gross_profit,
            gross_loss=gross_loss,
            win_rate=win_rate,
            profit_factor=profit_factor,
            avg_win=avg_win,
            avg_loss=avg_loss,
            expectancy=expectancy,
            max_drawdown=0,  # Requires equity curve
            avg_risk_per_trade=avg_risk,
            sharpe_ratio=0,  # Requires daily returns
            pnl_by_symbol=pnl_by_symbol,
            pnl_by_direction=pnl_by_direction
        )
    
    # ═══════════════════════════════════════════════════════════════════════════
    # SUMMARY & REPORTS
    # ═══════════════════════════════════════════════════════════════════════════
    
    def get_summary(
        self,
        period: str = "today"
    ) -> AuditSummary:
        """
        Audit özeti
        
        Args:
            period: "today", "week", "month"
            
        Returns:
            AuditSummary
        """
        now = datetime.now()
        
        if period == "today":
            start = now.replace(hour=0, minute=0, second=0, microsecond=0)
        elif period == "week":
            start = now - timedelta(days=7)
        elif period == "month":
            start = now - timedelta(days=30)
        else:
            start = now - timedelta(days=1)
        
        events = self.get_events(start_time=start, limit=10000)
        
        # Counts
        events_by_type = {}
        events_by_severity = {}
        events_by_category = {}
        
        for e in events:
            events_by_type[e.event_type.value] = events_by_type.get(e.event_type.value, 0) + 1
            events_by_severity[e.severity.value] = events_by_severity.get(e.severity.value, 0) + 1
            events_by_category[e.category.value] = events_by_category.get(e.category.value, 0) + 1
        
        # Trades & signals
        trades = self.get_trade_history(start_time=start)
        signals = self.get_signal_history(limit=10000)
        signals = [s for s in signals if s.generated_at >= start]
        
        signal_to_trade = len(trades) / len(signals) if signals else 0
        
        # Notable events
        notable = [e for e in events if e.severity in [AuditSeverity.WARNING, AuditSeverity.ERROR, AuditSeverity.CRITICAL]]
        notable = sorted(notable, key=lambda e: e.timestamp, reverse=True)[:10]
        
        return AuditSummary(
            period=period,
            start_date=start,
            end_date=now,
            total_events=len(events),
            events_by_type=events_by_type,
            events_by_severity=events_by_severity,
            events_by_category=events_by_category,
            total_trades=len(trades),
            total_signals=len(signals),
            signal_to_trade_ratio=signal_to_trade,
            error_count=events_by_severity.get('ERROR', 0),
            critical_count=events_by_severity.get('CRITICAL', 0),
            notable_events=notable
        )
    
    # ═══════════════════════════════════════════════════════════════════════════
    # PERSISTENCE
    # ═══════════════════════════════════════════════════════════════════════════
    
    def _persist_event(self, event: AuditEvent) -> None:
        """Single event'i dosyaya yaz"""
        filename = f"audit_{datetime.now().strftime('%Y%m%d')}.jsonl"
        filepath = self.log_dir / filename
        
        with open(filepath, 'a') as f:
            f.write(event.to_json() + '\n')
    
    def _persist_old_events(self) -> None:
        """Eski event'leri dosyaya taşı"""
        if len(self._events) <= self.max_events // 2:
            return
        
        # En eski yarısını persist et
        to_persist = self._events[:len(self._events) // 2]
        self._events = self._events[len(self._events) // 2:]
        
        filename = f"audit_{datetime.now().strftime('%Y%m%d')}.jsonl"
        filepath = self.log_dir / filename
        
        with open(filepath, 'a') as f:
            for event in to_persist:
                f.write(event.to_json() + '\n')
    
    def export_to_csv(
        self,
        filepath: Path,
        event_type: AuditEventType = None
    ) -> None:
        """
        CSV'ye export et
        
        Args:
            filepath: Dosya yolu
            event_type: Olay türü filtresi
        """
        events = self.get_events(event_type=event_type, limit=100000)
        
        df = pd.DataFrame([e.to_dict() for e in events])
        df.to_csv(filepath, index=False)
        
        logger.info(f"Exported {len(events)} events to {filepath}")


# ═══════════════════════════════════════════════════════════════════════════════
# SINGLETON INSTANCE
# ═══════════════════════════════════════════════════════════════════════════════

_audit_engine: Optional[AuditEngine] = None


def get_audit_engine() -> AuditEngine:
    """Global audit engine instance"""
    global _audit_engine
    if _audit_engine is None:
        _audit_engine = AuditEngine()
    return _audit_engine


# ═══════════════════════════════════════════════════════════════════════════════
# TEST
# ═══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    print("Audit Engine v4.2 - Test")
    print("=" * 60)
    
    engine = AuditEngine()
    
    # Signal log
    signal = engine.log_signal_generated(
        signal_id="SIG_001",
        symbol="THYAO",
        direction="LONG",
        scores={'smc': 75, 'orderflow': 68, 'alpha': 72},
        confidence=72.5
    )
    print(f"✅ Signal logged: {signal.signal_id}")
    
    # Trade log
    trade = engine.log_trade_opened(
        trade_id="TRD_001",
        symbol="THYAO",
        direction="LONG",
        entry_price=145.50,
        position_size=100,
        stop_loss=140.00,
        take_profits=[155.00, 160.00, 165.00],
        signal_id="SIG_001",
        entry_reason="SMC + OrderFlow confluence",
        scores={'smc': 75, 'orderflow': 68, 'alpha': 72, 'confidence': 72.5}
    )
    print(f"✅ Trade opened: {trade.trade_id}")
    
    # Close trade
    engine.log_trade_closed(
        trade_id="TRD_001",
        exit_price=155.00,
        exit_reason="TP1 HIT"
    )
    print(f"✅ Trade closed")
    
    # Risk warning
    engine.log_risk_warning(
        warning_type="PORTFOLIO_HEAT",
        current_value=8.5,
        threshold=10.0,
        message="Portfolio heat approaching limit"
    )
    print(f"✅ Risk warning logged")
    
    # Summary
    summary = engine.get_summary("today")
    print(f"\n📊 AUDIT ÖZET")
    print("=" * 60)
    print(f"📈 Toplam Olay: {summary.total_events}")
    print(f"📊 Sinyaller: {summary.total_signals}")
    print(f"💼 Trade'ler: {summary.total_trades}")
    print(f"⚠️ Hatalar: {summary.error_count}")
    
    # Performance
    perf = engine.calculate_performance_audit()
    print(f"\n📈 PERFORMANS")
    print(f"Win Rate: {perf.win_rate:.1f}%")
    print(f"Profit Factor: {perf.profit_factor:.2f}")
    print(f"Expectancy: {perf.expectancy:.2f}")
