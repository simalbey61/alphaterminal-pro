"""
AlphaTerminal Pro - Performance Monitor
=======================================

Canlı strateji performans takibi ve alert sistemi.

Author: AlphaTerminal Team
Version: 1.0.0
"""

from __future__ import annotations

import logging
from typing import Optional, List, Dict, Any, Callable
from datetime import datetime, timedelta
from dataclasses import dataclass, field
from enum import Enum
from collections import deque

import numpy as np

from app.ai_strategy.constants import (
    RetirementThresholds,
    PerformanceAlert,
    AlertSeverity,
    StrategyLifecycle,
)

logger = logging.getLogger(__name__)


class MonitorAction(str, Enum):
    """Monitor aksiyonu."""
    NONE = "none"
    WARNING = "warning"
    PAUSE = "pause"
    SANDBOX = "sandbox"
    RETIRE = "retire"


@dataclass
class PerformanceSnapshot:
    """Performans anlık görüntüsü."""
    timestamp: datetime
    total_trades: int
    wins: int
    losses: int
    win_rate: float
    total_pnl: float
    current_drawdown: float
    max_drawdown: float
    sharpe_ratio: float
    consecutive_losses: int
    last_trade_pnl: float
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "timestamp": self.timestamp.isoformat(),
            "total_trades": self.total_trades,
            "wins": self.wins,
            "losses": self.losses,
            "win_rate": self.win_rate,
            "total_pnl": self.total_pnl,
            "current_drawdown": self.current_drawdown,
            "max_drawdown": self.max_drawdown,
            "sharpe_ratio": self.sharpe_ratio,
            "consecutive_losses": self.consecutive_losses,
            "last_trade_pnl": self.last_trade_pnl,
        }


@dataclass
class Alert:
    """Performans alerti."""
    alert_type: PerformanceAlert
    severity: AlertSeverity
    strategy_id: str
    message: str
    details: Dict[str, Any]
    suggested_action: MonitorAction
    created_at: datetime = field(default_factory=datetime.utcnow)
    acknowledged: bool = False
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "alert_type": self.alert_type.value,
            "severity": self.severity.value,
            "strategy_id": self.strategy_id,
            "message": self.message,
            "details": self.details,
            "suggested_action": self.suggested_action.value,
            "created_at": self.created_at.isoformat(),
            "acknowledged": self.acknowledged,
        }


@dataclass
class MonitoringResult:
    """Monitoring sonucu."""
    strategy_id: str
    current_status: StrategyLifecycle
    suggested_status: StrategyLifecycle
    action_required: MonitorAction
    
    alerts: List[Alert]
    performance_snapshot: PerformanceSnapshot
    
    is_healthy: bool
    health_score: float  # 0-100
    
    checked_at: datetime = field(default_factory=datetime.utcnow)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "strategy_id": self.strategy_id,
            "current_status": self.current_status.value,
            "suggested_status": self.suggested_status.value,
            "action_required": self.action_required.value,
            "alerts": [a.to_dict() for a in self.alerts],
            "performance_snapshot": self.performance_snapshot.to_dict(),
            "is_healthy": self.is_healthy,
            "health_score": self.health_score,
            "checked_at": self.checked_at.isoformat(),
        }


class PerformanceMonitor:
    """
    Canlı strateji performans takibi.
    
    Takip edilen metrikler:
    - Consecutive losses
    - Drawdown (current vs expected)
    - Win rate deviation
    - Sharpe degradation
    - Regime mismatch
    
    Aksiyonlar:
    - WARNING: Log ve bildirim
    - PAUSE: Geçici durdurma
    - SANDBOX: Paper trading'e geri çekme
    - RETIRE: Kalıcı emeklilik
    
    Example:
        ```python
        monitor = PerformanceMonitor()
        
        # Trade sonrası kontrol
        result = monitor.check_strategy(
            strategy_id="abc123",
            trades=recent_trades,
            expected_metrics=backtest_metrics
        )
        
        if result.action_required != MonitorAction.NONE:
            handle_action(result)
        ```
    """
    
    def __init__(
        self,
        thresholds: Optional[RetirementThresholds] = None,
        window_size: int = 10,
        alert_callback: Optional[Callable[[Alert], None]] = None,
    ):
        """
        Initialize Performance Monitor.
        
        Args:
            thresholds: Emeklilik eşikleri
            window_size: Monitoring pencere boyutu
            alert_callback: Alert callback fonksiyonu
        """
        self.thresholds = thresholds or RetirementThresholds()
        self.window_size = window_size
        self.alert_callback = alert_callback
        
        # Trade history per strategy
        self._trade_history: Dict[str, deque] = {}
        self._snapshots: Dict[str, List[PerformanceSnapshot]] = {}
    
    def record_trade(
        self,
        strategy_id: str,
        pnl: float,
        is_win: bool,
        trade_data: Optional[Dict] = None,
    ) -> None:
        """
        Trade sonucunu kaydet.
        
        Args:
            strategy_id: Strateji ID
            pnl: P&L
            is_win: Kazanç mı?
            trade_data: Ek trade verisi
        """
        if strategy_id not in self._trade_history:
            self._trade_history[strategy_id] = deque(maxlen=100)
        
        self._trade_history[strategy_id].append({
            "pnl": pnl,
            "is_win": is_win,
            "timestamp": datetime.utcnow(),
            "data": trade_data or {},
        })
    
    def check_strategy(
        self,
        strategy_id: str,
        current_status: StrategyLifecycle,
        expected_win_rate: float,
        expected_sharpe: float,
        expected_max_dd: float,
        target_regime: Optional[str] = None,
        current_regime: Optional[str] = None,
    ) -> MonitoringResult:
        """
        Strateji sağlık kontrolü.
        
        Args:
            strategy_id: Strateji ID
            current_status: Mevcut durum
            expected_win_rate: Beklenen win rate
            expected_sharpe: Beklenen Sharpe
            expected_max_dd: Beklenen max drawdown
            target_regime: Hedef rejim
            current_regime: Mevcut rejim
            
        Returns:
            MonitoringResult: Monitoring sonucu
        """
        alerts = []
        action = MonitorAction.NONE
        suggested_status = current_status
        
        # Trade history al
        trades = list(self._trade_history.get(strategy_id, []))
        
        if len(trades) < 3:
            # Yeterli veri yok
            snapshot = self._create_empty_snapshot()
            return MonitoringResult(
                strategy_id=strategy_id,
                current_status=current_status,
                suggested_status=current_status,
                action_required=MonitorAction.NONE,
                alerts=[],
                performance_snapshot=snapshot,
                is_healthy=True,
                health_score=100.0,
            )
        
        # Performance snapshot
        snapshot = self._calculate_snapshot(trades)
        
        # =====================================================================
        # CHECK 1: Consecutive Losses
        # =====================================================================
        recent_trades = trades[-self.thresholds.monitoring_window:]
        consecutive_losses = self._count_consecutive_losses(recent_trades)
        
        if consecutive_losses >= self.thresholds.max_consecutive_losses:
            alert = Alert(
                alert_type=PerformanceAlert.CONSECUTIVE_LOSSES,
                severity=AlertSeverity.CRITICAL,
                strategy_id=strategy_id,
                message=f"{consecutive_losses} consecutive losses",
                details={"count": consecutive_losses},
                suggested_action=MonitorAction.SANDBOX,
            )
            alerts.append(alert)
            action = MonitorAction.SANDBOX
            suggested_status = StrategyLifecycle.SANDBOX
        
        # =====================================================================
        # CHECK 2: Win Rate Deviation
        # =====================================================================
        if len(recent_trades) >= self.window_size:
            recent_wr = sum(1 for t in recent_trades if t["is_win"]) / len(recent_trades)
            
            if recent_wr < self.thresholds.min_win_rate_10_trades:
                alert = Alert(
                    alert_type=PerformanceAlert.WIN_RATE_DEVIATION,
                    severity=AlertSeverity.HIGH,
                    strategy_id=strategy_id,
                    message=f"Win rate dropped to {recent_wr:.1%}",
                    details={
                        "actual": recent_wr,
                        "expected": expected_win_rate,
                        "threshold": self.thresholds.min_win_rate_10_trades,
                    },
                    suggested_action=MonitorAction.PAUSE,
                )
                alerts.append(alert)
                
                if action.value < MonitorAction.PAUSE.value:
                    action = MonitorAction.PAUSE
                    suggested_status = StrategyLifecycle.PAUSED
        
        # =====================================================================
        # CHECK 3: Drawdown
        # =====================================================================
        if snapshot.current_drawdown > expected_max_dd * 0.5:
            severity = AlertSeverity.HIGH if snapshot.current_drawdown > expected_max_dd * 0.75 else AlertSeverity.MEDIUM
            
            alert = Alert(
                alert_type=PerformanceAlert.DRAWDOWN_WARNING,
                severity=severity,
                strategy_id=strategy_id,
                message=f"Drawdown at {snapshot.current_drawdown:.1%}",
                details={
                    "current": snapshot.current_drawdown,
                    "expected_max": expected_max_dd,
                    "pct_of_expected": snapshot.current_drawdown / expected_max_dd,
                },
                suggested_action=MonitorAction.WARNING,
            )
            alerts.append(alert)
        
        if snapshot.current_drawdown > expected_max_dd:
            alert = Alert(
                alert_type=PerformanceAlert.DRAWDOWN_CRITICAL,
                severity=AlertSeverity.CRITICAL,
                strategy_id=strategy_id,
                message=f"Max drawdown exceeded: {snapshot.current_drawdown:.1%}",
                details={
                    "current": snapshot.current_drawdown,
                    "limit": expected_max_dd,
                },
                suggested_action=MonitorAction.RETIRE,
            )
            alerts.append(alert)
            action = MonitorAction.RETIRE
            suggested_status = StrategyLifecycle.RETIRED
        
        # =====================================================================
        # CHECK 4: Sharpe Degradation
        # =====================================================================
        if snapshot.sharpe_ratio < expected_sharpe * self.thresholds.sharpe_degradation_pct:
            alert = Alert(
                alert_type=PerformanceAlert.SHARPE_DEGRADATION,
                severity=AlertSeverity.HIGH,
                strategy_id=strategy_id,
                message=f"Sharpe degraded to {snapshot.sharpe_ratio:.2f}",
                details={
                    "actual": snapshot.sharpe_ratio,
                    "expected": expected_sharpe,
                    "degradation_pct": 1 - (snapshot.sharpe_ratio / expected_sharpe) if expected_sharpe > 0 else 1,
                },
                suggested_action=MonitorAction.SANDBOX,
            )
            alerts.append(alert)
            
            if action.value < MonitorAction.SANDBOX.value:
                action = MonitorAction.SANDBOX
                suggested_status = StrategyLifecycle.SANDBOX
        
        # =====================================================================
        # CHECK 5: Regime Mismatch
        # =====================================================================
        if target_regime and current_regime and target_regime != current_regime:
            alert = Alert(
                alert_type=PerformanceAlert.REGIME_MISMATCH,
                severity=AlertSeverity.MEDIUM,
                strategy_id=strategy_id,
                message=f"Regime changed from {target_regime} to {current_regime}",
                details={
                    "target": target_regime,
                    "current": current_regime,
                },
                suggested_action=MonitorAction.PAUSE,
            )
            alerts.append(alert)
            
            if action.value < MonitorAction.PAUSE.value:
                action = MonitorAction.PAUSE
                suggested_status = StrategyLifecycle.PAUSED
        
        # Health score
        health_score = self._calculate_health_score(
            snapshot=snapshot,
            expected_win_rate=expected_win_rate,
            expected_sharpe=expected_sharpe,
            expected_max_dd=expected_max_dd,
            alerts=alerts,
        )
        
        is_healthy = health_score >= 60 and action == MonitorAction.NONE
        
        # Trigger alert callbacks
        for alert in alerts:
            if self.alert_callback:
                try:
                    self.alert_callback(alert)
                except Exception as e:
                    logger.error(f"Alert callback error: {e}")
        
        return MonitoringResult(
            strategy_id=strategy_id,
            current_status=current_status,
            suggested_status=suggested_status,
            action_required=action,
            alerts=alerts,
            performance_snapshot=snapshot,
            is_healthy=is_healthy,
            health_score=health_score,
        )
    
    def _calculate_snapshot(self, trades: List[Dict]) -> PerformanceSnapshot:
        """Performance snapshot hesapla."""
        wins = sum(1 for t in trades if t["is_win"])
        losses = len(trades) - wins
        
        pnls = [t["pnl"] for t in trades]
        total_pnl = sum(pnls)
        
        # Win rate
        win_rate = wins / len(trades) if trades else 0
        
        # Drawdown
        cumulative = np.cumsum(pnls)
        running_max = np.maximum.accumulate(cumulative)
        drawdowns = running_max - cumulative
        current_dd = drawdowns[-1] / (running_max[-1] + 1) if running_max[-1] > 0 else 0
        max_dd = np.max(drawdowns) / (np.max(running_max) + 1) if np.max(running_max) > 0 else 0
        
        # Sharpe (simplified)
        if len(pnls) > 1 and np.std(pnls) > 0:
            sharpe = np.mean(pnls) / np.std(pnls) * np.sqrt(252)
        else:
            sharpe = 0
        
        # Consecutive losses
        consecutive = self._count_consecutive_losses(trades)
        
        return PerformanceSnapshot(
            timestamp=datetime.utcnow(),
            total_trades=len(trades),
            wins=wins,
            losses=losses,
            win_rate=win_rate,
            total_pnl=total_pnl,
            current_drawdown=current_dd,
            max_drawdown=max_dd,
            sharpe_ratio=sharpe,
            consecutive_losses=consecutive,
            last_trade_pnl=pnls[-1] if pnls else 0,
        )
    
    def _count_consecutive_losses(self, trades: List[Dict]) -> int:
        """Ardışık kayıp sayısını hesapla."""
        count = 0
        for trade in reversed(trades):
            if trade["is_win"]:
                break
            count += 1
        return count
    
    def _calculate_health_score(
        self,
        snapshot: PerformanceSnapshot,
        expected_win_rate: float,
        expected_sharpe: float,
        expected_max_dd: float,
        alerts: List[Alert],
    ) -> float:
        """Sağlık skoru hesapla (0-100)."""
        score = 100.0
        
        # Win rate impact
        if expected_win_rate > 0:
            wr_ratio = snapshot.win_rate / expected_win_rate
            score -= max(0, (1 - wr_ratio) * 30)
        
        # Sharpe impact
        if expected_sharpe > 0:
            sharpe_ratio = snapshot.sharpe_ratio / expected_sharpe
            score -= max(0, (1 - sharpe_ratio) * 20)
        
        # Drawdown impact
        if expected_max_dd > 0:
            dd_ratio = snapshot.current_drawdown / expected_max_dd
            score -= min(30, dd_ratio * 30)
        
        # Alert impact
        for alert in alerts:
            if alert.severity == AlertSeverity.CRITICAL:
                score -= 20
            elif alert.severity == AlertSeverity.HIGH:
                score -= 10
            elif alert.severity == AlertSeverity.MEDIUM:
                score -= 5
        
        return max(0, min(100, score))
    
    def _create_empty_snapshot(self) -> PerformanceSnapshot:
        """Boş snapshot oluştur."""
        return PerformanceSnapshot(
            timestamp=datetime.utcnow(),
            total_trades=0,
            wins=0,
            losses=0,
            win_rate=0,
            total_pnl=0,
            current_drawdown=0,
            max_drawdown=0,
            sharpe_ratio=0,
            consecutive_losses=0,
            last_trade_pnl=0,
        )
    
    def get_strategy_history(
        self,
        strategy_id: str,
        limit: int = 100,
    ) -> List[Dict]:
        """Strateji trade geçmişini al."""
        trades = self._trade_history.get(strategy_id, [])
        return [t for t in list(trades)[-limit:]]
    
    def clear_strategy(self, strategy_id: str) -> None:
        """Strateji verilerini temizle."""
        if strategy_id in self._trade_history:
            del self._trade_history[strategy_id]
        if strategy_id in self._snapshots:
            del self._snapshots[strategy_id]
