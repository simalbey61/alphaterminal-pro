"""
AlphaTerminal Pro - Retirement Manager
=====================================

Düşük performanslı stratejilerin yönetimi.

Author: AlphaTerminal Team
Version: 1.0.0
"""

from __future__ import annotations

import logging
from typing import Optional, List, Dict, Any
from datetime import datetime, timedelta
from dataclasses import dataclass, field
from enum import Enum

from app.ai_strategy.constants import (
    StrategyLifecycle,
    RetirementThresholds,
    PerformanceAlert,
)

logger = logging.getLogger(__name__)


class RetirementReason(str, Enum):
    """Emeklilik sebebi."""
    LOW_WIN_RATE = "low_win_rate"
    SHARPE_DEGRADATION = "sharpe_degradation"
    CONSECUTIVE_LOSSES = "consecutive_losses"
    MAX_DRAWDOWN_EXCEEDED = "max_drawdown_exceeded"
    REGIME_ENDED = "regime_ended"
    MANUAL = "manual"
    OBSOLETE = "obsolete"
    CORRELATION_REDUNDANCY = "correlation_redundancy"


@dataclass
class RetirementRecommendation:
    """Emeklilik önerisi."""
    strategy_id: str
    strategy_name: str
    current_lifecycle: StrategyLifecycle
    recommended_action: StrategyLifecycle
    reason: RetirementReason
    severity: str  # immediate, warning, monitor
    metrics: Dict[str, float]
    can_revive: bool
    revival_conditions: List[str]
    evaluated_at: datetime = field(default_factory=datetime.utcnow)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "strategy_id": self.strategy_id,
            "strategy_name": self.strategy_name,
            "current_lifecycle": self.current_lifecycle.value,
            "recommended_action": self.recommended_action.value,
            "reason": self.reason.value,
            "severity": self.severity,
            "metrics": self.metrics,
            "can_revive": self.can_revive,
            "revival_conditions": self.revival_conditions,
            "evaluated_at": self.evaluated_at.isoformat(),
        }


@dataclass
class StrategyHealth:
    """Strateji sağlık durumu."""
    strategy_id: str
    lifecycle: StrategyLifecycle
    
    # Performance metrics
    recent_win_rate: float
    recent_sharpe: float
    recent_pnl: float
    consecutive_losses: int
    current_drawdown: float
    
    # Backtest comparison
    expected_win_rate: float
    expected_sharpe: float
    expected_max_dd: float
    
    # Deviations
    win_rate_deviation: float
    sharpe_deviation: float
    
    # Regime info
    target_regime: Optional[str] = None
    current_regime: Optional[str] = None
    regime_match: bool = True
    
    # Health score
    health_score: float = 100.0
    
    last_updated: datetime = field(default_factory=datetime.utcnow)


class RetirementManager:
    """
    Strateji emeklilik yönetimi.
    
    Emeklilik süreci:
    1. ACTIVE → PROBATION (uyarı)
    2. PROBATION → SANDBOX (paper trading)
    3. SANDBOX → RETIRED (arşiv)
    
    Canlandırma süreci:
    - RETIRED → SANDBOX → PROBATION → ACTIVE
    (Regime dönerse ve backtest hala geçerli ise)
    
    Example:
        ```python
        manager = RetirementManager()
        
        # Sağlık değerlendirmesi
        health = manager.evaluate_health(
            strategy_id="abc123",
            recent_trades=trades,
            backtest_metrics=bt_metrics
        )
        
        # Emeklilik önerisi
        rec = manager.get_retirement_recommendation(health)
        
        if rec.severity == "immediate":
            strategy.lifecycle = rec.recommended_action
        ```
    """
    
    def __init__(
        self,
        thresholds: Optional[RetirementThresholds] = None,
        probation_days: int = 14,
        sandbox_days: int = 30,
    ):
        """
        Initialize Retirement Manager.
        
        Args:
            thresholds: Emeklilik eşikleri
            probation_days: Probation süresi (gün)
            sandbox_days: Sandbox süresi (gün)
        """
        self.thresholds = thresholds or RetirementThresholds()
        self.probation_days = probation_days
        self.sandbox_days = sandbox_days
        
        # Strategy tracking
        self._strategy_health: Dict[str, StrategyHealth] = {}
        self._lifecycle_history: Dict[str, List[Dict]] = {}
    
    def evaluate_health(
        self,
        strategy_id: str,
        strategy_name: str,
        current_lifecycle: StrategyLifecycle,
        recent_trades: List[Dict],
        expected_win_rate: float,
        expected_sharpe: float,
        expected_max_dd: float,
        target_regime: Optional[str] = None,
        current_regime: Optional[str] = None,
    ) -> StrategyHealth:
        """
        Strateji sağlık değerlendirmesi.
        
        Args:
            strategy_id: Strateji ID
            strategy_name: Strateji adı
            current_lifecycle: Mevcut lifecycle
            recent_trades: Son trade'ler (monitoring window)
            expected_win_rate: Beklenen win rate
            expected_sharpe: Beklenen Sharpe
            expected_max_dd: Beklenen max drawdown
            target_regime: Hedef rejim
            current_regime: Mevcut rejim
            
        Returns:
            StrategyHealth: Sağlık durumu
        """
        # Calculate recent metrics
        if not recent_trades:
            return self._create_empty_health(
                strategy_id, current_lifecycle, expected_win_rate, expected_sharpe, expected_max_dd
            )
        
        wins = sum(1 for t in recent_trades if t.get("is_win", False))
        recent_win_rate = wins / len(recent_trades)
        
        pnls = [t.get("pnl", 0) for t in recent_trades]
        recent_pnl = sum(pnls)
        
        # Sharpe (simplified)
        import numpy as np
        if len(pnls) > 1 and np.std(pnls) > 0:
            recent_sharpe = np.mean(pnls) / np.std(pnls) * np.sqrt(252)
        else:
            recent_sharpe = 0
        
        # Consecutive losses
        consecutive_losses = 0
        for trade in reversed(recent_trades):
            if trade.get("is_win", False):
                break
            consecutive_losses += 1
        
        # Drawdown
        cumulative = np.cumsum(pnls)
        running_max = np.maximum.accumulate(cumulative)
        drawdowns = running_max - cumulative
        current_drawdown = drawdowns[-1] / (running_max[-1] + 1) if running_max[-1] > 0 else 0
        
        # Deviations
        win_rate_dev = (expected_win_rate - recent_win_rate) / max(expected_win_rate, 0.01)
        sharpe_dev = (expected_sharpe - recent_sharpe) / max(expected_sharpe, 0.01) if expected_sharpe > 0 else 0
        
        # Regime match
        regime_match = target_regime == current_regime if target_regime and current_regime else True
        
        # Health score
        health_score = self._calculate_health_score(
            recent_win_rate, expected_win_rate,
            recent_sharpe, expected_sharpe,
            current_drawdown, expected_max_dd,
            consecutive_losses, regime_match
        )
        
        health = StrategyHealth(
            strategy_id=strategy_id,
            lifecycle=current_lifecycle,
            recent_win_rate=recent_win_rate,
            recent_sharpe=recent_sharpe,
            recent_pnl=recent_pnl,
            consecutive_losses=consecutive_losses,
            current_drawdown=current_drawdown,
            expected_win_rate=expected_win_rate,
            expected_sharpe=expected_sharpe,
            expected_max_dd=expected_max_dd,
            win_rate_deviation=win_rate_dev,
            sharpe_deviation=sharpe_dev,
            target_regime=target_regime,
            current_regime=current_regime,
            regime_match=regime_match,
            health_score=health_score,
        )
        
        self._strategy_health[strategy_id] = health
        
        return health
    
    def get_retirement_recommendation(
        self,
        health: StrategyHealth,
    ) -> Optional[RetirementRecommendation]:
        """
        Emeklilik önerisi al.
        
        Args:
            health: Strateji sağlık durumu
            
        Returns:
            Optional[RetirementRecommendation]: Öneri (None = sağlıklı)
        """
        # Check triggers
        trigger_reason = None
        severity = None
        
        # 1. Consecutive losses
        if health.consecutive_losses >= self.thresholds.max_consecutive_losses:
            trigger_reason = RetirementReason.CONSECUTIVE_LOSSES
            severity = "immediate"
        
        # 2. Low win rate
        elif health.recent_win_rate < self.thresholds.min_win_rate_10_trades:
            trigger_reason = RetirementReason.LOW_WIN_RATE
            severity = "warning"
        
        # 3. Sharpe degradation
        elif health.sharpe_deviation > (1 - self.thresholds.sharpe_degradation_pct):
            trigger_reason = RetirementReason.SHARPE_DEGRADATION
            severity = "warning"
        
        # 4. Max drawdown exceeded
        elif health.current_drawdown > health.expected_max_dd:
            trigger_reason = RetirementReason.MAX_DRAWDOWN_EXCEEDED
            severity = "immediate"
        
        # 5. Regime mismatch
        elif not health.regime_match:
            trigger_reason = RetirementReason.REGIME_ENDED
            severity = "monitor"
        
        if not trigger_reason:
            return None
        
        # Determine action based on current lifecycle
        recommended_action = self._determine_action(health.lifecycle, severity)
        
        # Revival conditions
        revival_conditions = self._get_revival_conditions(trigger_reason)
        can_revive = trigger_reason in [
            RetirementReason.REGIME_ENDED,
            RetirementReason.LOW_WIN_RATE,
            RetirementReason.SHARPE_DEGRADATION,
        ]
        
        return RetirementRecommendation(
            strategy_id=health.strategy_id,
            strategy_name=f"Strategy {health.strategy_id}",
            current_lifecycle=health.lifecycle,
            recommended_action=recommended_action,
            reason=trigger_reason,
            severity=severity,
            metrics={
                "recent_win_rate": health.recent_win_rate,
                "recent_sharpe": health.recent_sharpe,
                "consecutive_losses": health.consecutive_losses,
                "current_drawdown": health.current_drawdown,
                "health_score": health.health_score,
            },
            can_revive=can_revive,
            revival_conditions=revival_conditions,
        )
    
    def evaluate_for_revival(
        self,
        strategy_id: str,
        current_regime: str,
        latest_backtest_valid: bool,
    ) -> bool:
        """
        Canlandırma değerlendirmesi.
        
        Args:
            strategy_id: Strateji ID
            current_regime: Mevcut rejim
            latest_backtest_valid: Son backtest geçerli mi?
            
        Returns:
            bool: Canlandırılabilir mi?
        """
        health = self._strategy_health.get(strategy_id)
        
        if not health:
            return False
        
        if health.lifecycle != StrategyLifecycle.RETIRED:
            return False
        
        # Regime match
        regime_returned = health.target_regime == current_regime
        
        # Backtest still valid
        if not latest_backtest_valid:
            return False
        
        return regime_returned
    
    def process_lifecycle_transition(
        self,
        strategy_id: str,
        from_state: StrategyLifecycle,
        to_state: StrategyLifecycle,
        reason: str,
    ) -> None:
        """
        Lifecycle geçişini kaydet.
        
        Args:
            strategy_id: Strateji ID
            from_state: Önceki durum
            to_state: Yeni durum
            reason: Geçiş sebebi
        """
        if strategy_id not in self._lifecycle_history:
            self._lifecycle_history[strategy_id] = []
        
        self._lifecycle_history[strategy_id].append({
            "from": from_state.value,
            "to": to_state.value,
            "reason": reason,
            "timestamp": datetime.utcnow().isoformat(),
        })
        
        logger.info(f"Strategy {strategy_id}: {from_state.value} → {to_state.value} ({reason})")
    
    def get_lifecycle_history(self, strategy_id: str) -> List[Dict]:
        """Lifecycle geçmişini al."""
        return self._lifecycle_history.get(strategy_id, [])
    
    def get_candidates_for_retirement(self) -> List[str]:
        """Emeklilik adaylarını al."""
        candidates = []
        
        for strategy_id, health in self._strategy_health.items():
            rec = self.get_retirement_recommendation(health)
            if rec and rec.severity in ["immediate", "warning"]:
                candidates.append(strategy_id)
        
        return candidates
    
    def get_candidates_for_revival(self, current_regime: str) -> List[str]:
        """Canlandırma adaylarını al."""
        candidates = []
        
        for strategy_id, health in self._strategy_health.items():
            if health.lifecycle == StrategyLifecycle.RETIRED:
                if health.target_regime == current_regime:
                    candidates.append(strategy_id)
        
        return candidates
    
    def _determine_action(
        self,
        current_lifecycle: StrategyLifecycle,
        severity: str,
    ) -> StrategyLifecycle:
        """Önerilen aksiyonu belirle."""
        if severity == "immediate":
            if current_lifecycle == StrategyLifecycle.ACTIVE:
                return StrategyLifecycle.SANDBOX
            elif current_lifecycle == StrategyLifecycle.PROBATION:
                return StrategyLifecycle.SANDBOX
            elif current_lifecycle == StrategyLifecycle.SANDBOX:
                return StrategyLifecycle.RETIRED
            else:
                return StrategyLifecycle.RETIRED
        
        elif severity == "warning":
            if current_lifecycle == StrategyLifecycle.ACTIVE:
                return StrategyLifecycle.PROBATION
            elif current_lifecycle == StrategyLifecycle.PROBATION:
                return StrategyLifecycle.SANDBOX
            else:
                return current_lifecycle
        
        else:  # monitor
            return current_lifecycle
    
    def _get_revival_conditions(self, reason: RetirementReason) -> List[str]:
        """Canlandırma koşullarını al."""
        conditions = []
        
        if reason == RetirementReason.REGIME_ENDED:
            conditions.append("Target regime returns")
            conditions.append("Backtest validates on new data")
        
        elif reason == RetirementReason.LOW_WIN_RATE:
            conditions.append("Market conditions change")
            conditions.append("New backtest shows improved win rate")
        
        elif reason == RetirementReason.SHARPE_DEGRADATION:
            conditions.append("Volatility normalizes")
            conditions.append("New backtest shows improved Sharpe")
        
        elif reason in [RetirementReason.MAX_DRAWDOWN_EXCEEDED, RetirementReason.CONSECUTIVE_LOSSES]:
            conditions.append("Complete strategy review required")
            conditions.append("Parameter optimization needed")
        
        return conditions
    
    def _calculate_health_score(
        self,
        recent_wr: float, expected_wr: float,
        recent_sharpe: float, expected_sharpe: float,
        current_dd: float, expected_dd: float,
        consecutive_losses: int, regime_match: bool,
    ) -> float:
        """Sağlık skoru hesapla (0-100)."""
        score = 100.0
        
        # Win rate impact
        if expected_wr > 0:
            wr_ratio = recent_wr / expected_wr
            score -= max(0, (1 - wr_ratio) * 30)
        
        # Sharpe impact
        if expected_sharpe > 0:
            sharpe_ratio = recent_sharpe / expected_sharpe
            score -= max(0, (1 - sharpe_ratio) * 25)
        
        # Drawdown impact
        if expected_dd > 0:
            dd_ratio = current_dd / expected_dd
            score -= min(25, dd_ratio * 25)
        
        # Consecutive losses
        score -= consecutive_losses * 3
        
        # Regime mismatch
        if not regime_match:
            score -= 10
        
        return max(0, min(100, score))
    
    def _create_empty_health(
        self,
        strategy_id: str,
        lifecycle: StrategyLifecycle,
        expected_wr: float,
        expected_sharpe: float,
        expected_dd: float,
    ) -> StrategyHealth:
        """Boş sağlık kaydı oluştur."""
        return StrategyHealth(
            strategy_id=strategy_id,
            lifecycle=lifecycle,
            recent_win_rate=0,
            recent_sharpe=0,
            recent_pnl=0,
            consecutive_losses=0,
            current_drawdown=0,
            expected_win_rate=expected_wr,
            expected_sharpe=expected_sharpe,
            expected_max_dd=expected_dd,
            win_rate_deviation=1.0,
            sharpe_deviation=1.0,
            health_score=50.0,
        )
