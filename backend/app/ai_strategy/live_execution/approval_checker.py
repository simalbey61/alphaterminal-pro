"""
AlphaTerminal Pro - Approval Checker
====================================

Strateji canlı trading'e geçiş onay sistemi.

Author: AlphaTerminal Team
Version: 1.0.0
"""

from __future__ import annotations

import logging
from typing import Optional, List, Dict, Any
from datetime import datetime
from dataclasses import dataclass, field
from enum import Enum

from app.ai_strategy.constants import (
    ApprovalDecision,
    ApprovalThresholds,
    StrategyLifecycle,
)
from app.ai_strategy.validation_engine.monte_carlo import MonteCarloResult
from app.ai_strategy.validation_engine.walk_forward import WalkForwardResult

logger = logging.getLogger(__name__)


class CriterionSeverity(str, Enum):
    """Kriter şiddeti."""
    MANDATORY = "mandatory"  # Geçilmesi zorunlu
    SOFT = "soft"            # Uyarı, ama geçilebilir
    INFO = "info"            # Bilgi amaçlı


@dataclass
class CriterionResult:
    """Kriter değerlendirme sonucu."""
    name: str
    passed: bool
    severity: CriterionSeverity
    actual_value: float
    threshold_value: float
    message: str
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "passed": self.passed,
            "severity": self.severity.value,
            "actual_value": self.actual_value,
            "threshold_value": self.threshold_value,
            "message": self.message,
        }


@dataclass
class ApprovalResult:
    """Onay sonucu."""
    decision: ApprovalDecision
    mandatory_passed: int
    mandatory_failed: int
    soft_warnings: int
    
    criteria_results: List[CriterionResult]
    
    # Özet metrikler
    overall_score: float
    risk_assessment: str
    recommendation: str
    
    # Lifecycle önerisi
    suggested_lifecycle: StrategyLifecycle
    
    evaluated_at: datetime = field(default_factory=datetime.utcnow)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "decision": self.decision.value,
            "mandatory_passed": self.mandatory_passed,
            "mandatory_failed": self.mandatory_failed,
            "soft_warnings": self.soft_warnings,
            "criteria_results": [c.to_dict() for c in self.criteria_results],
            "overall_score": self.overall_score,
            "risk_assessment": self.risk_assessment,
            "recommendation": self.recommendation,
            "suggested_lifecycle": self.suggested_lifecycle.value,
            "evaluated_at": self.evaluated_at.isoformat(),
        }
    
    @property
    def is_approved(self) -> bool:
        return self.decision == ApprovalDecision.APPROVED


class ApprovalChecker:
    """
    Strateji canlı trading onay sistemi.
    
    Kontrol edilen kriterler:
    
    MANDATORY (zorunlu):
    - Win rate >= 55%
    - Profit factor >= 1.5
    - Sharpe ratio >= 1.0
    - Max drawdown <= 15%
    - Walk-forward consistency >= 60%
    - Monte Carlo VaR(95%) >= -10%
    - Robustness score >= 0.70
    - Sample size >= 50
    
    SOFT (uyarı):
    - Expected profit > 3x spread cost
    - Average trade duration > 4 hours
    - Min trades per month >= 5
    
    Example:
        ```python
        checker = ApprovalChecker()
        
        result = checker.evaluate(
            backtest_metrics=backtest,
            walk_forward_result=wf,
            monte_carlo_result=mc
        )
        
        if result.is_approved:
            strategy.lifecycle = StrategyLifecycle.ACTIVE
        elif result.decision == ApprovalDecision.SANDBOX:
            strategy.lifecycle = StrategyLifecycle.SANDBOX
        ```
    """
    
    def __init__(self, thresholds: Optional[ApprovalThresholds] = None):
        """
        Initialize Approval Checker.
        
        Args:
            thresholds: Onay eşikleri (None = varsayılan)
        """
        self.thresholds = thresholds or ApprovalThresholds()
    
    def evaluate(
        self,
        backtest_metrics: Dict[str, Any],
        walk_forward_result: Optional[WalkForwardResult] = None,
        monte_carlo_result: Optional[MonteCarloResult] = None,
        robustness_score: Optional[float] = None,
        execution_cost_pct: float = 0.004,
    ) -> ApprovalResult:
        """
        Stratejiyi değerlendir.
        
        Args:
            backtest_metrics: Backtest metrikleri
            walk_forward_result: Walk-forward analiz sonucu
            monte_carlo_result: Monte Carlo simülasyon sonucu
            robustness_score: Robustluk skoru
            execution_cost_pct: İşlem maliyeti yüzdesi
            
        Returns:
            ApprovalResult: Onay sonucu
        """
        criteria_results = []
        
        # =====================================================================
        # MANDATORY CRITERIA
        # =====================================================================
        
        # 1. Win Rate
        win_rate = backtest_metrics.get("win_rate", 0)
        criteria_results.append(CriterionResult(
            name="min_win_rate",
            passed=win_rate >= self.thresholds.min_win_rate,
            severity=CriterionSeverity.MANDATORY,
            actual_value=win_rate,
            threshold_value=self.thresholds.min_win_rate,
            message=f"Win rate {win_rate:.1%} {'≥' if win_rate >= self.thresholds.min_win_rate else '<'} {self.thresholds.min_win_rate:.1%}",
        ))
        
        # 2. Profit Factor
        profit_factor = backtest_metrics.get("profit_factor", 0)
        criteria_results.append(CriterionResult(
            name="min_profit_factor",
            passed=profit_factor >= self.thresholds.min_profit_factor,
            severity=CriterionSeverity.MANDATORY,
            actual_value=profit_factor,
            threshold_value=self.thresholds.min_profit_factor,
            message=f"Profit factor {profit_factor:.2f} {'≥' if profit_factor >= self.thresholds.min_profit_factor else '<'} {self.thresholds.min_profit_factor:.2f}",
        ))
        
        # 3. Sharpe Ratio
        sharpe = backtest_metrics.get("sharpe_ratio", 0)
        criteria_results.append(CriterionResult(
            name="min_sharpe_ratio",
            passed=sharpe >= self.thresholds.min_sharpe_ratio,
            severity=CriterionSeverity.MANDATORY,
            actual_value=sharpe,
            threshold_value=self.thresholds.min_sharpe_ratio,
            message=f"Sharpe ratio {sharpe:.2f} {'≥' if sharpe >= self.thresholds.min_sharpe_ratio else '<'} {self.thresholds.min_sharpe_ratio:.2f}",
        ))
        
        # 4. Max Drawdown
        max_dd = backtest_metrics.get("max_drawdown", 1.0)
        criteria_results.append(CriterionResult(
            name="max_drawdown",
            passed=max_dd <= self.thresholds.max_drawdown,
            severity=CriterionSeverity.MANDATORY,
            actual_value=max_dd,
            threshold_value=self.thresholds.max_drawdown,
            message=f"Max drawdown {max_dd:.1%} {'≤' if max_dd <= self.thresholds.max_drawdown else '>'} {self.thresholds.max_drawdown:.1%}",
        ))
        
        # 5. Sample Size
        sample_size = backtest_metrics.get("total_trades", 0)
        criteria_results.append(CriterionResult(
            name="min_sample_size",
            passed=sample_size >= self.thresholds.min_sample_size,
            severity=CriterionSeverity.MANDATORY,
            actual_value=sample_size,
            threshold_value=self.thresholds.min_sample_size,
            message=f"Sample size {sample_size} {'≥' if sample_size >= self.thresholds.min_sample_size else '<'} {self.thresholds.min_sample_size}",
        ))
        
        # 6. Walk-Forward Consistency
        wf_consistency = 0.0
        if walk_forward_result:
            wf_consistency = walk_forward_result.consistency_score
        criteria_results.append(CriterionResult(
            name="min_walkforward_consistency",
            passed=wf_consistency >= self.thresholds.min_walkforward_consistency,
            severity=CriterionSeverity.MANDATORY,
            actual_value=wf_consistency,
            threshold_value=self.thresholds.min_walkforward_consistency,
            message=f"Walk-forward consistency {wf_consistency:.1%} {'≥' if wf_consistency >= self.thresholds.min_walkforward_consistency else '<'} {self.thresholds.min_walkforward_consistency:.1%}",
        ))
        
        # 7. Monte Carlo VaR
        mc_var = 0.0
        if monte_carlo_result:
            mc_var = monte_carlo_result.var
        criteria_results.append(CriterionResult(
            name="min_monte_carlo_var95",
            passed=mc_var >= self.thresholds.min_monte_carlo_var95,
            severity=CriterionSeverity.MANDATORY,
            actual_value=mc_var,
            threshold_value=self.thresholds.min_monte_carlo_var95,
            message=f"MC VaR(95%) {mc_var:.1%} {'≥' if mc_var >= self.thresholds.min_monte_carlo_var95 else '<'} {self.thresholds.min_monte_carlo_var95:.1%}",
        ))
        
        # 8. Robustness Score
        robust = robustness_score or 0.0
        criteria_results.append(CriterionResult(
            name="min_robustness_score",
            passed=robust >= self.thresholds.min_robustness_score,
            severity=CriterionSeverity.MANDATORY,
            actual_value=robust,
            threshold_value=self.thresholds.min_robustness_score,
            message=f"Robustness score {robust:.2f} {'≥' if robust >= self.thresholds.min_robustness_score else '<'} {self.thresholds.min_robustness_score:.2f}",
        ))
        
        # =====================================================================
        # SOFT CRITERIA (Warnings)
        # =====================================================================
        
        # 9. Profit vs Cost
        avg_profit_pct = backtest_metrics.get("avg_profit_pct", 0)
        profit_vs_cost = avg_profit_pct / execution_cost_pct if execution_cost_pct > 0 else 0
        criteria_results.append(CriterionResult(
            name="min_profit_vs_spread",
            passed=profit_vs_cost >= self.thresholds.min_profit_vs_spread,
            severity=CriterionSeverity.SOFT,
            actual_value=profit_vs_cost,
            threshold_value=self.thresholds.min_profit_vs_spread,
            message=f"Profit/Cost ratio {profit_vs_cost:.1f}x {'≥' if profit_vs_cost >= self.thresholds.min_profit_vs_spread else '<'} {self.thresholds.min_profit_vs_spread:.1f}x",
        ))
        
        # 10. Trade Duration
        avg_duration = backtest_metrics.get("avg_trade_duration_hours", 0)
        criteria_results.append(CriterionResult(
            name="min_trade_duration",
            passed=avg_duration >= self.thresholds.min_trade_duration_hours,
            severity=CriterionSeverity.SOFT,
            actual_value=avg_duration,
            threshold_value=self.thresholds.min_trade_duration_hours,
            message=f"Avg duration {avg_duration:.1f}h {'≥' if avg_duration >= self.thresholds.min_trade_duration_hours else '<'} {self.thresholds.min_trade_duration_hours:.1f}h",
        ))
        
        # 11. Trades per Month
        trades_per_month = backtest_metrics.get("trades_per_month", 0)
        criteria_results.append(CriterionResult(
            name="min_trades_per_month",
            passed=trades_per_month >= self.thresholds.min_trades_per_month,
            severity=CriterionSeverity.SOFT,
            actual_value=trades_per_month,
            threshold_value=self.thresholds.min_trades_per_month,
            message=f"Trades/month {trades_per_month:.1f} {'≥' if trades_per_month >= self.thresholds.min_trades_per_month else '<'} {self.thresholds.min_trades_per_month}",
        ))
        
        # =====================================================================
        # COMPILE RESULTS
        # =====================================================================
        
        mandatory = [c for c in criteria_results if c.severity == CriterionSeverity.MANDATORY]
        soft = [c for c in criteria_results if c.severity == CriterionSeverity.SOFT]
        
        mandatory_passed = sum(1 for c in mandatory if c.passed)
        mandatory_failed = len(mandatory) - mandatory_passed
        soft_warnings = sum(1 for c in soft if not c.passed)
        
        # Overall score (0-100)
        total_criteria = len(criteria_results)
        passed_criteria = sum(1 for c in criteria_results if c.passed)
        overall_score = (passed_criteria / total_criteria * 100) if total_criteria > 0 else 0
        
        # Decision
        if mandatory_failed == 0:
            if soft_warnings == 0:
                decision = ApprovalDecision.APPROVED
                suggested_lifecycle = StrategyLifecycle.ACTIVE
            else:
                decision = ApprovalDecision.APPROVED
                suggested_lifecycle = StrategyLifecycle.PROBATION
        elif mandatory_failed <= 2:
            decision = ApprovalDecision.SANDBOX
            suggested_lifecycle = StrategyLifecycle.SANDBOX
        else:
            decision = ApprovalDecision.REJECTED
            suggested_lifecycle = StrategyLifecycle.PENDING
        
        # Risk assessment
        if overall_score >= 90:
            risk_assessment = "LOW RISK - Excellent metrics across all criteria"
        elif overall_score >= 75:
            risk_assessment = "MODERATE RISK - Good metrics with minor concerns"
        elif overall_score >= 60:
            risk_assessment = "ELEVATED RISK - Several areas need improvement"
        else:
            risk_assessment = "HIGH RISK - Significant concerns identified"
        
        # Recommendation
        recommendation = self._generate_recommendation(
            decision, criteria_results, mandatory_failed, soft_warnings
        )
        
        return ApprovalResult(
            decision=decision,
            mandatory_passed=mandatory_passed,
            mandatory_failed=mandatory_failed,
            soft_warnings=soft_warnings,
            criteria_results=criteria_results,
            overall_score=overall_score,
            risk_assessment=risk_assessment,
            recommendation=recommendation,
            suggested_lifecycle=suggested_lifecycle,
        )
    
    def _generate_recommendation(
        self,
        decision: ApprovalDecision,
        criteria: List[CriterionResult],
        mandatory_failed: int,
        soft_warnings: int,
    ) -> str:
        """Öneri oluştur."""
        if decision == ApprovalDecision.APPROVED:
            if soft_warnings > 0:
                failed_soft = [c.name for c in criteria if c.severity == CriterionSeverity.SOFT and not c.passed]
                return f"Approved with warnings. Monitor: {', '.join(failed_soft)}"
            return "Strategy approved for live trading. Monitor initial performance closely."
        
        elif decision == ApprovalDecision.SANDBOX:
            failed_mandatory = [c.name for c in criteria if c.severity == CriterionSeverity.MANDATORY and not c.passed]
            return f"Deploy to sandbox. Failed criteria: {', '.join(failed_mandatory)}"
        
        else:
            failed = [c.name for c in criteria if not c.passed]
            return f"Not ready for deployment. Failed {len(failed)} criteria. Requires significant improvement."
    
    def quick_check(
        self,
        win_rate: float,
        profit_factor: float,
        sharpe_ratio: float,
        max_drawdown: float,
    ) -> bool:
        """
        Hızlı ön kontrol.
        
        Returns:
            bool: Detaylı değerlendirmeye değer mi?
        """
        return (
            win_rate >= self.thresholds.min_win_rate * 0.9 and
            profit_factor >= self.thresholds.min_profit_factor * 0.9 and
            sharpe_ratio >= self.thresholds.min_sharpe_ratio * 0.8 and
            max_drawdown <= self.thresholds.max_drawdown * 1.2
        )
