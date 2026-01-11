"""
AlphaTerminal Pro - Risk Calculator
===================================

Kurumsal seviye risk hesaplama ve position sizing.
Slippage, spread ve worst-case senaryoları içerir.

Author: AlphaTerminal Team
Version: 1.0.0
"""

from __future__ import annotations

import logging
from typing import Optional, Dict, Any, List
from datetime import datetime
from dataclasses import dataclass, field
from decimal import Decimal

import numpy as np

from app.ai_strategy.constants import (
    PositionSizingLimits,
    BacktestDefaults,
)

logger = logging.getLogger(__name__)


@dataclass
class ExecutionCosts:
    """İşlem maliyetleri."""
    spread_pct: float = 0.001      # 0.1%
    commission_pct: float = 0.001   # 0.1%
    slippage_pct: float = 0.002     # 0.2%
    
    @property
    def total_cost_pct(self) -> float:
        """Toplam maliyet (round trip)."""
        return (self.spread_pct + self.commission_pct + self.slippage_pct) * 2
    
    def to_dict(self) -> Dict[str, float]:
        return {
            "spread_pct": self.spread_pct,
            "commission_pct": self.commission_pct,
            "slippage_pct": self.slippage_pct,
            "total_round_trip": self.total_cost_pct,
        }


@dataclass
class PositionSizeResult:
    """Position sizing sonucu."""
    shares: int
    position_value: float
    position_pct: float
    risk_amount: float
    risk_per_share: float
    
    # Stop levels
    stop_loss_price: float
    stop_distance_pct: float
    
    # Take profit levels
    tp1_price: float
    tp2_price: float
    tp3_price: float
    
    # Risk/Reward
    risk_reward_1: float
    risk_reward_2: float
    risk_reward_3: float
    
    # Execution costs impact
    breakeven_move_pct: float
    expected_slippage: float
    
    # Flags
    within_limits: bool
    warnings: List[str] = field(default_factory=list)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "shares": self.shares,
            "position_value": self.position_value,
            "position_pct": self.position_pct,
            "risk_amount": self.risk_amount,
            "risk_per_share": self.risk_per_share,
            "stop_loss_price": self.stop_loss_price,
            "stop_distance_pct": self.stop_distance_pct,
            "take_profits": {
                "tp1": self.tp1_price,
                "tp2": self.tp2_price,
                "tp3": self.tp3_price,
            },
            "risk_rewards": {
                "rr1": self.risk_reward_1,
                "rr2": self.risk_reward_2,
                "rr3": self.risk_reward_3,
            },
            "breakeven_move_pct": self.breakeven_move_pct,
            "expected_slippage": self.expected_slippage,
            "within_limits": self.within_limits,
            "warnings": self.warnings,
        }


@dataclass
class RiskAssessment:
    """Risk değerlendirmesi."""
    can_trade: bool
    risk_score: float  # 0-100, yüksek = daha riskli
    
    # Position limits
    remaining_capital: float
    remaining_risk_budget: float
    current_portfolio_heat: float
    
    # Trade-specific
    expected_profit: float
    expected_loss: float
    profit_after_costs: float
    
    # Viability
    profit_vs_cost_ratio: float
    is_cost_viable: bool
    
    # Kelly
    kelly_fraction: float
    adjusted_kelly: float
    
    reasons: List[str] = field(default_factory=list)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "can_trade": self.can_trade,
            "risk_score": self.risk_score,
            "remaining_capital": self.remaining_capital,
            "remaining_risk_budget": self.remaining_risk_budget,
            "current_portfolio_heat": self.current_portfolio_heat,
            "expected_profit": self.expected_profit,
            "expected_loss": self.expected_loss,
            "profit_after_costs": self.profit_after_costs,
            "profit_vs_cost_ratio": self.profit_vs_cost_ratio,
            "is_cost_viable": self.is_cost_viable,
            "kelly_fraction": self.kelly_fraction,
            "adjusted_kelly": self.adjusted_kelly,
            "reasons": self.reasons,
        }


class RiskCalculator:
    """
    Kurumsal seviye risk hesaplama.
    
    Özellikler:
    - Kelly Criterion position sizing
    - Slippage & spread modeling
    - Worst-case scenario analysis
    - Portfolio heat management
    - Correlation-adjusted sizing
    
    Example:
        ```python
        calculator = RiskCalculator(capital=100000)
        
        # Position size hesapla
        size = calculator.calculate_position_size(
            entry_price=150.0,
            stop_loss=145.0,
            atr=3.5
        )
        
        # Risk değerlendirmesi
        assessment = calculator.assess_trade_risk(
            entry_price=150.0,
            stop_loss=145.0,
            take_profit=160.0,
            win_rate=0.55
        )
        
        if assessment.can_trade:
            # Trade aç
            pass
        ```
    """
    
    def __init__(
        self,
        capital: float = 100000,
        max_risk_per_trade: float = 0.02,
        max_portfolio_heat: float = 0.20,
        max_position_size: float = 0.05,
        execution_costs: Optional[ExecutionCosts] = None,
        limits: Optional[PositionSizingLimits] = None,
    ):
        """
        Initialize Risk Calculator.
        
        Args:
            capital: Toplam sermaye
            max_risk_per_trade: Trade başına max risk (%)
            max_portfolio_heat: Toplam portföy riski (%)
            max_position_size: Max pozisyon boyutu (%)
            execution_costs: İşlem maliyetleri
            limits: Position sizing limitleri
        """
        self.capital = capital
        self.max_risk_per_trade = max_risk_per_trade
        self.max_portfolio_heat = max_portfolio_heat
        self.max_position_size = max_position_size
        self.execution_costs = execution_costs or ExecutionCosts()
        self.limits = limits or PositionSizingLimits()
        
        # Portfolio state
        self.current_heat = 0.0
        self.open_positions: List[Dict] = []
    
    def calculate_position_size(
        self,
        entry_price: float,
        stop_loss: float,
        atr: Optional[float] = None,
        direction: str = "long",
        volatility_adjustment: float = 1.0,
    ) -> PositionSizeResult:
        """
        Position size hesapla.
        
        Args:
            entry_price: Giriş fiyatı
            stop_loss: Stop loss fiyatı
            atr: Average True Range
            direction: long veya short
            volatility_adjustment: Volatilite ayarlama faktörü
            
        Returns:
            PositionSizeResult: Position sizing sonucu
        """
        warnings = []
        
        # Stop distance
        if direction == "long":
            stop_distance = entry_price - stop_loss
        else:
            stop_distance = stop_loss - entry_price
        
        if stop_distance <= 0:
            return self._create_zero_result(entry_price, "Invalid stop loss")
        
        stop_distance_pct = stop_distance / entry_price
        
        # Risk amount
        adjusted_risk = self.max_risk_per_trade * volatility_adjustment
        risk_amount = self.capital * adjusted_risk
        
        # Basic position size
        position_value = risk_amount / stop_distance_pct
        
        # Maximum position size constraint
        max_value = self.capital * self.max_position_size
        if position_value > max_value:
            position_value = max_value
            warnings.append(f"Position capped at {self.max_position_size:.1%} of capital")
        
        # Portfolio heat constraint
        remaining_heat = self.max_portfolio_heat - self.current_heat
        max_heat_value = self.capital * remaining_heat / stop_distance_pct
        if position_value > max_heat_value:
            position_value = max(0, max_heat_value)
            warnings.append(f"Position reduced due to portfolio heat limit")
        
        # Shares
        shares = int(position_value / entry_price)
        if shares <= 0:
            return self._create_zero_result(entry_price, "Calculated shares is zero")
        
        # Actual values
        actual_value = shares * entry_price
        actual_risk = actual_value * stop_distance_pct
        position_pct = actual_value / self.capital
        
        # Take profit levels (1.5x, 2.5x, 4x risk)
        if direction == "long":
            tp1 = entry_price + stop_distance * 1.5
            tp2 = entry_price + stop_distance * 2.5
            tp3 = entry_price + stop_distance * 4.0
        else:
            tp1 = entry_price - stop_distance * 1.5
            tp2 = entry_price - stop_distance * 2.5
            tp3 = entry_price - stop_distance * 4.0
        
        # Execution costs impact
        total_cost_pct = self.execution_costs.total_cost_pct
        breakeven_move = total_cost_pct / (1 - total_cost_pct)
        
        # Expected slippage (volume-based would be more accurate)
        expected_slippage = self.execution_costs.slippage_pct * entry_price
        
        # Within limits check
        within_limits = (
            position_pct <= self.max_position_size and
            actual_risk <= risk_amount * 1.1 and
            self.current_heat + (actual_risk / self.capital) <= self.max_portfolio_heat
        )
        
        return PositionSizeResult(
            shares=shares,
            position_value=actual_value,
            position_pct=position_pct,
            risk_amount=actual_risk,
            risk_per_share=stop_distance,
            stop_loss_price=stop_loss,
            stop_distance_pct=stop_distance_pct,
            tp1_price=tp1,
            tp2_price=tp2,
            tp3_price=tp3,
            risk_reward_1=1.5,
            risk_reward_2=2.5,
            risk_reward_3=4.0,
            breakeven_move_pct=breakeven_move,
            expected_slippage=expected_slippage,
            within_limits=within_limits,
            warnings=warnings,
        )
    
    def assess_trade_risk(
        self,
        entry_price: float,
        stop_loss: float,
        take_profit: float,
        win_rate: float,
        avg_win_pct: Optional[float] = None,
        avg_loss_pct: Optional[float] = None,
    ) -> RiskAssessment:
        """
        Trade risk değerlendirmesi.
        
        Args:
            entry_price: Giriş fiyatı
            stop_loss: Stop loss
            take_profit: Take profit
            win_rate: Win rate
            avg_win_pct: Ortalama kazanç %
            avg_loss_pct: Ortalama kayıp %
            
        Returns:
            RiskAssessment: Risk değerlendirmesi
        """
        reasons = []
        
        # Calculate distances
        loss_pct = abs(entry_price - stop_loss) / entry_price
        profit_pct = abs(take_profit - entry_price) / entry_price
        
        # Use provided or calculated
        avg_win = avg_win_pct or profit_pct
        avg_loss = avg_loss_pct or loss_pct
        
        # Expected values
        expected_profit = win_rate * avg_win * self.capital * self.max_risk_per_trade / loss_pct
        expected_loss = (1 - win_rate) * avg_loss * self.capital * self.max_risk_per_trade / loss_pct
        
        # Costs
        trade_cost = self.capital * self.max_risk_per_trade * self.execution_costs.total_cost_pct
        profit_after_costs = expected_profit - expected_loss - trade_cost
        
        # Cost viability (profit should be at least 3x costs)
        profit_vs_cost = expected_profit / trade_cost if trade_cost > 0 else 0
        is_cost_viable = profit_vs_cost >= 3.0
        
        if not is_cost_viable:
            reasons.append(f"Expected profit ({profit_vs_cost:.1f}x) should be at least 3x execution costs")
        
        # Kelly Criterion
        if avg_loss > 0:
            kelly = (win_rate * avg_win - (1 - win_rate) * avg_loss) / avg_win
        else:
            kelly = 0
        
        adjusted_kelly = kelly * self.limits.kelly_fraction
        adjusted_kelly = max(0, min(adjusted_kelly, self.max_risk_per_trade))
        
        # Portfolio state
        remaining_capital = self.capital * (1 - self.current_heat)
        remaining_risk = self.max_portfolio_heat - self.current_heat
        
        # Risk score (0-100)
        risk_score = self._calculate_risk_score(
            win_rate=win_rate,
            risk_reward=profit_pct / loss_pct,
            portfolio_heat=self.current_heat,
            cost_ratio=profit_vs_cost,
        )
        
        # Can trade decision
        can_trade = (
            is_cost_viable and
            kelly > 0 and
            remaining_risk > self.max_risk_per_trade * 0.5 and
            win_rate >= 0.40 and
            risk_score < 70
        )
        
        if not can_trade:
            if kelly <= 0:
                reasons.append("Negative Kelly criterion - edge is insufficient")
            if remaining_risk < self.max_risk_per_trade * 0.5:
                reasons.append("Insufficient remaining risk budget")
            if win_rate < 0.40:
                reasons.append("Win rate too low")
            if risk_score >= 70:
                reasons.append(f"Risk score too high: {risk_score:.0f}")
        
        return RiskAssessment(
            can_trade=can_trade,
            risk_score=risk_score,
            remaining_capital=remaining_capital,
            remaining_risk_budget=remaining_risk,
            current_portfolio_heat=self.current_heat,
            expected_profit=expected_profit,
            expected_loss=expected_loss,
            profit_after_costs=profit_after_costs,
            profit_vs_cost_ratio=profit_vs_cost,
            is_cost_viable=is_cost_viable,
            kelly_fraction=kelly,
            adjusted_kelly=adjusted_kelly,
            reasons=reasons,
        )
    
    def simulate_slippage(
        self,
        order_size: float,
        avg_volume: float,
        volatility: float,
        urgency: str = "normal",
    ) -> float:
        """
        Slippage simülasyonu (market impact).
        
        Args:
            order_size: Emir büyüklüğü (değer)
            avg_volume: Ortalama günlük hacim (değer)
            volatility: Volatilite (yıllık)
            urgency: normal, high, low
            
        Returns:
            float: Tahmini slippage (%)
        """
        # Almgren-Chriss simplified model
        participation_rate = order_size / avg_volume
        
        # Base slippage
        base_slippage = self.execution_costs.slippage_pct
        
        # Participation impact (linear + square root)
        linear_impact = participation_rate * 0.1
        sqrt_impact = np.sqrt(participation_rate) * 0.05
        
        # Volatility adjustment
        vol_factor = volatility / 0.20  # Normalized to 20% vol
        
        # Urgency adjustment
        urgency_multiplier = {"low": 0.5, "normal": 1.0, "high": 2.0}.get(urgency, 1.0)
        
        total_slippage = (base_slippage + linear_impact + sqrt_impact) * vol_factor * urgency_multiplier
        
        return min(total_slippage, 0.05)  # Cap at 5%
    
    def worst_case_scenario(
        self,
        position_value: float,
        stop_loss_pct: float,
        slippage_scenarios: List[float] = [0.001, 0.005, 0.01, 0.02],
    ) -> Dict[str, float]:
        """
        Worst-case senaryo analizi.
        
        Args:
            position_value: Pozisyon değeri
            stop_loss_pct: Stop loss yüzdesi
            slippage_scenarios: Slippage senaryoları
            
        Returns:
            Dict: Senaryo sonuçları
        """
        results = {}
        
        base_loss = position_value * stop_loss_pct
        
        for slip in slippage_scenarios:
            scenario_name = f"slippage_{slip*100:.1f}pct"
            
            # Stop loss + slippage
            total_loss_pct = stop_loss_pct + slip + self.execution_costs.spread_pct
            total_loss = position_value * total_loss_pct
            
            # Commission
            commission = position_value * self.execution_costs.commission_pct * 2
            
            results[scenario_name] = {
                "slippage_pct": slip,
                "total_loss": total_loss + commission,
                "loss_vs_capital_pct": (total_loss + commission) / self.capital,
                "additional_loss_vs_base": total_loss - base_loss,
            }
        
        # Worst case (gap down scenario - 2x stop)
        gap_down_loss = position_value * stop_loss_pct * 2
        results["gap_down_2x"] = {
            "total_loss": gap_down_loss,
            "loss_vs_capital_pct": gap_down_loss / self.capital,
        }
        
        return results
    
    def update_portfolio_heat(self, risk_amount: float) -> None:
        """Portföy heat'i güncelle (pozisyon açıldığında)."""
        self.current_heat += risk_amount / self.capital
    
    def release_portfolio_heat(self, risk_amount: float) -> None:
        """Portföy heat'i azalt (pozisyon kapandığında)."""
        self.current_heat = max(0, self.current_heat - risk_amount / self.capital)
    
    def _calculate_risk_score(
        self,
        win_rate: float,
        risk_reward: float,
        portfolio_heat: float,
        cost_ratio: float,
    ) -> float:
        """Risk skoru hesapla (0-100)."""
        score = 50  # Base
        
        # Win rate contribution
        if win_rate < 0.45:
            score += 20
        elif win_rate < 0.50:
            score += 10
        elif win_rate > 0.60:
            score -= 10
        
        # Risk/Reward contribution
        if risk_reward < 1.0:
            score += 25
        elif risk_reward < 1.5:
            score += 10
        elif risk_reward > 2.5:
            score -= 10
        
        # Portfolio heat
        if portfolio_heat > 0.15:
            score += 15
        elif portfolio_heat > 0.10:
            score += 5
        
        # Cost efficiency
        if cost_ratio < 2.0:
            score += 15
        elif cost_ratio < 3.0:
            score += 5
        
        return max(0, min(100, score))
    
    def _create_zero_result(self, entry_price: float, reason: str) -> PositionSizeResult:
        """Sıfır pozisyon sonucu."""
        return PositionSizeResult(
            shares=0,
            position_value=0,
            position_pct=0,
            risk_amount=0,
            risk_per_share=0,
            stop_loss_price=entry_price,
            stop_distance_pct=0,
            tp1_price=entry_price,
            tp2_price=entry_price,
            tp3_price=entry_price,
            risk_reward_1=0,
            risk_reward_2=0,
            risk_reward_3=0,
            breakeven_move_pct=0,
            expected_slippage=0,
            within_limits=False,
            warnings=[reason],
        )
