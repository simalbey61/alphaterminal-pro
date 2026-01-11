"""
AlphaTerminal Pro - Monte Carlo Simulator
=========================================

Monte Carlo simülasyonu ile strateji risk analizi.

Author: AlphaTerminal Team
Version: 1.0.0
"""

from __future__ import annotations

import logging
from typing import Optional, List, Dict, Any
from datetime import datetime
from dataclasses import dataclass, field

import numpy as np
from scipy import stats

logger = logging.getLogger(__name__)


@dataclass
class MonteCarloResult:
    """Monte Carlo simülasyon sonucu."""
    simulations: int
    confidence_level: float
    
    # Return dağılımı
    mean_return: float
    median_return: float
    std_return: float
    
    # Risk metrikleri
    var: float  # Value at Risk
    cvar: float  # Conditional VaR (Expected Shortfall)
    max_drawdown_mean: float
    max_drawdown_worst: float
    max_drawdown_95: float
    
    # Percentiles
    percentile_1: float
    percentile_5: float
    percentile_25: float
    percentile_75: float
    percentile_95: float
    percentile_99: float
    
    # Olasılıklar
    prob_positive: float
    prob_above_target: float
    prob_ruin: float
    target_return: float
    ruin_threshold: float
    
    # Distribution fit
    distribution_type: str
    skewness: float
    kurtosis: float
    
    # Time metrics
    avg_recovery_time: Optional[int] = None
    
    calculated_at: datetime = field(default_factory=datetime.utcnow)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "simulations": self.simulations,
            "confidence_level": self.confidence_level,
            "mean_return": self.mean_return,
            "median_return": self.median_return,
            "std_return": self.std_return,
            "var": self.var,
            "cvar": self.cvar,
            "max_drawdown_mean": self.max_drawdown_mean,
            "max_drawdown_worst": self.max_drawdown_worst,
            "max_drawdown_95": self.max_drawdown_95,
            "percentiles": {
                "1": self.percentile_1,
                "5": self.percentile_5,
                "25": self.percentile_25,
                "75": self.percentile_75,
                "95": self.percentile_95,
                "99": self.percentile_99,
            },
            "prob_positive": self.prob_positive,
            "prob_above_target": self.prob_above_target,
            "prob_ruin": self.prob_ruin,
            "distribution_type": self.distribution_type,
            "skewness": self.skewness,
            "kurtosis": self.kurtosis,
        }
    
    @property
    def risk_score(self) -> float:
        """Risk skoru (0-100, düşük = daha riskli)."""
        score = 100
        score -= self.prob_ruin * 50
        score -= max(0, -self.var) * 10
        score -= max(0, self.max_drawdown_mean - 0.1) * 100
        score += self.prob_positive * 20
        return max(0, min(100, score))


class MonteCarloSimulator:
    """
    Monte Carlo simülasyonu.
    
    Trade sonuçlarını rastgele sıralayarak olası
    sonuç dağılımını analiz eder.
    
    Example:
        ```python
        simulator = MonteCarloSimulator(simulations=10000)
        
        result = simulator.simulate(trade_returns)
        
        print(f"VaR 95%: {result.var:.2%}")
        print(f"Max DD Mean: {result.max_drawdown_mean:.2%}")
        print(f"Prob Ruin: {result.prob_ruin:.2%}")
        ```
    """
    
    def __init__(
        self,
        simulations: int = 10000,
        confidence_level: float = 0.95,
        target_return: float = 0.10,
        ruin_threshold: float = -0.30,
    ):
        """
        Initialize Monte Carlo Simulator.
        
        Args:
            simulations: Simülasyon sayısı
            confidence_level: VaR güven seviyesi
            target_return: Hedef getiri
            ruin_threshold: Ruin eşiği
        """
        self.simulations = simulations
        self.confidence_level = confidence_level
        self.target_return = target_return
        self.ruin_threshold = ruin_threshold
    
    def simulate(
        self,
        trade_returns: np.ndarray,
        initial_capital: float = 1.0,
    ) -> MonteCarloResult:
        """
        Monte Carlo simülasyonu çalıştır.
        
        Args:
            trade_returns: Trade returns array
            initial_capital: Başlangıç sermayesi
            
        Returns:
            MonteCarloResult: Simülasyon sonuçları
        """
        if len(trade_returns) < 5:
            logger.warning("Insufficient trades for Monte Carlo")
            return self._create_empty_result()
        
        n_trades = len(trade_returns)
        
        # Simülasyonları çalıştır
        final_returns = np.zeros(self.simulations)
        max_drawdowns = np.zeros(self.simulations)
        equity_curves = []
        
        for i in range(self.simulations):
            # Trade sırasını rastgele karıştır
            shuffled = np.random.permutation(trade_returns)
            
            # Equity curve hesapla
            equity = initial_capital * np.cumprod(1 + shuffled)
            equity = np.insert(equity, 0, initial_capital)
            
            # Final return
            final_returns[i] = equity[-1] / initial_capital - 1
            
            # Max drawdown
            running_max = np.maximum.accumulate(equity)
            drawdown = (running_max - equity) / running_max
            max_drawdowns[i] = np.max(drawdown)
            
            if i < 100:  # İlk 100 curve'ü sakla
                equity_curves.append(equity)
        
        # Return dağılımı
        mean_return = np.mean(final_returns)
        median_return = np.median(final_returns)
        std_return = np.std(final_returns)
        
        # VaR ve CVaR
        var_percentile = (1 - self.confidence_level) * 100
        var = np.percentile(final_returns, var_percentile)
        cvar = np.mean(final_returns[final_returns <= var])
        
        # Max drawdown statistics
        dd_mean = np.mean(max_drawdowns)
        dd_worst = np.max(max_drawdowns)
        dd_95 = np.percentile(max_drawdowns, 95)
        
        # Percentiles
        percentiles = np.percentile(final_returns, [1, 5, 25, 75, 95, 99])
        
        # Olasılıklar
        prob_positive = np.mean(final_returns > 0)
        prob_above_target = np.mean(final_returns > self.target_return)
        prob_ruin = np.mean(final_returns < self.ruin_threshold)
        
        # Distribution characteristics
        skewness = stats.skew(final_returns)
        kurtosis = stats.kurtosis(final_returns)
        
        # Distribution type
        if abs(skewness) < 0.5 and abs(kurtosis) < 1:
            distribution_type = "normal"
        elif skewness < -0.5:
            distribution_type = "left_skewed"
        elif skewness > 0.5:
            distribution_type = "right_skewed"
        elif kurtosis > 1:
            distribution_type = "fat_tailed"
        else:
            distribution_type = "mixed"
        
        return MonteCarloResult(
            simulations=self.simulations,
            confidence_level=self.confidence_level,
            mean_return=mean_return,
            median_return=median_return,
            std_return=std_return,
            var=var,
            cvar=cvar,
            max_drawdown_mean=dd_mean,
            max_drawdown_worst=dd_worst,
            max_drawdown_95=dd_95,
            percentile_1=percentiles[0],
            percentile_5=percentiles[1],
            percentile_25=percentiles[2],
            percentile_75=percentiles[3],
            percentile_95=percentiles[4],
            percentile_99=percentiles[5],
            prob_positive=prob_positive,
            prob_above_target=prob_above_target,
            prob_ruin=prob_ruin,
            target_return=self.target_return,
            ruin_threshold=self.ruin_threshold,
            distribution_type=distribution_type,
            skewness=skewness,
            kurtosis=kurtosis,
        )
    
    def simulate_with_sizing(
        self,
        trade_returns: np.ndarray,
        win_rate: float,
        avg_win: float,
        avg_loss: float,
        initial_capital: float = 100000,
        risk_per_trade: float = 0.02,
    ) -> MonteCarloResult:
        """
        Position sizing ile Monte Carlo.
        
        Args:
            trade_returns: Orijinal trade returns
            win_rate: Win rate
            avg_win: Ortalama kazanç
            avg_loss: Ortalama kayıp
            initial_capital: Başlangıç sermayesi
            risk_per_trade: Trade başına risk
            
        Returns:
            MonteCarloResult: Sonuçlar
        """
        # Parametrik simülasyon
        final_returns = np.zeros(self.simulations)
        max_drawdowns = np.zeros(self.simulations)
        
        n_trades = len(trade_returns)
        
        for i in range(self.simulations):
            capital = initial_capital
            peak = capital
            max_dd = 0
            
            for _ in range(n_trades):
                # Win/loss belirleme
                if np.random.random() < win_rate:
                    pnl = capital * risk_per_trade * (avg_win / avg_loss)
                else:
                    pnl = -capital * risk_per_trade
                
                capital += pnl
                
                if capital > peak:
                    peak = capital
                
                dd = (peak - capital) / peak
                max_dd = max(max_dd, dd)
                
                if capital <= 0:
                    capital = 0
                    break
            
            final_returns[i] = capital / initial_capital - 1
            max_drawdowns[i] = max_dd
        
        # Sonuçları derle (simulate metoduyla aynı)
        return self._compile_results(final_returns, max_drawdowns)
    
    def _compile_results(
        self,
        final_returns: np.ndarray,
        max_drawdowns: np.ndarray,
    ) -> MonteCarloResult:
        """Sonuçları derle."""
        mean_return = np.mean(final_returns)
        median_return = np.median(final_returns)
        std_return = np.std(final_returns)
        
        var_percentile = (1 - self.confidence_level) * 100
        var = np.percentile(final_returns, var_percentile)
        cvar = np.mean(final_returns[final_returns <= var]) if np.any(final_returns <= var) else var
        
        dd_mean = np.mean(max_drawdowns)
        dd_worst = np.max(max_drawdowns)
        dd_95 = np.percentile(max_drawdowns, 95)
        
        percentiles = np.percentile(final_returns, [1, 5, 25, 75, 95, 99])
        
        prob_positive = np.mean(final_returns > 0)
        prob_above_target = np.mean(final_returns > self.target_return)
        prob_ruin = np.mean(final_returns < self.ruin_threshold)
        
        skewness = stats.skew(final_returns)
        kurtosis = stats.kurtosis(final_returns)
        
        if abs(skewness) < 0.5 and abs(kurtosis) < 1:
            distribution_type = "normal"
        elif skewness < -0.5:
            distribution_type = "left_skewed"
        else:
            distribution_type = "mixed"
        
        return MonteCarloResult(
            simulations=self.simulations,
            confidence_level=self.confidence_level,
            mean_return=mean_return,
            median_return=median_return,
            std_return=std_return,
            var=var,
            cvar=cvar,
            max_drawdown_mean=dd_mean,
            max_drawdown_worst=dd_worst,
            max_drawdown_95=dd_95,
            percentile_1=percentiles[0],
            percentile_5=percentiles[1],
            percentile_25=percentiles[2],
            percentile_75=percentiles[3],
            percentile_95=percentiles[4],
            percentile_99=percentiles[5],
            prob_positive=prob_positive,
            prob_above_target=prob_above_target,
            prob_ruin=prob_ruin,
            target_return=self.target_return,
            ruin_threshold=self.ruin_threshold,
            distribution_type=distribution_type,
            skewness=skewness,
            kurtosis=kurtosis,
        )
    
    def _create_empty_result(self) -> MonteCarloResult:
        """Boş sonuç."""
        return MonteCarloResult(
            simulations=0,
            confidence_level=self.confidence_level,
            mean_return=0,
            median_return=0,
            std_return=0,
            var=0,
            cvar=0,
            max_drawdown_mean=0,
            max_drawdown_worst=0,
            max_drawdown_95=0,
            percentile_1=0,
            percentile_5=0,
            percentile_25=0,
            percentile_75=0,
            percentile_95=0,
            percentile_99=0,
            prob_positive=0,
            prob_above_target=0,
            prob_ruin=1,
            target_return=self.target_return,
            ruin_threshold=self.ruin_threshold,
            distribution_type="unknown",
            skewness=0,
            kurtosis=0,
        )
