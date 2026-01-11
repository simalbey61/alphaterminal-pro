"""
AlphaTerminal Pro - Alpha Engine v4.2
=====================================

Kurumsal Seviye Alpha & Performance Analiz Motoru

Özellikler:
- Jensen's Alpha hesaplama
- Risk-adjusted returns (Sharpe, Sortino)
- Relative Strength Analysis
- Maximum Drawdown tracking
- Value at Risk (VaR)
- Rolling metrics

Author: AlphaTerminal Team
Version: 4.2.0
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
from scipy import stats
from enum import Enum

from app.core.config import logger, ALPHA_CONFIG, get_sector, get_sector_symbols


# ═══════════════════════════════════════════════════════════════════════════════
# ENUMS
# ═══════════════════════════════════════════════════════════════════════════════

class PerformanceRating(Enum):
    EXCEPTIONAL = "EXCEPTIONAL"
    STRONG = "STRONG"
    GOOD = "GOOD"
    NEUTRAL = "NEUTRAL"
    WEAK = "WEAK"
    POOR = "POOR"


class AlphaCategory(Enum):
    LEADER = "LEADER"
    OUTPERFORMER = "OUTPERFORMER"
    MARKET_PERFORMER = "MARKET_PERFORMER"
    UNDERPERFORMER = "UNDERPERFORMER"
    LAGGARD = "LAGGARD"


class MomentumState(Enum):
    ACCELERATING = "ACCELERATING"
    STRONG = "STRONG"
    POSITIVE = "POSITIVE"
    NEUTRAL = "NEUTRAL"
    NEGATIVE = "NEGATIVE"
    WEAK = "WEAK"
    DECELERATING = "DECELERATING"


# ═══════════════════════════════════════════════════════════════════════════════
# DATA CLASSES
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass
class RiskMetrics:
    """Risk metrikleri"""
    volatility: float
    var_95: float
    var_99: float
    cvar_95: float
    max_drawdown: float
    drawdown_duration: int
    downside_deviation: float
    ulcer_index: float


@dataclass
class ReturnMetrics:
    """Getiri metrikleri"""
    total_return: float
    annual_return: float
    excess_return: float
    daily_return_avg: float
    weekly_return_avg: float
    monthly_return_avg: float
    best_day: float
    worst_day: float
    positive_days: int
    negative_days: int
    win_rate: float


@dataclass
class RatioMetrics:
    """Risk-adjusted ratio metrikleri"""
    sharpe_ratio: float
    sortino_ratio: float
    calmar_ratio: float
    omega_ratio: float
    information_ratio: float
    treynor_ratio: float


@dataclass
class RelativeStrength:
    """Relative Strength analizi"""
    rs_value: float
    rs_percentile: float
    rs_slope: str
    rs_momentum: float
    rs_ma: float
    outperforming: bool
    outperformance_streak: int


@dataclass
class RollingMetrics:
    """Rolling metrikler"""
    alpha_20d: float
    alpha_60d: float
    beta_20d: float
    beta_60d: float
    correlation_20d: float
    correlation_60d: float
    rs_20d: float
    rs_60d: float


@dataclass
class SectorAnalysis:
    """Sektör analizi"""
    sector: str
    sector_rank: int
    sector_percentile: float
    sector_correlation: float
    sector_beta: float
    vs_sector_return: float
    peer_count: int


@dataclass
class FactorExposure:
    """Faktör maruziyeti"""
    momentum_factor: float
    value_factor: float
    size_factor: float
    volatility_factor: float
    quality_factor: float


@dataclass
class AlphaAnalysis:
    """Kapsamlı Alpha analiz sonucu"""
    alpha_score: float
    alpha_category: AlphaCategory
    jensens_alpha: float
    beta: float
    r_squared: float
    returns: ReturnMetrics
    risk: RiskMetrics
    ratios: RatioMetrics
    relative_strength: RelativeStrength
    momentum_state: MomentumState
    rolling: RollingMetrics
    sector: Optional[SectorAnalysis]
    factors: Optional[FactorExposure]
    performance_rating: PerformanceRating
    strength_rating: int
    status: str
    recommendation: str


# ═══════════════════════════════════════════════════════════════════════════════
# ALPHA ENGINE CLASS
# ═══════════════════════════════════════════════════════════════════════════════

class AlphaEngine:
    """
    Kurumsal Seviye Alpha & Performance Analiz Motoru v4.2
    """
    
    def __init__(self, config=None):
        self.config = config or ALPHA_CONFIG
    
    def calculate_alpha_beta(
        self,
        stock_returns: pd.Series,
        index_returns: pd.Series
    ) -> Tuple[float, float, float]:
        """Jensen's Alpha ve Beta hesaplama"""
        aligned = pd.DataFrame({
            'stock': stock_returns,
            'index': index_returns
        }).dropna()
        
        if len(aligned) < self.config.min_data_points:
            return 0.0, 1.0, 0.0
        
        slope, intercept, r_value, _, _ = stats.linregress(
            aligned['index'], aligned['stock']
        )
        
        beta = slope
        r_squared = r_value ** 2
        
        daily_rf = (1 + self.config.risk_free_rate) ** (1/252) - 1
        stock_mean = aligned['stock'].mean()
        index_mean = aligned['index'].mean()
        
        alpha = stock_mean - (daily_rf + beta * (index_mean - daily_rf))
        annualized_alpha = alpha * 252
        
        return annualized_alpha, beta, r_squared
    
    def _analyze_returns(
        self,
        stock_returns: pd.Series,
        index_returns: pd.Series
    ) -> ReturnMetrics:
        """Detaylı getiri analizi"""
        returns = stock_returns.dropna()
        
        if len(returns) < 20:
            return ReturnMetrics(0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0)
        
        total_return = (1 + returns).prod() - 1
        n_periods = len(returns)
        annual_return = (1 + total_return) ** (252 / n_periods) - 1 if n_periods > 0 else 0
        
        index_total = (1 + index_returns.dropna()).prod() - 1
        excess_return = total_return - index_total
        
        daily_avg = returns.mean()
        best_day = returns.max()
        worst_day = returns.min()
        
        positive_days = (returns > 0).sum()
        negative_days = (returns < 0).sum()
        win_rate = positive_days / len(returns) if len(returns) > 0 else 0
        
        return ReturnMetrics(
            total_return=total_return * 100,
            annual_return=annual_return * 100,
            excess_return=excess_return * 100,
            daily_return_avg=daily_avg * 100,
            weekly_return_avg=daily_avg * 5 * 100,
            monthly_return_avg=daily_avg * 21 * 100,
            best_day=best_day * 100,
            worst_day=worst_day * 100,
            positive_days=positive_days,
            negative_days=negative_days,
            win_rate=win_rate * 100
        )
    
    def _calculate_risk_metrics(
        self,
        returns: pd.Series,
        prices: pd.Series
    ) -> RiskMetrics:
        """Risk metrikleri hesaplama"""
        returns = returns.dropna()
        
        if len(returns) < 20:
            return RiskMetrics(0, 0, 0, 0, 0, 0, 0, 0)
        
        volatility = returns.std() * np.sqrt(252) * 100
        var_95 = np.percentile(returns, 5) * 100
        var_99 = np.percentile(returns, 1) * 100
        
        var_threshold = np.percentile(returns, 5)
        below_var = returns[returns <= var_threshold]
        cvar_95 = below_var.mean() * 100 if len(below_var) > 0 else var_95
        
        rolling_max = prices.expanding().max()
        drawdown = (prices - rolling_max) / rolling_max
        max_dd = abs(drawdown.min()) * 100
        
        in_drawdown = drawdown < 0
        dd_duration = 0
        current_duration = 0
        for is_dd in in_drawdown:
            if is_dd:
                current_duration += 1
                dd_duration = max(dd_duration, current_duration)
            else:
                current_duration = 0
        
        negative_returns = returns[returns < 0]
        downside_dev = negative_returns.std() * np.sqrt(252) * 100 if len(negative_returns) > 0 else 0
        
        squared_drawdowns = (drawdown * 100) ** 2
        ulcer_index = np.sqrt(squared_drawdowns.mean())
        
        return RiskMetrics(
            volatility=volatility,
            var_95=abs(var_95),
            var_99=abs(var_99),
            cvar_95=abs(cvar_95),
            max_drawdown=max_dd,
            drawdown_duration=dd_duration,
            downside_deviation=downside_dev,
            ulcer_index=ulcer_index
        )
    
    def _calculate_ratio_metrics(
        self,
        returns: pd.Series,
        index_returns: pd.Series,
        max_drawdown: float,
        beta: float
    ) -> RatioMetrics:
        """Risk-adjusted ratio metrikleri hesaplama"""
        returns = returns.dropna()
        
        if len(returns) < 20:
            return RatioMetrics(0, 0, 0, 0, 0, 0)
        
        daily_rf = (1 + self.config.risk_free_rate) ** (1/252) - 1
        excess_returns = returns - daily_rf
        
        mean_excess = excess_returns.mean()
        std_excess = excess_returns.std()
        sharpe = (mean_excess / std_excess) * np.sqrt(252) if std_excess != 0 else 0
        
        negative_returns = returns[returns < 0]
        downside_std = negative_returns.std() if len(negative_returns) > 0 else returns.std()
        sortino = (mean_excess / downside_std) * np.sqrt(252) if downside_std != 0 else 0
        
        annual_return = (1 + returns).prod() ** (252 / len(returns)) - 1
        calmar = (annual_return * 100) / max_drawdown if max_drawdown != 0 else 0
        
        gains = returns[returns > 0].sum()
        losses = abs(returns[returns < 0].sum())
        omega = gains / losses if losses != 0 else float('inf')
        
        aligned = pd.DataFrame({'stock': returns, 'index': index_returns}).dropna()
        if len(aligned) >= 20:
            excess = aligned['stock'] - aligned['index']
            tracking_error = excess.std() * np.sqrt(252)
            info_ratio = (excess.mean() * 252) / tracking_error if tracking_error != 0 else 0
        else:
            info_ratio = 0
        
        treynor = (annual_return - self.config.risk_free_rate) / beta if beta != 0 else 0
        
        return RatioMetrics(
            sharpe_ratio=round(sharpe, 2),
            sortino_ratio=round(sortino, 2),
            calmar_ratio=round(calmar, 2),
            omega_ratio=round(min(omega, 10), 2),
            information_ratio=round(info_ratio, 2),
            treynor_ratio=round(treynor, 2)
        )
    
    def calculate_relative_strength(
        self,
        stock_prices: pd.Series,
        index_prices: pd.Series,
        lookback: int = None
    ) -> RelativeStrength:
        """Relative Strength hesaplama"""
        lookback = lookback or self.config.rs_lookback
        
        aligned = pd.DataFrame({
            'stock': stock_prices,
            'index': index_prices
        }).dropna()
        
        if len(aligned) < lookback:
            return RelativeStrength(100, 50, "NEUTRAL", 0, 100, False, 0)
        
        rs = aligned['stock'] / aligned['index']
        rs_normalized = rs / rs.iloc[0] * 100
        
        rs_value = rs_normalized.iloc[-1]
        rs_percentile = stats.percentileofscore(rs_normalized.values, rs_value)
        
        recent_rs = rs_normalized.iloc[-lookback:]
        x = np.arange(len(recent_rs))
        slope, _, _, _, _ = stats.linregress(x, recent_rs.values)
        
        if slope > 0.05:
            rs_slope = "POSITIVE"
        elif slope < -0.05:
            rs_slope = "NEGATIVE"
        else:
            rs_slope = "NEUTRAL"
        
        rs_momentum = (rs_normalized.iloc[-1] / rs_normalized.iloc[-lookback] - 1) * 100
        rs_ma = rs_normalized.rolling(lookback).mean().iloc[-1]
        outperforming = rs_value > rs_ma
        
        streak = 0
        for i in range(len(rs_normalized) - 1, -1, -1):
            if rs_normalized.iloc[i] > rs_ma:
                streak += 1
            else:
                break
        
        return RelativeStrength(
            rs_value=round(rs_value, 2),
            rs_percentile=round(rs_percentile, 2),
            rs_slope=rs_slope,
            rs_momentum=round(rs_momentum, 2),
            rs_ma=round(rs_ma, 2),
            outperforming=outperforming,
            outperformance_streak=streak
        )
    
    def _determine_momentum_state(self, rs: RelativeStrength) -> MomentumState:
        """Momentum durumu belirleme"""
        if rs.rs_momentum > 20 and rs.rs_slope == "POSITIVE":
            return MomentumState.ACCELERATING
        elif rs.rs_momentum > 10 and rs.outperforming:
            return MomentumState.STRONG
        elif rs.rs_momentum > 0:
            return MomentumState.POSITIVE
        elif rs.rs_momentum > -10:
            return MomentumState.NEUTRAL
        elif rs.rs_momentum > -20:
            return MomentumState.NEGATIVE
        elif rs.rs_slope == "NEGATIVE":
            return MomentumState.DECELERATING
        else:
            return MomentumState.WEAK
    
    def calculate_rolling_metrics(
        self,
        stock_returns: pd.Series,
        index_returns: pd.Series
    ) -> RollingMetrics:
        """Rolling metrikler hesaplama"""
        aligned = pd.DataFrame({
            'stock': stock_returns,
            'index': index_returns
        }).dropna()
        
        results = {}
        
        for w in [20, 60]:
            if len(aligned) >= w:
                recent = aligned.iloc[-w:]
                cov = recent['stock'].cov(recent['index'])
                var = recent['index'].var()
                beta = cov / var if var != 0 else 1.0
                
                daily_rf = (1 + self.config.risk_free_rate) ** (1/252) - 1
                alpha = recent['stock'].mean() - (daily_rf + beta * (recent['index'].mean() - daily_rf))
                
                corr = recent['stock'].corr(recent['index'])
                rs = (1 + recent['stock']).prod() / (1 + recent['index']).prod()
                
                results[f'alpha_{w}d'] = alpha * 252
                results[f'beta_{w}d'] = beta
                results[f'correlation_{w}d'] = corr
                results[f'rs_{w}d'] = rs * 100
            else:
                results[f'alpha_{w}d'] = 0
                results[f'beta_{w}d'] = 1
                results[f'correlation_{w}d'] = 0
                results[f'rs_{w}d'] = 100
        
        return RollingMetrics(
            alpha_20d=results.get('alpha_20d', 0),
            alpha_60d=results.get('alpha_60d', 0),
            beta_20d=results.get('beta_20d', 1),
            beta_60d=results.get('beta_60d', 1),
            correlation_20d=results.get('correlation_20d', 0),
            correlation_60d=results.get('correlation_60d', 0),
            rs_20d=results.get('rs_20d', 100),
            rs_60d=results.get('rs_60d', 100)
        )
    
    def _calculate_alpha_score(
        self,
        alpha: float,
        sharpe: float,
        sortino: float,
        excess_return: float,
        rs_momentum: float,
        info_ratio: float
    ) -> float:
        """Kompozit Alpha skoru hesapla (0-100)"""
        alpha_norm = np.clip((alpha + 0.5) * 100, 0, 100)
        sharpe_norm = np.clip((sharpe + 1) * 33, 0, 100)
        sortino_norm = np.clip((sortino + 1) * 25, 0, 100)
        excess_norm = np.clip((excess_return + 50), 0, 100)
        rs_norm = np.clip((rs_momentum + 20) * 2.5, 0, 100)
        ir_norm = np.clip((info_ratio + 1) * 33, 0, 100)
        
        score = (
            alpha_norm * 0.25 +
            sharpe_norm * 0.20 +
            sortino_norm * 0.15 +
            excess_norm * 0.20 +
            rs_norm * 0.10 +
            ir_norm * 0.10
        )
        
        return round(score, 2)
    
    def _determine_category(self, alpha_score: float) -> AlphaCategory:
        """Alpha kategorisi belirleme"""
        if alpha_score >= 80:
            return AlphaCategory.LEADER
        elif alpha_score >= 65:
            return AlphaCategory.OUTPERFORMER
        elif alpha_score >= 45:
            return AlphaCategory.MARKET_PERFORMER
        elif alpha_score >= 30:
            return AlphaCategory.UNDERPERFORMER
        else:
            return AlphaCategory.LAGGARD
    
    def _determine_rating(
        self,
        alpha: float,
        sharpe: float,
        excess_return: float,
        rs: RelativeStrength
    ) -> Tuple[PerformanceRating, int, str]:
        """Performance rating belirleme"""
        score = 0
        
        if alpha > self.config.alpha_strong_threshold:
            score += 2
        elif alpha > 0:
            score += 1
        elif alpha < -0.1:
            score -= 1
        
        if sharpe > self.config.sharpe_excellent_threshold:
            score += 2
        elif sharpe > self.config.sharpe_good_threshold:
            score += 1
        elif sharpe < 0:
            score -= 1
        
        if excess_return > 15:
            score += 2
        elif excess_return > 5:
            score += 1
        elif excess_return < -15:
            score -= 1
        
        if rs.outperforming and rs.rs_slope == "POSITIVE":
            score += 1
        elif rs.rs_slope == "NEGATIVE":
            score -= 1
        
        if score >= 6:
            return PerformanceRating.EXCEPTIONAL, 5, "OLAĞANÜSTÜ PERFORMANS"
        elif score >= 4:
            return PerformanceRating.STRONG, 4, "GÜÇLÜ PERFORMANS"
        elif score >= 2:
            return PerformanceRating.GOOD, 3, "İYİ PERFORMANS"
        elif score >= 0:
            return PerformanceRating.NEUTRAL, 2, "NÖTR PERFORMANS"
        elif score >= -2:
            return PerformanceRating.WEAK, 1, "ZAYIF PERFORMANS"
        else:
            return PerformanceRating.POOR, 0, "KÖTÜ PERFORMANS"
    
    def _generate_recommendation(
        self,
        alpha_category: AlphaCategory,
        momentum: MomentumState,
        rs: RelativeStrength
    ) -> str:
        """Öneri oluştur"""
        if alpha_category == AlphaCategory.LEADER:
            if momentum in [MomentumState.ACCELERATING, MomentumState.STRONG]:
                return "GÜÇLÜ AL - Lider hisse, momentum güçlü"
            return "TUT - Lider hisse, momentum beklemede"
        elif alpha_category == AlphaCategory.OUTPERFORMER:
            if momentum in [MomentumState.ACCELERATING, MomentumState.STRONG]:
                return "AL - Outperformer, momentum pozitif"
            return "TUT/BİRİKTİR - Outperformer"
        elif alpha_category == AlphaCategory.MARKET_PERFORMER:
            if rs.outperforming:
                return "NÖTR/BİRİKTİR - Outperform başlıyor"
            return "NÖTR - Endeksle uyumlu"
        elif alpha_category == AlphaCategory.UNDERPERFORMER:
            if momentum in [MomentumState.POSITIVE, MomentumState.ACCELERATING]:
                return "İZLE - Toparlanma sinyalleri"
            return "AZALT - Underperform devam"
        else:
            return "KAÇIN/SAT - Zayıf performans"
    
    def analyze(
        self,
        df: pd.DataFrame,
        df_index: pd.DataFrame,
        symbol: str = None,
        sector_data: Dict[str, pd.Series] = None
    ) -> AlphaAnalysis:
        """Kapsamlı Alpha analizi"""
        try:
            combined = pd.DataFrame({
                'stock': df['Close'],
                'index': df_index['Close']
            }).ffill().dropna()
            
            if len(combined) < self.config.min_data_points:
                return self._empty_analysis()
            
            combined['stock_ret'] = combined['stock'].pct_change()
            combined['index_ret'] = combined['index'].pct_change()
            combined = combined.dropna()
            
            stock_returns = combined['stock_ret']
            index_returns = combined['index_ret']
            stock_prices = combined['stock']
            index_prices = combined['index']
            
            alpha, beta, r_squared = self.calculate_alpha_beta(stock_returns, index_returns)
            returns_metrics = self._analyze_returns(stock_returns, index_returns)
            risk_metrics = self._calculate_risk_metrics(stock_returns, stock_prices)
            ratio_metrics = self._calculate_ratio_metrics(
                stock_returns, index_returns, risk_metrics.max_drawdown, beta
            )
            
            rs = self.calculate_relative_strength(stock_prices, index_prices)
            momentum_state = self._determine_momentum_state(rs)
            rolling = self.calculate_rolling_metrics(stock_returns, index_returns)
            
            alpha_score = self._calculate_alpha_score(
                alpha, ratio_metrics.sharpe_ratio, ratio_metrics.sortino_ratio,
                returns_metrics.excess_return, rs.rs_momentum, ratio_metrics.information_ratio
            )
            
            alpha_category = self._determine_category(alpha_score)
            rating, stars, status = self._determine_rating(
                alpha, ratio_metrics.sharpe_ratio, returns_metrics.excess_return, rs
            )
            recommendation = self._generate_recommendation(alpha_category, momentum_state, rs)
            
            return AlphaAnalysis(
                alpha_score=alpha_score,
                alpha_category=alpha_category,
                jensens_alpha=round(alpha, 4),
                beta=round(beta, 2),
                r_squared=round(r_squared, 2),
                returns=returns_metrics,
                risk=risk_metrics,
                ratios=ratio_metrics,
                relative_strength=rs,
                momentum_state=momentum_state,
                rolling=rolling,
                sector=None,
                factors=None,
                performance_rating=rating,
                strength_rating=stars,
                status=status,
                recommendation=recommendation
            )
        
        except Exception as e:
            logger.error(f"❌ Alpha Engine Error: {e}")
            return self._empty_analysis()
    
    def _empty_analysis(self) -> AlphaAnalysis:
        """Boş analiz sonucu"""
        return AlphaAnalysis(
            alpha_score=0,
            alpha_category=AlphaCategory.MARKET_PERFORMER,
            jensens_alpha=0,
            beta=1,
            r_squared=0,
            returns=ReturnMetrics(0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0),
            risk=RiskMetrics(0, 0, 0, 0, 0, 0, 0, 0),
            ratios=RatioMetrics(0, 0, 0, 0, 0, 0),
            relative_strength=RelativeStrength(100, 50, "NEUTRAL", 0, 100, False, 0),
            momentum_state=MomentumState.NEUTRAL,
            rolling=RollingMetrics(0, 0, 1, 1, 0, 0, 100, 100),
            sector=None,
            factors=None,
            performance_rating=PerformanceRating.NEUTRAL,
            strength_rating=0,
            status="VERİ YETERSİZ",
            recommendation="VERİ BEKLENİYOR"
        )
    
    def calculate_alpha_score(self, df: pd.DataFrame, df_index: pd.DataFrame) -> Dict:
        """Eski API uyumluluğu"""
        analysis = self.analyze(df, df_index)
        return {
            'alpha_score': analysis.alpha_score,
            'beta': analysis.beta,
            'rs_slope': analysis.relative_strength.rs_slope,
            'status': analysis.status,
            'sharpe': analysis.ratios.sharpe_ratio,
            'max_dd': analysis.risk.max_drawdown
        }


if __name__ == "__main__":
    print("Alpha Engine v4.2 - Test")
    print("=" * 60)
    
    import numpy as np
    
    dates = pd.date_range(start='2024-01-01', periods=252, freq='D')
    np.random.seed(42)
    
    stock_prices = 100 * np.exp(np.cumsum(np.random.randn(252) * 0.02 + 0.001))
    index_prices = 100 * np.exp(np.cumsum(np.random.randn(252) * 0.015))
    
    df_stock = pd.DataFrame({
        'Open': stock_prices * 0.99,
        'High': stock_prices * 1.02,
        'Low': stock_prices * 0.98,
        'Close': stock_prices,
        'Volume': np.random.randint(1000000, 5000000, 252)
    }, index=dates)
    
    df_index = pd.DataFrame({
        'Close': index_prices
    }, index=dates)
    
    engine = AlphaEngine()
    analysis = engine.analyze(df_stock, df_index, symbol="THYAO")
    
    print(f"\n📊 ALPHA ANALİZ SONUCU")
    print("=" * 60)
    print(f"🎯 Alpha Score: {analysis.alpha_score}/100")
    print(f"📊 Kategori: {analysis.alpha_category.value}")
    print(f"⭐ Rating: {'⭐' * analysis.strength_rating} ({analysis.status})")
    print(f"\n💡 Öneri: {analysis.recommendation}")
