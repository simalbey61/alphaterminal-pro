"""
AlphaTerminal Pro - AI Strategy Constants & Enums
=================================================

AI strateji sistemi için sabit değerler ve enum tanımları.

Author: AlphaTerminal Team
Version: 1.0.0
"""

from __future__ import annotations

from enum import Enum, auto
from typing import Dict, List, Any
from dataclasses import dataclass, field
from decimal import Decimal


# =============================================================================
# REGIME ENUMS
# =============================================================================

class TrendRegime(str, Enum):
    """Trend rejimi."""
    STRONG_BULL = "strong_bull"      # >20% yıllık
    BULL = "bull"                     # 10-20% yıllık
    WEAK_BULL = "weak_bull"          # 0-10% yıllık
    SIDEWAYS = "sideways"            # ±5%
    WEAK_BEAR = "weak_bear"          # -10-0% yıllık
    BEAR = "bear"                    # -20 to -10% yıllık
    STRONG_BEAR = "strong_bear"      # <-20% yıllık


class VolatilityRegime(str, Enum):
    """Volatilite rejimi."""
    VERY_LOW = "very_low"       # VIX < 12
    LOW = "low"                  # VIX 12-16
    NORMAL = "normal"            # VIX 16-20
    ELEVATED = "elevated"        # VIX 20-25
    HIGH = "high"                # VIX 25-30
    EXTREME = "extreme"          # VIX > 30


class LiquidityRegime(str, Enum):
    """Likidite rejimi."""
    THIN = "thin"                # Volume < 50% of avg
    LOW = "low"                  # Volume 50-80% of avg
    NORMAL = "normal"            # Volume 80-120% of avg
    HIGH = "high"                # Volume 120-200% of avg
    EXTREME = "extreme"          # Volume > 200% of avg


class MarketPhase(str, Enum):
    """Piyasa fazı (Wyckoff)."""
    ACCUMULATION = "accumulation"
    MARKUP = "markup"
    DISTRIBUTION = "distribution"
    MARKDOWN = "markdown"
    REACCUMULATION = "reaccumulation"
    REDISTRIBUTION = "redistribution"


# =============================================================================
# STRATEGY ENUMS
# =============================================================================

class StrategyType(str, Enum):
    """Strateji tipi."""
    TREND_FOLLOWING = "trend_following"
    MEAN_REVERSION = "mean_reversion"
    MOMENTUM = "momentum"
    BREAKOUT = "breakout"
    SMC_BASED = "smc_based"
    ORDERFLOW = "orderflow"
    HYBRID = "hybrid"
    ML_BASED = "ml_based"


class DiscoveryMethod(str, Enum):
    """Strateji keşif yöntemi."""
    DECISION_TREE = "decision_tree"
    CLUSTERING = "clustering"
    GENETIC_ALGORITHM = "genetic"
    NEURAL_NETWORK = "neural_network"
    RULE_MINING = "rule_mining"
    BREEDING = "breeding"
    MANUAL = "manual"


class StrategyLifecycle(str, Enum):
    """Strateji yaşam döngüsü."""
    DISCOVERED = "discovered"     # Yeni keşfedildi
    BACKTESTING = "backtesting"   # Backtest ediliyor
    PENDING = "pending"           # Onay bekliyor
    SANDBOX = "sandbox"           # Paper trading
    PROBATION = "probation"       # Deneme süresi
    ACTIVE = "active"             # Aktif trading
    PAUSED = "paused"             # Geçici durdurulmuş
    RETIRING = "retiring"         # Emeklilik süreci
    RETIRED = "retired"           # Emekli


class SignalType(str, Enum):
    """Sinyal tipi."""
    LONG = "long"
    SHORT = "short"
    CLOSE_LONG = "close_long"
    CLOSE_SHORT = "close_short"
    NEUTRAL = "neutral"


# =============================================================================
# FEATURE ENUMS
# =============================================================================

class FeatureCategory(str, Enum):
    """Feature kategorisi."""
    TREND = "trend"
    MOMENTUM = "momentum"
    VOLATILITY = "volatility"
    VOLUME = "volume"
    CANDLESTICK = "candlestick"
    SMC = "smc"
    ORDERFLOW = "orderflow"
    ALPHA = "alpha"
    REGIME = "regime"
    CUSTOM = "custom"


class FeatureTimeframe(str, Enum):
    """Feature timeframe."""
    M1 = "1m"
    M5 = "5m"
    M15 = "15m"
    M30 = "30m"
    H1 = "1h"
    H4 = "4h"
    D1 = "1d"
    W1 = "1w"
    MN1 = "1M"


class CalculationMode(str, Enum):
    """Hesaplama modu."""
    FULL = "full"                 # Tüm history
    INCREMENTAL = "incremental"   # Sadece son bar
    WINDOWED = "windowed"         # Son N bar


# =============================================================================
# VALIDATION ENUMS
# =============================================================================

class ValidationStatus(str, Enum):
    """Validasyon durumu."""
    NOT_STARTED = "not_started"
    IN_PROGRESS = "in_progress"
    PASSED = "passed"
    FAILED = "failed"
    NEEDS_REVIEW = "needs_review"


class ApprovalDecision(str, Enum):
    """Onay kararı."""
    APPROVED = "approved"
    SANDBOX = "sandbox"           # Paper trading'e yönlendir
    REJECTED = "rejected"
    NEEDS_MORE_DATA = "needs_more_data"


# =============================================================================
# ALERT & MONITORING ENUMS
# =============================================================================

class AlertSeverity(str, Enum):
    """Alert şiddeti."""
    CRITICAL = "critical"
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"
    INFO = "info"


class PerformanceAlert(str, Enum):
    """Performans alert tipi."""
    CONSECUTIVE_LOSSES = "consecutive_losses"
    DRAWDOWN_WARNING = "drawdown_warning"
    DRAWDOWN_CRITICAL = "drawdown_critical"
    WIN_RATE_DEVIATION = "win_rate_deviation"
    SHARPE_DEGRADATION = "sharpe_degradation"
    REGIME_MISMATCH = "regime_mismatch"


# =============================================================================
# THRESHOLDS & CONSTANTS
# =============================================================================

@dataclass(frozen=True)
class ApprovalThresholds:
    """Strateji onay eşikleri."""
    
    # Mandatory (must pass all)
    min_win_rate: float = 0.55
    min_profit_factor: float = 1.5
    min_sharpe_ratio: float = 1.0
    max_drawdown: float = 0.15
    min_walkforward_consistency: float = 0.60
    min_monte_carlo_var95: float = -0.10
    min_robustness_score: float = 0.70
    min_sample_size: int = 50
    
    # Soft (warning only)
    min_profit_vs_spread: float = 3.0  # Expected profit > 3x spread
    min_trade_duration_hours: float = 4.0
    min_trades_per_month: int = 5


@dataclass(frozen=True)
class RetirementThresholds:
    """Emeklilik eşikleri."""
    
    # Triggers (any one activates)
    min_win_rate_10_trades: float = 0.35
    sharpe_degradation_pct: float = 0.50  # < 50% of expected
    max_consecutive_losses: int = 5
    max_drawdown_exceeded: bool = True
    monitoring_window: int = 10  # trades


@dataclass(frozen=True)
class PositionSizingLimits:
    """Pozisyon boyutlandırma limitleri."""
    
    max_position_pct: float = 0.05       # Max 5% per position
    max_portfolio_heat: float = 0.20     # Max 20% total risk
    max_sector_exposure: float = 0.30    # Max 30% per sector
    max_correlated_exposure: float = 0.50  # Max 50% correlated
    kelly_fraction: float = 0.25         # Conservative Kelly


@dataclass(frozen=True)
class BacktestDefaults:
    """Backtest varsayılan değerleri."""
    
    initial_capital: Decimal = Decimal("100000")
    commission_pct: float = 0.001        # 0.1%
    slippage_pct: float = 0.002          # 0.2%
    spread_pct: float = 0.001            # 0.1%
    min_trade_duration_bars: int = 2
    max_holding_period_days: int = 30


@dataclass(frozen=True)
class ValidationDefaults:
    """Validasyon varsayılan değerleri."""
    
    # Purged K-Fold
    n_splits: int = 5
    purge_gap: int = 20
    embargo_pct: float = 0.01
    
    # Walk-Forward
    train_pct: float = 0.70
    min_windows: int = 8
    max_windows: int = 24
    
    # Monte Carlo
    simulations: int = 10000
    confidence_level: float = 0.95


@dataclass(frozen=True)
class EvolutionDefaults:
    """Evrim varsayılan değerleri."""
    
    # Genetic Algorithm
    population_size: int = 100
    generations: int = 50
    elite_pct: float = 0.10
    crossover_prob: float = 0.50
    mutation_prob: float = 0.10
    tournament_size: int = 3
    
    # Diversity
    max_correlation: float = 0.40  # Max pairwise correlation
    min_strategies_per_regime: int = 3


# =============================================================================
# FEATURE DEFINITIONS
# =============================================================================

TECHNICAL_FEATURES: Dict[str, Dict[str, Any]] = {
    # TREND
    "sma_5": {"category": "trend", "window": 5, "incremental": True},
    "sma_10": {"category": "trend", "window": 10, "incremental": True},
    "sma_20": {"category": "trend", "window": 20, "incremental": True},
    "sma_50": {"category": "trend", "window": 50, "incremental": True},
    "sma_100": {"category": "trend", "window": 100, "incremental": True},
    "sma_200": {"category": "trend", "window": 200, "incremental": True},
    
    "ema_9": {"category": "trend", "window": 9, "incremental": True},
    "ema_12": {"category": "trend", "window": 12, "incremental": True},
    "ema_21": {"category": "trend", "window": 21, "incremental": True},
    "ema_26": {"category": "trend", "window": 26, "incremental": True},
    "ema_50": {"category": "trend", "window": 50, "incremental": True},
    
    "adx_14": {"category": "trend", "window": 14, "incremental": False},
    "di_plus_14": {"category": "trend", "window": 14, "incremental": False},
    "di_minus_14": {"category": "trend", "window": 14, "incremental": False},
    
    # MOMENTUM
    "rsi_7": {"category": "momentum", "window": 7, "incremental": True},
    "rsi_14": {"category": "momentum", "window": 14, "incremental": True},
    "rsi_21": {"category": "momentum", "window": 21, "incremental": True},
    
    "stoch_k": {"category": "momentum", "window": 14, "incremental": False},
    "stoch_d": {"category": "momentum", "window": 14, "incremental": False},
    
    "macd": {"category": "momentum", "window": 26, "incremental": True},
    "macd_signal": {"category": "momentum", "window": 26, "incremental": True},
    "macd_hist": {"category": "momentum", "window": 26, "incremental": True},
    
    "cci_14": {"category": "momentum", "window": 14, "incremental": False},
    "mfi_14": {"category": "momentum", "window": 14, "incremental": False},
    "williams_r": {"category": "momentum", "window": 14, "incremental": False},
    
    # VOLATILITY
    "atr_7": {"category": "volatility", "window": 7, "incremental": True},
    "atr_14": {"category": "volatility", "window": 14, "incremental": True},
    "atr_21": {"category": "volatility", "window": 21, "incremental": True},
    
    "bb_upper": {"category": "volatility", "window": 20, "incremental": True},
    "bb_middle": {"category": "volatility", "window": 20, "incremental": True},
    "bb_lower": {"category": "volatility", "window": 20, "incremental": True},
    "bb_width": {"category": "volatility", "window": 20, "incremental": True},
    "bb_pct_b": {"category": "volatility", "window": 20, "incremental": True},
    
    "keltner_upper": {"category": "volatility", "window": 20, "incremental": True},
    "keltner_lower": {"category": "volatility", "window": 20, "incremental": True},
    
    "historical_vol_20": {"category": "volatility", "window": 20, "incremental": True},
    
    # VOLUME
    "obv": {"category": "volume", "window": 1, "incremental": True},
    "volume_sma_20": {"category": "volume", "window": 20, "incremental": True},
    "volume_ratio": {"category": "volume", "window": 20, "incremental": True},
    "vwap": {"category": "volume", "window": 1, "incremental": True},
    "cmf": {"category": "volume", "window": 20, "incremental": False},
    "adl": {"category": "volume", "window": 1, "incremental": True},
}

SMC_FEATURES: Dict[str, Dict[str, Any]] = {
    "structure_type": {"category": "smc", "window": 50, "incremental": False},
    "swing_high_distance": {"category": "smc", "window": 50, "incremental": False},
    "swing_low_distance": {"category": "smc", "window": 50, "incremental": False},
    "bos_bullish_count": {"category": "smc", "window": 50, "incremental": False},
    "bos_bearish_count": {"category": "smc", "window": 50, "incremental": False},
    "choch_detected": {"category": "smc", "window": 50, "incremental": False},
    
    "bullish_ob_count": {"category": "smc", "window": 100, "incremental": False},
    "bearish_ob_count": {"category": "smc", "window": 100, "incremental": False},
    "nearest_bullish_ob": {"category": "smc", "window": 100, "incremental": False},
    "nearest_bearish_ob": {"category": "smc", "window": 100, "incremental": False},
    "ob_strength": {"category": "smc", "window": 100, "incremental": False},
    
    "bullish_fvg_count": {"category": "smc", "window": 50, "incremental": False},
    "bearish_fvg_count": {"category": "smc", "window": 50, "incremental": False},
    "fvg_proximity": {"category": "smc", "window": 50, "incremental": False},
    
    "buy_liquidity_distance": {"category": "smc", "window": 100, "incremental": False},
    "sell_liquidity_distance": {"category": "smc", "window": 100, "incremental": False},
    "liquidity_sweep_count": {"category": "smc", "window": 50, "incremental": False},
}

ORDERFLOW_FEATURES: Dict[str, Dict[str, Any]] = {
    "delta": {"category": "orderflow", "window": 1, "incremental": True},
    "delta_pct": {"category": "orderflow", "window": 1, "incremental": True},
    "cvd": {"category": "orderflow", "window": 20, "incremental": True},
    "cvd_slope": {"category": "orderflow", "window": 10, "incremental": True},
    "delta_divergence": {"category": "orderflow", "window": 14, "incremental": False},
    "absorption_score": {"category": "orderflow", "window": 10, "incremental": False},
    "institutional_flow": {"category": "orderflow", "window": 20, "incremental": False},
}

ALPHA_FEATURES: Dict[str, Dict[str, Any]] = {
    "alpha_vs_index": {"category": "alpha", "window": 60, "incremental": False},
    "alpha_vs_sector": {"category": "alpha", "window": 60, "incremental": False},
    "sharpe_20": {"category": "alpha", "window": 20, "incremental": False},
    "sortino_20": {"category": "alpha", "window": 20, "incremental": False},
    "rs_rank": {"category": "alpha", "window": 20, "incremental": False},
    "momentum_20": {"category": "alpha", "window": 20, "incremental": True},
    "momentum_60": {"category": "alpha", "window": 60, "incremental": True},
}

# Tüm feature'ları birleştir
ALL_FEATURES = {
    **TECHNICAL_FEATURES,
    **SMC_FEATURES,
    **ORDERFLOW_FEATURES,
    **ALPHA_FEATURES,
}


# =============================================================================
# STRATEGY ZOO CATEGORIES
# =============================================================================

STRATEGY_ZOO_CATEGORIES: Dict[str, Dict[str, Any]] = {
    "bull_specialist": {
        "name": "Bull Market Specialist",
        "target_regime": [TrendRegime.STRONG_BULL, TrendRegime.BULL],
        "min_strategies": 3,
        "description": "Boğa piyasasına özel stratejiler",
    },
    "bear_specialist": {
        "name": "Bear Market Specialist",
        "target_regime": [TrendRegime.STRONG_BEAR, TrendRegime.BEAR],
        "min_strategies": 3,
        "description": "Ayı piyasasına özel stratejiler",
    },
    "sideways_trader": {
        "name": "Sideways/Range Trader",
        "target_regime": [TrendRegime.SIDEWAYS],
        "min_strategies": 3,
        "description": "Yatay piyasa stratejileri",
    },
    "high_vol_player": {
        "name": "High Volatility Player",
        "target_regime": [VolatilityRegime.HIGH, VolatilityRegime.EXTREME],
        "min_strategies": 2,
        "description": "Yüksek volatilite stratejileri",
    },
    "low_vol_player": {
        "name": "Low Volatility Player",
        "target_regime": [VolatilityRegime.LOW, VolatilityRegime.VERY_LOW],
        "min_strategies": 2,
        "description": "Düşük volatilite stratejileri",
    },
    "all_weather": {
        "name": "All-Weather Strategy",
        "target_regime": None,  # Tüm rejimlerde çalışır
        "min_strategies": 5,
        "description": "Her koşulda çalışan stratejiler",
    },
}
