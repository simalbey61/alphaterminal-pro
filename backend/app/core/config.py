"""
AlphaTerminal Pro - Core Engine Configuration
==============================================

Tüm analiz motorlarının merkezi konfigürasyon dosyası.
Kurumsal seviye parametreler ve sabitler.

Author: AlphaTerminal Team
Version: 4.2.0
"""

import os
import logging
from dataclasses import dataclass, field
from typing import List, Dict, Optional
from enum import Enum
from pathlib import Path

# ═══════════════════════════════════════════════════════════════════════════════
# LOGGING CONFIGURATION
# ═══════════════════════════════════════════════════════════════════════════════

LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")
LOG_FORMAT = "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s"
LOG_DATE_FORMAT = "%Y-%m-%d %H:%M:%S"

logging.basicConfig(
    level=getattr(logging, LOG_LEVEL),
    format=LOG_FORMAT,
    datefmt=LOG_DATE_FORMAT
)

logger = logging.getLogger("AlphaTerminal")

# ═══════════════════════════════════════════════════════════════════════════════
# PATHS & DIRECTORIES
# ═══════════════════════════════════════════════════════════════════════════════

BASE_DIR = Path(__file__).resolve().parent.parent.parent
CACHE_DIR = BASE_DIR / "cache"
DATA_DIR = BASE_DIR / "data"
LOGS_DIR = BASE_DIR / "logs"

# Klasörleri oluştur
for dir_path in [CACHE_DIR, DATA_DIR, LOGS_DIR]:
    dir_path.mkdir(parents=True, exist_ok=True)

# ═══════════════════════════════════════════════════════════════════════════════
# MARKET CONSTANTS
# ═══════════════════════════════════════════════════════════════════════════════

BIST_INDEX = "XU100"
BIST_30 = "XU030"
BIST_BANK = "XBANK"

# Desteklenen zaman dilimleri ve dönemler
SUPPORTED_INTERVALS = ["1m", "5m", "15m", "30m", "1h", "4h", "1d", "1w"]
DEFAULT_INTERVAL = "4h"
DEFAULT_PERIOD = "3mo"

SUPPORTED_PERIODS = {
    "1m": "5d",
    "5m": "5d",
    "15m": "1mo",
    "30m": "1mo",
    "1h": "3mo",
    "4h": "6mo",
    "1d": "2y",
    "1w": "5y"
}

# Piyasa saatleri (BIST)
MARKET_OPEN_HOUR = 10
MARKET_CLOSE_HOUR = 18
MARKET_TIMEZONE = "Europe/Istanbul"

# ═══════════════════════════════════════════════════════════════════════════════
# ENUMS
# ═══════════════════════════════════════════════════════════════════════════════

class MarketStructure(Enum):
    """Piyasa yapısı enumeration"""
    BULLISH = "BULLISH"
    BEARISH = "BEARISH"
    RANGING = "RANGING"
    UNDEFINED = "UNDEFINED"


class TrendStrength(Enum):
    """Trend gücü enumeration"""
    STRONG_BULL = "STRONG_BULL"
    BULL = "BULL"
    WEAK_BULL = "WEAK_BULL"
    NEUTRAL = "NEUTRAL"
    WEAK_BEAR = "WEAK_BEAR"
    BEAR = "BEAR"
    STRONG_BEAR = "STRONG_BEAR"


class ZoneType(Enum):
    """Fiyat bölgesi enumeration"""
    PREMIUM = "PREMIUM"
    DISCOUNT = "DISCOUNT"
    EQUILIBRIUM = "EQUILIBRIUM"


class SignalDirection(Enum):
    """Sinyal yönü enumeration"""
    LONG = "LONG"
    SHORT = "SHORT"
    NEUTRAL = "NEUTRAL"


class SignalStrength(Enum):
    """Sinyal gücü enumeration"""
    STRONG = "STRONG"
    MODERATE = "MODERATE"
    WEAK = "WEAK"


class MarketRegime(Enum):
    """Piyasa rejimi enumeration"""
    TRENDING_BULL = "TRENDING_BULL"
    TRENDING_BEAR = "TRENDING_BEAR"
    RANGING_HIGH_VOL = "RANGING_HIGH_VOL"
    RANGING_LOW_VOL = "RANGING_LOW_VOL"
    BREAKOUT = "BREAKOUT"
    REVERSAL = "REVERSAL"


class VolatilityRegime(Enum):
    """Volatilite rejimi enumeration"""
    VERY_LOW = "VERY_LOW"
    LOW = "LOW"
    NORMAL = "NORMAL"
    HIGH = "HIGH"
    EXTREME = "EXTREME"


class LiquidityRegime(Enum):
    """Likidite rejimi enumeration"""
    DRY = "DRY"
    LOW = "LOW"
    NORMAL = "NORMAL"
    HIGH = "HIGH"
    FLOOD = "FLOOD"


class WyckoffPhase(Enum):
    """Wyckoff fazı enumeration"""
    ACCUMULATION = "ACCUMULATION"
    MARKUP = "MARKUP"
    DISTRIBUTION = "DISTRIBUTION"
    MARKDOWN = "MARKDOWN"
    REACCUMULATION = "REACCUMULATION"
    REDISTRIBUTION = "REDISTRIBUTION"


class OrderBlockType(Enum):
    """Order Block tipi enumeration"""
    BULLISH_OB = "BULLISH_OB"
    BEARISH_OB = "BEARISH_OB"
    BULLISH_BREAKER = "BULLISH_BREAKER"
    BEARISH_BREAKER = "BEARISH_BREAKER"
    BULLISH_MITIGATION = "BULLISH_MITIGATION"
    BEARISH_MITIGATION = "BEARISH_MITIGATION"


class SessionType(Enum):
    """Trading session enumeration"""
    ASIAN = "ASIAN"
    LONDON = "LONDON"
    NEW_YORK = "NEW_YORK"
    BIST = "BIST"
    OVERLAP = "OVERLAP"


# ═══════════════════════════════════════════════════════════════════════════════
# SMC ENGINE CONFIGURATION
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass
class SMCConfig:
    """Smart Money Concepts Engine konfigürasyonu"""
    
    # Swing Point Detection
    swing_lookback: int = 50
    swing_strength: int = 3
    swing_min_distance: int = 5
    
    # Order Block Detection
    ob_lookback: int = 30
    ob_body_multiplier: float = 1.5
    ob_mitigation_threshold: float = 0.5
    ob_max_age_bars: int = 100
    ob_min_strength: float = 30.0
    
    # FVG Detection
    fvg_lookback: int = 30
    fvg_min_size_atr: float = 0.3
    fvg_max_fill_percent: float = 0.7
    
    # Liquidity Detection
    equal_level_tolerance: float = 0.002  # %0.2
    liquidity_lookback: int = 50
    sweep_confirmation_bars: int = 3
    
    # Premium/Discount
    equilibrium_period: int = 50
    premium_threshold: float = 0.618
    discount_threshold: float = 0.382
    
    # Multi-Timeframe
    mtf_timeframes: List[str] = field(default_factory=lambda: ["1h", "4h", "1d"])
    mtf_confluence_weight: float = 0.3
    
    # Wyckoff Analysis
    wyckoff_lookback: int = 100
    wyckoff_volume_threshold: float = 1.5
    
    # Session Analysis
    session_enabled: bool = True
    asian_session: tuple = (0, 8)  # UTC
    london_session: tuple = (8, 16)  # UTC
    ny_session: tuple = (13, 21)  # UTC
    bist_session: tuple = (7, 15)  # UTC (10:00-18:00 Istanbul)


SMC_CONFIG = SMCConfig()


# ═══════════════════════════════════════════════════════════════════════════════
# ORDERFLOW ENGINE CONFIGURATION
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass
class OrderFlowConfig:
    """Order Flow Engine konfigürasyonu"""
    
    # Volume Analysis
    volume_ma_period: int = 20
    volume_spike_threshold: float = 2.0
    volume_climax_threshold: float = 3.0
    
    # Delta Analysis
    delta_lookback: int = 14
    cvd_smoothing: int = 5
    delta_divergence_threshold: float = 0.3
    
    # VWAP
    vwap_std_bands: List[float] = field(default_factory=lambda: [1.0, 2.0, 3.0])
    vwap_anchor_types: List[str] = field(default_factory=lambda: ["session", "week", "month"])
    
    # Volume Profile
    profile_bins: int = 50
    poc_smoothing: int = 3
    value_area_percent: float = 0.70
    
    # Absorption/Exhaustion
    absorption_volume_threshold: float = 1.5
    absorption_price_threshold: float = 0.3
    exhaustion_wick_ratio: float = 0.6
    
    # Institutional Detection
    institutional_volume_mult: float = 2.0
    institutional_body_mult: float = 1.5
    institutional_consecutive_bars: int = 3
    
    # Whale Detection
    whale_volume_threshold: float = 3.0
    whale_body_threshold: float = 2.0
    whale_lookback: int = 20
    
    # Imbalance
    imbalance_ratio_threshold: float = 0.7
    imbalance_volume_threshold: float = 1.5


ORDERFLOW_CONFIG = OrderFlowConfig()


# ═══════════════════════════════════════════════════════════════════════════════
# ALPHA ENGINE CONFIGURATION
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass
class AlphaConfig:
    """Alpha & Performance Engine konfigürasyonu"""
    
    # Risk-Free Rate (Türkiye için)
    risk_free_rate: float = 0.25  # Yıllık %25
    
    # Return Calculation
    return_periods: List[int] = field(default_factory=lambda: [1, 5, 10, 20, 50, 100])
    annualization_factor: int = 252  # Trading days
    
    # Alpha/Beta
    rolling_window: int = 60
    min_data_points: int = 30
    
    # Relative Strength
    rs_lookback: int = 20
    rs_smoothing: int = 5
    rs_momentum_period: int = 10
    
    # VaR
    var_confidence_levels: List[float] = field(default_factory=lambda: [0.95, 0.99])
    var_lookback: int = 252
    
    # Rating Thresholds
    alpha_strong_threshold: float = 0.10
    sharpe_excellent_threshold: float = 2.0
    sharpe_good_threshold: float = 1.0
    sortino_excellent_threshold: float = 2.5
    
    # Correlation
    correlation_lookback: int = 60
    correlation_warning_threshold: float = 0.7
    
    # Sector Analysis
    sector_comparison_enabled: bool = True
    peer_min_count: int = 5


ALPHA_CONFIG = AlphaConfig()


# ═══════════════════════════════════════════════════════════════════════════════
# RISK ENGINE CONFIGURATION
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass
class RiskConfig:
    """Risk Management Engine konfigürasyonu"""
    
    # Position Sizing
    max_risk_per_trade: float = 0.02  # %2 per trade
    max_portfolio_heat: float = 0.10  # %10 total risk
    max_position_size: float = 0.20  # %20 of portfolio
    max_positions: int = 10
    
    # Kelly Criterion
    kelly_fraction_cap: float = 0.25  # Max %25 Kelly
    kelly_min_trades: int = 30
    
    # Stop Loss
    default_sl_atr_mult: float = 2.0
    max_sl_percent: float = 0.08  # Max %8 stop
    min_sl_percent: float = 0.01  # Min %1 stop
    trailing_stop_activation: float = 0.02  # %2 profit to activate
    trailing_stop_distance: float = 0.015  # %1.5 trailing
    
    # Take Profit
    default_rr_ratio: float = 2.0
    tp_levels: List[float] = field(default_factory=lambda: [1.0, 2.0, 3.0])
    tp_allocations: List[float] = field(default_factory=lambda: [0.33, 0.33, 0.34])
    
    # Drawdown Protection
    max_daily_loss: float = 0.03  # %3 daily max loss
    max_weekly_loss: float = 0.07  # %7 weekly max loss
    max_drawdown: float = 0.15  # %15 max drawdown
    
    # Circuit Breakers
    pause_after_consecutive_losses: int = 3
    pause_duration_hours: int = 24
    
    # Correlation Risk
    max_sector_exposure: float = 0.40  # %40 max in one sector
    max_correlation_exposure: float = 0.60  # %60 max correlated
    
    # Monte Carlo
    monte_carlo_simulations: int = 1000
    monte_carlo_confidence: float = 0.95
    
    # Stress Testing
    stress_test_scenarios: List[str] = field(default_factory=lambda: [
        "market_crash_20",
        "volatility_spike",
        "liquidity_crisis",
        "sector_rotation"
    ])


RISK_CONFIG = RiskConfig()


# ═══════════════════════════════════════════════════════════════════════════════
# DATA ENGINE CONFIGURATION
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass
class DataConfig:
    """Data Engine konfigürasyonu"""
    
    # Caching
    cache_ttl_seconds: int = 300  # 5 minutes
    disk_cache_ttl_hours: int = 24
    cache_enabled: bool = True
    
    # Rate Limiting
    rate_limit_delay: float = 0.2
    batch_size: int = 10
    batch_delay: float = 1.0
    
    # Data Sources
    primary_source: str = "tradingview"
    fallback_source: str = "yahoo"
    enable_data_fusion: bool = True
    
    # Data Quality
    min_bars_required: int = 30
    max_missing_percent: float = 0.05  # %5 max missing
    anomaly_detection_enabled: bool = True
    anomaly_std_threshold: float = 4.0
    
    # Indicators
    default_ema_periods: List[int] = field(default_factory=lambda: [9, 20, 50, 200])
    default_sma_periods: List[int] = field(default_factory=lambda: [20, 50, 200])
    rsi_period: int = 14
    atr_period: int = 14
    macd_fast: int = 12
    macd_slow: int = 26
    macd_signal: int = 9
    
    # Multi-Timeframe
    mtf_timeframes: List[str] = field(default_factory=lambda: ["15m", "1h", "4h", "1d"])


DATA_CONFIG = DataConfig()


# ═══════════════════════════════════════════════════════════════════════════════
# CORRELATION ENGINE CONFIGURATION
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass
class CorrelationConfig:
    """Correlation Engine konfigürasyonu"""
    
    # Calculation
    lookback_period: int = 60
    rolling_window: int = 20
    min_data_points: int = 30
    
    # Thresholds
    high_correlation: float = 0.7
    moderate_correlation: float = 0.4
    low_correlation: float = 0.2
    
    # Clustering
    cluster_method: str = "hierarchical"
    cluster_threshold: float = 0.5
    
    # Heatmap
    heatmap_size: int = 30  # Max stocks in heatmap


CORRELATION_CONFIG = CorrelationConfig()


# ═══════════════════════════════════════════════════════════════════════════════
# REGIME DETECTION CONFIGURATION
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass
class RegimeConfig:
    """Regime Detection konfigürasyonu"""
    
    # Trend Detection
    trend_ema_fast: int = 20
    trend_ema_slow: int = 50
    trend_adx_threshold: float = 25.0
    
    # Volatility Detection
    volatility_lookback: int = 20
    volatility_percentile_low: float = 20.0
    volatility_percentile_high: float = 80.0
    
    # Regime Change Detection
    regime_confirmation_bars: int = 5
    regime_min_duration: int = 10
    
    # Hidden Markov Model
    hmm_n_states: int = 4
    hmm_lookback: int = 100


REGIME_CONFIG = RegimeConfig()


# ═══════════════════════════════════════════════════════════════════════════════
# SIGNAL GENERATION CONFIGURATION
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass
class SignalConfig:
    """Signal Generation konfigürasyonu"""
    
    # Confluence
    min_confluence_score: float = 60.0
    strong_signal_threshold: float = 80.0
    
    # Weights
    smc_weight: float = 0.30
    orderflow_weight: float = 0.25
    alpha_weight: float = 0.20
    regime_weight: float = 0.15
    mtf_weight: float = 0.10
    
    # Filters
    min_volume_ratio: float = 0.8
    max_spread_percent: float = 0.01
    min_atr_percent: float = 0.005
    
    # Timing
    signal_validity_bars: int = 5
    signal_cooldown_bars: int = 10


SIGNAL_CONFIG = SignalConfig()


# ═══════════════════════════════════════════════════════════════════════════════
# SHADOW MODE CONFIGURATION
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass
class ShadowModeConfig:
    """Shadow Mode (Paper Trading) konfigürasyonu"""
    
    # Duration
    min_shadow_days: int = 5
    max_shadow_days: int = 30
    min_trades_required: int = 10
    
    # Performance Thresholds
    min_win_rate: float = 0.50
    min_profit_factor: float = 1.2
    min_sharpe_ratio: float = 0.5
    max_drawdown: float = 0.15
    
    # Consistency
    backtest_correlation_threshold: float = 0.7
    regime_consistency_required: bool = True
    
    # Auto-Approval
    auto_approve_enabled: bool = False
    auto_approve_min_score: float = 85.0


SHADOW_MODE_CONFIG = ShadowModeConfig()


# ═══════════════════════════════════════════════════════════════════════════════
# TELEGRAM BOT CONFIGURATION
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass
class TelegramConfig:
    """Telegram Bot konfigürasyonu"""
    
    # Rate Limiting
    max_messages_per_minute: int = 20
    message_delay_seconds: float = 0.1
    
    # Notifications
    signal_notifications: bool = True
    daily_summary: bool = True
    portfolio_alerts: bool = True
    
    # NLP
    nlp_enabled: bool = True
    nlp_confidence_threshold: float = 0.7
    
    # Commands
    admin_commands_enabled: bool = True


TELEGRAM_CONFIG = TelegramConfig()


# ═══════════════════════════════════════════════════════════════════════════════
# SECTOR MAPPING
# ═══════════════════════════════════════════════════════════════════════════════

SECTOR_MAPPING: Dict[str, List[str]] = {
    "BANKA": ["AKBNK", "GARAN", "YKBNK", "ISCTR", "HALKB", "VAKBN", "TSKB", "QNBFB", "SKBNK", "ALBRK"],
    "HOLDİNG": ["KCHOL", "SAHOL", "DOHOL", "SISE", "AGHOL", "GSDHO", "NTHOL", "POLHO"],
    "HAVAYOLU": ["THYAO", "PGSUS", "CLEBI", "TAVHL"],
    "OTOMOTİV": ["FROTO", "TOASO", "DOAS", "OTKAR", "ASUZU", "BRISA"],
    "DEMİR_ÇELİK": ["EREGL", "KRDMD", "KRDMA", "KRDMB", "BRSAN"],
    "ENERJİ": ["AKSEN", "AYDEM", "ENJSA", "AYEN", "ZOREN", "AKENR", "AKSA"],
    "PERAKENDE": ["BIMAS", "MGROS", "SOKM", "BIZIM", "MAVI"],
    "TEKNOLOJİ": ["LOGO", "INDES", "ARENA", "NETAS", "KRONT", "LINK"],
    "SAVUNMA": ["ASELS"],
    "KİMYA": ["PETKM", "SASA", "BAGFS", "GUBRF", "HEKTS"],
    "İNŞAAT": ["ENKAI", "TKFEN"],
    "GIDA": ["AEFES", "CCOLA", "ULKER", "BANVT", "TATGD"],
    "TELEKOMÜNİKASYON": ["TCELL", "TTKOM"],
    "GYO": ["EKGYO", "ISGYO", "HLGYO", "KLGYO", "TRGYO"],
    "SPOR": ["FENER", "GSRAY", "BJKAS", "TSPOR"],
}


def get_sector(symbol: str) -> Optional[str]:
    """Hisse sembolünün sektörünü döndür"""
    for sector, symbols in SECTOR_MAPPING.items():
        if symbol in symbols:
            return sector
    return None


def get_sector_symbols(sector: str) -> List[str]:
    """Sektördeki tüm hisseleri döndür"""
    return SECTOR_MAPPING.get(sector, [])


# ═══════════════════════════════════════════════════════════════════════════════
# BIST30 & BIST100 LISTS
# ═══════════════════════════════════════════════════════════════════════════════

BIST30_SYMBOLS = [
    "THYAO", "GARAN", "AKBNK", "YKBNK", "ISCTR", "EREGL", "BIMAS",
    "ASELS", "KCHOL", "TUPRS", "SISE", "SAHOL", "FROTO", "TOASO",
    "TCELL", "PGSUS", "ARCLK", "TAVHL", "PETKM", "SASA", "EKGYO",
    "HEKTS", "GUBRF", "KONTR", "ENKAI", "TKFEN", "TTKOM", "KRDMD",
    "SOKM", "MGROS"
]

BIST100_ADDITIONAL = [
    "DOAS", "MAVI", "VESTL", "OTKAR", "AEFES", "AKSA", "ALARK",
    "ANHYT", "ASTOR", "BERA", "BRISA", "CCOLA", "CEMTS", "DOHOL",
    "EGEEN", "ENJSA", "GESAN", "GLYHO", "GOLTS", "ISGYO", "KARSN",
    "OYAKC", "ANSGR", "AGHOL", "AKSEN", "ALBRK", "ALGYO", "ALKIM",
    "ASUZU", "AYDEM", "BAGFS", "BANVT", "BIENY", "BIZIM", "CANTE",
    "CWENE", "GWIND", "NATEN", "ODAS", "ZOREN", "LOGO", "INDES",
    "HALKB", "VAKBN", "TSKB", "QNBFB", "SKBNK", "AGESA", "AKGRT",
    "TURSG", "RAYSG", "FENER", "GSRAY", "BJKAS"
]

BIST100_SYMBOLS = BIST30_SYMBOLS + BIST100_ADDITIONAL


# ═══════════════════════════════════════════════════════════════════════════════
# EXPORTS
# ═══════════════════════════════════════════════════════════════════════════════

__all__ = [
    # Logger
    "logger",
    
    # Paths
    "BASE_DIR", "CACHE_DIR", "DATA_DIR", "LOGS_DIR",
    
    # Constants
    "BIST_INDEX", "BIST_30", "BIST_BANK",
    "SUPPORTED_INTERVALS", "SUPPORTED_PERIODS",
    "DEFAULT_INTERVAL", "DEFAULT_PERIOD",
    "MARKET_OPEN_HOUR", "MARKET_CLOSE_HOUR", "MARKET_TIMEZONE",
    
    # Enums
    "MarketStructure", "TrendStrength", "ZoneType",
    "SignalDirection", "SignalStrength", "MarketRegime",
    "VolatilityRegime", "LiquidityRegime", "WyckoffPhase",
    "OrderBlockType", "SessionType",
    
    # Configs
    "SMC_CONFIG", "ORDERFLOW_CONFIG", "ALPHA_CONFIG",
    "RISK_CONFIG", "DATA_CONFIG", "CORRELATION_CONFIG",
    "REGIME_CONFIG", "SIGNAL_CONFIG", "SHADOW_MODE_CONFIG",
    "TELEGRAM_CONFIG",
    
    # Sector Data
    "SECTOR_MAPPING", "get_sector", "get_sector_symbols",
    "BIST30_SYMBOLS", "BIST100_SYMBOLS",
]
