"""
AlphaTerminal Pro - Configuration Module
=========================================

Kurumsal seviye konfigürasyon yönetimi.
Environment variables, uygulama sabitleri ve yapılandırma sınıfları.

Author: AlphaTerminal Team
Version: 1.0.0
"""

from __future__ import annotations

import os
import logging
from pathlib import Path
from functools import lru_cache
from typing import List, Optional, Dict, Any
from enum import Enum

from pydantic import Field, field_validator, PostgresDsn, RedisDsn
from pydantic_settings import BaseSettings, SettingsConfigDict


# =============================================================================
# PATH CONFIGURATION
# =============================================================================

# Proje kök dizini
BASE_DIR = Path(__file__).resolve().parent.parent
APP_DIR = BASE_DIR / "app"
DATA_DIR = BASE_DIR / "data"
LOGS_DIR = BASE_DIR / "logs"
CACHE_DIR = DATA_DIR / "cache"
MODELS_DIR = DATA_DIR / "models"

# Dizinleri oluştur
for directory in [DATA_DIR, LOGS_DIR, CACHE_DIR, MODELS_DIR]:
    directory.mkdir(parents=True, exist_ok=True)


# =============================================================================
# ENUMS
# =============================================================================

class Environment(str, Enum):
    """Uygulama ortamları."""
    DEVELOPMENT = "development"
    STAGING = "staging"
    PRODUCTION = "production"
    TESTING = "testing"


class LogLevel(str, Enum):
    """Log seviyeleri."""
    DEBUG = "DEBUG"
    INFO = "INFO"
    WARNING = "WARNING"
    ERROR = "ERROR"
    CRITICAL = "CRITICAL"


class MarketStructure(str, Enum):
    """Piyasa yapısı türleri."""
    BULLISH = "BULLISH"
    BEARISH = "BEARISH"
    RANGING = "RANGING"
    UNDEFINED = "UNDEFINED"


class SignalType(str, Enum):
    """Sinyal türleri."""
    LONG = "LONG"
    SHORT = "SHORT"
    NEUTRAL = "NEUTRAL"


class SignalTier(str, Enum):
    """Sinyal kalite seviyeleri."""
    TIER1 = "TIER1"  # En yüksek kalite (>85 skor)
    TIER2 = "TIER2"  # Orta kalite (70-85 skor)
    TIER3 = "TIER3"  # Düşük kalite (55-70 skor)
    NONE = "NONE"    # Sinyal yok


class SignalStatus(str, Enum):
    """Sinyal durumları."""
    ACTIVE = "active"
    TP1_HIT = "tp1_hit"
    TP2_HIT = "tp2_hit"
    TP3_HIT = "tp3_hit"
    STOPPED = "stopped"
    EXPIRED = "expired"
    CLOSED = "closed"


class StrategyStatus(str, Enum):
    """AI strateji durumları."""
    PENDING_VALIDATION = "pending_validation"
    ACTIVE = "active"
    PAUSED = "paused"
    RETIRED = "retired"


class MarketRegime(str, Enum):
    """Piyasa rejimi türleri."""
    TRENDING_UP = "trending_up"
    TRENDING_DOWN = "trending_down"
    RANGING = "ranging"
    VOLATILE = "volatile"
    LOW_VOLATILITY = "low_volatility"


class ZoneType(str, Enum):
    """SMC bölge türleri."""
    ORDER_BLOCK = "order_block"
    BREAKER_BLOCK = "breaker_block"
    MITIGATION_BLOCK = "mitigation_block"
    FVG = "fvg"
    LIQUIDITY = "liquidity"


class FlowDirection(str, Enum):
    """Order flow yön türleri."""
    BULLISH = "BULLISH"
    BEARISH = "BEARISH"
    NEUTRAL = "NEUTRAL"


# =============================================================================
# SETTINGS CLASSES
# =============================================================================

class DatabaseSettings(BaseSettings):
    """Veritabanı ayarları."""
    
    model_config = SettingsConfigDict(
        env_prefix="DB_",
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore"
    )
    
    host: str = Field(default="localhost", description="Veritabanı sunucu adresi")
    port: int = Field(default=5432, description="Veritabanı portu")
    name: str = Field(default="alphaterminal", description="Veritabanı adı")
    user: str = Field(default="postgres", description="Veritabanı kullanıcısı")
    password: str = Field(default="postgres", description="Veritabanı şifresi")
    
    pool_size: int = Field(default=10, description="Connection pool boyutu")
    max_overflow: int = Field(default=20, description="Maksimum overflow bağlantısı")
    pool_timeout: int = Field(default=30, description="Pool timeout (saniye)")
    pool_recycle: int = Field(default=1800, description="Bağlantı yenileme süresi")
    echo: bool = Field(default=False, description="SQL sorgularını logla")
    
    @property
    def url(self) -> str:
        """PostgreSQL bağlantı URL'i."""
        return f"postgresql+asyncpg://{self.user}:{self.password}@{self.host}:{self.port}/{self.name}"
    
    @property
    def sync_url(self) -> str:
        """Senkron PostgreSQL bağlantı URL'i."""
        return f"postgresql://{self.user}:{self.password}@{self.host}:{self.port}/{self.name}"


class RedisSettings(BaseSettings):
    """Redis ayarları."""
    
    model_config = SettingsConfigDict(
        env_prefix="REDIS_",
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore"
    )
    
    host: str = Field(default="localhost", description="Redis sunucu adresi")
    port: int = Field(default=6379, description="Redis portu")
    db: int = Field(default=0, description="Redis veritabanı numarası")
    password: Optional[str] = Field(default=None, description="Redis şifresi")
    
    # TTL Ayarları (saniye)
    market_data_ttl: int = Field(default=300, description="Market data cache TTL")
    analysis_ttl: int = Field(default=300, description="Analiz sonuçları TTL")
    feature_ttl: int = Field(default=600, description="Feature cache TTL")
    signal_ttl: int = Field(default=3600, description="Sinyal cache TTL")
    strategy_ttl: int = Field(default=3600, description="Strateji cache TTL")
    
    @property
    def url(self) -> str:
        """Redis bağlantı URL'i."""
        if self.password:
            return f"redis://:{self.password}@{self.host}:{self.port}/{self.db}"
        return f"redis://{self.host}:{self.port}/{self.db}"


class TelegramSettings(BaseSettings):
    """Telegram bot ayarları."""
    
    model_config = SettingsConfigDict(
        env_prefix="TELEGRAM_",
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore"
    )
    
    token: str = Field(default="", description="Telegram bot token")
    chat_id: str = Field(default="", description="Ana kanal/grup chat ID")
    admin_ids: List[int] = Field(default_factory=list, description="Admin kullanıcı ID'leri")
    
    webhook_url: Optional[str] = Field(default=None, description="Webhook URL")
    use_webhook: bool = Field(default=False, description="Webhook kullan")
    
    rate_limit_messages: int = Field(default=30, description="Dakikada maksimum mesaj")
    rate_limit_window: int = Field(default=60, description="Rate limit penceresi (saniye)")


class JWTSettings(BaseSettings):
    """JWT authentication ayarları."""
    
    model_config = SettingsConfigDict(
        env_prefix="JWT_",
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore"
    )
    
    secret_key: str = Field(
        default="your-super-secret-key-change-in-production",
        description="JWT secret key"
    )
    algorithm: str = Field(default="HS256", description="JWT algoritması")
    access_token_expire_minutes: int = Field(default=30, description="Access token süresi (dakika)")
    refresh_token_expire_days: int = Field(default=7, description="Refresh token süresi (gün)")


class SMCSettings(BaseSettings):
    """Smart Money Concepts ayarları."""
    
    model_config = SettingsConfigDict(
        env_prefix="SMC_",
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore"
    )
    
    swing_lookback: int = Field(default=20, description="Swing point lookback periyodu")
    swing_strength: int = Field(default=3, description="Swing point onay bar sayısı")
    structure_lookback: int = Field(default=50, description="Yapı analizi lookback")
    
    ob_lookback: int = Field(default=50, description="Order block lookback")
    ob_body_multiplier: float = Field(default=1.5, description="OB body çarpanı")
    ob_max_age: int = Field(default=100, description="OB maksimum yaşı (bar)")
    ob_mitigation_threshold: float = Field(default=0.5, description="OB mitigation eşiği")
    
    fvg_lookback: int = Field(default=30, description="FVG lookback")
    fvg_min_size_atr: float = Field(default=0.3, description="Minimum FVG boyutu (ATR)")
    
    liquidity_lookback: int = Field(default=50, description="Likidite analizi lookback")
    equal_level_tolerance: float = Field(default=0.001, description="Eşit seviye toleransı")
    sweep_confirmation_candles: int = Field(default=3, description="Sweep onay bar sayısı")
    
    equilibrium_period: int = Field(default=50, description="Equilibrium hesaplama periyodu")


class OrderFlowSettings(BaseSettings):
    """Order Flow analiz ayarları."""
    
    model_config = SettingsConfigDict(
        env_prefix="ORDERFLOW_",
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore"
    )
    
    volume_ma_period: int = Field(default=20, description="Hacim MA periyodu")
    volume_spike_threshold: float = Field(default=2.0, description="Hacim spike eşiği")
    
    delta_lookback: int = Field(default=20, description="Delta lookback")
    cvd_smoothing: int = Field(default=5, description="CVD smoothing periyodu")
    
    vwap_std_bands: List[float] = Field(
        default=[1.0, 2.0, 3.0],
        description="VWAP standart sapma bantları"
    )
    
    absorption_volume_threshold: float = Field(default=1.5, description="Absorption hacim eşiği")
    absorption_price_threshold: float = Field(default=0.3, description="Absorption fiyat eşiği")


class RiskSettings(BaseSettings):
    """Risk yönetimi ayarları."""
    
    model_config = SettingsConfigDict(
        env_prefix="RISK_",
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore"
    )
    
    max_risk_per_trade: float = Field(default=0.02, description="Trade başına maksimum risk")
    max_portfolio_heat: float = Field(default=0.06, description="Maksimum portfolio heat")
    max_positions: int = Field(default=5, description="Maksimum açık pozisyon")
    
    default_sl_atr_mult: float = Field(default=1.5, description="Default stop loss ATR çarpanı")
    max_sl_percent: float = Field(default=0.05, description="Maksimum stop loss yüzdesi")
    
    default_rr_ratio: float = Field(default=2.0, description="Default risk/reward oranı")
    partial_tp_levels: List[float] = Field(
        default=[0.5, 0.75, 1.0],
        description="Kısmi kar alma seviyeleri"
    )
    
    max_daily_loss: float = Field(default=0.03, description="Maksimum günlük kayıp")
    max_weekly_loss: float = Field(default=0.06, description="Maksimum haftalık kayıp")
    pause_after_consecutive_losses: int = Field(default=3, description="Ardışık kayıp sonrası duraklama")


class SignalSettings(BaseSettings):
    """Sinyal üretim ayarları."""
    
    model_config = SettingsConfigDict(
        env_prefix="SIGNAL_",
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore"
    )
    
    # Ağırlıklar (toplam 1.0 olmalı)
    smc_weight: float = Field(default=0.30, description="SMC ağırlığı")
    orderflow_weight: float = Field(default=0.25, description="Order flow ağırlığı")
    alpha_weight: float = Field(default=0.20, description="Alpha ağırlığı")
    ml_weight: float = Field(default=0.15, description="ML ağırlığı")
    mtf_weight: float = Field(default=0.10, description="MTF ağırlığı")
    
    # Eşikler
    min_signal_score: float = Field(default=65.0, description="Minimum sinyal skoru")
    strong_signal_score: float = Field(default=80.0, description="Güçlü sinyal skoru")
    
    tier1_threshold: float = Field(default=85.0, description="Tier 1 eşiği")
    tier2_threshold: float = Field(default=70.0, description="Tier 2 eşiği")
    tier3_threshold: float = Field(default=55.0, description="Tier 3 eşiği")
    
    @field_validator("smc_weight", "orderflow_weight", "alpha_weight", "ml_weight", "mtf_weight")
    @classmethod
    def validate_weights(cls, v: float) -> float:
        """Ağırlıkların geçerli aralıkta olduğunu doğrula."""
        if not 0 <= v <= 1:
            raise ValueError("Ağırlık 0-1 arasında olmalı")
        return v


class AIStrategySettings(BaseSettings):
    """AI strateji sistemi ayarları."""
    
    model_config = SettingsConfigDict(
        env_prefix="AI_",
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore"
    )
    
    # Feature Factory
    feature_lookback_periods: List[int] = Field(
        default=[5, 10, 20, 50, 100],
        description="Feature lookback periyotları"
    )
    
    # Winner/Loser Detection
    daily_winner_threshold: float = Field(default=0.03, description="Günlük kazanan eşiği")
    weekly_winner_threshold: float = Field(default=0.10, description="Haftalık kazanan eşiği")
    monthly_winner_threshold: float = Field(default=0.25, description="Aylık kazanan eşiği")
    
    # Pattern Discovery
    min_pattern_support: float = Field(default=0.05, description="Minimum pattern desteği")
    min_pattern_confidence: float = Field(default=0.55, description="Minimum pattern güveni")
    max_tree_depth: int = Field(default=5, description="Karar ağacı maksimum derinliği")
    
    # Validation
    min_backtest_period_days: int = Field(default=180, description="Minimum backtest süresi (gün)")
    walk_forward_windows: int = Field(default=12, description="Walk-forward pencere sayısı")
    monte_carlo_simulations: int = Field(default=1000, description="Monte Carlo simülasyon sayısı")
    
    # Approval Criteria
    min_win_rate: float = Field(default=0.55, description="Minimum kazanma oranı")
    min_profit_factor: float = Field(default=1.5, description="Minimum profit factor")
    min_sharpe_ratio: float = Field(default=1.0, description="Minimum Sharpe oranı")
    max_drawdown: float = Field(default=0.20, description="Maksimum drawdown")
    
    # Evolution
    strategy_review_days: int = Field(default=60, description="Strateji değerlendirme süresi (gün)")
    genetic_population_size: int = Field(default=20, description="Genetik popülasyon boyutu")
    genetic_generations: int = Field(default=10, description="Genetik jenerasyon sayısı")
    mutation_rate: float = Field(default=0.1, description="Mutasyon oranı")


class MarketSettings(BaseSettings):
    """Piyasa ayarları."""
    
    model_config = SettingsConfigDict(
        env_prefix="MARKET_",
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore"
    )
    
    default_index: str = Field(default="XU100.IS", description="Varsayılan endeks")
    default_interval: str = Field(default="1h", description="Varsayılan zaman dilimi")
    default_period: str = Field(default="3mo", description="Varsayılan veri periyodu")
    
    supported_intervals: List[str] = Field(
        default=["15m", "1h", "4h", "1d", "1w"],
        description="Desteklenen zaman dilimleri"
    )
    
    interval_periods: Dict[str, str] = Field(
        default={
            "15m": "5d",
            "1h": "1mo",
            "4h": "3mo",
            "1d": "1y",
            "1w": "2y"
        },
        description="Zaman dilimi - dönem eşleştirmesi"
    )
    
    market_open_hour: int = Field(default=10, description="Piyasa açılış saati")
    market_close_hour: int = Field(default=18, description="Piyasa kapanış saati")
    
    risk_free_rate: float = Field(default=0.40, description="Risksiz faiz oranı (yıllık)")


class Settings(BaseSettings):
    """Ana uygulama ayarları."""
    
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore"
    )
    
    # Uygulama
    app_name: str = Field(default="AlphaTerminal Pro", description="Uygulama adı")
    app_version: str = Field(default="1.0.0", description="Uygulama versiyonu")
    environment: Environment = Field(default=Environment.DEVELOPMENT, description="Ortam")
    debug: bool = Field(default=False, description="Debug modu")
    
    # API
    api_host: str = Field(default="0.0.0.0", description="API host")
    api_port: int = Field(default=8000, description="API port")
    api_prefix: str = Field(default="/api/v1", description="API prefix")
    api_docs_enabled: bool = Field(default=True, description="API docs aktif")
    
    # CORS
    cors_origins: List[str] = Field(
        default=["http://localhost:3000", "http://localhost:8000"],
        description="İzin verilen CORS origin'leri"
    )
    cors_allow_credentials: bool = Field(default=True, description="CORS credentials")
    cors_allow_methods: List[str] = Field(default=["*"], description="İzin verilen HTTP metodları")
    cors_allow_headers: List[str] = Field(default=["*"], description="İzin verilen HTTP başlıkları")
    
    # Logging
    log_level: LogLevel = Field(default=LogLevel.INFO, description="Log seviyesi")
    log_format: str = Field(
        default="%(asctime)s | %(name)s | %(levelname)s | %(message)s",
        description="Log formatı"
    )
    log_file: Optional[str] = Field(default=None, description="Log dosyası yolu")
    
    # Alt ayarlar
    database: DatabaseSettings = Field(default_factory=DatabaseSettings)
    redis: RedisSettings = Field(default_factory=RedisSettings)
    telegram: TelegramSettings = Field(default_factory=TelegramSettings)
    jwt: JWTSettings = Field(default_factory=JWTSettings)
    smc: SMCSettings = Field(default_factory=SMCSettings)
    orderflow: OrderFlowSettings = Field(default_factory=OrderFlowSettings)
    risk: RiskSettings = Field(default_factory=RiskSettings)
    signal: SignalSettings = Field(default_factory=SignalSettings)
    ai_strategy: AIStrategySettings = Field(default_factory=AIStrategySettings)
    market: MarketSettings = Field(default_factory=MarketSettings)
    
    @property
    def is_production(self) -> bool:
        """Production ortamında mı kontrol et."""
        return self.environment == Environment.PRODUCTION
    
    @property
    def is_development(self) -> bool:
        """Development ortamında mı kontrol et."""
        return self.environment == Environment.DEVELOPMENT


# =============================================================================
# SINGLETON SETTINGS INSTANCE
# =============================================================================

@lru_cache()
def get_settings() -> Settings:
    """
    Cached settings instance döndür.
    
    Returns:
        Settings: Uygulama ayarları
    """
    return Settings()


# Global settings instance
settings = get_settings()


# =============================================================================
# LOGGING CONFIGURATION
# =============================================================================

def setup_logging() -> logging.Logger:
    """
    Uygulama logging yapılandırması.
    
    Returns:
        logging.Logger: Yapılandırılmış logger
    """
    # Root logger yapılandırması
    log_level = getattr(logging, settings.log_level.value)
    
    # Formatter
    formatter = logging.Formatter(
        fmt=settings.log_format,
        datefmt="%Y-%m-%d %H:%M:%S"
    )
    
    # Console handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(log_level)
    console_handler.setFormatter(formatter)
    
    # Root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(log_level)
    root_logger.addHandler(console_handler)
    
    # File handler (opsiyonel)
    if settings.log_file:
        file_handler = logging.FileHandler(
            settings.log_file,
            encoding="utf-8"
        )
        file_handler.setLevel(log_level)
        file_handler.setFormatter(formatter)
        root_logger.addHandler(file_handler)
    
    # Uygulama logger'ı
    logger = logging.getLogger("alphaterminal")
    logger.setLevel(log_level)
    
    # Diğer kütüphanelerin log seviyelerini ayarla
    logging.getLogger("uvicorn").setLevel(logging.WARNING)
    logging.getLogger("sqlalchemy").setLevel(logging.WARNING)
    logging.getLogger("httpx").setLevel(logging.WARNING)
    
    return logger


# Global logger instance
logger = setup_logging()


# =============================================================================
# SECTOR CONFIGURATION
# =============================================================================

SECTORS: Dict[str, List[str]] = {
    "BANKA": [
        "AKBNK", "ALBRK", "GARAN", "HALKB", "ICBCT", "ISATR", "ISBTR", "ISCTR",
        "ISKUR", "KLNMA", "QNBFB", "QNBFL", "SKBNK", "TSKB", "VAKBN", "YKBNK"
    ],
    "HOLDING": [
        "AGHOL", "AGROT", "AHGAZ", "ALARK", "ARDYZ", "ATATP", "AYGAZ", "BERA",
        "BRYAT", "CONSE", "DOHOL", "ECZYT", "ENKAI", "ESEN", "GLRYH", "GLYHO",
        "GOZDE", "GSDHO", "GUBRF", "HUBVC", "IEYHO", "IHLAS", "IHYAY", "INVEO",
        "INVES", "KCHOL", "KUVVA", "MARKA", "METRO", "MSGYO", "MZHLD", "NTHOL",
        "PAMEL", "PEKGY", "POLHO", "RYGYO", "SAHOL", "SISE", "TARKM", "TKFEN",
        "USAK", "VERUS", "VESTL", "YESIL", "YEOTK", "TAVHL"
    ],
    "SIGORTA": [
        "AGESA", "AKGRT", "ANHYT", "ANSGR", "RAYSG", "TURSG"
    ],
    "GYO": [
        "ADESE", "ADGYO", "AGYO", "AKFGY", "AKMGY", "AKSGY", "ALGYO", "ASGYO",
        "ATAGY", "AVGYO", "AVPGY", "BEGYO", "DGGYO", "DZGYO", "EKGYO", "EYGYO",
        "FZLGY", "HLGYO", "IDGYO", "ISGYO", "KLGYO", "KRGYO", "KZBGY", "KZGYO",
        "MHRGY", "MRGYO", "NUGYO", "OZGYO", "OZKGY", "PAGYO", "PSGYO", "SNGYO",
        "SRVGY", "SURGY", "TDGYO", "TRGYO", "TSGYO", "VKGYO", "VRGYO", "YGGYO",
        "YGYO", "ZRGYO"
    ],
    "OTOMOTIV": [
        "ARCLK", "ASUZU", "BFREN", "BRISA", "DOAS", "EGEEN", "FMIZP", "FROTO",
        "GOODY", "JANTS", "KARSN", "KATMR", "KLMSN", "MAKTK", "OTKAR", "PARSN",
        "TMSN", "TOASO", "TTRAK", "VESBE"
    ],
    "DEMIR_CELIK": [
        "BEYAZ", "BMSCH", "BMSTL", "BRSAN", "BURCE", "BURVA", "CELHA", "CEMTS",
        "CUSAN", "DITAS", "DMSAS", "DOKTA", "ERBOS", "EREGL", "ISDMR", "IZMDC",
        "KRDMA", "KRDMB", "KRDMD", "KOCMT", "OZRDN", "OZSUB", "PRKAB", "SANEL",
        "SARKY", "TUCLK"
    ],
    "ENERJI": [
        "AKSEN", "AKSA", "AKENR", "AYDEM", "AYEN", "CWENE", "ENJSA", "GWIND",
        "KARYE", "NATEN", "ODAS", "ORGE", "PAMEL", "ZOREN"
    ],
    "TEKNOLOJI": [
        "ARENA", "ARMDA", "ASELS", "DGATE", "ESCOM", "FONET", "INDES", "KRONT",
        "LINK", "LOGO", "NETAS", "PAPIL", "SMART"
    ],
    "GIDA": [
        "AEFES", "BANVT", "CCOLA", "EKIZ", "ERSU", "FADE", "FRIGO", "KENT",
        "KERVT", "KNFRT", "KRSTL", "MERKO", "OYLUM", "PENGD", "PETUN", "PINSU",
        "PNSUT", "SELGD", "TATGD", "TKURU", "TUKAS", "ULKER", "ULUUN", "VANGD"
    ],
    "SAGLIK": [
        "DEVA", "ECILC", "EGPRO", "LKMNH", "MPARK", "SELEC"
    ],
    "PERAKENDE": [
        "ADESE", "BIMAS", "BIZIM", "CRFSA", "MAVI", "MGROS", "SOKM", "TKNSA",
        "VAKKO"
    ],
    "HAVACILIK": [
        "CLEBI", "PGSUS", "TAVHL", "THYAO"
    ],
    "SPOR": [
        "BJKAS", "FENER", "GSRAY", "TSPOR"
    ],
}

SECTOR_META: Dict[str, Dict[str, str]] = {
    "BANKA": {"name": "Bankacılık", "emoji": "🏦", "color": "#1E88E5"},
    "HOLDING": {"name": "Holding", "emoji": "🏛️", "color": "#5E35B1"},
    "SIGORTA": {"name": "Sigorta", "emoji": "🛡️", "color": "#00897B"},
    "GYO": {"name": "GYO", "emoji": "🏢", "color": "#F4511E"},
    "OTOMOTIV": {"name": "Otomotiv", "emoji": "🚗", "color": "#D81B60"},
    "DEMIR_CELIK": {"name": "Demir Çelik", "emoji": "🔩", "color": "#546E7A"},
    "ENERJI": {"name": "Enerji", "emoji": "⚡", "color": "#FFD600"},
    "TEKNOLOJI": {"name": "Teknoloji", "emoji": "💻", "color": "#42A5F5"},
    "GIDA": {"name": "Gıda", "emoji": "🍞", "color": "#8D6E63"},
    "SAGLIK": {"name": "Sağlık", "emoji": "🏥", "color": "#EF5350"},
    "PERAKENDE": {"name": "Perakende", "emoji": "🛒", "color": "#26A69A"},
    "HAVACILIK": {"name": "Havacılık", "emoji": "✈️", "color": "#29B6F6"},
    "SPOR": {"name": "Spor", "emoji": "⚽", "color": "#9CCC65"},
}


def get_all_symbols() -> List[str]:
    """Tüm benzersiz sembolleri döndür."""
    all_symbols = set()
    for symbols in SECTORS.values():
        all_symbols.update(symbols)
    return sorted(list(all_symbols))


def get_sector_symbols(sector_code: str) -> List[str]:
    """Belirli bir sektörün sembollerini döndür."""
    return SECTORS.get(sector_code.upper(), [])


def get_symbol_sector(symbol: str) -> Optional[str]:
    """Sembolün sektörünü bul."""
    symbol = symbol.upper().replace(".IS", "")
    for sector_code, symbols in SECTORS.items():
        if symbol in symbols:
            return sector_code
    return None


def get_sector_info(sector_code: str) -> Optional[Dict[str, str]]:
    """Sektör bilgilerini döndür."""
    return SECTOR_META.get(sector_code.upper())
