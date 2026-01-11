"""
AlphaTerminal Pro - Data Providers
==================================

Enterprise-grade data provider system for market data.

Features:
- Multi-provider architecture with failover
- TradingView and Yahoo Finance support
- Intelligent caching (memory + disk)
- Rate limiting and health monitoring
- Batch fetching with parallelization

Quick Start:
    from app.data_providers import DataManager, DataInterval
    
    manager = DataManager()
    data = manager.get_data("THYAO", interval=DataInterval.D1)
    
    # Or use convenience function
    from app.data_providers import get_data
    data = get_data("THYAO")

Author: AlphaTerminal Team
Version: 1.0.0
"""

# Enums
from app.data_providers.enums import (
    DataInterval,
    DataSource,
    Market,
    DataQuality,
    ProviderStatus,
    CacheStrategy,
    AdjustmentType,
    DataField,
    SymbolType,
    BISTIndex,
    LiquidityTier,
    TRADINGVIEW_INTERVALS,
    YAHOO_INTERVALS,
)

# Models
from app.data_providers.models import (
    SymbolInfo,
    MarketData,
    DataRequest,
    ProviderHealth,
    RateLimitState,
    CacheEntry,
)

# Exceptions
from app.data_providers.exceptions import (
    DataProviderException,
    ConnectionError,
    TimeoutError,
    AuthenticationError,
    RateLimitError,
    SymbolNotFoundError,
    NoDataError,
    InsufficientDataError,
    DataValidationError,
    StaleDataError,
    ProviderError,
    ProviderUnavailableError,
    AllProvidersFailedError,
    CacheError,
    CacheMissError,
    InvalidIntervalError,
    InvalidDateRangeError,
    MarketClosedError,
)

# Providers
from app.data_providers.providers.base import (
    BaseDataProvider,
    ProviderRegistry,
    register_provider,
)

# Manager
from app.data_providers.manager import (
    DataManager,
    get_data,
    get_batch,
)

# Cache
from app.data_providers.cache.cache_manager import (
    DataCacheManager,
    TieredCache,
    CacheKeyBuilder,
)


__all__ = [
    # Enums
    "DataInterval",
    "DataSource",
    "Market",
    "DataQuality",
    "ProviderStatus",
    "CacheStrategy",
    "AdjustmentType",
    "DataField",
    "SymbolType",
    "BISTIndex",
    "LiquidityTier",
    "TRADINGVIEW_INTERVALS",
    "YAHOO_INTERVALS",
    
    # Models
    "SymbolInfo",
    "MarketData",
    "DataRequest",
    "ProviderHealth",
    "RateLimitState",
    "CacheEntry",
    
    # Exceptions
    "DataProviderException",
    "ConnectionError",
    "TimeoutError",
    "AuthenticationError",
    "RateLimitError",
    "SymbolNotFoundError",
    "NoDataError",
    "InsufficientDataError",
    "DataValidationError",
    "StaleDataError",
    "ProviderError",
    "ProviderUnavailableError",
    "AllProvidersFailedError",
    "CacheError",
    "CacheMissError",
    "InvalidIntervalError",
    "InvalidDateRangeError",
    "MarketClosedError",
    
    # Providers
    "BaseDataProvider",
    "ProviderRegistry",
    "register_provider",
    
    # Manager
    "DataManager",
    "get_data",
    "get_batch",
    
    # Cache
    "DataCacheManager",
    "TieredCache",
    "CacheKeyBuilder",
]

__version__ = "1.0.0"
