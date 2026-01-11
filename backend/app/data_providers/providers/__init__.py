"""
AlphaTerminal Pro - Data Providers
==================================

Concrete data provider implementations.

Available Providers:
    - TradingViewProvider: Primary source for BIST
    - YahooFinanceProvider: Fallback/secondary source
"""

from app.data_providers.providers.base import (
    BaseDataProvider,
    ProviderRegistry,
    register_provider,
)

# Import providers to trigger registration
try:
    from app.data_providers.providers.tradingview import (
        TradingViewProvider,
        TV_AVAILABLE,
        BIST_XU030,
        BIST_XU100_EXTRA,
    )
except ImportError:
    TV_AVAILABLE = False
    TradingViewProvider = None

try:
    from app.data_providers.providers.yahoo import (
        YahooFinanceProvider,
        YF_AVAILABLE,
    )
except ImportError:
    YF_AVAILABLE = False
    YahooFinanceProvider = None


__all__ = [
    "BaseDataProvider",
    "ProviderRegistry",
    "register_provider",
    "TradingViewProvider",
    "YahooFinanceProvider",
    "TV_AVAILABLE",
    "YF_AVAILABLE",
    "BIST_XU030",
    "BIST_XU100_EXTRA",
]
