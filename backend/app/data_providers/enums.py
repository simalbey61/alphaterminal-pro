"""
AlphaTerminal Pro - Data Provider Enums
=======================================

Enumerations for data provider operations.

Author: AlphaTerminal Team
Version: 1.0.0
"""

from enum import Enum, auto
from typing import Dict, List, Optional


class DataInterval(str, Enum):
    """
    Supported data intervals/timeframes.
    
    Value format matches common conventions for easy serialization.
    """
    
    # Intraday
    M1 = "1m"      # 1 minute
    M2 = "2m"      # 2 minutes
    M3 = "3m"      # 3 minutes
    M5 = "5m"      # 5 minutes
    M10 = "10m"    # 10 minutes
    M15 = "15m"    # 15 minutes
    M30 = "30m"    # 30 minutes
    M45 = "45m"    # 45 minutes
    
    # Hourly
    H1 = "1h"      # 1 hour
    H2 = "2h"      # 2 hours
    H3 = "3h"      # 3 hours
    H4 = "4h"      # 4 hours
    
    # Daily+
    D1 = "1d"      # 1 day
    W1 = "1w"      # 1 week
    MN1 = "1M"     # 1 month
    
    @property
    def minutes(self) -> int:
        """Get interval in minutes."""
        mapping = {
            "1m": 1, "2m": 2, "3m": 3, "5m": 5, "10m": 10,
            "15m": 15, "30m": 30, "45m": 45,
            "1h": 60, "2h": 120, "3h": 180, "4h": 240,
            "1d": 1440, "1w": 10080, "1M": 43200
        }
        return mapping.get(self.value, 1440)
    
    @property
    def is_intraday(self) -> bool:
        """Check if interval is intraday."""
        return self.minutes < 1440
    
    @property
    def bars_per_day(self) -> float:
        """Get approximate bars per trading day (6.5 hours)."""
        if self.minutes >= 1440:
            return 1440 / self.minutes
        return 390 / self.minutes  # 6.5 hours = 390 minutes
    
    @classmethod
    def from_string(cls, value: str) -> "DataInterval":
        """Create from string value."""
        value = value.lower().strip()
        
        # Handle variations
        mappings = {
            "1min": cls.M1, "5min": cls.M5, "15min": cls.M15, "30min": cls.M30,
            "1hour": cls.H1, "4hour": cls.H4,
            "daily": cls.D1, "day": cls.D1,
            "weekly": cls.W1, "week": cls.W1,
            "monthly": cls.MN1, "month": cls.MN1,
        }
        
        if value in mappings:
            return mappings[value]
        
        for interval in cls:
            if interval.value.lower() == value:
                return interval
        
        raise ValueError(f"Unknown interval: {value}")


class DataSource(str, Enum):
    """Available data sources/providers."""
    
    TRADINGVIEW = "tradingview"
    YAHOO_FINANCE = "yahoo_finance"
    INVESTING_COM = "investing_com"
    ISYATIRIM = "isyatirim"
    BIST_OFFICIAL = "bist_official"
    CACHE = "cache"
    SYNTHETIC = "synthetic"  # For testing
    
    @property
    def priority(self) -> int:
        """Default priority (lower = higher priority)."""
        priorities = {
            "tradingview": 1,
            "isyatirim": 2,
            "yahoo_finance": 3,
            "investing_com": 4,
            "bist_official": 5,
            "cache": 0,
            "synthetic": 99,
        }
        return priorities.get(self.value, 50)
    
    @property
    def is_realtime(self) -> bool:
        """Check if source provides real-time data."""
        return self in {
            DataSource.TRADINGVIEW,
            DataSource.ISYATIRIM,
            DataSource.BIST_OFFICIAL
        }


class Market(str, Enum):
    """Supported markets/exchanges."""
    
    BIST = "BIST"           # Borsa Istanbul
    BIST_VIOP = "VIOP"      # BIST Derivatives
    NYSE = "NYSE"           # New York Stock Exchange
    NASDAQ = "NASDAQ"       # NASDAQ
    CRYPTO = "CRYPTO"       # Cryptocurrency
    FOREX = "FOREX"         # Foreign Exchange
    
    @property
    def timezone(self) -> str:
        """Get market timezone."""
        timezones = {
            "BIST": "Europe/Istanbul",
            "VIOP": "Europe/Istanbul",
            "NYSE": "America/New_York",
            "NASDAQ": "America/New_York",
            "CRYPTO": "UTC",
            "FOREX": "UTC",
        }
        return timezones.get(self.value, "UTC")
    
    @property
    def trading_hours(self) -> Dict[str, str]:
        """Get trading hours."""
        hours = {
            "BIST": {"open": "10:00", "close": "18:00"},
            "VIOP": {"open": "09:30", "close": "18:15"},
            "NYSE": {"open": "09:30", "close": "16:00"},
            "NASDAQ": {"open": "09:30", "close": "16:00"},
            "CRYPTO": {"open": "00:00", "close": "23:59"},  # 24/7
            "FOREX": {"open": "00:00", "close": "23:59"},   # ~24/5
        }
        return hours.get(self.value, {"open": "09:00", "close": "17:00"})
    
    @property
    def currency(self) -> str:
        """Get primary currency."""
        currencies = {
            "BIST": "TRY",
            "VIOP": "TRY",
            "NYSE": "USD",
            "NASDAQ": "USD",
            "CRYPTO": "USDT",
            "FOREX": "USD",
        }
        return currencies.get(self.value, "USD")


class DataQuality(str, Enum):
    """Data quality levels."""
    
    HIGH = "high"           # Verified, complete, minimal gaps
    MEDIUM = "medium"       # Some gaps, unverified
    LOW = "low"             # Many gaps, possibly stale
    UNKNOWN = "unknown"     # Quality not assessed


class ProviderStatus(str, Enum):
    """Provider health status."""
    
    HEALTHY = "healthy"           # Fully operational
    DEGRADED = "degraded"         # Partial functionality
    UNHEALTHY = "unhealthy"       # Major issues
    MAINTENANCE = "maintenance"   # Planned downtime
    OFFLINE = "offline"           # Completely unavailable


class CacheStrategy(str, Enum):
    """Cache strategies for data retrieval."""
    
    CACHE_FIRST = "cache_first"       # Try cache, then provider
    PROVIDER_FIRST = "provider_first" # Try provider, cache as backup
    CACHE_ONLY = "cache_only"         # Only use cache
    NO_CACHE = "no_cache"             # Never use cache
    STALE_WHILE_REVALIDATE = "stale_while_revalidate"  # Return stale, refresh async


class AdjustmentType(str, Enum):
    """Price adjustment types."""
    
    NONE = "none"           # No adjustment
    SPLIT = "split"         # Split adjusted
    DIVIDEND = "dividend"   # Dividend adjusted
    FULL = "full"           # Fully adjusted (split + dividend)


class DataField(str, Enum):
    """Standard OHLCV+ data fields."""
    
    # Core OHLCV
    OPEN = "Open"
    HIGH = "High"
    LOW = "Low"
    CLOSE = "Close"
    VOLUME = "Volume"
    
    # Extended
    ADJUSTED_CLOSE = "Adj Close"
    VWAP = "VWAP"
    TRADES = "Trades"
    TURNOVER = "Turnover"
    
    # Metadata
    TIMESTAMP = "Timestamp"
    SYMBOL = "Symbol"
    
    @classmethod
    def ohlcv_fields(cls) -> List["DataField"]:
        """Get core OHLCV fields."""
        return [cls.OPEN, cls.HIGH, cls.LOW, cls.CLOSE, cls.VOLUME]
    
    @classmethod
    def required_fields(cls) -> List[str]:
        """Get required field names as strings."""
        return [f.value for f in cls.ohlcv_fields()]


class SymbolType(str, Enum):
    """Types of tradeable symbols."""
    
    STOCK = "stock"             # Common stock
    ETF = "etf"                 # Exchange Traded Fund
    INDEX = "index"             # Market index
    FUTURES = "futures"         # Futures contract
    OPTIONS = "options"         # Options contract
    FOREX = "forex"             # Currency pair
    CRYPTO = "crypto"           # Cryptocurrency
    WARRANT = "warrant"         # Warrant
    REIT = "reit"               # Real Estate Investment Trust
    BOND = "bond"               # Bond/Fixed income


class BISTIndex(str, Enum):
    """BIST market indices."""
    
    XU100 = "XU100"     # BIST 100
    XU050 = "XU050"     # BIST 50
    XU030 = "XU030"     # BIST 30
    XUTEK = "XUTEK"     # BIST Technology
    XBANK = "XBANK"     # BIST Banks
    XUSIN = "XUSIN"     # BIST Industrials
    XHOLD = "XHOLD"     # BIST Holdings
    XGIDA = "XGIDA"     # BIST Food & Beverage
    XULAS = "XULAS"     # BIST Transportation
    XTEKS = "XTEKS"     # BIST Textile
    XKMYA = "XKMYA"     # BIST Chemicals
    XMANA = "XMANA"     # BIST Metals
    XUMAL = "XUMAL"     # BIST Financials
    XSGRT = "XSGRT"     # BIST Insurance
    XILTM = "XILTM"     # BIST Telecommunications


class LiquidityTier(str, Enum):
    """Liquidity classification tiers."""
    
    TIER_1 = "tier_1"   # Highest liquidity (XU030 stocks)
    TIER_2 = "tier_2"   # High liquidity (XU050 stocks)
    TIER_3 = "tier_3"   # Medium liquidity (XU100 stocks)
    TIER_4 = "tier_4"   # Lower liquidity (other BIST stocks)
    TIER_5 = "tier_5"   # Low liquidity (thinly traded)
    
    @property
    def avg_daily_volume_min(self) -> int:
        """Minimum average daily volume for tier."""
        minimums = {
            "tier_1": 100_000_000,
            "tier_2": 25_000_000,
            "tier_3": 5_000_000,
            "tier_4": 1_000_000,
            "tier_5": 0,
        }
        return minimums.get(self.value, 0)


# =============================================================================
# INTERVAL MAPPINGS FOR PROVIDERS
# =============================================================================

# TradingView interval mapping
TRADINGVIEW_INTERVALS: Dict[DataInterval, str] = {
    DataInterval.M1: "1",
    DataInterval.M3: "3",
    DataInterval.M5: "5",
    DataInterval.M15: "15",
    DataInterval.M30: "30",
    DataInterval.M45: "45",
    DataInterval.H1: "60",
    DataInterval.H2: "120",
    DataInterval.H3: "180",
    DataInterval.H4: "240",
    DataInterval.D1: "D",
    DataInterval.W1: "W",
    DataInterval.MN1: "M",
}

# Yahoo Finance interval mapping
YAHOO_INTERVALS: Dict[DataInterval, str] = {
    DataInterval.M1: "1m",
    DataInterval.M2: "2m",
    DataInterval.M5: "5m",
    DataInterval.M15: "15m",
    DataInterval.M30: "30m",
    DataInterval.H1: "1h",
    DataInterval.D1: "1d",
    DataInterval.W1: "1wk",
    DataInterval.MN1: "1mo",
}


# =============================================================================
# ALL EXPORTS
# =============================================================================

__all__ = [
    # Core enums
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
    
    # Mappings
    "TRADINGVIEW_INTERVALS",
    "YAHOO_INTERVALS",
]
