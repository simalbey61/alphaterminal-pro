"""
AlphaTerminal Pro - Data Provider Models
========================================

Data models for market data and provider operations.

Author: AlphaTerminal Team
Version: 1.0.0
"""

from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Optional, Dict, Any, List, Union
import pandas as pd
import numpy as np
from enum import Enum

from app.data_providers.enums import (
    DataInterval, DataSource, Market, DataQuality,
    ProviderStatus, AdjustmentType, SymbolType, LiquidityTier, DataField
)


# =============================================================================
# SYMBOL INFO
# =============================================================================

@dataclass
class SymbolInfo:
    """
    Complete information about a tradeable symbol.
    
    Attributes:
        symbol: Trading symbol (e.g., "THYAO")
        name: Full company/asset name
        market: Market/exchange
        symbol_type: Type of instrument
        sector: Industry sector
        currency: Trading currency
        lot_size: Minimum lot size
        tick_size: Minimum price movement
        liquidity_tier: Liquidity classification
        is_active: Whether symbol is actively traded
        metadata: Additional provider-specific data
    """
    
    symbol: str
    name: str
    market: Market
    symbol_type: SymbolType = SymbolType.STOCK
    sector: Optional[str] = None
    industry: Optional[str] = None
    currency: str = "TRY"
    lot_size: int = 1
    tick_size: float = 0.01
    liquidity_tier: LiquidityTier = LiquidityTier.TIER_3
    is_active: bool = True
    isin: Optional[str] = None
    description: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def __post_init__(self):
        self.symbol = self.symbol.upper().strip()
    
    @property
    def full_symbol(self) -> str:
        """Get full symbol with market prefix."""
        if self.market == Market.BIST:
            return f"BIST:{self.symbol}"
        return f"{self.market.value}:{self.symbol}"
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "symbol": self.symbol,
            "name": self.name,
            "market": self.market.value,
            "symbol_type": self.symbol_type.value,
            "sector": self.sector,
            "industry": self.industry,
            "currency": self.currency,
            "lot_size": self.lot_size,
            "tick_size": self.tick_size,
            "liquidity_tier": self.liquidity_tier.value,
            "is_active": self.is_active,
            "isin": self.isin,
            "description": self.description,
            "metadata": self.metadata,
        }


# =============================================================================
# MARKET DATA
# =============================================================================

@dataclass
class MarketData:
    """
    Container for OHLCV market data with metadata.
    
    Attributes:
        symbol: Trading symbol
        interval: Data timeframe
        data: DataFrame with OHLCV data
        source: Data source provider
        quality: Data quality assessment
        start_time: First timestamp
        end_time: Last timestamp
        fetch_time: When data was fetched
        adjustment: Price adjustment type
        metadata: Additional information
    """
    
    symbol: str
    interval: DataInterval
    data: pd.DataFrame
    source: DataSource
    quality: DataQuality = DataQuality.UNKNOWN
    start_time: Optional[datetime] = None
    end_time: Optional[datetime] = None
    fetch_time: datetime = field(default_factory=datetime.now)
    adjustment: AdjustmentType = AdjustmentType.FULL
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def __post_init__(self):
        if not self.data.empty:
            if self.start_time is None:
                self.start_time = self.data.index[0]
            if self.end_time is None:
                self.end_time = self.data.index[-1]
    
    @property
    def rows(self) -> int:
        """Number of data rows."""
        return len(self.data)
    
    @property
    def is_empty(self) -> bool:
        """Check if data is empty."""
        return self.data.empty
    
    @property
    def has_gaps(self) -> bool:
        """Check if data has gaps (simplified check)."""
        if self.rows < 2:
            return False
        
        # For daily data, check for missing trading days
        if self.interval == DataInterval.D1:
            date_range = pd.date_range(
                start=self.start_time,
                end=self.end_time,
                freq='B'  # Business days
            )
            return len(date_range) > self.rows * 1.1  # Allow 10% tolerance
        
        return False
    
    @property
    def age_seconds(self) -> float:
        """Age of data in seconds."""
        return (datetime.now() - self.fetch_time).total_seconds()
    
    @property
    def is_stale(self) -> bool:
        """Check if data is stale (> 1 hour for daily, > 5 min for intraday)."""
        max_age = 300 if self.interval.is_intraday else 3600
        return self.age_seconds > max_age
    
    def get_latest(self) -> Optional[Dict[str, float]]:
        """Get latest data point as dict."""
        if self.is_empty:
            return None
        
        row = self.data.iloc[-1]
        return {
            "open": row.get("Open"),
            "high": row.get("High"),
            "low": row.get("Low"),
            "close": row.get("Close"),
            "volume": row.get("Volume"),
            "timestamp": self.data.index[-1],
        }
    
    def slice(
        self,
        start: Optional[datetime] = None,
        end: Optional[datetime] = None
    ) -> "MarketData":
        """Return a slice of the data."""
        sliced = self.data.copy()
        
        if start:
            sliced = sliced[sliced.index >= start]
        if end:
            sliced = sliced[sliced.index <= end]
        
        return MarketData(
            symbol=self.symbol,
            interval=self.interval,
            data=sliced,
            source=self.source,
            quality=self.quality,
            adjustment=self.adjustment,
            metadata=self.metadata.copy()
        )
    
    def validate(self) -> List[str]:
        """
        Validate data integrity.
        
        Returns:
            List of validation issues (empty if valid)
        """
        issues = []
        
        if self.is_empty:
            issues.append("Data is empty")
            return issues
        
        required = DataField.required_fields()
        missing = [f for f in required if f not in self.data.columns]
        if missing:
            issues.append(f"Missing columns: {missing}")
        
        # Check for NaN values
        if self.data[required].isnull().any().any():
            nan_counts = self.data[required].isnull().sum()
            nan_cols = nan_counts[nan_counts > 0].to_dict()
            issues.append(f"NaN values found: {nan_cols}")
        
        # Check OHLC consistency
        if "High" in self.data.columns and "Low" in self.data.columns:
            invalid_hl = (self.data["High"] < self.data["Low"]).sum()
            if invalid_hl > 0:
                issues.append(f"High < Low in {invalid_hl} rows")
        
        # Check for negative prices
        price_cols = ["Open", "High", "Low", "Close"]
        for col in price_cols:
            if col in self.data.columns:
                neg = (self.data[col] < 0).sum()
                if neg > 0:
                    issues.append(f"Negative {col} values: {neg}")
        
        # Check volume
        if "Volume" in self.data.columns:
            neg_vol = (self.data["Volume"] < 0).sum()
            if neg_vol > 0:
                issues.append(f"Negative volume: {neg_vol}")
        
        return issues
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary (without DataFrame)."""
        return {
            "symbol": self.symbol,
            "interval": self.interval.value,
            "source": self.source.value,
            "quality": self.quality.value,
            "rows": self.rows,
            "start_time": self.start_time.isoformat() if self.start_time else None,
            "end_time": self.end_time.isoformat() if self.end_time else None,
            "fetch_time": self.fetch_time.isoformat(),
            "adjustment": self.adjustment.value,
            "metadata": self.metadata,
        }


# =============================================================================
# DATA REQUEST
# =============================================================================

@dataclass
class DataRequest:
    """
    Request specification for market data.
    
    Attributes:
        symbol: Trading symbol(s)
        interval: Data timeframe
        start_date: Start of date range
        end_date: End of date range
        bars: Number of bars (alternative to date range)
        source: Preferred data source
        adjustment: Price adjustment type
        include_premarket: Include pre-market data
        include_afterhours: Include after-hours data
    """
    
    symbol: Union[str, List[str]]
    interval: DataInterval = DataInterval.D1
    start_date: Optional[datetime] = None
    end_date: Optional[datetime] = None
    bars: Optional[int] = None
    source: Optional[DataSource] = None
    adjustment: AdjustmentType = AdjustmentType.FULL
    include_premarket: bool = False
    include_afterhours: bool = False
    timeout: float = 30.0
    
    def __post_init__(self):
        # Normalize symbol(s)
        if isinstance(self.symbol, str):
            self.symbol = self.symbol.upper().strip()
        else:
            self.symbol = [s.upper().strip() for s in self.symbol]
        
        # Set defaults for date range
        if self.end_date is None:
            self.end_date = datetime.now()
        
        if self.start_date is None and self.bars is None:
            # Default to 1 year of data
            self.start_date = self.end_date - timedelta(days=365)
    
    @property
    def is_batch(self) -> bool:
        """Check if this is a batch request (multiple symbols)."""
        return isinstance(self.symbol, list)
    
    @property
    def symbols(self) -> List[str]:
        """Get symbols as list."""
        if isinstance(self.symbol, str):
            return [self.symbol]
        return self.symbol
    
    def validate(self) -> List[str]:
        """Validate request parameters."""
        issues = []
        
        if not self.symbol:
            issues.append("Symbol is required")
        
        if self.start_date and self.end_date:
            if self.start_date >= self.end_date:
                issues.append("start_date must be before end_date")
        
        if self.bars is not None and self.bars <= 0:
            issues.append("bars must be positive")
        
        if self.timeout <= 0:
            issues.append("timeout must be positive")
        
        return issues


# =============================================================================
# PROVIDER HEALTH
# =============================================================================

@dataclass
class ProviderHealth:
    """
    Health status of a data provider.
    
    Attributes:
        provider: Provider identifier
        status: Current status
        latency_ms: Average response latency
        success_rate: Success rate (0-1)
        last_check: Last health check time
        last_success: Last successful request time
        last_error: Last error message
        error_count: Recent error count
        metadata: Additional health info
    """
    
    provider: DataSource
    status: ProviderStatus = ProviderStatus.HEALTHY
    latency_ms: float = 0.0
    success_rate: float = 1.0
    last_check: datetime = field(default_factory=datetime.now)
    last_success: Optional[datetime] = None
    last_error: Optional[str] = None
    error_count: int = 0
    consecutive_failures: int = 0
    total_requests: int = 0
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    @property
    def is_healthy(self) -> bool:
        """Check if provider is healthy."""
        return self.status == ProviderStatus.HEALTHY
    
    @property
    def is_available(self) -> bool:
        """Check if provider can handle requests."""
        return self.status in {
            ProviderStatus.HEALTHY,
            ProviderStatus.DEGRADED
        }
    
    def record_success(self, latency_ms: float):
        """Record successful request."""
        self.total_requests += 1
        self.last_success = datetime.now()
        self.consecutive_failures = 0
        self.latency_ms = (self.latency_ms * 0.9) + (latency_ms * 0.1)  # EMA
        self.success_rate = min(1.0, self.success_rate + 0.01)
        self._update_status()
    
    def record_failure(self, error: str):
        """Record failed request."""
        self.total_requests += 1
        self.error_count += 1
        self.consecutive_failures += 1
        self.last_error = error
        self.success_rate = max(0.0, self.success_rate - 0.05)
        self._update_status()
    
    def _update_status(self):
        """Update status based on metrics."""
        if self.consecutive_failures >= 5:
            self.status = ProviderStatus.UNHEALTHY
        elif self.consecutive_failures >= 3 or self.success_rate < 0.8:
            self.status = ProviderStatus.DEGRADED
        elif self.success_rate >= 0.95:
            self.status = ProviderStatus.HEALTHY
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "provider": self.provider.value,
            "status": self.status.value,
            "latency_ms": round(self.latency_ms, 2),
            "success_rate": round(self.success_rate, 4),
            "last_check": self.last_check.isoformat(),
            "last_success": self.last_success.isoformat() if self.last_success else None,
            "last_error": self.last_error,
            "error_count": self.error_count,
            "consecutive_failures": self.consecutive_failures,
            "total_requests": self.total_requests,
        }


# =============================================================================
# RATE LIMIT STATE
# =============================================================================

@dataclass
class RateLimitState:
    """
    Rate limiting state for a provider.
    
    Attributes:
        provider: Provider identifier
        requests_per_minute: Max requests per minute
        requests_per_day: Max requests per day
        current_minute_count: Current minute request count
        current_day_count: Current day request count
        minute_reset_time: When minute limit resets
        day_reset_time: When day limit resets
        is_limited: Whether currently rate limited
        retry_after: Seconds until can retry
    """
    
    provider: DataSource
    requests_per_minute: int = 60
    requests_per_day: int = 10000
    current_minute_count: int = 0
    current_day_count: int = 0
    minute_reset_time: datetime = field(default_factory=datetime.now)
    day_reset_time: datetime = field(default_factory=datetime.now)
    is_limited: bool = False
    retry_after: float = 0.0
    
    def can_request(self) -> bool:
        """Check if a request can be made."""
        self._check_reset()
        
        if self.current_minute_count >= self.requests_per_minute:
            self.is_limited = True
            self.retry_after = (self.minute_reset_time - datetime.now()).total_seconds()
            return False
        
        if self.current_day_count >= self.requests_per_day:
            self.is_limited = True
            self.retry_after = (self.day_reset_time - datetime.now()).total_seconds()
            return False
        
        self.is_limited = False
        self.retry_after = 0.0
        return True
    
    def record_request(self):
        """Record a request."""
        self._check_reset()
        self.current_minute_count += 1
        self.current_day_count += 1
    
    def _check_reset(self):
        """Check and reset counters if needed."""
        now = datetime.now()
        
        if now >= self.minute_reset_time:
            self.current_minute_count = 0
            self.minute_reset_time = now + timedelta(minutes=1)
        
        if now >= self.day_reset_time:
            self.current_day_count = 0
            self.day_reset_time = now.replace(
                hour=0, minute=0, second=0, microsecond=0
            ) + timedelta(days=1)
    
    @property
    def remaining_minute(self) -> int:
        """Remaining requests this minute."""
        return max(0, self.requests_per_minute - self.current_minute_count)
    
    @property
    def remaining_day(self) -> int:
        """Remaining requests today."""
        return max(0, self.requests_per_day - self.current_day_count)


# =============================================================================
# CACHE ENTRY
# =============================================================================

@dataclass
class CacheEntry:
    """
    Cache entry for market data.
    
    Attributes:
        key: Cache key
        data: Cached MarketData
        created_at: When entry was created
        expires_at: When entry expires
        hits: Number of cache hits
        source: Original data source
    """
    
    key: str
    data: MarketData
    created_at: datetime = field(default_factory=datetime.now)
    expires_at: Optional[datetime] = None
    hits: int = 0
    source: Optional[DataSource] = None
    
    def __post_init__(self):
        if self.source is None:
            self.source = self.data.source
        
        # Default TTL: 5 min for intraday, 1 hour for daily
        if self.expires_at is None:
            ttl = 300 if self.data.interval.is_intraday else 3600
            self.expires_at = self.created_at + timedelta(seconds=ttl)
    
    @property
    def is_expired(self) -> bool:
        """Check if entry is expired."""
        return datetime.now() >= self.expires_at
    
    @property
    def ttl_seconds(self) -> float:
        """Remaining TTL in seconds."""
        return max(0, (self.expires_at - datetime.now()).total_seconds())
    
    def touch(self):
        """Record a cache hit."""
        self.hits += 1
    
    def extend_ttl(self, seconds: float):
        """Extend TTL by given seconds."""
        self.expires_at = self.expires_at + timedelta(seconds=seconds)


# =============================================================================
# ALL EXPORTS
# =============================================================================

__all__ = [
    "SymbolInfo",
    "MarketData",
    "DataRequest",
    "ProviderHealth",
    "RateLimitState",
    "CacheEntry",
]
