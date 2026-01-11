"""
AlphaTerminal Pro - Base Data Provider
======================================

Abstract base class for all data providers.

Author: AlphaTerminal Team
Version: 1.0.0
"""

from abc import ABC, abstractmethod
from datetime import datetime, timedelta
from typing import Optional, Dict, List, Any, Type
import logging
import time
from contextlib import contextmanager

import pandas as pd

from app.data_providers.enums import (
    DataInterval, DataSource, Market, DataQuality,
    ProviderStatus, AdjustmentType
)
from app.data_providers.models import (
    SymbolInfo, MarketData, DataRequest,
    ProviderHealth, RateLimitState
)
from app.data_providers.exceptions import (
    DataProviderException,
    RateLimitError,
    TimeoutError,
    SymbolNotFoundError,
    NoDataError,
    ProviderUnavailableError,
    ConnectionError
)


logger = logging.getLogger(__name__)


class BaseDataProvider(ABC):
    """
    Abstract base class for all data providers.
    
    Implements common functionality:
    - Rate limiting
    - Health monitoring
    - Request tracking
    - Error handling
    - Logging
    
    Subclasses must implement:
    - _fetch_data: Core data fetching logic
    - _get_symbol_info: Symbol information retrieval
    - _get_available_symbols: List of available symbols
    
    Attributes:
        name: Provider name
        source: DataSource enum value
        supported_intervals: List of supported intervals
        supported_markets: List of supported markets
        requires_auth: Whether authentication is required
        is_realtime: Whether provider offers real-time data
    """
    
    # Class attributes (override in subclasses)
    name: str = "BaseProvider"
    source: DataSource = DataSource.SYNTHETIC
    supported_intervals: List[DataInterval] = []
    supported_markets: List[Market] = []
    requires_auth: bool = False
    is_realtime: bool = False
    
    # Rate limiting defaults
    default_requests_per_minute: int = 60
    default_requests_per_day: int = 10000
    
    def __init__(
        self,
        rate_limit_rpm: Optional[int] = None,
        rate_limit_daily: Optional[int] = None,
        timeout: float = 30.0,
        max_retries: int = 3,
        retry_delay: float = 1.0,
        **kwargs
    ):
        """
        Initialize provider.
        
        Args:
            rate_limit_rpm: Requests per minute limit
            rate_limit_daily: Daily request limit
            timeout: Request timeout in seconds
            max_retries: Maximum retry attempts
            retry_delay: Delay between retries in seconds
            **kwargs: Additional provider-specific arguments
        """
        self.timeout = timeout
        self.max_retries = max_retries
        self.retry_delay = retry_delay
        self._extra_config = kwargs
        
        # Initialize rate limiter
        self._rate_limit = RateLimitState(
            provider=self.source,
            requests_per_minute=rate_limit_rpm or self.default_requests_per_minute,
            requests_per_day=rate_limit_daily or self.default_requests_per_day
        )
        
        # Initialize health tracker
        self._health = ProviderHealth(provider=self.source)
        
        # Request tracking
        self._total_requests = 0
        self._total_errors = 0
        self._last_request_time: Optional[datetime] = None
        
        # Initialization
        self._initialized = False
        self._init_error: Optional[str] = None
    
    # =========================================================================
    # PUBLIC API
    # =========================================================================
    
    def get_data(self, request: DataRequest) -> MarketData:
        """
        Get market data for the given request.
        
        Args:
            request: Data request specification
            
        Returns:
            MarketData with OHLCV data
            
        Raises:
            RateLimitError: If rate limited
            TimeoutError: If request times out
            SymbolNotFoundError: If symbol not found
            NoDataError: If no data available
            DataProviderException: For other errors
        """
        self._ensure_initialized()
        
        # Validate request
        issues = request.validate()
        if issues:
            raise DataProviderException(
                f"Invalid request: {issues}",
                provider=self.name
            )
        
        # Check interval support
        if request.interval not in self.supported_intervals:
            raise DataProviderException(
                f"Interval {request.interval.value} not supported",
                provider=self.name,
                details={"supported": [i.value for i in self.supported_intervals]}
            )
        
        # Check rate limit
        if not self._rate_limit.can_request():
            raise RateLimitError(
                f"Rate limit exceeded for {self.name}",
                provider=self.name,
                retry_after=self._rate_limit.retry_after,
                limit=self._rate_limit.requests_per_minute,
                remaining=self._rate_limit.remaining_minute
            )
        
        # Execute with timing and error handling
        start_time = time.time()
        last_error: Optional[Exception] = None
        
        for attempt in range(1, self.max_retries + 1):
            try:
                self._rate_limit.record_request()
                self._total_requests += 1
                self._last_request_time = datetime.now()
                
                # Fetch data
                data = self._fetch_with_timeout(request)
                
                # Validate result
                if data.is_empty:
                    raise NoDataError(
                        symbol=request.symbols[0] if not request.is_batch else str(request.symbols),
                        interval=request.interval.value,
                        provider=self.name
                    )
                
                # Record success
                latency_ms = (time.time() - start_time) * 1000
                self._health.record_success(latency_ms)
                
                logger.debug(
                    f"{self.name}: Fetched {data.rows} bars for {data.symbol} "
                    f"in {latency_ms:.0f}ms"
                )
                
                return data
                
            except (RateLimitError, SymbolNotFoundError):
                # Don't retry these
                raise
            
            except Exception as e:
                last_error = e
                self._total_errors += 1
                self._health.record_failure(str(e))
                
                if attempt < self.max_retries:
                    delay = self.retry_delay * (2 ** (attempt - 1))  # Exponential backoff
                    logger.warning(
                        f"{self.name}: Attempt {attempt} failed for {request.symbol}: {e}. "
                        f"Retrying in {delay:.1f}s..."
                    )
                    time.sleep(delay)
                else:
                    logger.error(
                        f"{self.name}: All {self.max_retries} attempts failed for {request.symbol}"
                    )
        
        # All retries exhausted
        if last_error:
            if isinstance(last_error, DataProviderException):
                raise last_error
            raise DataProviderException(
                str(last_error),
                provider=self.name,
                symbol=request.symbols[0] if not request.is_batch else None
            )
        
        raise DataProviderException(
            "Unknown error occurred",
            provider=self.name
        )
    
    def get_symbol_info(self, symbol: str) -> SymbolInfo:
        """
        Get information about a symbol.
        
        Args:
            symbol: Trading symbol
            
        Returns:
            SymbolInfo with symbol details
            
        Raises:
            SymbolNotFoundError: If symbol not found
        """
        self._ensure_initialized()
        return self._get_symbol_info(symbol.upper().strip())
    
    def get_available_symbols(
        self,
        market: Optional[Market] = None
    ) -> List[str]:
        """
        Get list of available symbols.
        
        Args:
            market: Filter by market (optional)
            
        Returns:
            List of symbol strings
        """
        self._ensure_initialized()
        return self._get_available_symbols(market)
    
    def get_health(self) -> ProviderHealth:
        """Get current provider health status."""
        return self._health
    
    def get_rate_limit_status(self) -> Dict[str, Any]:
        """Get current rate limit status."""
        return {
            "remaining_minute": self._rate_limit.remaining_minute,
            "remaining_day": self._rate_limit.remaining_day,
            "is_limited": self._rate_limit.is_limited,
            "retry_after": self._rate_limit.retry_after,
        }
    
    def is_available(self) -> bool:
        """Check if provider is available for requests."""
        return (
            self._initialized and
            self._health.is_available and
            self._rate_limit.can_request()
        )
    
    # =========================================================================
    # ABSTRACT METHODS (implement in subclasses)
    # =========================================================================
    
    @abstractmethod
    def _fetch_data(self, request: DataRequest) -> MarketData:
        """
        Core data fetching implementation.
        
        Args:
            request: Validated data request
            
        Returns:
            MarketData with fetched data
        """
        pass
    
    @abstractmethod
    def _get_symbol_info(self, symbol: str) -> SymbolInfo:
        """
        Get symbol information implementation.
        
        Args:
            symbol: Trading symbol (already normalized)
            
        Returns:
            SymbolInfo
            
        Raises:
            SymbolNotFoundError: If symbol not found
        """
        pass
    
    @abstractmethod
    def _get_available_symbols(self, market: Optional[Market] = None) -> List[str]:
        """
        Get available symbols implementation.
        
        Args:
            market: Optional market filter
            
        Returns:
            List of symbol strings
        """
        pass
    
    # =========================================================================
    # LIFECYCLE METHODS
    # =========================================================================
    
    def initialize(self) -> bool:
        """
        Initialize the provider.
        
        Returns:
            True if initialization successful
        """
        if self._initialized:
            return True
        
        try:
            self._do_initialize()
            self._initialized = True
            self._health.status = ProviderStatus.HEALTHY
            logger.info(f"{self.name}: Initialized successfully")
            return True
            
        except Exception as e:
            self._init_error = str(e)
            self._health.status = ProviderStatus.UNHEALTHY
            self._health.record_failure(str(e))
            logger.error(f"{self.name}: Initialization failed: {e}")
            return False
    
    def shutdown(self):
        """Shutdown the provider and cleanup resources."""
        try:
            self._do_shutdown()
            self._initialized = False
            logger.info(f"{self.name}: Shutdown complete")
        except Exception as e:
            logger.error(f"{self.name}: Error during shutdown: {e}")
    
    def _do_initialize(self):
        """
        Provider-specific initialization.
        Override in subclasses for custom initialization.
        """
        pass
    
    def _do_shutdown(self):
        """
        Provider-specific shutdown.
        Override in subclasses for custom cleanup.
        """
        pass
    
    # =========================================================================
    # HELPER METHODS
    # =========================================================================
    
    def _ensure_initialized(self):
        """Ensure provider is initialized."""
        if not self._initialized:
            if not self.initialize():
                raise ProviderUnavailableError(
                    self.name,
                    reason=self._init_error or "Initialization failed"
                )
    
    def _fetch_with_timeout(self, request: DataRequest) -> MarketData:
        """
        Fetch data with timeout handling.
        
        Note: For proper async timeout, override in subclass.
        """
        # Simple implementation - subclasses may override for proper async
        return self._fetch_data(request)
    
    def _normalize_dataframe(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Normalize DataFrame to standard format.
        
        Ensures:
        - Standard column names (Open, High, Low, Close, Volume)
        - DatetimeIndex
        - Sorted by timestamp
        - No duplicate indices
        """
        if df.empty:
            return df
        
        # Copy to avoid modifying original
        df = df.copy()
        
        # Normalize column names
        column_mapping = {
            'open': 'Open', 'OPEN': 'Open',
            'high': 'High', 'HIGH': 'High',
            'low': 'Low', 'LOW': 'Low',
            'close': 'Close', 'CLOSE': 'Close',
            'volume': 'Volume', 'VOLUME': 'Volume',
            'adj close': 'Adj Close', 'adjclose': 'Adj Close',
        }
        
        df.rename(columns=column_mapping, inplace=True)
        
        # Ensure datetime index
        if not isinstance(df.index, pd.DatetimeIndex):
            if 'Date' in df.columns:
                df.set_index('Date', inplace=True)
            elif 'Timestamp' in df.columns:
                df.set_index('Timestamp', inplace=True)
            elif 'datetime' in df.columns:
                df.set_index('datetime', inplace=True)
            
            df.index = pd.to_datetime(df.index)
        
        # Sort by index
        df.sort_index(inplace=True)
        
        # Remove duplicates
        df = df[~df.index.duplicated(keep='last')]
        
        return df
    
    def _assess_data_quality(self, df: pd.DataFrame) -> DataQuality:
        """Assess quality of fetched data."""
        if df.empty:
            return DataQuality.UNKNOWN
        
        # Check for issues
        nan_ratio = df.isnull().sum().sum() / (len(df) * len(df.columns))
        
        if nan_ratio > 0.1:
            return DataQuality.LOW
        elif nan_ratio > 0.01:
            return DataQuality.MEDIUM
        
        return DataQuality.HIGH
    
    @contextmanager
    def _request_context(self, symbol: str):
        """Context manager for request tracking."""
        start = time.time()
        try:
            yield
        finally:
            elapsed = time.time() - start
            logger.debug(f"{self.name}: Request for {symbol} took {elapsed:.2f}s")
    
    # =========================================================================
    # REPRESENTATION
    # =========================================================================
    
    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}("
            f"name={self.name!r}, "
            f"status={self._health.status.value}, "
            f"initialized={self._initialized})"
        )
    
    def __str__(self) -> str:
        return f"{self.name} ({self.source.value})"


# =============================================================================
# PROVIDER REGISTRY
# =============================================================================

class ProviderRegistry:
    """
    Registry for data providers.
    
    Manages provider instances and provides discovery/routing.
    """
    
    _providers: Dict[DataSource, BaseDataProvider] = {}
    _provider_classes: Dict[DataSource, Type[BaseDataProvider]] = {}
    
    @classmethod
    def register_class(
        cls,
        source: DataSource,
        provider_class: Type[BaseDataProvider]
    ):
        """Register a provider class."""
        cls._provider_classes[source] = provider_class
    
    @classmethod
    def register_instance(
        cls,
        provider: BaseDataProvider
    ):
        """Register a provider instance."""
        cls._providers[provider.source] = provider
    
    @classmethod
    def get(cls, source: DataSource) -> Optional[BaseDataProvider]:
        """Get provider by source."""
        return cls._providers.get(source)
    
    @classmethod
    def get_or_create(
        cls,
        source: DataSource,
        **kwargs
    ) -> Optional[BaseDataProvider]:
        """Get existing provider or create new instance."""
        if source in cls._providers:
            return cls._providers[source]
        
        if source in cls._provider_classes:
            provider = cls._provider_classes[source](**kwargs)
            cls._providers[source] = provider
            return provider
        
        return None
    
    @classmethod
    def get_all(cls) -> List[BaseDataProvider]:
        """Get all registered provider instances."""
        return list(cls._providers.values())
    
    @classmethod
    def get_available(cls) -> List[BaseDataProvider]:
        """Get all available (healthy) providers."""
        return [p for p in cls._providers.values() if p.is_available()]
    
    @classmethod
    def get_for_market(cls, market: Market) -> List[BaseDataProvider]:
        """Get providers supporting a specific market."""
        return [
            p for p in cls._providers.values()
            if market in p.supported_markets
        ]
    
    @classmethod
    def clear(cls):
        """Clear all registered providers."""
        for provider in cls._providers.values():
            try:
                provider.shutdown()
            except Exception:
                pass
        cls._providers.clear()


# =============================================================================
# DECORATOR FOR PROVIDER REGISTRATION
# =============================================================================

def register_provider(source: DataSource):
    """
    Decorator to register a provider class.
    
    Usage:
        @register_provider(DataSource.TRADINGVIEW)
        class TradingViewProvider(BaseDataProvider):
            ...
    """
    def decorator(cls: Type[BaseDataProvider]) -> Type[BaseDataProvider]:
        ProviderRegistry.register_class(source, cls)
        return cls
    return decorator


# =============================================================================
# ALL EXPORTS
# =============================================================================

__all__ = [
    "BaseDataProvider",
    "ProviderRegistry",
    "register_provider",
]
