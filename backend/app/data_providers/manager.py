"""
AlphaTerminal Pro - Data Manager
================================

Central orchestrator for data fetching with failover,
caching, and provider management.

Author: AlphaTerminal Team
Version: 1.0.0
"""

import logging
from datetime import datetime, timedelta
from typing import Optional, Dict, List, Any, Union
import time
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed

import pandas as pd

from app.data_providers.enums import (
    DataInterval, DataSource, Market, DataQuality,
    CacheStrategy, ProviderStatus
)
from app.data_providers.models import (
    SymbolInfo, MarketData, DataRequest, ProviderHealth
)
from app.data_providers.exceptions import (
    DataProviderException, AllProvidersFailedError,
    SymbolNotFoundError, NoDataError, RateLimitError,
    CacheMissError
)
from app.data_providers.providers.base import (
    BaseDataProvider, ProviderRegistry
)
from app.data_providers.cache.cache_manager import (
    DataCacheManager, CacheKeyBuilder
)


logger = logging.getLogger(__name__)


class DataManager:
    """
    Central data manager with intelligent provider routing.
    
    Features:
    - Multi-provider failover
    - Automatic cache management
    - Rate limit awareness
    - Health monitoring
    - Batch fetching
    - Provider priority routing
    
    Usage:
        manager = DataManager()
        
        # Single symbol
        data = manager.get_data("THYAO", interval=DataInterval.D1)
        
        # Multiple symbols
        batch = manager.get_batch(["THYAO", "GARAN", "AKBNK"])
        
        # With specific source
        data = manager.get_data("THYAO", source=DataSource.TRADINGVIEW)
    """
    
    _instance: Optional["DataManager"] = None
    
    def __init__(
        self,
        providers: Optional[List[BaseDataProvider]] = None,
        cache_manager: Optional[DataCacheManager] = None,
        cache_strategy: CacheStrategy = CacheStrategy.CACHE_FIRST,
        max_workers: int = 5,
        default_timeout: float = 30.0
    ):
        """
        Initialize data manager.
        
        Args:
            providers: List of data providers (auto-detect if None)
            cache_manager: Cache manager instance
            cache_strategy: Default caching strategy
            max_workers: Max parallel workers for batch operations
            default_timeout: Default request timeout
        """
        self.cache_strategy = cache_strategy
        self.max_workers = max_workers
        self.default_timeout = default_timeout
        
        # Initialize providers
        self._providers: Dict[DataSource, BaseDataProvider] = {}
        self._provider_priority: List[DataSource] = []
        
        if providers:
            for provider in providers:
                self._register_provider(provider)
        else:
            self._auto_detect_providers()
        
        # Initialize cache
        self._cache = cache_manager or DataCacheManager.get_instance()
        
        # Statistics
        self._stats = {
            "total_requests": 0,
            "cache_hits": 0,
            "provider_requests": 0,
            "failures": 0,
            "failovers": 0,
        }
        
        self._lock = threading.RLock()
    
    @classmethod
    def get_instance(cls) -> "DataManager":
        """Get singleton instance."""
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance
    
    # =========================================================================
    # PUBLIC API
    # =========================================================================
    
    def get_data(
        self,
        symbol: str,
        interval: DataInterval = DataInterval.D1,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None,
        bars: Optional[int] = None,
        source: Optional[DataSource] = None,
        cache_strategy: Optional[CacheStrategy] = None,
        timeout: Optional[float] = None
    ) -> MarketData:
        """
        Get market data for a symbol.
        
        Args:
            symbol: Trading symbol
            interval: Data interval/timeframe
            start_date: Start of date range
            end_date: End of date range
            bars: Number of bars (alternative to date range)
            source: Preferred data source
            cache_strategy: Cache strategy override
            timeout: Request timeout override
            
        Returns:
            MarketData with OHLCV data
            
        Raises:
            AllProvidersFailedError: If all providers fail
            SymbolNotFoundError: If symbol not found
        """
        with self._lock:
            self._stats["total_requests"] += 1
        
        strategy = cache_strategy or self.cache_strategy
        
        # Build request
        request = DataRequest(
            symbol=symbol,
            interval=interval,
            start_date=start_date,
            end_date=end_date,
            bars=bars,
            source=source,
            timeout=timeout or self.default_timeout
        )
        
        # Try cache first (if strategy allows)
        if strategy in {CacheStrategy.CACHE_FIRST, CacheStrategy.CACHE_ONLY}:
            cached = self._get_from_cache(request)
            if cached is not None:
                with self._lock:
                    self._stats["cache_hits"] += 1
                logger.debug(f"Cache hit for {symbol}")
                return cached
            
            if strategy == CacheStrategy.CACHE_ONLY:
                raise CacheMissError(
                    f"No cached data for {symbol}",
                    details={"interval": interval.value}
                )
        
        # Fetch from providers
        data = self._fetch_with_failover(request, source)
        
        # Cache result (if strategy allows)
        if strategy != CacheStrategy.NO_CACHE:
            self._save_to_cache(data, request)
        
        return data
    
    def get_batch(
        self,
        symbols: List[str],
        interval: DataInterval = DataInterval.D1,
        bars: int = 500,
        source: Optional[DataSource] = None,
        parallel: bool = True
    ) -> Dict[str, Union[MarketData, Exception]]:
        """
        Get data for multiple symbols.
        
        Args:
            symbols: List of symbols
            interval: Data interval
            bars: Number of bars
            source: Preferred source
            parallel: Use parallel fetching
            
        Returns:
            Dict mapping symbol to MarketData or Exception
        """
        results: Dict[str, Union[MarketData, Exception]] = {}
        
        if parallel and len(symbols) > 1:
            with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
                futures = {
                    executor.submit(
                        self.get_data,
                        sym, interval, bars=bars, source=source
                    ): sym
                    for sym in symbols
                }
                
                for future in as_completed(futures):
                    sym = futures[future]
                    try:
                        results[sym] = future.result()
                    except Exception as e:
                        results[sym] = e
        else:
            for sym in symbols:
                try:
                    results[sym] = self.get_data(
                        sym, interval, bars=bars, source=source
                    )
                except Exception as e:
                    results[sym] = e
                
                # Small delay between sequential requests
                time.sleep(0.1)
        
        return results
    
    def get_symbol_info(
        self,
        symbol: str,
        source: Optional[DataSource] = None
    ) -> SymbolInfo:
        """
        Get symbol information.
        
        Args:
            symbol: Trading symbol
            source: Preferred source
            
        Returns:
            SymbolInfo with symbol details
        """
        providers = self._get_providers_for_request(source)
        
        for provider in providers:
            try:
                return provider.get_symbol_info(symbol)
            except SymbolNotFoundError:
                continue
            except Exception as e:
                logger.warning(f"{provider.name}: Failed to get info for {symbol}: {e}")
                continue
        
        raise SymbolNotFoundError(
            symbol=symbol,
            details={"tried_providers": [p.name for p in providers]}
        )
    
    def get_available_symbols(
        self,
        market: Optional[Market] = None
    ) -> List[str]:
        """
        Get list of available symbols.
        
        Args:
            market: Filter by market
            
        Returns:
            List of symbol strings
        """
        all_symbols = set()
        
        for provider in self._providers.values():
            try:
                symbols = provider.get_available_symbols(market)
                all_symbols.update(symbols)
            except Exception as e:
                logger.warning(f"{provider.name}: Failed to get symbols: {e}")
        
        return sorted(all_symbols)
    
    def get_bist_symbols(self) -> List[str]:
        """Get all BIST symbols."""
        return self.get_available_symbols(Market.BIST)
    
    # =========================================================================
    # PROVIDER MANAGEMENT
    # =========================================================================
    
    def register_provider(self, provider: BaseDataProvider):
        """Register a new data provider."""
        self._register_provider(provider)
    
    def get_provider(self, source: DataSource) -> Optional[BaseDataProvider]:
        """Get specific provider by source."""
        return self._providers.get(source)
    
    def get_provider_health(self) -> Dict[str, ProviderHealth]:
        """Get health status of all providers."""
        return {
            source.value: provider.get_health()
            for source, provider in self._providers.items()
        }
    
    def get_available_providers(self) -> List[DataSource]:
        """Get list of available (healthy) providers."""
        return [
            source for source, provider in self._providers.items()
            if provider.is_available()
        ]
    
    # =========================================================================
    # CACHE MANAGEMENT
    # =========================================================================
    
    def invalidate_cache(
        self,
        symbol: Optional[str] = None,
        interval: Optional[DataInterval] = None
    ) -> int:
        """
        Invalidate cache entries.
        
        Args:
            symbol: Symbol to invalidate (all if None)
            interval: Interval to invalidate (all if None)
            
        Returns:
            Number of entries invalidated
        """
        return self._cache.invalidate(symbol, interval)
    
    def get_cache_stats(self) -> Dict[str, Any]:
        """Get cache statistics."""
        return self._cache.get_stats()
    
    # =========================================================================
    # STATISTICS
    # =========================================================================
    
    def get_stats(self) -> Dict[str, Any]:
        """Get manager statistics."""
        with self._lock:
            stats = self._stats.copy()
        
        stats["cache"] = self.get_cache_stats()
        stats["providers"] = {
            source.value: provider.get_health().to_dict()
            for source, provider in self._providers.items()
        }
        
        return stats
    
    # =========================================================================
    # INTERNAL METHODS
    # =========================================================================
    
    def _register_provider(self, provider: BaseDataProvider):
        """Register provider internally."""
        self._providers[provider.source] = provider
        ProviderRegistry.register_instance(provider)
        
        # Initialize provider
        if not provider._initialized:
            try:
                provider.initialize()
            except Exception as e:
                logger.warning(f"Failed to initialize {provider.name}: {e}")
        
        # Update priority list
        self._update_provider_priority()
        
        logger.info(f"Registered provider: {provider.name}")
    
    def _auto_detect_providers(self):
        """Auto-detect and register available providers."""
        # Try TradingView
        try:
            from app.data_providers.providers.tradingview import (
                TradingViewProvider, TV_AVAILABLE
            )
            if TV_AVAILABLE:
                self._register_provider(TradingViewProvider())
        except ImportError:
            pass
        
        # Try Yahoo Finance
        try:
            from app.data_providers.providers.yahoo import (
                YahooFinanceProvider, YF_AVAILABLE
            )
            if YF_AVAILABLE:
                self._register_provider(YahooFinanceProvider())
        except ImportError:
            pass
        
        if not self._providers:
            logger.warning("No data providers available!")
    
    def _update_provider_priority(self):
        """Update provider priority based on health and config."""
        # Sort by: availability, health, default priority
        def priority_key(source: DataSource) -> tuple:
            provider = self._providers[source]
            health = provider.get_health()
            
            return (
                0 if health.is_available else 1,          # Available first
                health.consecutive_failures,               # Fewer failures better
                source.priority,                          # Default priority
            )
        
        self._provider_priority = sorted(
            self._providers.keys(),
            key=priority_key
        )
    
    def _get_providers_for_request(
        self,
        preferred_source: Optional[DataSource] = None
    ) -> List[BaseDataProvider]:
        """Get ordered list of providers to try for a request."""
        # Update priority
        self._update_provider_priority()
        
        providers = []
        
        # Add preferred source first if specified and available
        if preferred_source and preferred_source in self._providers:
            provider = self._providers[preferred_source]
            if provider.is_available():
                providers.append(provider)
        
        # Add remaining providers in priority order
        for source in self._provider_priority:
            if source not in [p.source for p in providers]:
                provider = self._providers[source]
                if provider.is_available():
                    providers.append(provider)
        
        return providers
    
    def _get_from_cache(self, request: DataRequest) -> Optional[MarketData]:
        """Try to get data from cache."""
        return self._cache.get_market_data(
            symbol=request.symbols[0] if not request.is_batch else str(request.symbols),
            interval=request.interval,
            source=request.source,
            start_date=request.start_date,
            end_date=request.end_date
        )
    
    def _save_to_cache(self, data: MarketData, request: DataRequest):
        """Save data to cache."""
        self._cache.set_market_data(
            data=data,
            start_date=request.start_date,
            end_date=request.end_date
        )
    
    def _fetch_with_failover(
        self,
        request: DataRequest,
        preferred_source: Optional[DataSource] = None
    ) -> MarketData:
        """
        Fetch data with automatic failover between providers.
        
        Args:
            request: Data request
            preferred_source: Preferred provider
            
        Returns:
            MarketData from first successful provider
            
        Raises:
            AllProvidersFailedError: If all providers fail
        """
        providers = self._get_providers_for_request(preferred_source)
        
        if not providers:
            raise AllProvidersFailedError(
                symbol=request.symbols[0] if not request.is_batch else str(request.symbols),
                provider_errors={"none": Exception("No providers available")}
            )
        
        errors: Dict[str, Exception] = {}
        
        for i, provider in enumerate(providers):
            try:
                with self._lock:
                    self._stats["provider_requests"] += 1
                
                logger.debug(f"Trying {provider.name} for {request.symbol}")
                data = provider.get_data(request)
                
                if i > 0:
                    with self._lock:
                        self._stats["failovers"] += 1
                    logger.info(f"Failover to {provider.name} successful for {request.symbol}")
                
                return data
                
            except RateLimitError as e:
                logger.warning(f"{provider.name}: Rate limited, trying next provider")
                errors[provider.name] = e
                continue
                
            except SymbolNotFoundError as e:
                # Symbol not found - don't try other providers
                errors[provider.name] = e
                # But continue in case another provider has it
                continue
                
            except Exception as e:
                logger.warning(f"{provider.name}: Failed for {request.symbol}: {e}")
                errors[provider.name] = e
                continue
        
        # All providers failed
        with self._lock:
            self._stats["failures"] += 1
        
        raise AllProvidersFailedError(
            symbol=request.symbols[0] if not request.is_batch else str(request.symbols),
            provider_errors=errors
        )
    
    # =========================================================================
    # CONTEXT MANAGER
    # =========================================================================
    
    def __enter__(self):
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.shutdown()
    
    def shutdown(self):
        """Shutdown all providers."""
        for provider in self._providers.values():
            try:
                provider.shutdown()
            except Exception as e:
                logger.error(f"Error shutting down {provider.name}: {e}")
        
        self._providers.clear()


# =============================================================================
# CONVENIENCE FUNCTIONS
# =============================================================================

def get_data(
    symbol: str,
    interval: DataInterval = DataInterval.D1,
    bars: int = 500,
    **kwargs
) -> MarketData:
    """
    Convenience function to get market data.
    
    Args:
        symbol: Trading symbol
        interval: Data interval
        bars: Number of bars
        **kwargs: Additional arguments
        
    Returns:
        MarketData
    """
    manager = DataManager.get_instance()
    return manager.get_data(symbol, interval, bars=bars, **kwargs)


def get_batch(
    symbols: List[str],
    interval: DataInterval = DataInterval.D1,
    bars: int = 500,
    **kwargs
) -> Dict[str, Union[MarketData, Exception]]:
    """
    Convenience function to get batch data.
    
    Args:
        symbols: List of symbols
        interval: Data interval
        bars: Number of bars
        **kwargs: Additional arguments
        
    Returns:
        Dict mapping symbol to MarketData or Exception
    """
    manager = DataManager.get_instance()
    return manager.get_batch(symbols, interval, bars, **kwargs)


# =============================================================================
# ALL EXPORTS
# =============================================================================

__all__ = [
    "DataManager",
    "get_data",
    "get_batch",
]
