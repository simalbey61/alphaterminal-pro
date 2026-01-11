"""
AlphaTerminal Pro - Data Provider Tests
=======================================

Unit tests for data provider system.

Author: AlphaTerminal Team
"""

import pytest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from unittest.mock import Mock, patch, MagicMock

from app.data_providers.enums import (
    DataInterval, DataSource, Market, DataQuality,
    ProviderStatus, CacheStrategy, SymbolType, LiquidityTier
)
from app.data_providers.models import (
    SymbolInfo, MarketData, DataRequest,
    ProviderHealth, RateLimitState, CacheEntry
)
from app.data_providers.exceptions import (
    DataProviderException, SymbolNotFoundError, NoDataError,
    RateLimitError, AllProvidersFailedError, CacheMissError
)
from app.data_providers.cache import (
    CacheKeyBuilder, MemoryCacheBackend, TieredCache, DataCacheManager
)
from app.data_providers.providers.base import BaseDataProvider, ProviderRegistry


# =============================================================================
# FIXTURES
# =============================================================================

@pytest.fixture
def sample_ohlcv_df():
    """Create sample OHLCV DataFrame."""
    dates = pd.date_range(start='2023-01-01', periods=100, freq='D')
    np.random.seed(42)
    
    close = 100 + np.cumsum(np.random.randn(100) * 2)
    
    return pd.DataFrame({
        'Open': close * (1 + np.random.uniform(-0.01, 0.01, 100)),
        'High': close * (1 + np.abs(np.random.randn(100) * 0.01)),
        'Low': close * (1 - np.abs(np.random.randn(100) * 0.01)),
        'Close': close,
        'Volume': np.random.randint(100000, 1000000, 100)
    }, index=dates)


@pytest.fixture
def sample_market_data(sample_ohlcv_df):
    """Create sample MarketData."""
    return MarketData(
        symbol="THYAO",
        interval=DataInterval.D1,
        data=sample_ohlcv_df,
        source=DataSource.TRADINGVIEW,
        quality=DataQuality.HIGH
    )


@pytest.fixture
def memory_cache():
    """Create memory cache backend."""
    return MemoryCacheBackend(max_entries=100)


# =============================================================================
# ENUM TESTS
# =============================================================================

class TestDataInterval:
    """Test DataInterval enum."""
    
    def test_interval_values(self):
        """Test interval string values."""
        assert DataInterval.M1.value == "1m"
        assert DataInterval.M5.value == "5m"
        assert DataInterval.H1.value == "1h"
        assert DataInterval.D1.value == "1d"
        assert DataInterval.W1.value == "1w"
    
    def test_interval_minutes(self):
        """Test minutes property."""
        assert DataInterval.M1.minutes == 1
        assert DataInterval.M5.minutes == 5
        assert DataInterval.H1.minutes == 60
        assert DataInterval.H4.minutes == 240
        assert DataInterval.D1.minutes == 1440
    
    def test_is_intraday(self):
        """Test intraday detection."""
        assert DataInterval.M1.is_intraday
        assert DataInterval.M15.is_intraday
        assert DataInterval.H4.is_intraday
        assert not DataInterval.D1.is_intraday
        assert not DataInterval.W1.is_intraday
    
    def test_from_string(self):
        """Test string parsing."""
        assert DataInterval.from_string("1m") == DataInterval.M1
        assert DataInterval.from_string("1h") == DataInterval.H1
        assert DataInterval.from_string("daily") == DataInterval.D1


class TestDataSource:
    """Test DataSource enum."""
    
    def test_source_priority(self):
        """Test source priority ordering."""
        assert DataSource.TRADINGVIEW.priority < DataSource.YAHOO_FINANCE.priority
        assert DataSource.CACHE.priority < DataSource.TRADINGVIEW.priority
    
    def test_is_realtime(self):
        """Test realtime flag."""
        assert DataSource.TRADINGVIEW.is_realtime
        assert not DataSource.YAHOO_FINANCE.is_realtime


class TestMarket:
    """Test Market enum."""
    
    def test_market_properties(self):
        """Test market properties."""
        assert Market.BIST.timezone == "Europe/Istanbul"
        assert Market.BIST.currency == "TRY"
        assert Market.NYSE.currency == "USD"
    
    def test_trading_hours(self):
        """Test trading hours."""
        bist_hours = Market.BIST.trading_hours
        assert "open" in bist_hours
        assert "close" in bist_hours


# =============================================================================
# MODEL TESTS
# =============================================================================

class TestSymbolInfo:
    """Test SymbolInfo model."""
    
    def test_creation(self):
        """Test symbol info creation."""
        info = SymbolInfo(
            symbol="THYAO",
            name="Türk Hava Yolları",
            market=Market.BIST,
            symbol_type=SymbolType.STOCK,
            currency="TRY"
        )
        
        assert info.symbol == "THYAO"
        assert info.market == Market.BIST
        assert info.currency == "TRY"
    
    def test_full_symbol(self):
        """Test full symbol generation."""
        info = SymbolInfo(
            symbol="THYAO",
            name="THY",
            market=Market.BIST
        )
        assert info.full_symbol == "BIST:THYAO"
    
    def test_symbol_normalization(self):
        """Test symbol is normalized to uppercase."""
        info = SymbolInfo(symbol="thyao", name="THY", market=Market.BIST)
        assert info.symbol == "THYAO"
    
    def test_to_dict(self):
        """Test serialization."""
        info = SymbolInfo(symbol="THYAO", name="THY", market=Market.BIST)
        d = info.to_dict()
        
        assert d["symbol"] == "THYAO"
        assert d["market"] == "BIST"


class TestMarketData:
    """Test MarketData model."""
    
    def test_creation(self, sample_ohlcv_df):
        """Test market data creation."""
        data = MarketData(
            symbol="THYAO",
            interval=DataInterval.D1,
            data=sample_ohlcv_df,
            source=DataSource.TRADINGVIEW
        )
        
        assert data.symbol == "THYAO"
        assert data.rows == 100
        assert not data.is_empty
    
    def test_auto_dates(self, sample_ohlcv_df):
        """Test automatic date detection."""
        data = MarketData(
            symbol="TEST",
            interval=DataInterval.D1,
            data=sample_ohlcv_df,
            source=DataSource.TRADINGVIEW
        )
        
        assert data.start_time == sample_ohlcv_df.index[0]
        assert data.end_time == sample_ohlcv_df.index[-1]
    
    def test_get_latest(self, sample_market_data):
        """Test getting latest data point."""
        latest = sample_market_data.get_latest()
        
        assert latest is not None
        assert "open" in latest
        assert "close" in latest
        assert "volume" in latest
    
    def test_slice(self, sample_market_data):
        """Test data slicing."""
        start = sample_market_data.data.index[20]
        end = sample_market_data.data.index[50]
        
        sliced = sample_market_data.slice(start=start, end=end)
        
        assert sliced.rows == 31  # Inclusive
        assert sliced.symbol == sample_market_data.symbol
    
    def test_validate(self, sample_ohlcv_df):
        """Test validation."""
        data = MarketData(
            symbol="TEST",
            interval=DataInterval.D1,
            data=sample_ohlcv_df,
            source=DataSource.TRADINGVIEW
        )
        
        issues = data.validate()
        # Should have minimal issues with clean data
        assert len(issues) <= 1  # May have minor OHLC inconsistencies
    
    def test_empty_data(self):
        """Test empty data handling."""
        data = MarketData(
            symbol="TEST",
            interval=DataInterval.D1,
            data=pd.DataFrame(),
            source=DataSource.TRADINGVIEW
        )
        
        assert data.is_empty
        assert data.rows == 0
        assert data.get_latest() is None


class TestDataRequest:
    """Test DataRequest model."""
    
    def test_single_symbol(self):
        """Test single symbol request."""
        request = DataRequest(symbol="THYAO", interval=DataInterval.D1)
        
        assert not request.is_batch
        assert request.symbols == ["THYAO"]
    
    def test_batch_symbols(self):
        """Test batch symbol request."""
        request = DataRequest(
            symbol=["THYAO", "GARAN", "AKBNK"],
            interval=DataInterval.D1
        )
        
        assert request.is_batch
        assert len(request.symbols) == 3
    
    def test_default_dates(self):
        """Test default date range."""
        request = DataRequest(symbol="THYAO")
        
        assert request.end_date is not None
        assert request.start_date is not None
        # Default is 1 year
        assert (request.end_date - request.start_date).days >= 360
    
    def test_validation(self):
        """Test request validation."""
        # Valid request
        request = DataRequest(symbol="THYAO", bars=500)
        assert len(request.validate()) == 0
        
        # Invalid: empty symbol
        request = DataRequest(symbol="", bars=500)
        issues = request.validate()
        assert len(issues) > 0


class TestProviderHealth:
    """Test ProviderHealth model."""
    
    def test_initial_state(self):
        """Test initial healthy state."""
        health = ProviderHealth(provider=DataSource.TRADINGVIEW)
        
        assert health.is_healthy
        assert health.is_available
        assert health.consecutive_failures == 0
    
    def test_record_success(self):
        """Test recording success."""
        health = ProviderHealth(provider=DataSource.TRADINGVIEW)
        health.record_success(latency_ms=100)
        
        assert health.last_success is not None
        assert health.latency_ms > 0
        assert health.total_requests == 1
    
    def test_record_failure(self):
        """Test recording failure."""
        health = ProviderHealth(provider=DataSource.TRADINGVIEW)
        health.record_failure("Connection error")
        
        assert health.last_error == "Connection error"
        assert health.error_count == 1
        assert health.consecutive_failures == 1
    
    def test_status_degradation(self):
        """Test status degrades with failures."""
        health = ProviderHealth(provider=DataSource.TRADINGVIEW)
        
        # Record multiple failures
        for i in range(5):
            health.record_failure(f"Error {i}")
        
        assert health.status == ProviderStatus.UNHEALTHY
        assert not health.is_healthy


class TestRateLimitState:
    """Test RateLimitState model."""
    
    def test_can_request(self):
        """Test request allowance."""
        state = RateLimitState(
            provider=DataSource.TRADINGVIEW,
            requests_per_minute=60
        )
        
        assert state.can_request()
        assert state.remaining_minute == 60
    
    def test_record_request(self):
        """Test request recording."""
        state = RateLimitState(
            provider=DataSource.TRADINGVIEW,
            requests_per_minute=60
        )
        
        state.record_request()
        assert state.remaining_minute == 59
    
    def test_rate_limit_reached(self):
        """Test rate limit reached."""
        state = RateLimitState(
            provider=DataSource.TRADINGVIEW,
            requests_per_minute=3
        )
        
        for _ in range(3):
            state.record_request()
        
        assert not state.can_request()
        assert state.is_limited


# =============================================================================
# EXCEPTION TESTS
# =============================================================================

class TestExceptions:
    """Test exception classes."""
    
    def test_symbol_not_found(self):
        """Test SymbolNotFoundError."""
        exc = SymbolNotFoundError(
            symbol="INVALID",
            provider="TradingView",
            suggestions=["THYAO", "TCELL"]
        )
        
        assert exc.symbol == "INVALID"
        assert exc.provider == "TradingView"
        assert exc.suggestions == ["THYAO", "TCELL"]
        assert "INVALID" in str(exc)
    
    def test_rate_limit_error(self):
        """Test RateLimitError."""
        exc = RateLimitError(
            "Rate limit exceeded",
            retry_after=60,
            limit=30,
            remaining=0
        )
        
        assert exc.retry_after == 60
        assert exc.limit == 30
    
    def test_all_providers_failed(self):
        """Test AllProvidersFailedError."""
        errors = {
            "TradingView": ValueError("Timeout"),
            "Yahoo": ValueError("Rate limited")
        }
        
        exc = AllProvidersFailedError(symbol="THYAO", provider_errors=errors)
        
        assert exc.symbol == "THYAO"
        assert len(exc.provider_errors) == 2


# =============================================================================
# CACHE TESTS
# =============================================================================

class TestCacheKeyBuilder:
    """Test cache key builder."""
    
    def test_build_key(self):
        """Test key building."""
        key = CacheKeyBuilder.build(
            symbol="THYAO",
            interval=DataInterval.D1,
            source=DataSource.TRADINGVIEW
        )
        
        assert "v1" in key
        assert "THYAO" in key
        assert "1d" in key
        assert "tradingview" in key
    
    def test_key_consistency(self):
        """Test keys are consistent."""
        key1 = CacheKeyBuilder.build(symbol="THYAO", interval=DataInterval.D1)
        key2 = CacheKeyBuilder.build(symbol="THYAO", interval=DataInterval.D1)
        
        assert key1 == key2
    
    def test_different_intervals(self):
        """Test different intervals produce different keys."""
        key1 = CacheKeyBuilder.build(symbol="THYAO", interval=DataInterval.D1)
        key2 = CacheKeyBuilder.build(symbol="THYAO", interval=DataInterval.H1)
        
        assert key1 != key2


class TestMemoryCacheBackend:
    """Test memory cache backend."""
    
    def test_set_and_get(self, memory_cache, sample_market_data):
        """Test basic set and get."""
        entry = CacheEntry(
            key="test",
            data=sample_market_data,
            expires_at=datetime.now() + timedelta(hours=1)
        )
        
        memory_cache.set("test", entry)
        retrieved = memory_cache.get("test")
        
        assert retrieved is not None
        assert retrieved.data.symbol == "THYAO"
    
    def test_expiration(self, memory_cache, sample_market_data):
        """Test cache expiration."""
        entry = CacheEntry(
            key="test",
            data=sample_market_data,
            expires_at=datetime.now() - timedelta(hours=1)  # Already expired
        )
        
        memory_cache.set("test", entry)
        retrieved = memory_cache.get("test")
        
        assert retrieved is None
    
    def test_lru_eviction(self, sample_market_data):
        """Test LRU eviction."""
        cache = MemoryCacheBackend(max_entries=3)
        
        for i in range(5):
            entry = CacheEntry(
                key=f"key{i}",
                data=sample_market_data,
                expires_at=datetime.now() + timedelta(hours=1)
            )
            cache.set(f"key{i}", entry)
        
        # Should have max 3 entries
        assert cache.size() == 3
        
        # Earlier keys should be evicted
        assert not cache.exists("key0")
        assert not cache.exists("key1")
        assert cache.exists("key4")
    
    def test_delete(self, memory_cache, sample_market_data):
        """Test deletion."""
        entry = CacheEntry(
            key="test",
            data=sample_market_data,
            expires_at=datetime.now() + timedelta(hours=1)
        )
        
        memory_cache.set("test", entry)
        assert memory_cache.exists("test")
        
        memory_cache.delete("test")
        assert not memory_cache.exists("test")
    
    def test_clear(self, memory_cache, sample_market_data):
        """Test clearing cache."""
        for i in range(5):
            entry = CacheEntry(
                key=f"key{i}",
                data=sample_market_data,
                expires_at=datetime.now() + timedelta(hours=1)
            )
            memory_cache.set(f"key{i}", entry)
        
        cleared = memory_cache.clear()
        
        assert cleared == 5
        assert memory_cache.size() == 0


class TestTieredCache:
    """Test tiered cache."""
    
    def test_l1_hit(self, sample_market_data):
        """Test L1 (memory) cache hit."""
        cache = TieredCache(memory_max_entries=100)
        
        cache.set("test", sample_market_data)
        
        # Should hit L1
        result = cache.get("test")
        
        assert result is not None
        stats = cache.get_stats()
        assert stats["l1_hits"] == 1


# =============================================================================
# INTEGRATION TESTS
# =============================================================================

class TestDataManagerIntegration:
    """Integration tests for DataManager."""
    
    def test_manager_creation(self):
        """Test manager can be created."""
        from app.data_providers.manager import DataManager
        
        manager = DataManager(providers=[])
        assert manager is not None
    
    def test_cache_stats(self):
        """Test cache statistics."""
        from app.data_providers.manager import DataManager
        
        manager = DataManager(providers=[])
        stats = manager.get_cache_stats()
        
        assert "l1_hits" in stats
        assert "l2_hits" in stats
        assert "misses" in stats
