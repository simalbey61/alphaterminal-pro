"""
AlphaTerminal Pro - Core Utility Tests
======================================

Unit tests for validators, error handlers, and circuit breaker.

Author: AlphaTerminal Team
"""

import pytest
import time
import pandas as pd
import numpy as np

from app.core.validators import (
    DataFrameValidator,
    NumericValidator,
    ConfigValidator,
    validate_ohlcv,
    validate_price,
    validate_quantity,
    validate_percentage
)
from app.core.error_handlers import (
    retry,
    handle_exceptions,
    graceful_degradation,
    timed,
    safe_call,
    error_context,
    ErrorHandler
)
from app.core.circuit_breaker import (
    CircuitBreaker,
    CircuitBreakerConfig,
    CircuitBreakerError,
    CircuitState,
    circuit_breaker
)
from app.backtest.exceptions import (
    InvalidDataError,
    InsufficientDataError,
    InvalidConfigError
)


class TestDataFrameValidator:
    """Test DataFrame validation."""
    
    @pytest.fixture
    def valid_ohlcv(self):
        """Create valid OHLCV data."""
        dates = pd.date_range(start='2023-01-01', periods=100, freq='D')
        return pd.DataFrame({
            'Open': np.random.uniform(100, 110, 100),
            'High': np.random.uniform(110, 120, 100),
            'Low': np.random.uniform(90, 100, 100),
            'Close': np.random.uniform(100, 110, 100),
            'Volume': np.random.randint(100000, 1000000, 100)
        }, index=dates)
    
    def test_validate_valid_data(self, valid_ohlcv):
        """Test validation of valid data."""
        # Fix OHLC consistency
        valid_ohlcv['High'] = valid_ohlcv[['Open', 'Close', 'High']].max(axis=1)
        valid_ohlcv['Low'] = valid_ohlcv[['Open', 'Close', 'Low']].min(axis=1)
        
        is_valid, issues = DataFrameValidator.validate_ohlcv(
            valid_ohlcv,
            symbol="TEST",
            raise_on_error=False
        )
        
        assert is_valid
        assert len(issues) == 0
    
    def test_validate_missing_columns(self):
        """Test validation with missing columns."""
        dates = pd.date_range(start='2023-01-01', periods=100, freq='D')
        bad_data = pd.DataFrame({
            'Open': [100] * 100,
            'Close': [100] * 100
            # Missing High, Low, Volume
        }, index=dates)
        
        is_valid, issues = DataFrameValidator.validate_ohlcv(
            bad_data,
            raise_on_error=False
        )
        
        assert not is_valid
        assert any('Missing' in issue for issue in issues)
    
    def test_validate_empty_data(self):
        """Test validation of empty DataFrame."""
        empty = pd.DataFrame()
        
        is_valid, issues = DataFrameValidator.validate_ohlcv(
            empty,
            raise_on_error=False
        )
        
        assert not is_valid
    
    def test_validate_insufficient_rows(self, valid_ohlcv):
        """Test validation with insufficient rows."""
        is_valid, issues = DataFrameValidator.validate_ohlcv(
            valid_ohlcv[:10],
            min_rows=100,
            raise_on_error=False
        )
        
        assert not is_valid
    
    def test_validate_raises_on_error(self):
        """Test that validation raises on error when configured."""
        empty = pd.DataFrame()
        
        with pytest.raises(InsufficientDataError):
            DataFrameValidator.validate_ohlcv(
                empty,
                raise_on_error=True
            )


class TestNumericValidator:
    """Test numeric validation."""
    
    def test_validate_positive(self):
        """Test positive number validation."""
        result = NumericValidator.validate_positive(100, "price")
        assert result == 100
        
        with pytest.raises(InvalidConfigError):
            NumericValidator.validate_positive(-10, "price")
        
        with pytest.raises(InvalidConfigError):
            NumericValidator.validate_positive(0, "price")
    
    def test_validate_positive_allow_zero(self):
        """Test positive with zero allowed."""
        result = NumericValidator.validate_positive(0, "value", allow_zero=True)
        assert result == 0
    
    def test_validate_range(self):
        """Test range validation."""
        result = NumericValidator.validate_range(50, "value", min_val=0, max_val=100)
        assert result == 50
        
        with pytest.raises(InvalidConfigError):
            NumericValidator.validate_range(150, "value", min_val=0, max_val=100)
    
    def test_validate_percentage(self):
        """Test percentage validation."""
        # Value between 0 and 1
        result = NumericValidator.validate_percentage(0.5, "rate")
        assert result == 0.5
        
        # Value > 1 should be normalized
        result = NumericValidator.validate_percentage(50, "rate")
        assert result == 0.5
    
    def test_validate_integer(self):
        """Test integer validation."""
        result = NumericValidator.validate_integer(100, "quantity")
        assert result == 100
        assert isinstance(result, int)
        
        # Float should be converted
        result = NumericValidator.validate_integer(100.5, "quantity")
        assert result == 100
    
    def test_safe_divide(self):
        """Test safe division."""
        assert NumericValidator.safe_divide(10, 2) == 5
        assert NumericValidator.safe_divide(10, 0) == 0  # Default
        assert NumericValidator.safe_divide(10, 0, default=-1) == -1


class TestRetryDecorator:
    """Test retry decorator."""
    
    def test_retry_success(self):
        """Test retry with eventual success."""
        attempts = 0
        
        @retry(max_attempts=3, delay=0.01)
        def flaky_function():
            nonlocal attempts
            attempts += 1
            if attempts < 2:
                raise ValueError("Temporary failure")
            return "success"
        
        result = flaky_function()
        
        assert result == "success"
        assert attempts == 2
    
    def test_retry_failure(self):
        """Test retry with ultimate failure."""
        @retry(max_attempts=3, delay=0.01)
        def always_fails():
            raise ValueError("Always fails")
        
        with pytest.raises(ValueError):
            always_fails()
    
    def test_retry_specific_exceptions(self):
        """Test retry with specific exception types."""
        @retry(max_attempts=3, delay=0.01, exceptions=(ValueError,))
        def raises_type_error():
            raise TypeError("Wrong type")
        
        # TypeError should not be retried
        with pytest.raises(TypeError):
            raises_type_error()


class TestHandleExceptions:
    """Test exception handling decorator."""
    
    def test_handle_with_default(self):
        """Test exception handling with default return."""
        @handle_exceptions(default_return=[])
        def fails():
            raise ValueError("Error")
        
        result = fails()
        assert result == []
    
    def test_handle_with_reraise(self):
        """Test exception handling with reraise."""
        @handle_exceptions(reraise=True)
        def fails():
            raise ValueError("Error")
        
        with pytest.raises(ValueError):
            fails()
    
    def test_handle_success(self):
        """Test that successful calls pass through."""
        @handle_exceptions(default_return=None)
        def succeeds():
            return "success"
        
        assert succeeds() == "success"


class TestGracefulDegradation:
    """Test graceful degradation decorator."""
    
    def test_degradation_to_fallback(self):
        """Test degradation to fallback function."""
        def fallback():
            return "fallback"
        
        @graceful_degradation(fallback=fallback)
        def primary():
            raise ValueError("Primary failed")
        
        result = primary()
        assert result == "fallback"
    
    def test_no_degradation_on_success(self):
        """Test no degradation when primary succeeds."""
        def fallback():
            return "fallback"
        
        @graceful_degradation(fallback=fallback)
        def primary():
            return "primary"
        
        result = primary()
        assert result == "primary"


class TestTimedDecorator:
    """Test timing decorator."""
    
    def test_timed_function(self):
        """Test that timing works."""
        @timed()
        def slow_function():
            time.sleep(0.01)
            return "done"
        
        result = slow_function()
        assert result == "done"


class TestSafeCall:
    """Test safe_call utility."""
    
    def test_safe_call_success(self):
        """Test safe_call with successful function."""
        def succeeds(x):
            return x * 2
        
        result = safe_call(succeeds, 5)
        assert result == 10
    
    def test_safe_call_failure(self):
        """Test safe_call with failing function."""
        def fails():
            raise ValueError("Error")
        
        result = safe_call(fails, default="default", log_error=False)
        assert result == "default"


class TestErrorContext:
    """Test error context manager."""
    
    def test_error_context_success(self):
        """Test error context with success."""
        with error_context("test operation"):
            result = 1 + 1
        
        assert result == 2
    
    def test_error_context_failure_suppressed(self):
        """Test error context with suppressed failure."""
        ctx = error_context("test", reraise=False, default="default")
        
        with ctx:
            raise ValueError("Error")
        
        assert ctx.exception is not None
        assert ctx.result == "default"
    
    def test_error_context_failure_raised(self):
        """Test error context with raised failure."""
        with pytest.raises(ValueError):
            with error_context("test", reraise=True):
                raise ValueError("Error")


class TestCircuitBreaker:
    """Test circuit breaker."""
    
    def test_circuit_breaker_closed(self):
        """Test circuit breaker in closed state."""
        cb = CircuitBreaker("test", CircuitBreakerConfig(failure_threshold=3))
        
        assert cb.state == CircuitState.CLOSED
        assert cb.is_closed
        
        # Successful call
        result = cb.call(lambda: "success")
        assert result == "success"
    
    def test_circuit_breaker_opens(self):
        """Test circuit breaker opens after failures."""
        config = CircuitBreakerConfig(failure_threshold=3, timeout=1.0)
        cb = CircuitBreaker("test", config)
        
        def failing():
            raise ValueError("Fail")
        
        # Trigger failures
        for _ in range(3):
            try:
                cb.call(failing)
            except ValueError:
                pass
        
        # Should be open now
        assert cb.state == CircuitState.OPEN
        
        # Calls should fail fast
        with pytest.raises(CircuitBreakerError):
            cb.call(lambda: "should fail fast")
    
    def test_circuit_breaker_half_open(self):
        """Test circuit breaker half-open state."""
        config = CircuitBreakerConfig(
            failure_threshold=2,
            success_threshold=1,
            timeout=0.01  # Very short for testing
        )
        cb = CircuitBreaker("test", config)
        
        # Open the circuit
        for _ in range(2):
            try:
                cb.call(lambda: (_ for _ in ()).throw(ValueError("Fail")))
            except ValueError:
                pass
        
        assert cb.is_open
        
        # Wait for timeout
        time.sleep(0.02)
        
        # Next call should transition to half-open
        result = cb.call(lambda: "success")
        assert result == "success"
        assert cb.is_closed
    
    def test_circuit_breaker_stats(self):
        """Test circuit breaker statistics."""
        cb = CircuitBreaker("test")
        
        # Successful calls
        cb.call(lambda: "a")
        cb.call(lambda: "b")
        
        # Failed call
        try:
            cb.call(lambda: (_ for _ in ()).throw(ValueError()))
        except ValueError:
            pass
        
        stats = cb.stats
        assert stats.total_calls == 3
        assert stats.successful_calls == 2
        assert stats.failed_calls == 1
    
    def test_circuit_breaker_reset(self):
        """Test circuit breaker manual reset."""
        config = CircuitBreakerConfig(failure_threshold=1)
        cb = CircuitBreaker("test", config)
        
        # Open the circuit
        try:
            cb.call(lambda: (_ for _ in ()).throw(ValueError()))
        except ValueError:
            pass
        
        assert cb.is_open
        
        # Reset
        cb.reset()
        
        assert cb.is_closed


class TestCircuitBreakerDecorator:
    """Test circuit breaker decorator."""
    
    def test_decorator_usage(self):
        """Test circuit breaker as decorator."""
        call_count = 0
        
        @circuit_breaker("test_func", failure_threshold=3, timeout=0.1)
        def test_function():
            nonlocal call_count
            call_count += 1
            return "success"
        
        result = test_function()
        assert result == "success"
        assert call_count == 1
        
        # Check circuit breaker is attached
        assert hasattr(test_function, 'circuit_breaker')
        assert isinstance(test_function.circuit_breaker, CircuitBreaker)
