"""
AlphaTerminal Pro - Error Handling Decorators
=============================================

Decorators for consistent error handling across the application.

Features:
- Automatic retry with backoff
- Exception transformation
- Graceful degradation
- Logging integration
- Performance tracking

Author: AlphaTerminal Team
Version: 1.0.0
"""

from __future__ import annotations

import logging
import time
import functools
from typing import Any, Callable, Dict, Optional, Tuple, Type, TypeVar, Union
from datetime import datetime
import traceback

from app.backtest.exceptions import BacktestError

logger = logging.getLogger(__name__)

T = TypeVar('T')


# =============================================================================
# RETRY DECORATOR
# =============================================================================

def retry(
    max_attempts: int = 3,
    delay: float = 1.0,
    backoff: float = 2.0,
    max_delay: float = 60.0,
    exceptions: Tuple[Type[Exception], ...] = (Exception,),
    on_retry: Optional[Callable[[Exception, int], None]] = None
):
    """
    Retry decorator with exponential backoff.
    
    Args:
        max_attempts: Maximum retry attempts
        delay: Initial delay between retries (seconds)
        backoff: Backoff multiplier
        max_delay: Maximum delay cap
        exceptions: Exception types to retry on
        on_retry: Callback called on each retry (exception, attempt)
        
    Example:
        ```python
        @retry(max_attempts=3, delay=1.0, exceptions=(ConnectionError,))
        def fetch_data():
            return requests.get(url)
        ```
    """
    def decorator(func: Callable[..., T]) -> Callable[..., T]:
        @functools.wraps(func)
        def wrapper(*args, **kwargs) -> T:
            current_delay = delay
            last_exception: Optional[Exception] = None
            
            for attempt in range(1, max_attempts + 1):
                try:
                    return func(*args, **kwargs)
                    
                except exceptions as e:
                    last_exception = e
                    
                    if attempt == max_attempts:
                        logger.error(
                            f"Function '{func.__name__}' failed after {max_attempts} attempts: {e}"
                        )
                        raise
                    
                    logger.warning(
                        f"Function '{func.__name__}' attempt {attempt}/{max_attempts} failed: {e}. "
                        f"Retrying in {current_delay:.1f}s..."
                    )
                    
                    if on_retry:
                        on_retry(e, attempt)
                    
                    time.sleep(current_delay)
                    current_delay = min(current_delay * backoff, max_delay)
            
            # Should never reach here
            raise last_exception  # type: ignore
        
        return wrapper
    return decorator


# =============================================================================
# EXCEPTION HANDLER DECORATOR
# =============================================================================

def handle_exceptions(
    default_return: Any = None,
    log_level: int = logging.ERROR,
    reraise: bool = False,
    transform_to: Optional[Type[Exception]] = None
):
    """
    Exception handling decorator.
    
    Args:
        default_return: Value to return on exception
        log_level: Logging level for exceptions
        reraise: Whether to reraise the exception
        transform_to: Transform exception to this type
        
    Example:
        ```python
        @handle_exceptions(default_return=[], log_level=logging.WARNING)
        def get_trades():
            # If this fails, returns [] instead of raising
            return db.fetch_trades()
        
        @handle_exceptions(transform_to=BacktestError, reraise=True)
        def process_data():
            # Transforms any exception to BacktestError
            return data.process()
        ```
    """
    def decorator(func: Callable[..., T]) -> Callable[..., T]:
        @functools.wraps(func)
        def wrapper(*args, **kwargs) -> T:
            try:
                return func(*args, **kwargs)
                
            except Exception as e:
                # Log the exception
                logger.log(
                    log_level,
                    f"Exception in '{func.__name__}': {type(e).__name__}: {e}",
                    exc_info=(log_level >= logging.ERROR)
                )
                
                # Transform if specified
                if transform_to is not None:
                    transformed = transform_to(str(e))
                    transformed.__cause__ = e
                    if reraise:
                        raise transformed from e
                    return default_return
                
                # Reraise if specified
                if reraise:
                    raise
                
                # Return default
                return default_return
        
        return wrapper
    return decorator


# =============================================================================
# GRACEFUL DEGRADATION DECORATOR
# =============================================================================

def graceful_degradation(
    fallback: Callable[..., T],
    exceptions: Tuple[Type[Exception], ...] = (Exception,),
    log_degradation: bool = True
):
    """
    Graceful degradation decorator.
    
    Falls back to alternative function on failure.
    
    Args:
        fallback: Fallback function to call
        exceptions: Exceptions that trigger fallback
        log_degradation: Whether to log degradation
        
    Example:
        ```python
        def fetch_from_cache(symbol):
            return cache.get(symbol)
        
        @graceful_degradation(fallback=fetch_from_cache)
        def fetch_from_api(symbol):
            return api.get_price(symbol)
        ```
    """
    def decorator(func: Callable[..., T]) -> Callable[..., T]:
        @functools.wraps(func)
        def wrapper(*args, **kwargs) -> T:
            try:
                return func(*args, **kwargs)
                
            except exceptions as e:
                if log_degradation:
                    logger.warning(
                        f"Function '{func.__name__}' failed, degrading to '{fallback.__name__}': {e}"
                    )
                
                return fallback(*args, **kwargs)
        
        return wrapper
    return decorator


# =============================================================================
# TIMING DECORATOR
# =============================================================================

def timed(
    log_level: int = logging.DEBUG,
    warn_threshold: Optional[float] = None
):
    """
    Timing decorator.
    
    Args:
        log_level: Logging level for timing info
        warn_threshold: Seconds threshold for warning
        
    Example:
        ```python
        @timed(warn_threshold=5.0)
        def slow_function():
            time.sleep(6)  # Will log warning
        ```
    """
    def decorator(func: Callable[..., T]) -> Callable[..., T]:
        @functools.wraps(func)
        def wrapper(*args, **kwargs) -> T:
            start = time.perf_counter()
            
            try:
                return func(*args, **kwargs)
            finally:
                elapsed = time.perf_counter() - start
                
                if warn_threshold and elapsed > warn_threshold:
                    logger.warning(
                        f"Function '{func.__name__}' took {elapsed:.3f}s "
                        f"(threshold: {warn_threshold}s)"
                    )
                else:
                    logger.log(
                        log_level,
                        f"Function '{func.__name__}' completed in {elapsed:.3f}s"
                    )
        
        return wrapper
    return decorator


# =============================================================================
# VALIDATION DECORATOR
# =============================================================================

def validate_args(**validators: Callable[[Any], bool]):
    """
    Argument validation decorator.
    
    Args:
        **validators: Dict of arg_name -> validator_function
        
    Example:
        ```python
        @validate_args(
            quantity=lambda x: x > 0,
            price=lambda x: x > 0
        )
        def place_order(quantity: int, price: float):
            pass
        ```
    """
    def decorator(func: Callable[..., T]) -> Callable[..., T]:
        @functools.wraps(func)
        def wrapper(*args, **kwargs) -> T:
            import inspect
            sig = inspect.signature(func)
            bound = sig.bind(*args, **kwargs)
            bound.apply_defaults()
            
            for param_name, validator in validators.items():
                if param_name in bound.arguments:
                    value = bound.arguments[param_name]
                    if not validator(value):
                        raise ValueError(
                            f"Invalid value for '{param_name}': {value}"
                        )
            
            return func(*args, **kwargs)
        
        return wrapper
    return decorator


# =============================================================================
# ENSURE NOT NONE DECORATOR
# =============================================================================

def ensure_not_none(
    result_name: str = "result",
    message: Optional[str] = None
):
    """
    Ensure function doesn't return None.
    
    Args:
        result_name: Name for error message
        message: Custom error message
        
    Example:
        ```python
        @ensure_not_none(result_name="trade")
        def get_trade(trade_id: str):
            return db.find_trade(trade_id)  # Raises if None
        ```
    """
    def decorator(func: Callable[..., T]) -> Callable[..., T]:
        @functools.wraps(func)
        def wrapper(*args, **kwargs) -> T:
            result = func(*args, **kwargs)
            
            if result is None:
                error_msg = message or f"Expected {result_name} but got None"
                raise ValueError(error_msg)
            
            return result
        
        return wrapper
    return decorator


# =============================================================================
# DEPRECATION DECORATOR
# =============================================================================

def deprecated(
    reason: str = "",
    replacement: Optional[str] = None,
    remove_in: Optional[str] = None
):
    """
    Mark function as deprecated.
    
    Args:
        reason: Reason for deprecation
        replacement: Suggested replacement function
        remove_in: Version where it will be removed
        
    Example:
        ```python
        @deprecated(reason="Use new_function instead", remove_in="2.0.0")
        def old_function():
            pass
        ```
    """
    def decorator(func: Callable[..., T]) -> Callable[..., T]:
        @functools.wraps(func)
        def wrapper(*args, **kwargs) -> T:
            msg = f"Function '{func.__name__}' is deprecated"
            
            if reason:
                msg += f": {reason}"
            if replacement:
                msg += f". Use '{replacement}' instead"
            if remove_in:
                msg += f". Will be removed in version {remove_in}"
            
            import warnings
            warnings.warn(msg, DeprecationWarning, stacklevel=2)
            logger.warning(msg)
            
            return func(*args, **kwargs)
        
        return wrapper
    return decorator


# =============================================================================
# SAFE CALL UTILITY
# =============================================================================

def safe_call(
    func: Callable[..., T],
    *args,
    default: T = None,
    exceptions: Tuple[Type[Exception], ...] = (Exception,),
    log_error: bool = True,
    **kwargs
) -> T:
    """
    Safely call a function with exception handling.
    
    Args:
        func: Function to call
        *args: Positional arguments
        default: Default value on exception
        exceptions: Exception types to catch
        log_error: Whether to log errors
        **kwargs: Keyword arguments
        
    Returns:
        Function result or default value
        
    Example:
        ```python
        result = safe_call(risky_function, arg1, arg2, default=[])
        ```
    """
    try:
        return func(*args, **kwargs)
    except exceptions as e:
        if log_error:
            logger.error(f"safe_call failed for '{func.__name__}': {e}")
        return default


# =============================================================================
# ERROR CONTEXT MANAGER
# =============================================================================

class error_context:
    """
    Context manager for error handling.
    
    Example:
        ```python
        with error_context("Processing trades", reraise=False, default=[]):
            trades = process_trades()
        ```
    """
    
    def __init__(
        self,
        operation: str,
        reraise: bool = True,
        default: Any = None,
        log_level: int = logging.ERROR,
        transform_to: Optional[Type[Exception]] = None
    ):
        self.operation = operation
        self.reraise = reraise
        self.default = default
        self.log_level = log_level
        self.transform_to = transform_to
        self.exception: Optional[Exception] = None
        self.result: Any = None
    
    def __enter__(self):
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        if exc_val is not None:
            self.exception = exc_val
            
            logger.log(
                self.log_level,
                f"Error during '{self.operation}': {exc_type.__name__}: {exc_val}",
                exc_info=(self.log_level >= logging.ERROR)
            )
            
            if self.transform_to is not None:
                transformed = self.transform_to(f"{self.operation}: {exc_val}")
                transformed.__cause__ = exc_val
                raise transformed from exc_val
            
            if not self.reraise:
                self.result = self.default
                return True  # Suppress exception
        
        return False


# =============================================================================
# COMBINED ERROR HANDLER CLASS
# =============================================================================

class ErrorHandler:
    """
    Centralized error handler for consistent error management.
    
    Example:
        ```python
        handler = ErrorHandler(
            log_errors=True,
            collect_stats=True
        )
        
        @handler.wrap
        def my_function():
            pass
        
        # Get error stats
        print(handler.get_stats())
        ```
    """
    
    def __init__(
        self,
        log_errors: bool = True,
        collect_stats: bool = True,
        default_retry_config: Optional[Dict] = None
    ):
        self.log_errors = log_errors
        self.collect_stats = collect_stats
        self.default_retry_config = default_retry_config or {}
        
        self._error_counts: Dict[str, int] = {}
        self._last_errors: Dict[str, Tuple[datetime, Exception]] = {}
    
    def wrap(
        self,
        func: Optional[Callable] = None,
        *,
        default_return: Any = None,
        reraise: bool = True,
        do_retry: bool = False
    ):
        """
        Wrap function with error handling.
        
        Can be used as decorator with or without arguments.
        """
        def decorator(f: Callable[..., T]) -> Callable[..., T]:
            @functools.wraps(f)
            def wrapper(*args, **kwargs) -> T:
                func_name = f.__name__
                
                try:
                    return f(*args, **kwargs)
                    
                except Exception as e:
                    # Record error
                    if self.collect_stats:
                        error_type = type(e).__name__
                        self._error_counts[error_type] = self._error_counts.get(error_type, 0) + 1
                        self._last_errors[func_name] = (datetime.utcnow(), e)
                    
                    # Log error
                    if self.log_errors:
                        logger.error(f"Error in '{func_name}': {e}", exc_info=True)
                    
                    # Handle
                    if reraise:
                        raise
                    return default_return
            
            return wrapper
        
        if func is not None:
            return decorator(func)
        return decorator
    
    def get_stats(self) -> Dict[str, Any]:
        """Get error statistics."""
        return {
            "error_counts": self._error_counts.copy(),
            "total_errors": sum(self._error_counts.values()),
            "last_errors": {
                func: {
                    "time": err_time.isoformat(),
                    "error": str(error)
                }
                for func, (err_time, error) in self._last_errors.items()
            }
        }
    
    def clear_stats(self) -> None:
        """Clear error statistics."""
        self._error_counts.clear()
        self._last_errors.clear()


# =============================================================================
# GLOBAL ERROR HANDLER
# =============================================================================

# Global error handler instance
global_error_handler = ErrorHandler(log_errors=True, collect_stats=True)
