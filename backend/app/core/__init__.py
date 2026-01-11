"""
AlphaTerminal Pro - Core Module
===============================

Core utilities and infrastructure components.

Modules:
    - validators: Input validation utilities
    - error_handlers: Error handling decorators
    - circuit_breaker: Circuit breaker pattern

Author: AlphaTerminal Team
Version: 1.0.0
"""

# Validators
from app.core.validators import (
    DataFrameValidator,
    NumericValidator,
    ConfigValidator,
    validate_ohlcv,
    validate_price,
    validate_quantity,
    validate_percentage,
    validate_dataframe,
    validate_positive_params,
    validate_range_params
)

# Error handlers
from app.core.error_handlers import (
    retry,
    handle_exceptions,
    graceful_degradation,
    timed,
    validate_args,
    ensure_not_none,
    deprecated,
    safe_call,
    error_context,
    ErrorHandler,
    global_error_handler
)

# Circuit breaker
from app.core.circuit_breaker import (
    CircuitBreaker,
    CircuitBreakerConfig,
    CircuitBreakerError,
    CircuitBreakerStats,
    CircuitState,
    circuit_breaker,
    circuit_breaker_registry
)

__all__ = [
    # Validators
    "DataFrameValidator",
    "NumericValidator",
    "ConfigValidator",
    "validate_ohlcv",
    "validate_price",
    "validate_quantity",
    "validate_percentage",
    "validate_dataframe",
    "validate_positive_params",
    "validate_range_params",
    
    # Error handlers
    "retry",
    "handle_exceptions",
    "graceful_degradation",
    "timed",
    "validate_args",
    "ensure_not_none",
    "deprecated",
    "safe_call",
    "error_context",
    "ErrorHandler",
    "global_error_handler",
    
    # Circuit breaker
    "CircuitBreaker",
    "CircuitBreakerConfig",
    "CircuitBreakerError",
    "CircuitBreakerStats",
    "CircuitState",
    "circuit_breaker",
    "circuit_breaker_registry"
]

__version__ = "1.0.0"
