"""
AlphaTerminal Pro - Circuit Breaker
====================================

Circuit breaker pattern for fault tolerance.

States:
- CLOSED: Normal operation, requests pass through
- OPEN: Failure threshold exceeded, requests fail fast
- HALF_OPEN: Testing if service recovered

Author: AlphaTerminal Team
Version: 1.0.0
"""

from __future__ import annotations

import logging
import time
import threading
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Any, Callable, Dict, Optional, TypeVar, Generic
from enum import Enum
from functools import wraps

logger = logging.getLogger(__name__)

T = TypeVar('T')


class CircuitState(Enum):
    """Circuit breaker states."""
    CLOSED = "closed"       # Normal operation
    OPEN = "open"           # Failing fast
    HALF_OPEN = "half_open" # Testing recovery


@dataclass
class CircuitBreakerConfig:
    """
    Circuit breaker configuration.
    
    Attributes:
        failure_threshold: Number of failures before opening
        success_threshold: Successes needed to close from half-open
        timeout: Seconds to wait before half-open
        exclude_exceptions: Exceptions that don't count as failures
    """
    failure_threshold: int = 5
    success_threshold: int = 2
    timeout: float = 30.0
    exclude_exceptions: tuple = ()
    
    # Advanced options
    failure_rate_threshold: Optional[float] = None  # 0.5 = 50% failure rate
    min_calls: int = 5  # Minimum calls before rate calculation


@dataclass
class CircuitBreakerStats:
    """Circuit breaker statistics."""
    total_calls: int = 0
    successful_calls: int = 0
    failed_calls: int = 0
    rejected_calls: int = 0
    last_failure_time: Optional[datetime] = None
    last_success_time: Optional[datetime] = None
    state_changes: int = 0
    
    @property
    def failure_rate(self) -> float:
        """Calculate failure rate."""
        if self.total_calls == 0:
            return 0.0
        return self.failed_calls / self.total_calls
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "total_calls": self.total_calls,
            "successful_calls": self.successful_calls,
            "failed_calls": self.failed_calls,
            "rejected_calls": self.rejected_calls,
            "failure_rate": round(self.failure_rate, 4),
            "last_failure_time": self.last_failure_time.isoformat() if self.last_failure_time else None,
            "last_success_time": self.last_success_time.isoformat() if self.last_success_time else None,
            "state_changes": self.state_changes
        }


class CircuitBreakerError(Exception):
    """Raised when circuit breaker is open."""
    
    def __init__(self, message: str, circuit_name: str, retry_after: float):
        self.circuit_name = circuit_name
        self.retry_after = retry_after
        super().__init__(message)


class CircuitBreaker:
    """
    Circuit breaker implementation.
    
    Prevents cascade failures by failing fast when a service is unhealthy.
    
    Example:
        ```python
        cb = CircuitBreaker("external_api", config=CircuitBreakerConfig(
            failure_threshold=5,
            timeout=30.0
        ))
        
        try:
            result = cb.call(external_api_function, arg1, arg2)
        except CircuitBreakerError as e:
            print(f"Circuit open, retry after {e.retry_after}s")
        ```
    """
    
    def __init__(
        self,
        name: str,
        config: Optional[CircuitBreakerConfig] = None
    ):
        self.name = name
        self.config = config or CircuitBreakerConfig()
        
        self._state = CircuitState.CLOSED
        self._failure_count = 0
        self._success_count = 0
        self._last_failure_time: Optional[float] = None
        
        self._stats = CircuitBreakerStats()
        self._lock = threading.RLock()
        
        logger.debug(f"CircuitBreaker '{name}' initialized")
    
    @property
    def state(self) -> CircuitState:
        """Get current state (may trigger half-open transition)."""
        with self._lock:
            if self._state == CircuitState.OPEN:
                # Check if timeout has passed
                if self._should_attempt_reset():
                    self._transition_to_half_open()
            return self._state
    
    @property
    def is_closed(self) -> bool:
        """Check if circuit is closed (normal operation)."""
        return self.state == CircuitState.CLOSED
    
    @property
    def is_open(self) -> bool:
        """Check if circuit is open (failing fast)."""
        return self.state == CircuitState.OPEN
    
    @property
    def is_half_open(self) -> bool:
        """Check if circuit is half-open (testing)."""
        return self.state == CircuitState.HALF_OPEN
    
    @property
    def stats(self) -> CircuitBreakerStats:
        """Get statistics."""
        return self._stats
    
    def call(self, func: Callable[..., T], *args, **kwargs) -> T:
        """
        Call function through circuit breaker.
        
        Args:
            func: Function to call
            *args: Positional arguments
            **kwargs: Keyword arguments
            
        Returns:
            Function result
            
        Raises:
            CircuitBreakerError: If circuit is open
            Exception: If function raises exception
        """
        with self._lock:
            state = self.state  # This may trigger half-open
            
            if state == CircuitState.OPEN:
                self._stats.rejected_calls += 1
                retry_after = self._get_retry_after()
                raise CircuitBreakerError(
                    f"Circuit '{self.name}' is open",
                    self.name,
                    retry_after
                )
        
        try:
            result = func(*args, **kwargs)
            self._on_success()
            return result
            
        except self.config.exclude_exceptions:
            # Don't count as failure
            raise
            
        except Exception as e:
            self._on_failure(e)
            raise
    
    def _on_success(self) -> None:
        """Handle successful call."""
        with self._lock:
            self._stats.total_calls += 1
            self._stats.successful_calls += 1
            self._stats.last_success_time = datetime.utcnow()
            
            if self._state == CircuitState.HALF_OPEN:
                self._success_count += 1
                if self._success_count >= self.config.success_threshold:
                    self._transition_to_closed()
            else:
                self._failure_count = 0
    
    def _on_failure(self, exception: Exception) -> None:
        """Handle failed call."""
        with self._lock:
            self._stats.total_calls += 1
            self._stats.failed_calls += 1
            self._stats.last_failure_time = datetime.utcnow()
            self._last_failure_time = time.time()
            
            self._failure_count += 1
            self._success_count = 0
            
            if self._state == CircuitState.HALF_OPEN:
                # Any failure in half-open goes back to open
                self._transition_to_open()
            elif self._state == CircuitState.CLOSED:
                # Check if should open
                if self._should_open():
                    self._transition_to_open()
            
            logger.warning(
                f"CircuitBreaker '{self.name}' recorded failure "
                f"({self._failure_count}/{self.config.failure_threshold}): {exception}"
            )
    
    def _should_open(self) -> bool:
        """Check if circuit should open."""
        # Check failure count threshold
        if self._failure_count >= self.config.failure_threshold:
            return True
        
        # Check failure rate threshold
        if self.config.failure_rate_threshold is not None:
            if self._stats.total_calls >= self.config.min_calls:
                if self._stats.failure_rate >= self.config.failure_rate_threshold:
                    return True
        
        return False
    
    def _should_attempt_reset(self) -> bool:
        """Check if should attempt reset (move to half-open)."""
        if self._last_failure_time is None:
            return False
        
        elapsed = time.time() - self._last_failure_time
        return elapsed >= self.config.timeout
    
    def _get_retry_after(self) -> float:
        """Get seconds until retry is allowed."""
        if self._last_failure_time is None:
            return 0.0
        
        elapsed = time.time() - self._last_failure_time
        remaining = self.config.timeout - elapsed
        return max(0.0, remaining)
    
    def _transition_to_open(self) -> None:
        """Transition to OPEN state."""
        self._state = CircuitState.OPEN
        self._stats.state_changes += 1
        logger.warning(f"CircuitBreaker '{self.name}' OPENED")
    
    def _transition_to_half_open(self) -> None:
        """Transition to HALF_OPEN state."""
        self._state = CircuitState.HALF_OPEN
        self._success_count = 0
        self._stats.state_changes += 1
        logger.info(f"CircuitBreaker '{self.name}' HALF-OPEN (testing)")
    
    def _transition_to_closed(self) -> None:
        """Transition to CLOSED state."""
        self._state = CircuitState.CLOSED
        self._failure_count = 0
        self._success_count = 0
        self._stats.state_changes += 1
        logger.info(f"CircuitBreaker '{self.name}' CLOSED (recovered)")
    
    def reset(self) -> None:
        """Manually reset circuit breaker."""
        with self._lock:
            self._state = CircuitState.CLOSED
            self._failure_count = 0
            self._success_count = 0
            self._last_failure_time = None
            logger.info(f"CircuitBreaker '{self.name}' manually reset")
    
    def force_open(self) -> None:
        """Manually force circuit open."""
        with self._lock:
            self._transition_to_open()
            self._last_failure_time = time.time()
            logger.info(f"CircuitBreaker '{self.name}' manually opened")


# =============================================================================
# DECORATOR
# =============================================================================

def circuit_breaker(
    name: str,
    failure_threshold: int = 5,
    timeout: float = 30.0,
    exclude_exceptions: tuple = ()
):
    """
    Decorator to wrap function with circuit breaker.
    
    Example:
        ```python
        @circuit_breaker("api_call", failure_threshold=3, timeout=60)
        def call_api():
            return requests.get("https://api.example.com")
        ```
    """
    config = CircuitBreakerConfig(
        failure_threshold=failure_threshold,
        timeout=timeout,
        exclude_exceptions=exclude_exceptions
    )
    cb = CircuitBreaker(name, config)
    
    def decorator(func: Callable[..., T]) -> Callable[..., T]:
        @wraps(func)
        def wrapper(*args, **kwargs) -> T:
            return cb.call(func, *args, **kwargs)
        
        # Attach circuit breaker for inspection
        wrapper.circuit_breaker = cb
        return wrapper
    
    return decorator


# =============================================================================
# CIRCUIT BREAKER REGISTRY
# =============================================================================

class CircuitBreakerRegistry:
    """
    Registry for managing multiple circuit breakers.
    
    Example:
        ```python
        registry = CircuitBreakerRegistry()
        
        cb = registry.get_or_create("api_service")
        result = cb.call(api_function)
        
        # Get all stats
        all_stats = registry.get_all_stats()
        ```
    """
    
    _instance: Optional["CircuitBreakerRegistry"] = None
    _lock = threading.Lock()
    
    def __new__(cls) -> "CircuitBreakerRegistry":
        """Singleton pattern."""
        with cls._lock:
            if cls._instance is None:
                cls._instance = super().__new__(cls)
                cls._instance._circuit_breakers = {}
                cls._instance._registry_lock = threading.RLock()
            return cls._instance
    
    def get_or_create(
        self,
        name: str,
        config: Optional[CircuitBreakerConfig] = None
    ) -> CircuitBreaker:
        """Get existing or create new circuit breaker."""
        with self._registry_lock:
            if name not in self._circuit_breakers:
                self._circuit_breakers[name] = CircuitBreaker(name, config)
            return self._circuit_breakers[name]
    
    def get(self, name: str) -> Optional[CircuitBreaker]:
        """Get circuit breaker by name."""
        return self._circuit_breakers.get(name)
    
    def list_all(self) -> Dict[str, CircuitBreaker]:
        """Get all circuit breakers."""
        return self._circuit_breakers.copy()
    
    def get_all_stats(self) -> Dict[str, Dict[str, Any]]:
        """Get stats for all circuit breakers."""
        return {
            name: {
                "state": cb.state.value,
                **cb.stats.to_dict()
            }
            for name, cb in self._circuit_breakers.items()
        }
    
    def reset_all(self) -> None:
        """Reset all circuit breakers."""
        with self._registry_lock:
            for cb in self._circuit_breakers.values():
                cb.reset()
    
    def remove(self, name: str) -> None:
        """Remove circuit breaker."""
        with self._registry_lock:
            self._circuit_breakers.pop(name, None)


# Singleton instance
circuit_breaker_registry = CircuitBreakerRegistry()
