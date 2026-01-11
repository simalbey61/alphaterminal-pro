"""
AlphaTerminal Pro - Data Provider Exceptions
============================================

Comprehensive exception hierarchy for data provider operations.

Author: AlphaTerminal Team
Version: 1.0.0
"""

from typing import Optional, Dict, Any, List
from datetime import datetime


class DataProviderException(Exception):
    """Base exception for all data provider errors."""
    
    def __init__(
        self,
        message: str,
        provider: Optional[str] = None,
        symbol: Optional[str] = None,
        details: Optional[Dict[str, Any]] = None
    ):
        self.message = message
        self.provider = provider
        self.symbol = symbol
        self.details = details or {}
        self.timestamp = datetime.now()
        
        super().__init__(self._format_message())
    
    def _format_message(self) -> str:
        parts = [self.message]
        if self.provider:
            parts.append(f"Provider: {self.provider}")
        if self.symbol:
            parts.append(f"Symbol: {self.symbol}")
        return " | ".join(parts)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "error_type": self.__class__.__name__,
            "message": self.message,
            "provider": self.provider,
            "symbol": self.symbol,
            "details": self.details,
            "timestamp": self.timestamp.isoformat()
        }


# =============================================================================
# CONNECTION ERRORS
# =============================================================================

class ConnectionError(DataProviderException):
    """Network connection failed."""
    pass


class TimeoutError(DataProviderException):
    """Request timed out."""
    
    def __init__(
        self,
        message: str = "Request timed out",
        timeout_seconds: Optional[float] = None,
        **kwargs
    ):
        self.timeout_seconds = timeout_seconds
        if timeout_seconds:
            kwargs.setdefault('details', {})['timeout_seconds'] = timeout_seconds
        super().__init__(message, **kwargs)


class AuthenticationError(DataProviderException):
    """Authentication failed."""
    pass


class SSLError(ConnectionError):
    """SSL/TLS error occurred."""
    pass


# =============================================================================
# RATE LIMITING ERRORS
# =============================================================================

class RateLimitError(DataProviderException):
    """Rate limit exceeded."""
    
    def __init__(
        self,
        message: str = "Rate limit exceeded",
        retry_after: Optional[float] = None,
        limit: Optional[int] = None,
        remaining: Optional[int] = None,
        **kwargs
    ):
        self.retry_after = retry_after
        self.limit = limit
        self.remaining = remaining
        
        kwargs.setdefault('details', {}).update({
            'retry_after_seconds': retry_after,
            'rate_limit': limit,
            'remaining': remaining
        })
        super().__init__(message, **kwargs)


class QuotaExceededError(RateLimitError):
    """Daily/monthly quota exceeded."""
    pass


# =============================================================================
# DATA ERRORS
# =============================================================================

class DataError(DataProviderException):
    """Base class for data-related errors."""
    pass


class SymbolNotFoundError(DataError):
    """Symbol does not exist or is not supported."""
    
    def __init__(
        self,
        symbol: str,
        provider: Optional[str] = None,
        suggestions: Optional[List[str]] = None,
        **kwargs
    ):
        self.suggestions = suggestions
        message = f"Symbol '{symbol}' not found"
        if suggestions:
            kwargs.setdefault('details', {})['suggestions'] = suggestions
        super().__init__(message, provider=provider, symbol=symbol, **kwargs)


class NoDataError(DataError):
    """No data available for the requested parameters."""
    
    def __init__(
        self,
        symbol: str,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None,
        interval: Optional[str] = None,
        **kwargs
    ):
        self.start_date = start_date
        self.end_date = end_date
        self.interval = interval
        
        message = f"No data available for '{symbol}'"
        kwargs.setdefault('details', {}).update({
            'start_date': start_date.isoformat() if start_date else None,
            'end_date': end_date.isoformat() if end_date else None,
            'interval': interval
        })
        super().__init__(message, symbol=symbol, **kwargs)


class InsufficientDataError(DataError):
    """Not enough data points for the requested operation."""
    
    def __init__(
        self,
        symbol: str,
        required: int,
        received: int,
        **kwargs
    ):
        self.required = required
        self.received = received
        
        message = f"Insufficient data for '{symbol}': required {required}, got {received}"
        kwargs.setdefault('details', {}).update({
            'required_rows': required,
            'received_rows': received
        })
        super().__init__(message, symbol=symbol, **kwargs)


class DataValidationError(DataError):
    """Data failed validation checks."""
    
    def __init__(
        self,
        message: str,
        validation_errors: Optional[List[str]] = None,
        **kwargs
    ):
        self.validation_errors = validation_errors or []
        kwargs.setdefault('details', {})['validation_errors'] = self.validation_errors
        super().__init__(message, **kwargs)


class StaleDataError(DataError):
    """Data is outdated."""
    
    def __init__(
        self,
        symbol: str,
        last_update: datetime,
        max_age_seconds: float,
        **kwargs
    ):
        self.last_update = last_update
        self.max_age_seconds = max_age_seconds
        
        age = (datetime.now() - last_update).total_seconds()
        message = f"Stale data for '{symbol}': last update {age:.0f}s ago (max: {max_age_seconds}s)"
        
        kwargs.setdefault('details', {}).update({
            'last_update': last_update.isoformat(),
            'age_seconds': age,
            'max_age_seconds': max_age_seconds
        })
        super().__init__(message, symbol=symbol, **kwargs)


class DataCorruptionError(DataError):
    """Data is corrupted or malformed."""
    pass


# =============================================================================
# PROVIDER ERRORS
# =============================================================================

class ProviderError(DataProviderException):
    """Base class for provider-specific errors."""
    pass


class ProviderUnavailableError(ProviderError):
    """Provider is temporarily unavailable."""
    
    def __init__(
        self,
        provider: str,
        reason: Optional[str] = None,
        estimated_recovery: Optional[datetime] = None,
        **kwargs
    ):
        self.reason = reason
        self.estimated_recovery = estimated_recovery
        
        message = f"Provider '{provider}' is unavailable"
        if reason:
            message += f": {reason}"
        
        kwargs.setdefault('details', {}).update({
            'reason': reason,
            'estimated_recovery': estimated_recovery.isoformat() if estimated_recovery else None
        })
        super().__init__(message, provider=provider, **kwargs)


class ProviderMaintenanceError(ProviderUnavailableError):
    """Provider is under maintenance."""
    pass


class ProviderDeprecatedError(ProviderError):
    """Provider is deprecated."""
    
    def __init__(
        self,
        provider: str,
        replacement: Optional[str] = None,
        sunset_date: Optional[datetime] = None,
        **kwargs
    ):
        self.replacement = replacement
        self.sunset_date = sunset_date
        
        message = f"Provider '{provider}' is deprecated"
        kwargs.setdefault('details', {}).update({
            'replacement': replacement,
            'sunset_date': sunset_date.isoformat() if sunset_date else None
        })
        super().__init__(message, provider=provider, **kwargs)


class AllProvidersFailedError(ProviderError):
    """All configured providers failed."""
    
    def __init__(
        self,
        symbol: str,
        provider_errors: Dict[str, Exception],
        **kwargs
    ):
        self.provider_errors = provider_errors
        
        message = f"All providers failed for '{symbol}'"
        kwargs.setdefault('details', {})['provider_errors'] = {
            p: str(e) for p, e in provider_errors.items()
        }
        super().__init__(message, symbol=symbol, **kwargs)


# =============================================================================
# CACHE ERRORS
# =============================================================================

class CacheError(DataProviderException):
    """Base class for cache-related errors."""
    pass


class CacheMissError(CacheError):
    """Requested data not in cache."""
    pass


class CacheWriteError(CacheError):
    """Failed to write to cache."""
    pass


class CacheConnectionError(CacheError):
    """Failed to connect to cache backend."""
    pass


class CacheSerializationError(CacheError):
    """Failed to serialize/deserialize cache data."""
    pass


# =============================================================================
# CONFIGURATION ERRORS
# =============================================================================

class ConfigurationError(DataProviderException):
    """Configuration is invalid or missing."""
    pass


class InvalidIntervalError(ConfigurationError):
    """Invalid time interval specified."""
    
    def __init__(
        self,
        interval: str,
        valid_intervals: List[str],
        provider: Optional[str] = None,
        **kwargs
    ):
        self.interval = interval
        self.valid_intervals = valid_intervals
        
        message = f"Invalid interval '{interval}'. Valid: {valid_intervals}"
        kwargs.setdefault('details', {})['valid_intervals'] = valid_intervals
        super().__init__(message, provider=provider, **kwargs)


class InvalidDateRangeError(ConfigurationError):
    """Invalid date range specified."""
    
    def __init__(
        self,
        start_date: datetime,
        end_date: datetime,
        reason: str,
        **kwargs
    ):
        self.start_date = start_date
        self.end_date = end_date
        self.reason = reason
        
        message = f"Invalid date range: {reason}"
        kwargs.setdefault('details', {}).update({
            'start_date': start_date.isoformat(),
            'end_date': end_date.isoformat(),
            'reason': reason
        })
        super().__init__(message, **kwargs)


# =============================================================================
# MARKET ERRORS
# =============================================================================

class MarketError(DataProviderException):
    """Base class for market-related errors."""
    pass


class MarketClosedError(MarketError):
    """Market is closed."""
    
    def __init__(
        self,
        market: str,
        next_open: Optional[datetime] = None,
        **kwargs
    ):
        self.market = market
        self.next_open = next_open
        
        message = f"Market '{market}' is closed"
        if next_open:
            message += f". Opens at {next_open.isoformat()}"
        
        kwargs.setdefault('details', {}).update({
            'market': market,
            'next_open': next_open.isoformat() if next_open else None
        })
        super().__init__(message, **kwargs)


class TradingHaltedError(MarketError):
    """Trading is halted for the symbol."""
    
    def __init__(
        self,
        symbol: str,
        reason: Optional[str] = None,
        **kwargs
    ):
        self.halt_reason = reason
        
        message = f"Trading halted for '{symbol}'"
        if reason:
            message += f": {reason}"
        
        kwargs.setdefault('details', {})['halt_reason'] = reason
        super().__init__(message, symbol=symbol, **kwargs)


# =============================================================================
# EXCEPTION MAPPING
# =============================================================================

# Map HTTP status codes to exceptions
HTTP_STATUS_EXCEPTIONS = {
    400: DataValidationError,
    401: AuthenticationError,
    403: AuthenticationError,
    404: SymbolNotFoundError,
    408: TimeoutError,
    429: RateLimitError,
    500: ProviderError,
    502: ProviderUnavailableError,
    503: ProviderMaintenanceError,
    504: TimeoutError,
}


def exception_from_status_code(
    status_code: int,
    message: str = "",
    **kwargs
) -> DataProviderException:
    """Create appropriate exception from HTTP status code."""
    exc_class = HTTP_STATUS_EXCEPTIONS.get(status_code, DataProviderException)
    return exc_class(message or f"HTTP {status_code}", **kwargs)


# =============================================================================
# ALL EXPORTS
# =============================================================================

__all__ = [
    # Base
    "DataProviderException",
    
    # Connection
    "ConnectionError",
    "TimeoutError",
    "AuthenticationError",
    "SSLError",
    
    # Rate limiting
    "RateLimitError",
    "QuotaExceededError",
    
    # Data
    "DataError",
    "SymbolNotFoundError",
    "NoDataError",
    "InsufficientDataError",
    "DataValidationError",
    "StaleDataError",
    "DataCorruptionError",
    
    # Provider
    "ProviderError",
    "ProviderUnavailableError",
    "ProviderMaintenanceError",
    "ProviderDeprecatedError",
    "AllProvidersFailedError",
    
    # Cache
    "CacheError",
    "CacheMissError",
    "CacheWriteError",
    "CacheConnectionError",
    "CacheSerializationError",
    
    # Configuration
    "ConfigurationError",
    "InvalidIntervalError",
    "InvalidDateRangeError",
    
    # Market
    "MarketError",
    "MarketClosedError",
    "TradingHaltedError",
    
    # Utilities
    "HTTP_STATUS_EXCEPTIONS",
    "exception_from_status_code",
]
