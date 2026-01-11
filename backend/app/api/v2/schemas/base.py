"""
AlphaTerminal Pro - API v2 Base Schemas
=======================================

Standard request/response schemas for API consistency.

Author: AlphaTerminal Team
Version: 2.0.0
"""

from datetime import datetime
from typing import Optional, Dict, Any, List, Generic, TypeVar, Union
from enum import Enum
from pydantic import BaseModel, Field, ConfigDict, field_validator
import uuid


# =============================================================================
# TYPE VARIABLES
# =============================================================================

T = TypeVar('T')


# =============================================================================
# ENUMS
# =============================================================================

class ResponseStatus(str, Enum):
    """API response status codes."""
    SUCCESS = "success"
    ERROR = "error"
    PARTIAL = "partial"
    PENDING = "pending"


class ErrorCode(str, Enum):
    """Standardized error codes."""
    # General
    UNKNOWN_ERROR = "UNKNOWN_ERROR"
    INTERNAL_ERROR = "INTERNAL_ERROR"
    
    # Validation
    VALIDATION_ERROR = "VALIDATION_ERROR"
    INVALID_REQUEST = "INVALID_REQUEST"
    MISSING_FIELD = "MISSING_FIELD"
    INVALID_FORMAT = "INVALID_FORMAT"
    
    # Authentication
    UNAUTHORIZED = "UNAUTHORIZED"
    FORBIDDEN = "FORBIDDEN"
    TOKEN_EXPIRED = "TOKEN_EXPIRED"
    INVALID_TOKEN = "INVALID_TOKEN"
    
    # Resource
    NOT_FOUND = "NOT_FOUND"
    ALREADY_EXISTS = "ALREADY_EXISTS"
    CONFLICT = "CONFLICT"
    
    # Rate limiting
    RATE_LIMITED = "RATE_LIMITED"
    QUOTA_EXCEEDED = "QUOTA_EXCEEDED"
    
    # Data
    NO_DATA = "NO_DATA"
    STALE_DATA = "STALE_DATA"
    DATA_FETCH_ERROR = "DATA_FETCH_ERROR"
    
    # Provider
    PROVIDER_ERROR = "PROVIDER_ERROR"
    PROVIDER_UNAVAILABLE = "PROVIDER_UNAVAILABLE"
    
    # Business logic
    INSUFFICIENT_FUNDS = "INSUFFICIENT_FUNDS"
    MARKET_CLOSED = "MARKET_CLOSED"
    SYMBOL_NOT_FOUND = "SYMBOL_NOT_FOUND"


# =============================================================================
# BASE MODELS
# =============================================================================

class BaseSchema(BaseModel):
    """Base schema with common configuration."""
    
    model_config = ConfigDict(
        from_attributes=True,
        populate_by_name=True,
        use_enum_values=True,
        json_encoders={
            datetime: lambda v: v.isoformat()
        }
    )


class TimestampMixin(BaseModel):
    """Mixin for timestamp fields."""
    created_at: datetime = Field(default_factory=datetime.utcnow)
    updated_at: Optional[datetime] = None


# =============================================================================
# ERROR RESPONSE
# =============================================================================

class ErrorDetail(BaseSchema):
    """Detailed error information."""
    code: ErrorCode
    message: str
    field: Optional[str] = None
    details: Optional[Dict[str, Any]] = None


class ErrorResponse(BaseSchema):
    """Standard error response."""
    status: ResponseStatus = ResponseStatus.ERROR
    error: ErrorDetail
    request_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    path: Optional[str] = None
    
    @classmethod
    def create(
        cls,
        code: ErrorCode,
        message: str,
        field: Optional[str] = None,
        details: Optional[Dict[str, Any]] = None,
        path: Optional[str] = None,
        request_id: Optional[str] = None
    ) -> "ErrorResponse":
        """Factory method to create error response."""
        return cls(
            error=ErrorDetail(
                code=code,
                message=message,
                field=field,
                details=details
            ),
            path=path,
            request_id=request_id or str(uuid.uuid4())
        )


# =============================================================================
# SUCCESS RESPONSE
# =============================================================================

class MetaInfo(BaseSchema):
    """Response metadata."""
    request_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    duration_ms: Optional[float] = None
    version: str = "2.0.0"
    
    # Pagination
    page: Optional[int] = None
    page_size: Optional[int] = None
    total_items: Optional[int] = None
    total_pages: Optional[int] = None
    
    # Additional info
    cache_hit: Optional[bool] = None
    data_source: Optional[str] = None


class APIResponse(BaseSchema, Generic[T]):
    """Standard API response wrapper."""
    status: ResponseStatus = ResponseStatus.SUCCESS
    data: Optional[T] = None
    meta: MetaInfo = Field(default_factory=MetaInfo)
    errors: Optional[List[ErrorDetail]] = None
    warnings: Optional[List[str]] = None
    
    @classmethod
    def success(
        cls,
        data: T,
        meta: Optional[MetaInfo] = None,
        warnings: Optional[List[str]] = None
    ) -> "APIResponse[T]":
        """Create success response."""
        return cls(
            status=ResponseStatus.SUCCESS,
            data=data,
            meta=meta or MetaInfo(),
            warnings=warnings
        )
    
    @classmethod
    def partial(
        cls,
        data: T,
        errors: List[ErrorDetail],
        meta: Optional[MetaInfo] = None
    ) -> "APIResponse[T]":
        """Create partial success response."""
        return cls(
            status=ResponseStatus.PARTIAL,
            data=data,
            meta=meta or MetaInfo(),
            errors=errors
        )
    
    @classmethod
    def error(
        cls,
        code: ErrorCode,
        message: str,
        details: Optional[Dict[str, Any]] = None
    ) -> "APIResponse[None]":
        """Create error response."""
        return cls(
            status=ResponseStatus.ERROR,
            data=None,
            errors=[ErrorDetail(code=code, message=message, details=details)]
        )


# =============================================================================
# PAGINATION
# =============================================================================

class PaginationParams(BaseSchema):
    """Pagination parameters."""
    page: int = Field(default=1, ge=1, description="Page number (1-indexed)")
    page_size: int = Field(default=20, ge=1, le=100, description="Items per page")
    
    @property
    def offset(self) -> int:
        """Calculate offset for database queries."""
        return (self.page - 1) * self.page_size
    
    @property
    def limit(self) -> int:
        """Get limit for database queries."""
        return self.page_size


class PaginatedResponse(BaseSchema, Generic[T]):
    """Paginated response wrapper."""
    status: ResponseStatus = ResponseStatus.SUCCESS
    data: List[T]
    meta: MetaInfo
    
    @classmethod
    def create(
        cls,
        items: List[T],
        total: int,
        page: int,
        page_size: int,
        request_id: Optional[str] = None
    ) -> "PaginatedResponse[T]":
        """Create paginated response."""
        total_pages = (total + page_size - 1) // page_size
        
        return cls(
            data=items,
            meta=MetaInfo(
                request_id=request_id or str(uuid.uuid4()),
                page=page,
                page_size=page_size,
                total_items=total,
                total_pages=total_pages
            )
        )


# =============================================================================
# COMMON REQUEST SCHEMAS
# =============================================================================

class DateRangeParams(BaseSchema):
    """Date range parameters."""
    start_date: Optional[datetime] = Field(
        None,
        description="Start date (inclusive)"
    )
    end_date: Optional[datetime] = Field(
        None,
        description="End date (inclusive)"
    )
    
    @field_validator('end_date')
    @classmethod
    def validate_date_range(cls, v, info):
        """Validate end_date is after start_date."""
        start = info.data.get('start_date')
        if start and v and v < start:
            raise ValueError('end_date must be after start_date')
        return v


class SymbolParams(BaseSchema):
    """Symbol-related parameters."""
    symbol: str = Field(
        ...,
        min_length=1,
        max_length=20,
        description="Trading symbol (e.g., THYAO)"
    )
    
    @field_validator('symbol')
    @classmethod
    def normalize_symbol(cls, v):
        """Normalize symbol to uppercase."""
        return v.upper().strip()


class MultiSymbolParams(BaseSchema):
    """Multiple symbols parameters."""
    symbols: List[str] = Field(
        ...,
        min_length=1,
        max_length=100,
        description="List of trading symbols"
    )
    
    @field_validator('symbols')
    @classmethod
    def normalize_symbols(cls, v):
        """Normalize all symbols to uppercase."""
        return [s.upper().strip() for s in v]


class IntervalParams(BaseSchema):
    """Interval/timeframe parameters."""
    interval: str = Field(
        default="1d",
        description="Data interval (1m, 5m, 15m, 1h, 4h, 1d, 1w)"
    )
    
    @field_validator('interval')
    @classmethod
    def validate_interval(cls, v):
        """Validate interval value."""
        valid = ['1m', '5m', '15m', '30m', '1h', '2h', '4h', '1d', '1w', '1M']
        if v not in valid:
            raise ValueError(f'Invalid interval. Must be one of: {valid}')
        return v


# =============================================================================
# HEALTH CHECK
# =============================================================================

class HealthStatus(str, Enum):
    """Health check status."""
    HEALTHY = "healthy"
    DEGRADED = "degraded"
    UNHEALTHY = "unhealthy"


class ComponentHealth(BaseSchema):
    """Individual component health."""
    name: str
    status: HealthStatus
    latency_ms: Optional[float] = None
    message: Optional[str] = None
    details: Optional[Dict[str, Any]] = None


class HealthCheckResponse(BaseSchema):
    """Health check response."""
    status: HealthStatus
    version: str
    uptime_seconds: float
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    components: Dict[str, ComponentHealth] = {}


# =============================================================================
# BATCH OPERATIONS
# =============================================================================

class BatchItemResult(BaseSchema, Generic[T]):
    """Result for a single item in batch operation."""
    id: str
    success: bool
    data: Optional[T] = None
    error: Optional[ErrorDetail] = None


class BatchResponse(BaseSchema, Generic[T]):
    """Batch operation response."""
    status: ResponseStatus
    total: int
    succeeded: int
    failed: int
    results: List[BatchItemResult[T]]
    meta: MetaInfo = Field(default_factory=MetaInfo)


# =============================================================================
# ALL EXPORTS
# =============================================================================

__all__ = [
    # Enums
    "ResponseStatus",
    "ErrorCode",
    "HealthStatus",
    
    # Base
    "BaseSchema",
    "TimestampMixin",
    
    # Error
    "ErrorDetail",
    "ErrorResponse",
    
    # Success
    "MetaInfo",
    "APIResponse",
    
    # Pagination
    "PaginationParams",
    "PaginatedResponse",
    
    # Common params
    "DateRangeParams",
    "SymbolParams",
    "MultiSymbolParams",
    "IntervalParams",
    
    # Health
    "ComponentHealth",
    "HealthCheckResponse",
    
    # Batch
    "BatchItemResult",
    "BatchResponse",
]
