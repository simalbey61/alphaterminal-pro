"""
AlphaTerminal Pro - API v2
==========================

Enterprise-grade REST API for trading system.

Features:
- Standardized request/response schemas
- Rate limiting
- Request tracing and logging
- Comprehensive error handling
- OpenAPI documentation

Author: AlphaTerminal Team
Version: 2.0.0
"""

from app.api.v2.router import router

# Schemas
from app.api.v2.schemas.base import (
    APIResponse,
    ErrorResponse,
    ErrorCode,
    ResponseStatus,
    MetaInfo,
    PaginationParams,
    PaginatedResponse,
    HealthStatus,
    HealthCheckResponse,
)

# Middleware
from app.api.v2.middleware import (
    RateLimitConfig,
    RateLimiterMiddleware,
    RequestLoggingMiddleware,
    ErrorHandlingMiddleware,
    get_cors_config,
)


__all__ = [
    # Router
    "router",
    
    # Schemas
    "APIResponse",
    "ErrorResponse",
    "ErrorCode",
    "ResponseStatus",
    "MetaInfo",
    "PaginationParams",
    "PaginatedResponse",
    "HealthStatus",
    "HealthCheckResponse",
    
    # Middleware
    "RateLimitConfig",
    "RateLimiterMiddleware",
    "RequestLoggingMiddleware",
    "ErrorHandlingMiddleware",
    "get_cors_config",
]

__version__ = "2.0.0"
