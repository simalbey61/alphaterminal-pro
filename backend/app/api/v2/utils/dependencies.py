"""
AlphaTerminal Pro - API v2 Dependencies
=======================================

FastAPI dependency injection utilities.

Author: AlphaTerminal Team
Version: 2.0.0
"""

import logging
from typing import Optional, Generator, AsyncGenerator
from functools import lru_cache

from fastapi import Depends, Request, HTTPException
from fastapi.security import APIKeyHeader

from app.api.v2.schemas.base import ErrorCode, PaginationParams
from app.api.v2.middleware.logging import RequestContext


logger = logging.getLogger(__name__)


# =============================================================================
# API KEY AUTHENTICATION
# =============================================================================

api_key_header = APIKeyHeader(name="X-API-Key", auto_error=False)


async def get_api_key(
    api_key: Optional[str] = Depends(api_key_header)
) -> Optional[str]:
    """
    Extract API key from request header.
    
    Returns None if no key provided (for public endpoints).
    """
    return api_key


async def require_api_key(
    api_key: Optional[str] = Depends(get_api_key)
) -> str:
    """
    Require valid API key for protected endpoints.
    
    Raises HTTPException if no key provided.
    """
    if not api_key:
        raise HTTPException(
            status_code=401,
            detail={
                "code": ErrorCode.UNAUTHORIZED.value,
                "message": "API key required"
            }
        )
    
    # In production, validate against database
    # For now, accept any non-empty key
    return api_key


# =============================================================================
# REQUEST CONTEXT
# =============================================================================

async def get_request_context(request: Request) -> Optional[RequestContext]:
    """Get request context from request state."""
    return getattr(request.state, "context", None)


async def get_request_id(request: Request) -> str:
    """Get request ID from request state."""
    return getattr(request.state, "request_id", "unknown")


# =============================================================================
# DATA MANAGER
# =============================================================================

@lru_cache()
def get_data_manager_instance():
    """Get cached DataManager instance."""
    from app.data_providers import DataManager
    return DataManager.get_instance()


async def get_data_manager():
    """
    Dependency to get DataManager instance.
    
    Usage:
        @router.get("/data")
        async def get_data(manager = Depends(get_data_manager)):
            ...
    """
    return get_data_manager_instance()


# =============================================================================
# PAGINATION
# =============================================================================

async def get_pagination(
    page: int = 1,
    page_size: int = 20
) -> PaginationParams:
    """
    Parse pagination parameters from query.
    
    Usage:
        @router.get("/items")
        async def list_items(pagination: PaginationParams = Depends(get_pagination)):
            ...
    """
    # Enforce limits
    page = max(1, page)
    page_size = max(1, min(100, page_size))
    
    return PaginationParams(page=page, page_size=page_size)


# =============================================================================
# RATE LIMIT INFO
# =============================================================================

async def get_rate_limit_info(request: Request) -> dict:
    """Get rate limit info from response headers."""
    # This would be populated by the rate limiter middleware
    return {
        "limit": request.headers.get("X-RateLimit-Limit"),
        "remaining": request.headers.get("X-RateLimit-Remaining"),
        "reset": request.headers.get("X-RateLimit-Reset")
    }


# =============================================================================
# VALIDATION HELPERS
# =============================================================================

def validate_symbol(symbol: str) -> str:
    """Validate and normalize symbol."""
    if not symbol:
        raise HTTPException(
            status_code=400,
            detail={
                "code": ErrorCode.VALIDATION_ERROR.value,
                "message": "Symbol is required"
            }
        )
    
    symbol = symbol.upper().strip()
    
    if len(symbol) < 1 or len(symbol) > 20:
        raise HTTPException(
            status_code=400,
            detail={
                "code": ErrorCode.VALIDATION_ERROR.value,
                "message": "Symbol must be 1-20 characters"
            }
        )
    
    return symbol


def validate_interval(interval: str) -> str:
    """Validate interval string."""
    valid_intervals = ['1m', '5m', '15m', '30m', '1h', '2h', '4h', '1d', '1w', '1M']
    
    if interval not in valid_intervals:
        raise HTTPException(
            status_code=400,
            detail={
                "code": ErrorCode.VALIDATION_ERROR.value,
                "message": f"Invalid interval. Must be one of: {valid_intervals}"
            }
        )
    
    return interval


# =============================================================================
# ALL EXPORTS
# =============================================================================

__all__ = [
    # Authentication
    "api_key_header",
    "get_api_key",
    "require_api_key",
    
    # Context
    "get_request_context",
    "get_request_id",
    
    # Data
    "get_data_manager",
    "get_data_manager_instance",
    
    # Pagination
    "get_pagination",
    
    # Rate limit
    "get_rate_limit_info",
    
    # Validation
    "validate_symbol",
    "validate_interval",
]
