"""
AlphaTerminal Pro - API v2 Rate Limiter Middleware
=================================================

Token bucket rate limiter with Redis support.

Author: AlphaTerminal Team
Version: 2.0.0
"""

import time
import asyncio
import logging
from datetime import datetime, timedelta
from typing import Optional, Dict, Any, Callable, Tuple
from dataclasses import dataclass, field
from collections import defaultdict
import hashlib

from fastapi import Request, Response, HTTPException
from fastapi.responses import JSONResponse
from starlette.middleware.base import BaseHTTPMiddleware

from app.api.v2.schemas.base import ErrorResponse, ErrorCode


logger = logging.getLogger(__name__)


# =============================================================================
# RATE LIMIT CONFIGURATION
# =============================================================================

@dataclass
class RateLimitConfig:
    """Rate limit configuration."""
    requests_per_minute: int = 60
    requests_per_hour: int = 1000
    requests_per_day: int = 10000
    burst_size: int = 10  # Allow burst above limit
    
    # Per-endpoint overrides
    endpoint_limits: Dict[str, int] = field(default_factory=dict)
    
    # Whitelist/blacklist
    whitelisted_ips: set = field(default_factory=set)
    whitelisted_api_keys: set = field(default_factory=set)
    blacklisted_ips: set = field(default_factory=set)


@dataclass
class RateLimitState:
    """State for a single rate limit bucket."""
    tokens: float
    last_update: float
    request_count_minute: int = 0
    request_count_hour: int = 0
    request_count_day: int = 0
    minute_reset: float = 0
    hour_reset: float = 0
    day_reset: float = 0


# =============================================================================
# IN-MEMORY RATE LIMITER
# =============================================================================

class InMemoryRateLimiter:
    """
    In-memory token bucket rate limiter.
    
    Suitable for single-instance deployments.
    For multi-instance, use Redis-based limiter.
    """
    
    def __init__(self, config: RateLimitConfig):
        self.config = config
        self._buckets: Dict[str, RateLimitState] = {}
        self._lock = asyncio.Lock()
    
    def _get_bucket_key(
        self,
        identifier: str,
        endpoint: Optional[str] = None
    ) -> str:
        """Generate bucket key."""
        if endpoint:
            return f"{identifier}:{endpoint}"
        return identifier
    
    def _get_limit_for_endpoint(self, endpoint: str) -> int:
        """Get rate limit for specific endpoint."""
        return self.config.endpoint_limits.get(
            endpoint,
            self.config.requests_per_minute
        )
    
    async def check_rate_limit(
        self,
        identifier: str,
        endpoint: Optional[str] = None
    ) -> Tuple[bool, Dict[str, Any]]:
        """
        Check if request should be allowed.
        
        Returns:
            Tuple of (allowed, info_dict)
        """
        async with self._lock:
            key = self._get_bucket_key(identifier, endpoint)
            now = time.time()
            
            # Get or create bucket
            if key not in self._buckets:
                self._buckets[key] = RateLimitState(
                    tokens=self.config.burst_size,
                    last_update=now,
                    minute_reset=now + 60,
                    hour_reset=now + 3600,
                    day_reset=now + 86400
                )
            
            bucket = self._buckets[key]
            
            # Reset counters if needed
            if now >= bucket.minute_reset:
                bucket.request_count_minute = 0
                bucket.minute_reset = now + 60
            
            if now >= bucket.hour_reset:
                bucket.request_count_hour = 0
                bucket.hour_reset = now + 3600
            
            if now >= bucket.day_reset:
                bucket.request_count_day = 0
                bucket.day_reset = now + 86400
            
            # Refill tokens (token bucket algorithm)
            time_passed = now - bucket.last_update
            limit = self._get_limit_for_endpoint(endpoint or "default")
            refill_rate = limit / 60  # tokens per second
            bucket.tokens = min(
                self.config.burst_size,
                bucket.tokens + time_passed * refill_rate
            )
            bucket.last_update = now
            
            # Check limits
            info = {
                "limit_minute": limit,
                "remaining_minute": max(0, limit - bucket.request_count_minute),
                "limit_hour": self.config.requests_per_hour,
                "remaining_hour": max(0, self.config.requests_per_hour - bucket.request_count_hour),
                "limit_day": self.config.requests_per_day,
                "remaining_day": max(0, self.config.requests_per_day - bucket.request_count_day),
                "reset_minute": int(bucket.minute_reset - now),
                "reset_hour": int(bucket.hour_reset - now),
            }
            
            # Check if rate limited
            if bucket.request_count_minute >= limit:
                info["retry_after"] = int(bucket.minute_reset - now)
                return False, info
            
            if bucket.request_count_hour >= self.config.requests_per_hour:
                info["retry_after"] = int(bucket.hour_reset - now)
                return False, info
            
            if bucket.request_count_day >= self.config.requests_per_day:
                info["retry_after"] = int(bucket.day_reset - now)
                return False, info
            
            # Check token bucket (for burst control)
            if bucket.tokens < 1:
                info["retry_after"] = int(1 / refill_rate)
                return False, info
            
            # Allow request
            bucket.tokens -= 1
            bucket.request_count_minute += 1
            bucket.request_count_hour += 1
            bucket.request_count_day += 1
            
            return True, info
    
    async def get_stats(self, identifier: str) -> Dict[str, Any]:
        """Get rate limit stats for identifier."""
        async with self._lock:
            stats = {}
            for key, bucket in self._buckets.items():
                if key.startswith(identifier):
                    stats[key] = {
                        "tokens": bucket.tokens,
                        "minute_count": bucket.request_count_minute,
                        "hour_count": bucket.request_count_hour,
                        "day_count": bucket.request_count_day,
                    }
            return stats
    
    async def reset(self, identifier: str):
        """Reset rate limit for identifier."""
        async with self._lock:
            keys_to_delete = [
                k for k in self._buckets.keys()
                if k.startswith(identifier)
            ]
            for key in keys_to_delete:
                del self._buckets[key]


# =============================================================================
# RATE LIMITER MIDDLEWARE
# =============================================================================

class RateLimiterMiddleware(BaseHTTPMiddleware):
    """
    FastAPI middleware for rate limiting.
    
    Usage:
        app.add_middleware(
            RateLimiterMiddleware,
            config=RateLimitConfig(requests_per_minute=60)
        )
    """
    
    def __init__(
        self,
        app,
        config: Optional[RateLimitConfig] = None,
        limiter: Optional[InMemoryRateLimiter] = None,
        key_func: Optional[Callable[[Request], str]] = None,
        exclude_paths: Optional[set] = None
    ):
        super().__init__(app)
        
        self.config = config or RateLimitConfig()
        self.limiter = limiter or InMemoryRateLimiter(self.config)
        self.key_func = key_func or self._default_key_func
        self.exclude_paths = exclude_paths or {
            "/health",
            "/docs",
            "/redoc",
            "/openapi.json",
            "/favicon.ico"
        }
    
    def _default_key_func(self, request: Request) -> str:
        """Default key function - use client IP."""
        # Try to get real IP from headers (for proxied requests)
        forwarded = request.headers.get("X-Forwarded-For")
        if forwarded:
            return forwarded.split(",")[0].strip()
        
        real_ip = request.headers.get("X-Real-IP")
        if real_ip:
            return real_ip
        
        # Fall back to direct client IP
        if request.client:
            return request.client.host
        
        return "unknown"
    
    def _get_api_key(self, request: Request) -> Optional[str]:
        """Extract API key from request."""
        # Check header
        api_key = request.headers.get("X-API-Key")
        if api_key:
            return api_key
        
        # Check query param
        api_key = request.query_params.get("api_key")
        if api_key:
            return api_key
        
        return None
    
    async def dispatch(
        self,
        request: Request,
        call_next: Callable
    ) -> Response:
        """Process request with rate limiting."""
        
        # Skip excluded paths
        path = request.url.path
        if any(path.startswith(p) for p in self.exclude_paths):
            return await call_next(request)
        
        # Get identifier
        identifier = self.key_func(request)
        api_key = self._get_api_key(request)
        
        # Check whitelist
        if identifier in self.config.whitelisted_ips:
            return await call_next(request)
        
        if api_key and api_key in self.config.whitelisted_api_keys:
            return await call_next(request)
        
        # Check blacklist
        if identifier in self.config.blacklisted_ips:
            return self._rate_limit_response(
                {"retry_after": 3600},
                "IP address blocked"
            )
        
        # Use API key as identifier if available (higher limits for authenticated)
        if api_key:
            identifier = f"apikey:{hashlib.md5(api_key.encode()).hexdigest()[:16]}"
        
        # Check rate limit
        allowed, info = await self.limiter.check_rate_limit(
            identifier,
            endpoint=path
        )
        
        if not allowed:
            return self._rate_limit_response(info)
        
        # Process request
        response = await call_next(request)
        
        # Add rate limit headers
        response.headers["X-RateLimit-Limit"] = str(info["limit_minute"])
        response.headers["X-RateLimit-Remaining"] = str(info["remaining_minute"])
        response.headers["X-RateLimit-Reset"] = str(info["reset_minute"])
        
        return response
    
    def _rate_limit_response(
        self,
        info: Dict[str, Any],
        message: str = "Rate limit exceeded"
    ) -> JSONResponse:
        """Create rate limit error response."""
        retry_after = info.get("retry_after", 60)
        
        error_response = ErrorResponse.create(
            code=ErrorCode.RATE_LIMITED,
            message=message,
            details={
                "retry_after_seconds": retry_after,
                "limit": info.get("limit_minute"),
                "remaining": info.get("remaining_minute", 0)
            }
        )
        
        return JSONResponse(
            status_code=429,
            content=error_response.model_dump(),
            headers={
                "Retry-After": str(retry_after),
                "X-RateLimit-Limit": str(info.get("limit_minute", 60)),
                "X-RateLimit-Remaining": "0",
                "X-RateLimit-Reset": str(info.get("reset_minute", 60))
            }
        )


# =============================================================================
# RATE LIMIT DECORATOR
# =============================================================================

def rate_limit(
    requests_per_minute: int = 60,
    key_func: Optional[Callable[[Request], str]] = None
):
    """
    Decorator for endpoint-level rate limiting.
    
    Usage:
        @router.get("/data")
        @rate_limit(requests_per_minute=30)
        async def get_data(request: Request):
            ...
    """
    def decorator(func):
        # Store rate limit config on function
        func._rate_limit = requests_per_minute
        func._rate_limit_key_func = key_func
        return func
    return decorator


# =============================================================================
# ALL EXPORTS
# =============================================================================

__all__ = [
    "RateLimitConfig",
    "RateLimitState",
    "InMemoryRateLimiter",
    "RateLimiterMiddleware",
    "rate_limit",
]
