"""
AlphaTerminal Pro - API v2 Request Logging Middleware
====================================================

Comprehensive request/response logging with tracing.

Author: AlphaTerminal Team
Version: 2.0.0
"""

import time
import uuid
import logging
import json
from datetime import datetime
from typing import Optional, Callable, Dict, Any, Set
from contextvars import ContextVar

from fastapi import Request, Response
from fastapi.responses import JSONResponse
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.types import ASGIApp


logger = logging.getLogger(__name__)

# Context variable for request ID
request_id_var: ContextVar[str] = ContextVar("request_id", default="")


# =============================================================================
# REQUEST CONTEXT
# =============================================================================

class RequestContext:
    """Request context for logging and tracing."""
    
    def __init__(self, request_id: str):
        self.request_id = request_id
        self.start_time = time.time()
        self.attributes: Dict[str, Any] = {}
    
    def set_attribute(self, key: str, value: Any):
        """Set context attribute."""
        self.attributes[key] = value
    
    def get_attribute(self, key: str, default: Any = None) -> Any:
        """Get context attribute."""
        return self.attributes.get(key, default)
    
    @property
    def elapsed_ms(self) -> float:
        """Get elapsed time in milliseconds."""
        return (time.time() - self.start_time) * 1000


def get_request_id() -> str:
    """Get current request ID from context."""
    return request_id_var.get()


# =============================================================================
# LOGGING MIDDLEWARE
# =============================================================================

class RequestLoggingMiddleware(BaseHTTPMiddleware):
    """
    Middleware for comprehensive request/response logging.
    
    Features:
    - Unique request ID generation/propagation
    - Request/response timing
    - Structured logging
    - Sensitive data masking
    - Error tracking
    """
    
    # Headers to mask in logs
    SENSITIVE_HEADERS: Set[str] = {
        "authorization",
        "x-api-key",
        "cookie",
        "set-cookie",
    }
    
    # Query params to mask
    SENSITIVE_PARAMS: Set[str] = {
        "api_key",
        "token",
        "password",
        "secret",
    }
    
    # Paths to skip logging
    SKIP_PATHS: Set[str] = {
        "/health",
        "/metrics",
        "/favicon.ico",
    }
    
    def __init__(
        self,
        app: ASGIApp,
        log_request_body: bool = False,
        log_response_body: bool = False,
        max_body_log_size: int = 1000,
        skip_paths: Optional[Set[str]] = None,
        logger_name: str = "api.access"
    ):
        super().__init__(app)
        
        self.log_request_body = log_request_body
        self.log_response_body = log_response_body
        self.max_body_log_size = max_body_log_size
        self.skip_paths = skip_paths or self.SKIP_PATHS
        self.access_logger = logging.getLogger(logger_name)
    
    async def dispatch(
        self,
        request: Request,
        call_next: Callable
    ) -> Response:
        """Process request with logging."""
        
        # Skip paths
        if request.url.path in self.skip_paths:
            return await call_next(request)
        
        # Generate or propagate request ID
        request_id = (
            request.headers.get("X-Request-ID") or
            request.headers.get("X-Correlation-ID") or
            str(uuid.uuid4())
        )
        
        # Set context variable
        token = request_id_var.set(request_id)
        
        # Create context
        ctx = RequestContext(request_id)
        
        # Store in request state
        request.state.request_id = request_id
        request.state.context = ctx
        
        # Log request
        await self._log_request(request, ctx)
        
        # Process request
        try:
            response = await call_next(request)
            
            # Add request ID to response headers
            response.headers["X-Request-ID"] = request_id
            
            # Log response
            self._log_response(request, response, ctx)
            
            return response
            
        except Exception as e:
            # Log error
            self._log_error(request, e, ctx)
            raise
            
        finally:
            # Reset context
            request_id_var.reset(token)
    
    async def _log_request(self, request: Request, ctx: RequestContext):
        """Log incoming request."""
        
        # Build log data
        log_data = {
            "event": "request_start",
            "request_id": ctx.request_id,
            "method": request.method,
            "path": request.url.path,
            "query": self._mask_query_params(dict(request.query_params)),
            "client_ip": self._get_client_ip(request),
            "user_agent": request.headers.get("user-agent", ""),
            "content_type": request.headers.get("content-type", ""),
            "content_length": request.headers.get("content-length", "0"),
        }
        
        # Add masked headers if needed
        if self.access_logger.isEnabledFor(logging.DEBUG):
            log_data["headers"] = self._mask_headers(dict(request.headers))
        
        # Log request body if enabled
        if self.log_request_body:
            try:
                body = await request.body()
                if body:
                    log_data["body"] = self._truncate_body(body.decode())
            except Exception:
                pass
        
        self.access_logger.info(
            f"{request.method} {request.url.path}",
            extra={"json_data": log_data}
        )
    
    def _log_response(
        self,
        request: Request,
        response: Response,
        ctx: RequestContext
    ):
        """Log response."""
        
        log_data = {
            "event": "request_complete",
            "request_id": ctx.request_id,
            "method": request.method,
            "path": request.url.path,
            "status_code": response.status_code,
            "duration_ms": round(ctx.elapsed_ms, 2),
        }
        
        # Determine log level based on status code
        if response.status_code >= 500:
            level = logging.ERROR
        elif response.status_code >= 400:
            level = logging.WARNING
        else:
            level = logging.INFO
        
        self.access_logger.log(
            level,
            f"{request.method} {request.url.path} - {response.status_code} ({ctx.elapsed_ms:.0f}ms)",
            extra={"json_data": log_data}
        )
    
    def _log_error(
        self,
        request: Request,
        error: Exception,
        ctx: RequestContext
    ):
        """Log error."""
        
        log_data = {
            "event": "request_error",
            "request_id": ctx.request_id,
            "method": request.method,
            "path": request.url.path,
            "duration_ms": round(ctx.elapsed_ms, 2),
            "error_type": type(error).__name__,
            "error_message": str(error),
        }
        
        self.access_logger.error(
            f"{request.method} {request.url.path} - ERROR: {error}",
            extra={"json_data": log_data},
            exc_info=True
        )
    
    def _get_client_ip(self, request: Request) -> str:
        """Get client IP address."""
        forwarded = request.headers.get("X-Forwarded-For")
        if forwarded:
            return forwarded.split(",")[0].strip()
        
        real_ip = request.headers.get("X-Real-IP")
        if real_ip:
            return real_ip
        
        if request.client:
            return request.client.host
        
        return "unknown"
    
    def _mask_headers(self, headers: Dict[str, str]) -> Dict[str, str]:
        """Mask sensitive headers."""
        masked = {}
        for key, value in headers.items():
            if key.lower() in self.SENSITIVE_HEADERS:
                masked[key] = "***MASKED***"
            else:
                masked[key] = value
        return masked
    
    def _mask_query_params(self, params: Dict[str, str]) -> Dict[str, str]:
        """Mask sensitive query parameters."""
        masked = {}
        for key, value in params.items():
            if key.lower() in self.SENSITIVE_PARAMS:
                masked[key] = "***MASKED***"
            else:
                masked[key] = value
        return masked
    
    def _truncate_body(self, body: str) -> str:
        """Truncate body for logging."""
        if len(body) > self.max_body_log_size:
            return body[:self.max_body_log_size] + "...[TRUNCATED]"
        return body


# =============================================================================
# ERROR HANDLING MIDDLEWARE
# =============================================================================

class ErrorHandlingMiddleware(BaseHTTPMiddleware):
    """
    Middleware for consistent error handling.
    
    Catches unhandled exceptions and returns standardized error responses.
    """
    
    def __init__(
        self,
        app: ASGIApp,
        debug: bool = False,
        include_traceback: bool = False
    ):
        super().__init__(app)
        self.debug = debug
        self.include_traceback = include_traceback
    
    async def dispatch(
        self,
        request: Request,
        call_next: Callable
    ) -> Response:
        """Process request with error handling."""
        
        try:
            return await call_next(request)
            
        except Exception as e:
            return self._handle_error(request, e)
    
    def _handle_error(self, request: Request, error: Exception) -> JSONResponse:
        """Handle unhandled exception."""
        
        from app.api.v2.schemas.base import ErrorResponse, ErrorCode
        
        # Get request ID if available
        request_id = getattr(request.state, "request_id", str(uuid.uuid4()))
        
        # Log error
        logger.exception(
            f"Unhandled error in {request.method} {request.url.path}",
            extra={"request_id": request_id}
        )
        
        # Build error response
        details = None
        if self.debug or self.include_traceback:
            import traceback
            details = {
                "exception_type": type(error).__name__,
                "traceback": traceback.format_exc() if self.include_traceback else None
            }
        
        error_response = ErrorResponse.create(
            code=ErrorCode.INTERNAL_ERROR,
            message="An internal error occurred" if not self.debug else str(error),
            details=details,
            path=request.url.path,
            request_id=request_id
        )
        
        return JSONResponse(
            status_code=500,
            content=error_response.model_dump(),
            headers={"X-Request-ID": request_id}
        )


# =============================================================================
# CORS MIDDLEWARE CONFIG
# =============================================================================

def get_cors_config(
    allowed_origins: Optional[list] = None,
    allow_credentials: bool = True
) -> Dict[str, Any]:
    """Get CORS middleware configuration."""
    
    return {
        "allow_origins": allowed_origins or ["*"],
        "allow_credentials": allow_credentials,
        "allow_methods": ["*"],
        "allow_headers": ["*"],
        "expose_headers": [
            "X-Request-ID",
            "X-RateLimit-Limit",
            "X-RateLimit-Remaining",
            "X-RateLimit-Reset",
        ]
    }


# =============================================================================
# ALL EXPORTS
# =============================================================================

__all__ = [
    "RequestContext",
    "request_id_var",
    "get_request_id",
    "RequestLoggingMiddleware",
    "ErrorHandlingMiddleware",
    "get_cors_config",
]
