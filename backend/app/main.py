"""
AlphaTerminal Pro - FastAPI Application
=======================================

Kurumsal seviye BIST analiz ve trading platformu ana uygulama dosyası.

Author: AlphaTerminal Team
Version: 1.0.0
"""

from __future__ import annotations

import logging
import time
from contextlib import asynccontextmanager
from typing import AsyncGenerator

import uvicorn
from fastapi import FastAPI, Request, Response, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.gzip import GZipMiddleware
from fastapi.responses import JSONResponse
from fastapi.exceptions import RequestValidationError
from starlette.exceptions import HTTPException as StarletteHTTPException

from app.config import settings, logger
from app.db import init_db, close_db, db
from app.cache import init_cache, close_cache, cache
from app.schemas import (
    ErrorResponse,
    ErrorDetail,
    HealthStatus,
    DetailedHealthStatus,
    ServiceHealth,
    APIInfo,
)

# API Router import
from app.api.v1.router import api_router


# =============================================================================
# LIFESPAN
# =============================================================================

@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    """
    Application lifespan manager.
    
    Startup ve shutdown işlemlerini yönetir.
    """
    # =========================================================================
    # STARTUP
    # =========================================================================
    logger.info("=" * 60)
    logger.info(f"Starting {settings.app_name} v{settings.app_version}")
    logger.info(f"Environment: {settings.environment.value}")
    logger.info("=" * 60)
    
    # Database başlat
    try:
        await init_db()
        logger.info("✅ Database connection established")
    except Exception as e:
        logger.error(f"❌ Database connection failed: {e}")
        raise
    
    # Redis cache başlat
    try:
        await init_cache()
        logger.info("✅ Redis cache connection established")
    except Exception as e:
        logger.warning(f"⚠️ Redis connection failed (continuing without cache): {e}")
    
    # Scheduler başlat (ileride eklenecek)
    # await start_scheduler()
    # logger.info("✅ Task scheduler started")
    
    logger.info("=" * 60)
    logger.info(f"🚀 {settings.app_name} is ready!")
    logger.info(f"📡 API: http://{settings.api_host}:{settings.api_port}")
    logger.info(f"📚 Docs: http://{settings.api_host}:{settings.api_port}/docs")
    logger.info("=" * 60)
    
    yield
    
    # =========================================================================
    # SHUTDOWN
    # =========================================================================
    logger.info("=" * 60)
    logger.info(f"Shutting down {settings.app_name}...")
    logger.info("=" * 60)
    
    # Scheduler durdur
    # await stop_scheduler()
    # logger.info("✅ Task scheduler stopped")
    
    # Redis kapat
    try:
        await close_cache()
        logger.info("✅ Redis connection closed")
    except Exception as e:
        logger.warning(f"⚠️ Redis close error: {e}")
    
    # Database kapat
    try:
        await close_db()
        logger.info("✅ Database connection closed")
    except Exception as e:
        logger.warning(f"⚠️ Database close error: {e}")
    
    logger.info(f"👋 {settings.app_name} shutdown complete")


# =============================================================================
# APPLICATION
# =============================================================================

def create_application() -> FastAPI:
    """
    FastAPI uygulamasını oluştur.
    
    Returns:
        FastAPI: Yapılandırılmış uygulama
    """
    app = FastAPI(
        title=settings.app_name,
        description="""
## AlphaTerminal Pro API

Kurumsal seviye BIST analiz ve trading platformu.

### Özellikler

* 🧠 **Smart Money Concepts (SMC)**: Order Block, FVG, CHoCH, BOS, Liquidity Sweep
* 📊 **Order Flow Analysis**: Delta, CVD, VWAP, Absorption Detection
* 📈 **Alpha Engine**: Jensen Alpha, Sharpe, Sortino, Relative Strength
* ⚠️ **Risk Management**: Kelly Criterion, Position Sizing, Portfolio Heat
* 🤖 **AI Strategy System**: 7 katmanlı otomatik strateji keşfi ve evrim
* 📱 **Telegram Integration**: Real-time sinyal bildirimleri

### Authentication

API, JWT tabanlı authentication kullanır. Access token `Authorization: Bearer <token>` 
header'ı ile gönderilmelidir.

### Rate Limiting

- API: 100 istek/dakika
- WebSocket: 10 bağlantı/kullanıcı
        """,
        version=settings.app_version,
        docs_url="/docs" if settings.api_docs_enabled else None,
        redoc_url="/redoc" if settings.api_docs_enabled else None,
        openapi_url="/openapi.json" if settings.api_docs_enabled else None,
        lifespan=lifespan,
    )
    
    # =========================================================================
    # MIDDLEWARE
    # =========================================================================
    
    # CORS
    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.cors_origins,
        allow_credentials=settings.cors_allow_credentials,
        allow_methods=settings.cors_allow_methods,
        allow_headers=settings.cors_allow_headers,
    )
    
    # GZip compression
    app.add_middleware(GZipMiddleware, minimum_size=1000)
    
    # Request timing middleware
    @app.middleware("http")
    async def add_process_time_header(request: Request, call_next):
        """Request süresini header'a ekle."""
        start_time = time.time()
        response = await call_next(request)
        process_time = time.time() - start_time
        response.headers["X-Process-Time"] = f"{process_time:.4f}"
        return response
    
    # Request logging middleware
    @app.middleware("http")
    async def log_requests(request: Request, call_next):
        """Request'leri logla."""
        # Health check endpoint'lerini loglama
        if request.url.path in ["/health", "/api/v1/health"]:
            return await call_next(request)
        
        logger.debug(f"➡️ {request.method} {request.url.path}")
        response = await call_next(request)
        logger.debug(f"⬅️ {request.method} {request.url.path} - {response.status_code}")
        return response
    
    # =========================================================================
    # EXCEPTION HANDLERS
    # =========================================================================
    
    @app.exception_handler(StarletteHTTPException)
    async def http_exception_handler(request: Request, exc: StarletteHTTPException):
        """HTTP exception handler."""
        return JSONResponse(
            status_code=exc.status_code,
            content=ErrorResponse(
                error="http_error",
                message=str(exc.detail),
            ).model_dump(),
        )
    
    @app.exception_handler(RequestValidationError)
    async def validation_exception_handler(request: Request, exc: RequestValidationError):
        """Validation exception handler."""
        details = [
            ErrorDetail(
                field=".".join(str(loc) for loc in err["loc"]),
                message=err["msg"],
                code=err["type"],
            )
            for err in exc.errors()
        ]
        
        return JSONResponse(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            content=ErrorResponse(
                error="validation_error",
                message="Validation failed",
                details=details,
            ).model_dump(),
        )
    
    @app.exception_handler(Exception)
    async def general_exception_handler(request: Request, exc: Exception):
        """Genel exception handler."""
        logger.exception(f"Unhandled exception: {exc}")
        
        return JSONResponse(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            content=ErrorResponse(
                error="internal_error",
                message="An internal error occurred" if settings.is_production else str(exc),
            ).model_dump(),
        )
    
    # =========================================================================
    # ROUTES
    # =========================================================================
    
    # API router
    app.include_router(api_router, prefix=settings.api_prefix)
    
    # Health check endpoints
    @app.get(
        "/health",
        tags=["Health"],
        response_model=HealthStatus,
        summary="Basic health check",
    )
    async def health_check():
        """Basit sağlık kontrolü."""
        return HealthStatus(
            status="healthy",
            version=settings.app_version,
            environment=settings.environment.value,
        )
    
    @app.get(
        "/health/detailed",
        tags=["Health"],
        response_model=DetailedHealthStatus,
        summary="Detailed health check",
    )
    async def detailed_health_check():
        """Detaylı sağlık kontrolü."""
        import time as t
        start = t.time()
        
        # Database check
        db_start = t.time()
        db_healthy = await db.health_check()
        db_latency = (t.time() - db_start) * 1000
        
        database = ServiceHealth(
            name="PostgreSQL",
            status="healthy" if db_healthy else "unhealthy",
            latency_ms=db_latency,
        )
        
        # Redis check
        redis_start = t.time()
        redis_healthy = await cache.health_check()
        redis_latency = (t.time() - redis_start) * 1000
        
        redis_status = ServiceHealth(
            name="Redis",
            status="healthy" if redis_healthy else "unhealthy",
            latency_ms=redis_latency,
        )
        
        # Services (placeholder)
        services = [
            ServiceHealth(name="SMC Engine", status="healthy"),
            ServiceHealth(name="OrderFlow Engine", status="healthy"),
            ServiceHealth(name="Alpha Engine", status="healthy"),
            ServiceHealth(name="AI Strategy", status="healthy"),
        ]
        
        # Overall status
        overall = "healthy"
        if not db_healthy:
            overall = "unhealthy"
        elif not redis_healthy:
            overall = "degraded"
        
        return DetailedHealthStatus(
            status=overall,
            version=settings.app_version,
            environment=settings.environment.value,
            services=services,
            database=database,
            redis=redis_status,
            uptime_seconds=t.time() - start,
        )
    
    @app.get(
        "/",
        tags=["Info"],
        response_model=APIInfo,
        summary="API information",
    )
    async def api_info():
        """API bilgilerini döndür."""
        return APIInfo(
            version=settings.app_version,
            environment=settings.environment.value,
        )
    
    return app


# =============================================================================
# APPLICATION INSTANCE
# =============================================================================

app = create_application()


# =============================================================================
# MAIN
# =============================================================================

if __name__ == "__main__":
    uvicorn.run(
        "app.main:app",
        host=settings.api_host,
        port=settings.api_port,
        reload=settings.is_development,
        workers=1 if settings.is_development else 4,
        log_level=settings.log_level.value.lower(),
    )
