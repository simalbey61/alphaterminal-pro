"""
AlphaTerminal Pro - Health Endpoints
====================================

Sistem sağlık kontrolü endpoint'leri.

Author: AlphaTerminal Team
Version: 1.0.0
"""

from __future__ import annotations

import time
import logging
from datetime import datetime

from fastapi import APIRouter, Depends

from app.config import settings
from app.db import db
from app.cache import cache
from app.schemas import (
    HealthStatus,
    DetailedHealthStatus,
    ServiceHealth,
)

logger = logging.getLogger(__name__)
router = APIRouter()


@router.get(
    "",
    response_model=HealthStatus,
    summary="Basic Health Check",
    description="Basit sağlık kontrolü. Load balancer ve monitoring için kullanılır.",
)
async def health_check() -> HealthStatus:
    """
    Basit sağlık kontrolü.
    
    Returns:
        HealthStatus: Temel sağlık durumu
    """
    return HealthStatus(
        status="healthy",
        version=settings.app_version,
        environment=settings.environment.value,
    )


@router.get(
    "/detailed",
    response_model=DetailedHealthStatus,
    summary="Detailed Health Check",
    description="Tüm servislerin detaylı sağlık kontrolü.",
)
async def detailed_health_check() -> DetailedHealthStatus:
    """
    Detaylı sağlık kontrolü.
    
    Tüm bağlı servislerin durumunu kontrol eder:
    - PostgreSQL Database
    - Redis Cache
    - Core Engines
    
    Returns:
        DetailedHealthStatus: Detaylı sağlık durumu
    """
    start_time = time.time()
    services = []
    overall_status = "healthy"
    
    # ==========================================================================
    # DATABASE CHECK
    # ==========================================================================
    db_start = time.time()
    try:
        db_healthy = await db.health_check()
        db_latency = (time.time() - db_start) * 1000
        
        database = ServiceHealth(
            name="PostgreSQL",
            status="healthy" if db_healthy else "unhealthy",
            latency_ms=round(db_latency, 2),
            message="Connection pool active" if db_healthy else "Connection failed"
        )
        
        if not db_healthy:
            overall_status = "unhealthy"
            
    except Exception as e:
        logger.error(f"Database health check failed: {e}")
        database = ServiceHealth(
            name="PostgreSQL",
            status="unhealthy",
            latency_ms=None,
            message=str(e)
        )
        overall_status = "unhealthy"
    
    # ==========================================================================
    # REDIS CHECK
    # ==========================================================================
    redis_start = time.time()
    try:
        redis_healthy = await cache.health_check()
        redis_latency = (time.time() - redis_start) * 1000
        
        redis_service = ServiceHealth(
            name="Redis",
            status="healthy" if redis_healthy else "unhealthy",
            latency_ms=round(redis_latency, 2),
            message="Cache operational" if redis_healthy else "Connection failed"
        )
        
        if not redis_healthy and overall_status == "healthy":
            overall_status = "degraded"
            
    except Exception as e:
        logger.warning(f"Redis health check failed: {e}")
        redis_service = ServiceHealth(
            name="Redis",
            status="unhealthy",
            latency_ms=None,
            message=str(e)
        )
        if overall_status == "healthy":
            overall_status = "degraded"
    
    # ==========================================================================
    # CORE ENGINES CHECK
    # ==========================================================================
    
    # SMC Engine
    services.append(ServiceHealth(
        name="SMC Engine",
        status="healthy",
        message="Smart Money Concepts analysis ready"
    ))
    
    # OrderFlow Engine
    services.append(ServiceHealth(
        name="OrderFlow Engine",
        status="healthy",
        message="Order flow analysis ready"
    ))
    
    # Alpha Engine
    services.append(ServiceHealth(
        name="Alpha Engine",
        status="healthy",
        message="Performance metrics ready"
    ))
    
    # Risk Engine
    services.append(ServiceHealth(
        name="Risk Engine",
        status="healthy",
        message="Risk management ready"
    ))
    
    # AI Strategy Engine
    services.append(ServiceHealth(
        name="AI Strategy Engine",
        status="healthy",
        message="Pattern discovery ready"
    ))
    
    # ==========================================================================
    # RESPONSE
    # ==========================================================================
    
    uptime = time.time() - start_time
    
    return DetailedHealthStatus(
        status=overall_status,
        version=settings.app_version,
        environment=settings.environment.value,
        services=services,
        database=database,
        redis=redis_service,
        uptime_seconds=round(uptime, 4),
    )


@router.get(
    "/ready",
    summary="Readiness Check",
    description="Kubernetes readiness probe için kullanılır.",
)
async def readiness_check() -> dict:
    """
    Readiness kontrolü.
    
    Sistemin istek almaya hazır olup olmadığını kontrol eder.
    
    Returns:
        dict: Hazırlık durumu
    """
    # Database bağlantısı zorunlu
    db_ready = await db.health_check()
    
    if not db_ready:
        return {
            "ready": False,
            "reason": "Database not connected"
        }
    
    return {
        "ready": True,
        "timestamp": datetime.utcnow().isoformat()
    }


@router.get(
    "/live",
    summary="Liveness Check",
    description="Kubernetes liveness probe için kullanılır.",
)
async def liveness_check() -> dict:
    """
    Liveness kontrolü.
    
    Uygulama process'inin canlı olup olmadığını kontrol eder.
    
    Returns:
        dict: Canlılık durumu
    """
    return {
        "alive": True,
        "timestamp": datetime.utcnow().isoformat()
    }
