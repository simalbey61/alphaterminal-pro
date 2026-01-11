"""
AlphaTerminal Pro - API v2 Health Endpoints
==========================================

Health check and monitoring endpoints.

Author: AlphaTerminal Team
Version: 2.0.0
"""

import logging
import time
import os
import platform
import psutil
from datetime import datetime
from typing import Optional, Dict, Any

from fastapi import APIRouter, Request

from app.api.v2.schemas.base import (
    APIResponse, HealthStatus, ComponentHealth, HealthCheckResponse
)


logger = logging.getLogger(__name__)

router = APIRouter(tags=["Health"])

# Track startup time
_startup_time = time.time()


# =============================================================================
# HEALTH CHECK ENDPOINTS
# =============================================================================

@router.get(
    "/health",
    response_model=HealthCheckResponse,
    summary="Health check",
    description="Basic health check endpoint."
)
async def health_check(request: Request = None):
    """
    Basic health check.
    
    Returns overall system health status.
    Used by load balancers and monitoring systems.
    """
    uptime = time.time() - _startup_time
    
    return HealthCheckResponse(
        status=HealthStatus.HEALTHY,
        version="2.0.0",
        uptime_seconds=uptime,
        timestamp=datetime.now()
    )


@router.get(
    "/health/ready",
    response_model=HealthCheckResponse,
    summary="Readiness check",
    description="Check if service is ready to accept traffic."
)
async def readiness_check(request: Request = None):
    """
    Readiness probe for Kubernetes.
    
    Checks if all required dependencies are available:
    - Data providers
    - Cache
    - Database (if applicable)
    """
    components: Dict[str, ComponentHealth] = {}
    overall_status = HealthStatus.HEALTHY
    
    # Check data providers
    try:
        from app.data_providers import DataManager
        
        manager = DataManager.get_instance()
        available = manager.get_available_providers()
        
        if available:
            components["data_providers"] = ComponentHealth(
                name="Data Providers",
                status=HealthStatus.HEALTHY,
                message=f"{len(available)} providers available",
                details={"providers": [p.value for p in available]}
            )
        else:
            components["data_providers"] = ComponentHealth(
                name="Data Providers",
                status=HealthStatus.DEGRADED,
                message="No providers available"
            )
            overall_status = HealthStatus.DEGRADED
            
    except Exception as e:
        components["data_providers"] = ComponentHealth(
            name="Data Providers",
            status=HealthStatus.UNHEALTHY,
            message=str(e)
        )
        overall_status = HealthStatus.UNHEALTHY
    
    # Check cache
    try:
        from app.data_providers import DataCacheManager
        
        cache = DataCacheManager.get_instance()
        stats = cache.get_stats()
        
        components["cache"] = ComponentHealth(
            name="Cache",
            status=HealthStatus.HEALTHY,
            message=f"L1: {stats['l1_size']} entries, L2: {stats['l2_size']} entries",
            details=stats
        )
    except Exception as e:
        components["cache"] = ComponentHealth(
            name="Cache",
            status=HealthStatus.DEGRADED,
            message=str(e)
        )
    
    uptime = time.time() - _startup_time
    
    return HealthCheckResponse(
        status=overall_status,
        version="2.0.0",
        uptime_seconds=uptime,
        timestamp=datetime.now(),
        components=components
    )


@router.get(
    "/health/live",
    response_model=HealthCheckResponse,
    summary="Liveness check",
    description="Check if service is alive."
)
async def liveness_check(request: Request = None):
    """
    Liveness probe for Kubernetes.
    
    Simply confirms the service is responding.
    If this fails, the container should be restarted.
    """
    uptime = time.time() - _startup_time
    
    return HealthCheckResponse(
        status=HealthStatus.HEALTHY,
        version="2.0.0",
        uptime_seconds=uptime,
        timestamp=datetime.now()
    )


@router.get(
    "/health/detailed",
    response_model=APIResponse[dict],
    summary="Detailed health check",
    description="Comprehensive health check with system metrics."
)
async def detailed_health(request: Request = None):
    """
    Detailed health check with system metrics.
    
    Includes:
    - Component status
    - System resources (CPU, memory, disk)
    - Runtime information
    """
    components: Dict[str, ComponentHealth] = {}
    
    # Data providers
    try:
        from app.data_providers import DataManager
        
        manager = DataManager.get_instance()
        provider_health = manager.get_provider_health()
        
        for name, health in provider_health.items():
            status = HealthStatus.HEALTHY
            if health.status.value == "unhealthy":
                status = HealthStatus.UNHEALTHY
            elif health.status.value == "degraded":
                status = HealthStatus.DEGRADED
            
            components[f"provider_{name}"] = ComponentHealth(
                name=f"Provider: {name}",
                status=status,
                latency_ms=health.latency_ms,
                message=f"Success rate: {health.success_rate:.1%}",
                details={
                    "total_requests": health.total_requests,
                    "error_count": health.error_count,
                    "consecutive_failures": health.consecutive_failures
                }
            )
    except Exception as e:
        components["data_providers"] = ComponentHealth(
            name="Data Providers",
            status=HealthStatus.UNHEALTHY,
            message=str(e)
        )
    
    # Cache
    try:
        from app.data_providers import DataCacheManager
        
        cache = DataCacheManager.get_instance()
        stats = cache.get_stats()
        
        hit_rate = stats.get("hit_rate", 0)
        status = HealthStatus.HEALTHY if hit_rate > 0.5 else HealthStatus.DEGRADED
        
        components["cache"] = ComponentHealth(
            name="Cache",
            status=status,
            message=f"Hit rate: {hit_rate:.1%}",
            details=stats
        )
    except Exception as e:
        components["cache"] = ComponentHealth(
            name="Cache",
            status=HealthStatus.DEGRADED,
            message=str(e)
        )
    
    # System metrics
    system_metrics = {}
    
    try:
        # CPU
        cpu_percent = psutil.cpu_percent(interval=0.1)
        system_metrics["cpu"] = {
            "percent": cpu_percent,
            "count": psutil.cpu_count()
        }
        
        # Memory
        memory = psutil.virtual_memory()
        system_metrics["memory"] = {
            "total_gb": round(memory.total / (1024**3), 2),
            "available_gb": round(memory.available / (1024**3), 2),
            "percent": memory.percent
        }
        
        # Disk
        disk = psutil.disk_usage('/')
        system_metrics["disk"] = {
            "total_gb": round(disk.total / (1024**3), 2),
            "free_gb": round(disk.free / (1024**3), 2),
            "percent": disk.percent
        }
        
    except Exception as e:
        logger.warning(f"Failed to get system metrics: {e}")
    
    # Runtime info
    runtime_info = {
        "python_version": platform.python_version(),
        "platform": platform.platform(),
        "processor": platform.processor(),
        "pid": os.getpid(),
        "uptime_seconds": time.time() - _startup_time
    }
    
    # Determine overall status
    unhealthy = [c for c in components.values() if c.status == HealthStatus.UNHEALTHY]
    degraded = [c for c in components.values() if c.status == HealthStatus.DEGRADED]
    
    if unhealthy:
        overall_status = HealthStatus.UNHEALTHY
    elif degraded:
        overall_status = HealthStatus.DEGRADED
    else:
        overall_status = HealthStatus.HEALTHY
    
    return APIResponse.success(data={
        "status": overall_status.value,
        "version": "2.0.0",
        "timestamp": datetime.now().isoformat(),
        "uptime_seconds": time.time() - _startup_time,
        "components": {k: v.model_dump() for k, v in components.items()},
        "system": system_metrics,
        "runtime": runtime_info
    })


# =============================================================================
# METRICS ENDPOINT
# =============================================================================

@router.get(
    "/metrics",
    summary="Prometheus metrics",
    description="Prometheus-compatible metrics endpoint."
)
async def prometheus_metrics(request: Request = None):
    """
    Prometheus metrics endpoint.
    
    Returns metrics in Prometheus text format.
    """
    metrics = []
    
    # Uptime
    uptime = time.time() - _startup_time
    metrics.append(f'alphaterminal_uptime_seconds {uptime}')
    
    # System metrics
    try:
        cpu_percent = psutil.cpu_percent(interval=0.1)
        metrics.append(f'alphaterminal_cpu_percent {cpu_percent}')
        
        memory = psutil.virtual_memory()
        metrics.append(f'alphaterminal_memory_percent {memory.percent}')
        metrics.append(f'alphaterminal_memory_available_bytes {memory.available}')
        
    except Exception:
        pass
    
    # Provider metrics
    try:
        from app.data_providers import DataManager
        
        manager = DataManager.get_instance()
        stats = manager.get_stats()
        
        metrics.append(f'alphaterminal_requests_total {stats.get("total_requests", 0)}')
        metrics.append(f'alphaterminal_cache_hits_total {stats.get("cache_hits", 0)}')
        metrics.append(f'alphaterminal_failures_total {stats.get("failures", 0)}')
        
    except Exception:
        pass
    
    # Cache metrics
    try:
        from app.data_providers import DataCacheManager
        
        cache = DataCacheManager.get_instance()
        cache_stats = cache.get_stats()
        
        metrics.append(f'alphaterminal_cache_l1_size {cache_stats.get("l1_size", 0)}')
        metrics.append(f'alphaterminal_cache_l2_size {cache_stats.get("l2_size", 0)}')
        metrics.append(f'alphaterminal_cache_hit_rate {cache_stats.get("hit_rate", 0)}')
        
    except Exception:
        pass
    
    from fastapi.responses import PlainTextResponse
    return PlainTextResponse(
        content="\n".join(metrics),
        media_type="text/plain"
    )


# =============================================================================
# INFO ENDPOINT
# =============================================================================

@router.get(
    "/info",
    response_model=APIResponse[dict],
    summary="API information",
    description="Get API version and capabilities."
)
async def api_info(request: Request = None):
    """
    Get API information.
    
    Returns version, available endpoints, and capabilities.
    """
    info = {
        "name": "AlphaTerminal Pro API",
        "version": "2.0.0",
        "description": "Enterprise-grade trading system API",
        "documentation": "/docs",
        "openapi": "/openapi.json",
        "features": [
            "Market data (OHLCV, quotes, symbols)",
            "Backtesting engine",
            "Multiple data providers",
            "Rate limiting",
            "Request tracing",
            "WebSocket support"
        ],
        "data_providers": ["TradingView", "Yahoo Finance"],
        "supported_markets": ["BIST", "NYSE", "NASDAQ"],
        "supported_intervals": ["1m", "5m", "15m", "30m", "1h", "4h", "1d", "1w"],
        "rate_limits": {
            "requests_per_minute": 60,
            "requests_per_hour": 1000,
            "requests_per_day": 10000
        }
    }
    
    return APIResponse.success(data=info)


# =============================================================================
# ALL EXPORTS
# =============================================================================

__all__ = ["router"]
