"""
AlphaTerminal Pro - API v2 Router
=================================

Main router configuration for API v2.

Author: AlphaTerminal Team
Version: 2.0.0
"""

from fastapi import APIRouter

from app.api.v2.endpoints.health import router as health_router
from app.api.v2.endpoints.market import router as market_router
from app.api.v2.endpoints.backtest import router as backtest_router


# Create main v2 router
router = APIRouter(prefix="/api/v2")

# Include sub-routers
router.include_router(health_router)
router.include_router(market_router)
router.include_router(backtest_router)


__all__ = ["router"]
