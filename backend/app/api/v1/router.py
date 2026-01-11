"""
AlphaTerminal Pro - API v1 Router
=================================

Tüm API v1 endpoint'lerinin merkezi router'ı.

Author: AlphaTerminal Team
Version: 1.0.0
"""

from __future__ import annotations

from fastapi import APIRouter

from app.api.v1.endpoints import (
    health,
    auth,
    stocks,
    signals,
    strategies,
    analysis,
    backtest,
    portfolio,
    market,
    users,
    ai_strategy,
)

# Ana API router
api_router = APIRouter()

# Health endpoints
api_router.include_router(
    health.router,
    prefix="/health",
    tags=["Health"]
)

# Authentication endpoints
api_router.include_router(
    auth.router,
    prefix="/auth",
    tags=["Authentication"]
)

# Stock endpoints
api_router.include_router(
    stocks.router,
    prefix="/stocks",
    tags=["Stocks"]
)

# Signal endpoints
api_router.include_router(
    signals.router,
    prefix="/signals",
    tags=["Signals"]
)

# Strategy endpoints
api_router.include_router(
    strategies.router,
    prefix="/strategies",
    tags=["AI Strategies"]
)

# Analysis endpoints
api_router.include_router(
    analysis.router,
    prefix="/analysis",
    tags=["Analysis"]
)

# Backtest endpoints
api_router.include_router(
    backtest.router,
    prefix="/backtest",
    tags=["Backtest"]
)

# Portfolio endpoints
api_router.include_router(
    portfolio.router,
    prefix="/portfolio",
    tags=["Portfolio"]
)

# Market endpoints
api_router.include_router(
    market.router,
    prefix="/market",
    tags=["Market"]
)

# User endpoints
api_router.include_router(
    users.router,
    prefix="/users",
    tags=["Users"]
)

# AI Strategy endpoints (7-Layer AI System)
api_router.include_router(
    ai_strategy.router,
    tags=["AI Strategy"]
)
