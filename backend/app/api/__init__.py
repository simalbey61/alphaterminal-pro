"""
AlphaTerminal Pro - API Module
==============================

REST API modülü.

Author: AlphaTerminal Team
Version: 1.0.0
"""

from app.api.dependencies import (
    get_current_user,
    get_current_user_optional,
    get_current_admin_user,
    get_current_premium_user,
    get_stock_repository,
    get_signal_repository,
    get_strategy_repository,
    create_access_token,
    create_refresh_token,
    verify_password,
    get_password_hash,
    RateLimiter,
    rate_limiter_default,
    rate_limiter_strict,
    rate_limiter_relaxed,
    PaginationParams,
    CurrentUser,
    CurrentUserOptional,
    CurrentAdmin,
    CurrentPremium,
    DbSession,
    StockRepo,
    SignalRepo,
    StrategyRepo,
    Pagination,
)

from app.api.v1 import api_router

__all__ = [
    # Router
    "api_router",
    
    # Auth dependencies
    "get_current_user",
    "get_current_user_optional",
    "get_current_admin_user",
    "get_current_premium_user",
    "create_access_token",
    "create_refresh_token",
    "verify_password",
    "get_password_hash",
    
    # Repository dependencies
    "get_stock_repository",
    "get_signal_repository",
    "get_strategy_repository",
    
    # Rate limiting
    "RateLimiter",
    "rate_limiter_default",
    "rate_limiter_strict",
    "rate_limiter_relaxed",
    
    # Pagination
    "PaginationParams",
    
    # Type aliases
    "CurrentUser",
    "CurrentUserOptional",
    "CurrentAdmin",
    "CurrentPremium",
    "DbSession",
    "StockRepo",
    "SignalRepo",
    "StrategyRepo",
    "Pagination",
]
