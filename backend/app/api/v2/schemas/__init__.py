"""
AlphaTerminal Pro - API v2 Schemas
==================================

Pydantic schemas for API request/response validation.
"""

from app.api.v2.schemas.base import (
    # Enums
    ResponseStatus,
    ErrorCode,
    HealthStatus,
    
    # Base
    BaseSchema,
    TimestampMixin,
    
    # Error
    ErrorDetail,
    ErrorResponse,
    
    # Success
    MetaInfo,
    APIResponse,
    
    # Pagination
    PaginationParams,
    PaginatedResponse,
    
    # Common params
    DateRangeParams,
    SymbolParams,
    MultiSymbolParams,
    IntervalParams,
    
    # Health
    ComponentHealth,
    HealthCheckResponse,
    
    # Batch
    BatchItemResult,
    BatchResponse,
)

from app.api.v2.schemas.market import (
    OHLCVBar,
    OHLCVResponse,
    MarketDataRequest,
    BatchMarketDataRequest,
    SymbolInfo,
    SymbolSearchRequest,
    SymbolListRequest,
    Quote,
    QuoteRequest,
    BatchQuoteRequest,
    MarketIndexData,
    MarketOverview,
    MarketOverviewRequest,
    FundamentalData,
    IndicatorRequest,
    IndicatorValue,
    IndicatorResponse,
    MultiIndicatorRequest,
)

from app.api.v2.schemas.backtest import (
    StrategyType,
    BacktestStatus,
    BacktestConfig,
    StrategyConfig,
    BacktestRequest,
    MultiSymbolBacktestRequest,
    StrategyOptimizationRequest,
    TradeResult,
    PerformanceMetrics,
    DrawdownInfo,
    MonthlyReturn,
    EquityCurvePoint,
    BacktestResponse,
    MultiSymbolBacktestResponse,
    OptimizationResult,
    StrategyOptimizationResponse,
    BacktestJobStatus,
)


__all__ = [
    # Base
    "ResponseStatus",
    "ErrorCode",
    "HealthStatus",
    "BaseSchema",
    "TimestampMixin",
    "ErrorDetail",
    "ErrorResponse",
    "MetaInfo",
    "APIResponse",
    "PaginationParams",
    "PaginatedResponse",
    "DateRangeParams",
    "SymbolParams",
    "MultiSymbolParams",
    "IntervalParams",
    "ComponentHealth",
    "HealthCheckResponse",
    "BatchItemResult",
    "BatchResponse",
    
    # Market
    "OHLCVBar",
    "OHLCVResponse",
    "MarketDataRequest",
    "BatchMarketDataRequest",
    "SymbolInfo",
    "SymbolSearchRequest",
    "SymbolListRequest",
    "Quote",
    "QuoteRequest",
    "BatchQuoteRequest",
    "MarketIndexData",
    "MarketOverview",
    "MarketOverviewRequest",
    "FundamentalData",
    "IndicatorRequest",
    "IndicatorValue",
    "IndicatorResponse",
    "MultiIndicatorRequest",
    
    # Backtest
    "StrategyType",
    "BacktestStatus",
    "BacktestConfig",
    "StrategyConfig",
    "BacktestRequest",
    "MultiSymbolBacktestRequest",
    "StrategyOptimizationRequest",
    "TradeResult",
    "PerformanceMetrics",
    "DrawdownInfo",
    "MonthlyReturn",
    "EquityCurvePoint",
    "BacktestResponse",
    "MultiSymbolBacktestResponse",
    "OptimizationResult",
    "StrategyOptimizationResponse",
    "BacktestJobStatus",
]
