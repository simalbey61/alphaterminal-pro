"""
AlphaTerminal Pro - Pydantic Schemas
====================================

Tüm Pydantic schema'ların merkezi export noktası.

Author: AlphaTerminal Team
Version: 1.0.0
"""

# Stock schemas
from app.schemas.stock import (
    StockBase,
    StockCreate,
    StockUpdate,
    StockPriceUpdate,
    StockResponse,
    StockListResponse,
    StockSummary,
    StockMover,
    SectorSummary,
    SectorDetailResponse,
    MarketStatistics,
    MarketOverview,
)

# Signal schemas
from app.schemas.signal import (
    SignalBase,
    SignalCreate,
    SignalUpdate,
    SignalClose,
    SignalResponse,
    SignalListResponse,
    SignalSummary,
    SignalFilter,
    SignalPerformanceStats,
    TierPerformance,
    SignalDistribution,
    SymbolPerformance,
    SignalTelegramMessage,
    SignalAlert,
)

# Strategy schemas
from app.schemas.strategy import (
    StrategyCondition,
    StrategyBase,
    StrategyCreate,
    StrategyUpdate,
    StrategyResponse,
    StrategyListResponse,
    StrategySummary,
    BacktestRequest,
    BacktestTrade,
    BacktestResult,
    EvolutionLogResponse,
    StrategyEvolutionHistory,
    StrategyStatistics,
    DiscoveryMethodStats,
    GenerationStats,
    StrategyApprovalCriteria,
    StrategyApprovalResult,
)

# Common schemas
from app.schemas.common import (
    PaginationParams,
    PaginatedResponse,
    ErrorDetail,
    ErrorResponse,
    ValidationErrorResponse,
    SuccessResponse,
    DeleteResponse,
    BulkOperationResponse,
    HealthStatus,
    ServiceHealth,
    DetailedHealthStatus,
    APIInfo,
    WebSocketMessage,
    WebSocketSubscription,
    NotificationCreate,
    NotificationResponse,
    DateRangeFilter,
    NumericRangeFilter,
    BatchDeleteRequest,
    BatchUpdateRequest,
)

__all__ = [
    # Stock
    "StockBase",
    "StockCreate",
    "StockUpdate",
    "StockPriceUpdate",
    "StockResponse",
    "StockListResponse",
    "StockSummary",
    "StockMover",
    "SectorSummary",
    "SectorDetailResponse",
    "MarketStatistics",
    "MarketOverview",
    
    # Signal
    "SignalBase",
    "SignalCreate",
    "SignalUpdate",
    "SignalClose",
    "SignalResponse",
    "SignalListResponse",
    "SignalSummary",
    "SignalFilter",
    "SignalPerformanceStats",
    "TierPerformance",
    "SignalDistribution",
    "SymbolPerformance",
    "SignalTelegramMessage",
    "SignalAlert",
    
    # Strategy
    "StrategyCondition",
    "StrategyBase",
    "StrategyCreate",
    "StrategyUpdate",
    "StrategyResponse",
    "StrategyListResponse",
    "StrategySummary",
    "BacktestRequest",
    "BacktestTrade",
    "BacktestResult",
    "EvolutionLogResponse",
    "StrategyEvolutionHistory",
    "StrategyStatistics",
    "DiscoveryMethodStats",
    "GenerationStats",
    "StrategyApprovalCriteria",
    "StrategyApprovalResult",
    
    # Common
    "PaginationParams",
    "PaginatedResponse",
    "ErrorDetail",
    "ErrorResponse",
    "ValidationErrorResponse",
    "SuccessResponse",
    "DeleteResponse",
    "BulkOperationResponse",
    "HealthStatus",
    "ServiceHealth",
    "DetailedHealthStatus",
    "APIInfo",
    "WebSocketMessage",
    "WebSocketSubscription",
    "NotificationCreate",
    "NotificationResponse",
    "DateRangeFilter",
    "NumericRangeFilter",
    "BatchDeleteRequest",
    "BatchUpdateRequest",
]
