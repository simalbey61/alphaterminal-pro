"""
AlphaTerminal Pro - API v2 Backtest Schemas
==========================================

Schemas for backtest endpoints.

Author: AlphaTerminal Team
Version: 2.0.0
"""

from datetime import datetime
from typing import Optional, Dict, Any, List
from enum import Enum
from pydantic import Field, field_validator

from app.api.v2.schemas.base import BaseSchema, DateRangeParams


# =============================================================================
# ENUMS
# =============================================================================

class StrategyType(str, Enum):
    """Available strategy types."""
    SMA_CROSSOVER = "sma_crossover"
    DUAL_SMA_CROSSOVER = "dual_sma_crossover"
    RSI_MEAN_REVERSION = "rsi_mean_reversion"
    RSI_EXTREMES = "rsi_extremes"
    BOLLINGER_BANDS = "bollinger_bands"
    MACD_CROSSOVER = "macd_crossover"
    CUSTOM = "custom"


class BacktestStatus(str, Enum):
    """Backtest job status."""
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


# =============================================================================
# CONFIGURATION
# =============================================================================

class BacktestConfig(BaseSchema):
    """Backtest configuration."""
    initial_capital: float = Field(
        default=100_000,
        ge=1000,
        description="Starting capital"
    )
    commission_rate: float = Field(
        default=0.001,
        ge=0,
        le=0.1,
        description="Commission rate (0.001 = 0.1%)"
    )
    slippage_rate: float = Field(
        default=0.0005,
        ge=0,
        le=0.1,
        description="Slippage rate"
    )
    max_position_size: float = Field(
        default=0.25,
        ge=0.01,
        le=1.0,
        description="Maximum position size as fraction of capital"
    )
    max_positions: int = Field(
        default=5,
        ge=1,
        le=100,
        description="Maximum concurrent positions"
    )
    risk_per_trade: float = Field(
        default=0.02,
        ge=0.001,
        le=0.5,
        description="Risk per trade as fraction of capital"
    )
    allow_shorting: bool = Field(
        default=False,
        description="Allow short positions"
    )


class StrategyConfig(BaseSchema):
    """Strategy configuration."""
    strategy_type: StrategyType
    params: Dict[str, Any] = Field(
        default={},
        description="Strategy-specific parameters"
    )
    
    @field_validator('params')
    @classmethod
    def validate_params(cls, v, info):
        """Validate params based on strategy type."""
        # Basic validation - could be extended per strategy
        return v


# =============================================================================
# BACKTEST REQUEST
# =============================================================================

class BacktestRequest(DateRangeParams):
    """Backtest request."""
    symbol: str = Field(
        ...,
        min_length=1,
        max_length=20,
        description="Symbol to backtest"
    )
    interval: str = Field(
        default="1d",
        description="Data interval"
    )
    strategy: StrategyConfig
    config: BacktestConfig = Field(default_factory=BacktestConfig)
    
    # Options
    log_trades: bool = Field(
        default=True,
        description="Include detailed trade log"
    )
    include_equity_curve: bool = Field(
        default=True,
        description="Include equity curve data"
    )
    
    @field_validator('symbol')
    @classmethod
    def normalize_symbol(cls, v):
        return v.upper().strip()


class MultiSymbolBacktestRequest(BaseSchema):
    """Backtest multiple symbols with same strategy."""
    symbols: List[str] = Field(
        ...,
        min_length=1,
        max_length=50,
        description="Symbols to backtest"
    )
    interval: str = Field(default="1d")
    strategy: StrategyConfig
    config: BacktestConfig = Field(default_factory=BacktestConfig)
    start_date: Optional[datetime] = None
    end_date: Optional[datetime] = None
    
    @field_validator('symbols')
    @classmethod
    def normalize_symbols(cls, v):
        return [s.upper().strip() for s in v]


class StrategyOptimizationRequest(BaseSchema):
    """Strategy optimization request."""
    symbol: str
    interval: str = "1d"
    strategy_type: StrategyType
    
    # Parameter ranges to optimize
    param_ranges: Dict[str, Dict[str, Any]] = Field(
        ...,
        description="Parameter ranges: {'param_name': {'min': 5, 'max': 50, 'step': 5}}"
    )
    
    # Optimization settings
    optimization_target: str = Field(
        default="sharpe_ratio",
        description="Metric to optimize (sharpe_ratio, total_return, profit_factor)"
    )
    max_iterations: int = Field(
        default=100,
        ge=10,
        le=1000
    )
    
    config: BacktestConfig = Field(default_factory=BacktestConfig)
    start_date: Optional[datetime] = None
    end_date: Optional[datetime] = None


# =============================================================================
# TRADE RESULT
# =============================================================================

class TradeResult(BaseSchema):
    """Single trade result."""
    trade_id: str
    symbol: str
    direction: str  # "long" or "short"
    
    # Entry
    entry_time: datetime
    entry_price: float
    quantity: int
    
    # Exit
    exit_time: datetime
    exit_price: float
    exit_reason: str
    
    # P&L
    gross_pnl: float
    net_pnl: float
    pnl_pct: float
    
    # Risk
    initial_stop_loss: Optional[float] = None
    initial_take_profit: Optional[float] = None
    r_multiple: Optional[float] = None
    
    # Duration
    bars_held: int
    holding_hours: float
    
    # Excursion
    max_favorable_excursion: Optional[float] = None
    max_adverse_excursion: Optional[float] = None


# =============================================================================
# BACKTEST METRICS
# =============================================================================

class PerformanceMetrics(BaseSchema):
    """Comprehensive performance metrics."""
    
    # Returns
    total_return: float
    total_return_pct: float
    annualized_return: float
    
    # Risk
    volatility: float
    max_drawdown: float
    max_drawdown_duration_days: Optional[int] = None
    
    # Risk-adjusted
    sharpe_ratio: float
    sortino_ratio: float
    calmar_ratio: float
    omega_ratio: Optional[float] = None
    
    # Trade statistics
    total_trades: int
    winning_trades: int
    losing_trades: int
    win_rate: float
    
    # Profit/Loss
    profit_factor: float
    avg_trade_pnl: float
    avg_winner: float
    avg_loser: float
    largest_winner: float
    largest_loser: float
    
    # Ratios
    avg_win_loss_ratio: float
    expectancy: float
    expectancy_ratio: Optional[float] = None
    
    # R-multiples
    avg_r_multiple: Optional[float] = None
    r_expectancy: Optional[float] = None
    
    # Duration
    avg_holding_period_hours: float
    avg_winner_holding_hours: Optional[float] = None
    avg_loser_holding_hours: Optional[float] = None
    
    # Advanced
    var_95: Optional[float] = None
    cvar_95: Optional[float] = None
    skewness: Optional[float] = None
    kurtosis: Optional[float] = None


class DrawdownInfo(BaseSchema):
    """Drawdown information."""
    max_drawdown: float
    max_drawdown_pct: float
    peak_date: datetime
    trough_date: datetime
    recovery_date: Optional[datetime] = None
    duration_days: int
    recovery_days: Optional[int] = None


class MonthlyReturn(BaseSchema):
    """Monthly return data."""
    year: int
    month: int
    return_pct: float
    trades: int


# =============================================================================
# BACKTEST RESPONSE
# =============================================================================

class EquityCurvePoint(BaseSchema):
    """Single point on equity curve."""
    timestamp: datetime
    equity: float
    cash: float
    positions_value: float
    drawdown: float


class BacktestResponse(BaseSchema):
    """Backtest result response."""
    backtest_id: str
    status: BacktestStatus
    
    # Request info
    symbol: str
    interval: str
    strategy_type: str
    strategy_params: Dict[str, Any]
    
    # Time range
    start_date: datetime
    end_date: datetime
    total_bars: int
    
    # Configuration
    config: BacktestConfig
    
    # Metrics
    metrics: PerformanceMetrics
    
    # Trades
    trades: List[TradeResult]
    
    # Equity curve (optional)
    equity_curve: Optional[List[EquityCurvePoint]] = None
    
    # Drawdown info
    drawdowns: Optional[List[DrawdownInfo]] = None
    
    # Monthly returns
    monthly_returns: Optional[List[MonthlyReturn]] = None
    
    # Execution info
    execution_time_seconds: float
    data_source: str


class MultiSymbolBacktestResponse(BaseSchema):
    """Multi-symbol backtest response."""
    job_id: str
    status: BacktestStatus
    
    # Summary
    total_symbols: int
    completed: int
    failed: int
    
    # Individual results
    results: Dict[str, BacktestResponse]
    
    # Aggregate metrics
    aggregate_metrics: Optional[Dict[str, float]] = None
    
    # Best/worst performers
    best_symbol: Optional[str] = None
    worst_symbol: Optional[str] = None


class OptimizationResult(BaseSchema):
    """Single optimization iteration result."""
    params: Dict[str, Any]
    metrics: Dict[str, float]
    rank: int


class StrategyOptimizationResponse(BaseSchema):
    """Strategy optimization response."""
    job_id: str
    status: BacktestStatus
    
    # Request info
    symbol: str
    strategy_type: str
    optimization_target: str
    
    # Results
    best_params: Dict[str, Any]
    best_metrics: PerformanceMetrics
    
    # All results
    all_results: List[OptimizationResult]
    
    # Statistics
    total_iterations: int
    execution_time_seconds: float


# =============================================================================
# ASYNC JOB
# =============================================================================

class BacktestJobStatus(BaseSchema):
    """Backtest job status."""
    job_id: str
    status: BacktestStatus
    progress: float = Field(ge=0, le=100)
    message: Optional[str] = None
    created_at: datetime
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    result_url: Optional[str] = None


# =============================================================================
# ALL EXPORTS
# =============================================================================

__all__ = [
    # Enums
    "StrategyType",
    "BacktestStatus",
    
    # Config
    "BacktestConfig",
    "StrategyConfig",
    
    # Requests
    "BacktestRequest",
    "MultiSymbolBacktestRequest",
    "StrategyOptimizationRequest",
    
    # Trade
    "TradeResult",
    
    # Metrics
    "PerformanceMetrics",
    "DrawdownInfo",
    "MonthlyReturn",
    
    # Responses
    "EquityCurvePoint",
    "BacktestResponse",
    "MultiSymbolBacktestResponse",
    "OptimizationResult",
    "StrategyOptimizationResponse",
    
    # Job
    "BacktestJobStatus",
]
