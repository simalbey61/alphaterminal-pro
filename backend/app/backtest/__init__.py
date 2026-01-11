"""
AlphaTerminal Pro - Backtest Module
====================================

Kurumsal seviye backtesting framework.

Usage:
    ```python
    from app.backtest import BacktestEngine, BacktestConfig, BaseStrategy, Signal
    
    class MyStrategy(BaseStrategy):
        name = "My Strategy"
        
        def generate_signal(self, data):
            # Strategy logic
            return Signal.no_action()
    
    # Configure
    config = BacktestConfig(
        initial_capital=100000,
        commission_rate=0.001
    )
    
    # Run backtest
    engine = BacktestEngine(config)
    result = engine.run(data, MyStrategy(), "THYAO")
    
    # View results
    print(result.summary())
    ```

Author: AlphaTerminal Team
Version: 1.0.0
"""

# Exceptions
from app.backtest.exceptions import (
    BacktestError,
    ConfigurationError,
    DataError,
    InsufficientDataError,
    InvalidDataError,
    ExecutionError,
    InsufficientFundsError,
    StrategyError,
    MetricsError
)

# Enums
from app.backtest.enums import (
    OrderType,
    OrderSide,
    OrderStatus,
    PositionSide,
    TradeDirection,
    ExitReason,
    SignalType,
    FillMode,
    BacktestMode,
    Timeframe,
    LiquidityTier
)

# Models
from app.backtest.models import (
    Order,
    Position,
    Trade,
    TradeList,
    create_market_order,
    create_limit_order,
    create_long_position,
    create_short_position
)

# Costs
from app.backtest.costs import (
    BISTCostCalculator,
    BISTCommissionConfig,
    BISTSlippageConfig,
    CommissionResult,
    SlippageResult
)

# Engine
from app.backtest.engine import (
    BacktestEngine,
    BacktestConfig,
    BacktestState,
    BacktestResult,
    BaseStrategy,
    Signal
)

# Metrics
from app.backtest.metrics import (
    calculate_sharpe_ratio,
    calculate_sortino_ratio,
    calculate_max_drawdown,
    calculate_all_metrics,
    PerformanceMetrics
)

__all__ = [
    # Exceptions
    "BacktestError",
    "ConfigurationError",
    "DataError",
    "InsufficientDataError",
    "InvalidDataError",
    "ExecutionError",
    "InsufficientFundsError",
    "StrategyError",
    "MetricsError",
    
    # Enums
    "OrderType",
    "OrderSide",
    "OrderStatus",
    "PositionSide",
    "TradeDirection",
    "ExitReason",
    "SignalType",
    "FillMode",
    "BacktestMode",
    "Timeframe",
    
    # Models
    "Order",
    "Position",
    "Trade",
    "TradeList",
    "create_market_order",
    "create_limit_order",
    "create_long_position",
    "create_short_position",
    
    # Costs
    "BISTCostCalculator",
    "BISTCommissionConfig",
    "BISTSlippageConfig",
    
    # Engine
    "BacktestEngine",
    "BacktestConfig",
    "BacktestState",
    "BacktestResult",
    "BaseStrategy",
    "Signal",
    
    # Metrics
    "calculate_sharpe_ratio",
    "calculate_sortino_ratio",
    "calculate_max_drawdown",
    "calculate_all_metrics",
    "PerformanceMetrics"
]

__version__ = "1.0.0"
