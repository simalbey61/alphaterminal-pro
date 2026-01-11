"""
AlphaTerminal Pro - Backtest Engine
====================================

Kurumsal seviye backtesting motoru.

Features:
- Event-driven or vectorized execution
- Realistic order execution simulation
- Full transaction cost modeling (BIST-specific)
- Position management with stop/target
- Comprehensive performance metrics
- Trade-by-trade logging
- Equity curve tracking

Author: AlphaTerminal Team
Version: 1.0.0
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime, date, time, timedelta
from typing import Any, Callable, Dict, List, Optional, Tuple, Type
from uuid import uuid4
from copy import deepcopy
from enum import Enum

import pandas as pd
import numpy as np

from app.backtest.exceptions import (
    BacktestError,
    InsufficientDataError,
    InvalidDataError,
    ExecutionError,
    InsufficientFundsError,
    StrategyError
)
from app.backtest.enums import (
    OrderType, OrderSide, OrderStatus, PositionSide,
    TradeDirection, ExitReason, FillMode, BacktestMode,
    SignalType
)
from app.backtest.models import (
    Order, Position, Trade, TradeList,
    create_market_order, create_long_position, create_short_position
)
from app.backtest.costs import (
    BISTCostCalculator, BISTCommissionConfig, BISTSlippageConfig
)

logger = logging.getLogger(__name__)


# =============================================================================
# CONFIGURATION
# =============================================================================

@dataclass
class BacktestConfig:
    """
    Backtest configuration.
    
    Attributes:
        initial_capital: Starting capital (TRY)
        commission_rate: Commission rate (0.001 = 0.1%)
        slippage_rate: Slippage rate (0.0005 = 0.05%)
        fill_mode: When to fill orders
        allow_shorting: Allow short selling
        max_position_size: Maximum position as % of capital
        max_positions: Maximum concurrent positions
        risk_per_trade: Risk per trade as % of capital
    """
    # Capital
    initial_capital: float = 100_000.0
    
    # Transaction costs
    commission_rate: float = 0.001      # 0.1%
    slippage_rate: float = 0.0005       # 0.05%
    spread_rate: float = 0.0002         # 0.02%
    use_bist_cost_model: bool = True    # Use detailed BIST costs
    
    # Execution
    fill_mode: FillMode = FillMode.NEXT_OPEN
    allow_shorting: bool = False        # BIST doesn't allow retail shorting
    allow_margin: bool = False
    
    # Position sizing
    max_position_size: float = 0.20     # Max 20% per position
    max_positions: int = 10             # Max 10 concurrent positions
    min_position_value: float = 1000    # Minimum position 1000 TRY
    
    # Risk management
    risk_per_trade: float = 0.02        # 2% risk per trade
    max_portfolio_risk: float = 0.10    # 10% max portfolio risk
    max_drawdown_limit: Optional[float] = 0.20  # Stop trading at 20% DD
    
    # BIST-specific
    lot_size: int = 1                   # Minimum lot size
    trading_hours_only: bool = True     # Only trade during market hours
    market_open: time = time(10, 0)     # BIST opens 10:00
    market_close: time = time(18, 0)    # BIST closes 18:00
    
    # Backtesting
    warmup_period: int = 50             # Bars for indicator warmup
    data_frequency: str = "1d"          # Data frequency
    
    # Logging
    log_trades: bool = True
    log_orders: bool = False
    verbose: bool = False
    
    def validate(self) -> None:
        """Validate configuration."""
        if self.initial_capital <= 0:
            raise ValueError(f"initial_capital must be positive: {self.initial_capital}")
        if not 0 <= self.commission_rate <= 0.1:
            raise ValueError(f"commission_rate out of range: {self.commission_rate}")
        if not 0 <= self.max_position_size <= 1:
            raise ValueError(f"max_position_size must be 0-1: {self.max_position_size}")
        if self.max_positions < 1:
            raise ValueError(f"max_positions must be >= 1: {self.max_positions}")
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "initial_capital": self.initial_capital,
            "commission_rate": self.commission_rate,
            "slippage_rate": self.slippage_rate,
            "fill_mode": self.fill_mode.value,
            "allow_shorting": self.allow_shorting,
            "max_position_size": self.max_position_size,
            "max_positions": self.max_positions,
            "risk_per_trade": self.risk_per_trade,
            "warmup_period": self.warmup_period
        }


# =============================================================================
# BACKTEST STATE
# =============================================================================

@dataclass
class BacktestState:
    """
    Current state of the backtest.
    
    Tracks capital, positions, orders, and history.
    """
    # Capital tracking
    cash: float = 0.0
    initial_capital: float = 0.0
    
    # Positions and orders
    positions: Dict[str, Position] = field(default_factory=dict)
    pending_orders: List[Order] = field(default_factory=list)
    
    # Trade history
    trades: TradeList = field(default_factory=TradeList)
    closed_orders: List[Order] = field(default_factory=list)
    
    # Equity tracking
    equity_history: List[float] = field(default_factory=list)
    cash_history: List[float] = field(default_factory=list)
    timestamp_history: List[datetime] = field(default_factory=list)
    
    # Performance tracking
    high_water_mark: float = 0.0
    max_drawdown: float = 0.0
    current_drawdown: float = 0.0
    
    # Counters
    total_trades: int = 0
    winning_trades: int = 0
    losing_trades: int = 0
    
    # Current bar info
    current_bar_index: int = 0
    current_timestamp: Optional[datetime] = None
    
    # Daily tracking
    daily_pnl: float = 0.0
    cumulative_volume: float = 0.0
    
    @property
    def equity(self) -> float:
        """Current total equity (cash + positions market value)."""
        positions_value = sum(
            pos.market_value for pos in self.positions.values()
        )
        return self.cash + positions_value
    
    @property
    def unrealized_pnl(self) -> float:
        """Total unrealized P&L from open positions."""
        return sum(pos.unrealized_pnl for pos in self.positions.values())
    
    @property
    def realized_pnl(self) -> float:
        """Total realized P&L from closed trades."""
        return self.trades.total_pnl if self.trades else 0.0
    
    @property
    def total_pnl(self) -> float:
        """Total P&L (realized + unrealized)."""
        return self.realized_pnl + self.unrealized_pnl
    
    @property
    def return_pct(self) -> float:
        """Return percentage."""
        if self.initial_capital == 0:
            return 0.0
        return (self.equity - self.initial_capital) / self.initial_capital
    
    @property
    def num_positions(self) -> int:
        """Number of open positions."""
        return len(self.positions)
    
    @property
    def exposure(self) -> float:
        """Current market exposure as % of equity."""
        positions_value = sum(
            abs(pos.market_value) for pos in self.positions.values()
        )
        return positions_value / self.equity if self.equity > 0 else 0.0
    
    def update_drawdown(self) -> None:
        """Update drawdown tracking."""
        if self.equity > self.high_water_mark:
            self.high_water_mark = self.equity
        
        if self.high_water_mark > 0:
            self.current_drawdown = (self.high_water_mark - self.equity) / self.high_water_mark
            self.max_drawdown = max(self.max_drawdown, self.current_drawdown)
    
    def record_equity(self, timestamp: datetime) -> None:
        """Record current equity to history."""
        self.equity_history.append(self.equity)
        self.cash_history.append(self.cash)
        self.timestamp_history.append(timestamp)
        self.update_drawdown()
    
    def reset(self, initial_capital: float) -> None:
        """Reset state for new backtest."""
        self.cash = initial_capital
        self.initial_capital = initial_capital
        self.high_water_mark = initial_capital
        self.positions.clear()
        self.pending_orders.clear()
        self.trades = TradeList()
        self.closed_orders.clear()
        self.equity_history = [initial_capital]
        self.cash_history = [initial_capital]
        self.timestamp_history = []
        self.max_drawdown = 0.0
        self.current_drawdown = 0.0
        self.total_trades = 0
        self.winning_trades = 0
        self.losing_trades = 0
        self.current_bar_index = 0


# =============================================================================
# BACKTEST RESULT
# =============================================================================

@dataclass
class BacktestResult:
    """
    Comprehensive backtest result.
    
    Contains all metrics, equity curves, and trade history.
    """
    # Configuration
    config: BacktestConfig
    symbol: str
    timeframe: str
    start_date: datetime
    end_date: datetime
    
    # Return metrics
    total_return: float = 0.0
    total_return_pct: float = 0.0
    annualized_return: float = 0.0
    
    # Risk metrics
    volatility: float = 0.0
    max_drawdown: float = 0.0
    max_drawdown_duration: int = 0  # Days
    
    # Risk-adjusted metrics
    sharpe_ratio: float = 0.0
    sortino_ratio: float = 0.0
    calmar_ratio: float = 0.0
    
    # Trade statistics
    total_trades: int = 0
    winning_trades: int = 0
    losing_trades: int = 0
    win_rate: float = 0.0
    
    # P&L statistics
    profit_factor: float = 0.0
    avg_trade_pnl: float = 0.0
    avg_winner: float = 0.0
    avg_loser: float = 0.0
    largest_winner: float = 0.0
    largest_loser: float = 0.0
    
    # Trade timing
    avg_bars_in_trade: float = 0.0
    avg_trade_duration: float = 0.0  # Hours
    
    # Exposure metrics
    time_in_market: float = 0.0
    avg_exposure: float = 0.0
    
    # Detailed data
    equity_curve: pd.Series = field(default_factory=pd.Series)
    drawdown_curve: pd.Series = field(default_factory=pd.Series)
    monthly_returns: pd.Series = field(default_factory=pd.Series)
    trades: TradeList = field(default_factory=TradeList)
    
    # Benchmark (optional)
    benchmark_return: Optional[float] = None
    alpha: Optional[float] = None
    beta: Optional[float] = None
    
    # Execution info
    execution_time_seconds: float = 0.0
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "symbol": self.symbol,
            "timeframe": self.timeframe,
            "start_date": self.start_date.isoformat(),
            "end_date": self.end_date.isoformat(),
            
            # Returns
            "total_return": round(self.total_return, 2),
            "total_return_pct": round(self.total_return_pct, 4),
            "annualized_return": round(self.annualized_return, 4),
            
            # Risk
            "volatility": round(self.volatility, 4),
            "max_drawdown": round(self.max_drawdown, 4),
            "max_drawdown_duration_days": self.max_drawdown_duration,
            
            # Risk-adjusted
            "sharpe_ratio": round(self.sharpe_ratio, 2),
            "sortino_ratio": round(self.sortino_ratio, 2),
            "calmar_ratio": round(self.calmar_ratio, 2),
            
            # Trades
            "total_trades": self.total_trades,
            "winning_trades": self.winning_trades,
            "losing_trades": self.losing_trades,
            "win_rate": round(self.win_rate, 4),
            
            # P&L
            "profit_factor": round(self.profit_factor, 2),
            "avg_trade_pnl": round(self.avg_trade_pnl, 2),
            "avg_winner": round(self.avg_winner, 2),
            "avg_loser": round(self.avg_loser, 2),
            "largest_winner": round(self.largest_winner, 2),
            "largest_loser": round(self.largest_loser, 2),
            
            # Timing
            "avg_bars_in_trade": round(self.avg_bars_in_trade, 1),
            "avg_trade_duration_hours": round(self.avg_trade_duration, 1),
            
            # Exposure
            "time_in_market": round(self.time_in_market, 4),
            "avg_exposure": round(self.avg_exposure, 4),
            
            # Benchmark
            "benchmark_return": round(self.benchmark_return, 4) if self.benchmark_return else None,
            "alpha": round(self.alpha, 4) if self.alpha else None,
            "beta": round(self.beta, 2) if self.beta else None,
            
            # Meta
            "execution_time_seconds": round(self.execution_time_seconds, 2)
        }
    
    def summary(self) -> str:
        """Generate summary string."""
        return f"""
╔══════════════════════════════════════════════════════════════════╗
║                    BACKTEST RESULT SUMMARY                       ║
╠══════════════════════════════════════════════════════════════════╣
║  Symbol: {self.symbol:<12}  Timeframe: {self.timeframe:<8}                    ║
║  Period: {self.start_date.strftime('%Y-%m-%d')} to {self.end_date.strftime('%Y-%m-%d')}                    ║
╠══════════════════════════════════════════════════════════════════╣
║  RETURNS                                                          ║
║    Total Return:     {self.total_return_pct:>+8.2%}                               ║
║    Annualized:       {self.annualized_return:>+8.2%}                               ║
╠══════════════════════════════════════════════════════════════════╣
║  RISK                                                             ║
║    Max Drawdown:     {self.max_drawdown:>8.2%}                                ║
║    Volatility:       {self.volatility:>8.2%}                                ║
╠══════════════════════════════════════════════════════════════════╣
║  RISK-ADJUSTED                                                    ║
║    Sharpe Ratio:     {self.sharpe_ratio:>8.2f}                                ║
║    Sortino Ratio:    {self.sortino_ratio:>8.2f}                                ║
║    Calmar Ratio:     {self.calmar_ratio:>8.2f}                                ║
╠══════════════════════════════════════════════════════════════════╣
║  TRADES                                                           ║
║    Total:            {self.total_trades:>8}                                   ║
║    Win Rate:         {self.win_rate:>8.1%}                                ║
║    Profit Factor:    {self.profit_factor:>8.2f}                                ║
║    Avg Trade:        {self.avg_trade_pnl:>+8.2f} TRY                           ║
╚══════════════════════════════════════════════════════════════════╝
"""
