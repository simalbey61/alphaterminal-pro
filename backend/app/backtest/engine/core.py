"""
AlphaTerminal Pro - Backtest Engine Core
=========================================

Part 2: Signal, BaseStrategy, and BacktestEngine class.

Author: AlphaTerminal Team
Version: 1.0.0
"""

from __future__ import annotations

import logging
import time as time_module
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime, date, time, timedelta
from typing import Any, Callable, Dict, List, Optional, Tuple, Type, Union

import pandas as pd
import numpy as np

from app.backtest.engine.backtest_engine import (
    BacktestConfig, BacktestState, BacktestResult
)
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
    TradeDirection, ExitReason, FillMode, SignalType
)
from app.backtest.models import (
    Order, Position, Trade, TradeList,
    create_market_order, create_long_position, create_short_position
)
from app.backtest.costs import (
    BISTCostCalculator, BISTCommissionConfig
)

logger = logging.getLogger(__name__)


# =============================================================================
# SIGNAL CLASS
# =============================================================================

@dataclass
class Signal:
    """
    Trading signal from strategy.
    
    Attributes:
        signal_type: Type of signal (entry/exit)
        direction: Long or Short for entries
        strength: Signal strength (0-1)
        entry_price: Suggested entry price
        stop_loss: Stop loss price
        take_profit: Take profit price(s)
        position_size: Suggested position size (0-1)
    """
    signal_type: SignalType
    direction: Optional[str] = None  # "long" or "short"
    strength: float = 1.0
    
    # Prices
    entry_price: Optional[float] = None
    stop_loss: Optional[float] = None
    take_profit: Optional[float] = None
    take_profit_2: Optional[float] = None
    take_profit_3: Optional[float] = None
    
    # Sizing
    position_size: float = 0.10  # Default 10% of capital
    
    # Metadata
    reason: str = ""
    strategy_name: str = ""
    timestamp: datetime = field(default_factory=datetime.utcnow)
    
    @classmethod
    def no_action(cls) -> "Signal":
        """Create a no-action signal."""
        return cls(signal_type=SignalType.NO_ACTION)
    
    @classmethod
    def long_entry(
        cls,
        entry_price: float,
        stop_loss: float,
        take_profit: Optional[float] = None,
        position_size: float = 0.10,
        strength: float = 1.0,
        reason: str = ""
    ) -> "Signal":
        """Create a long entry signal."""
        return cls(
            signal_type=SignalType.ENTRY_LONG,
            direction="long",
            strength=strength,
            entry_price=entry_price,
            stop_loss=stop_loss,
            take_profit=take_profit,
            position_size=position_size,
            reason=reason
        )
    
    @classmethod
    def short_entry(
        cls,
        entry_price: float,
        stop_loss: float,
        take_profit: Optional[float] = None,
        position_size: float = 0.10,
        strength: float = 1.0,
        reason: str = ""
    ) -> "Signal":
        """Create a short entry signal."""
        return cls(
            signal_type=SignalType.ENTRY_SHORT,
            direction="short",
            strength=strength,
            entry_price=entry_price,
            stop_loss=stop_loss,
            take_profit=take_profit,
            position_size=position_size,
            reason=reason
        )
    
    @classmethod
    def exit_long(cls, reason: str = "") -> "Signal":
        """Create a long exit signal."""
        return cls(signal_type=SignalType.EXIT_LONG, reason=reason)
    
    @classmethod
    def exit_short(cls, reason: str = "") -> "Signal":
        """Create a short exit signal."""
        return cls(signal_type=SignalType.EXIT_SHORT, reason=reason)
    
    @classmethod
    def exit_all(cls, reason: str = "") -> "Signal":
        """Create an exit-all signal."""
        return cls(signal_type=SignalType.EXIT_ALL, reason=reason)
    
    @property
    def is_entry(self) -> bool:
        """Check if signal is an entry signal."""
        return self.signal_type.is_entry
    
    @property
    def is_exit(self) -> bool:
        """Check if signal is an exit signal."""
        return self.signal_type.is_exit
    
    @property
    def is_long(self) -> bool:
        """Check if signal is for long direction."""
        return self.direction == "long"
    
    @property
    def is_short(self) -> bool:
        """Check if signal is for short direction."""
        return self.direction == "short"
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "signal_type": self.signal_type.value,
            "direction": self.direction,
            "strength": self.strength,
            "entry_price": self.entry_price,
            "stop_loss": self.stop_loss,
            "take_profit": self.take_profit,
            "position_size": self.position_size,
            "reason": self.reason,
            "timestamp": self.timestamp.isoformat()
        }


# =============================================================================
# BASE STRATEGY
# =============================================================================

class BaseStrategy(ABC):
    """
    Abstract base class for trading strategies.
    
    All strategies must inherit from this class and implement:
    - generate_signal(): Generate trading signals
    
    Optionally override:
    - initialize(): Setup before backtest
    - on_trade_open(): Called when trade opens
    - on_trade_close(): Called when trade closes
    
    Example:
        ```python
        class MyStrategy(BaseStrategy):
            name = "My Strategy"
            
            def __init__(self, fast_period=10, slow_period=20):
                super().__init__()
                self.fast_period = fast_period
                self.slow_period = slow_period
            
            def generate_signal(self, data: pd.DataFrame) -> Signal:
                fast_ma = data['Close'].rolling(self.fast_period).mean()
                slow_ma = data['Close'].rolling(self.slow_period).mean()
                
                if fast_ma.iloc[-1] > slow_ma.iloc[-1] and fast_ma.iloc[-2] <= slow_ma.iloc[-2]:
                    return Signal.long_entry(
                        entry_price=data['Close'].iloc[-1],
                        stop_loss=data['Close'].iloc[-1] * 0.98,
                        take_profit=data['Close'].iloc[-1] * 1.04
                    )
                
                return Signal.no_action()
        ```
    """
    
    # Strategy metadata
    name: str = "Base Strategy"
    version: str = "1.0.0"
    description: str = ""
    
    # Warmup period (minimum bars needed)
    warmup_period: int = 50
    
    def __init__(self):
        """Initialize strategy."""
        self._is_initialized = False
        self._current_position: Optional[str] = None  # "long", "short", or None
        self._trade_count = 0
        self._parameters: Dict[str, Any] = {}
    
    def initialize(self) -> None:
        """
        Initialize strategy before backtest.
        
        Override this to setup indicators, load data, etc.
        Called once before backtest starts.
        """
        self._is_initialized = True
        logger.debug(f"Strategy '{self.name}' initialized")
    
    @abstractmethod
    def generate_signal(self, data: pd.DataFrame) -> Signal:
        """
        Generate trading signal based on current data.
        
        Args:
            data: Historical OHLCV data up to current bar
            
        Returns:
            Signal object with entry/exit information
            
        Note:
            - data.iloc[-1] is the current bar
            - data includes all historical data up to now
            - Use data.iloc[-self.warmup_period:] if you only need recent data
        """
        raise NotImplementedError("Subclasses must implement generate_signal()")
    
    def on_trade_open(self, trade_info: Dict[str, Any]) -> None:
        """
        Called when a new trade opens.
        
        Override to track trade opens, update state, etc.
        
        Args:
            trade_info: Dictionary with trade details
        """
        self._trade_count += 1
        self._current_position = trade_info.get("direction")
    
    def on_trade_close(self, trade: Trade) -> None:
        """
        Called when a trade closes.
        
        Override to analyze closed trades, update state, etc.
        
        Args:
            trade: The closed Trade object
        """
        self._current_position = None
    
    def on_bar(self, bar: pd.Series, bar_index: int) -> None:
        """
        Called on each bar (optional).
        
        Override for per-bar logic like trailing stop updates.
        
        Args:
            bar: Current bar data
            bar_index: Current bar index
        """
        pass
    
    @property
    def is_in_position(self) -> bool:
        """Check if strategy has an open position."""
        return self._current_position is not None
    
    @property
    def is_long(self) -> bool:
        """Check if strategy is in long position."""
        return self._current_position == "long"
    
    @property
    def is_short(self) -> bool:
        """Check if strategy is in short position."""
        return self._current_position == "short"
    
    def get_parameters(self) -> Dict[str, Any]:
        """Get strategy parameters."""
        return self._parameters.copy()
    
    def set_parameters(self, **params) -> None:
        """Set strategy parameters."""
        self._parameters.update(params)
    
    def __str__(self) -> str:
        return f"{self.name} v{self.version}"
    
    def __repr__(self) -> str:
        return f"<{self.__class__.__name__}(name='{self.name}')>"


# =============================================================================
# BACKTEST ENGINE
# =============================================================================

class BacktestEngine:
    """
    Kurumsal seviye backtesting motoru.
    
    Features:
    - Event-driven bar-by-bar execution
    - Realistic order filling with slippage
    - Full transaction cost modeling
    - Position management with stops
    - Comprehensive metrics calculation
    
    Example:
        ```python
        # Create engine
        engine = BacktestEngine(config=BacktestConfig(
            initial_capital=100000,
            commission_rate=0.001
        ))
        
        # Run backtest
        result = engine.run(
            data=historical_data,
            strategy=MyStrategy(),
            symbol="THYAO"
        )
        
        # View results
        print(result.summary())
        print(f"Sharpe: {result.sharpe_ratio:.2f}")
        ```
    """
    
    def __init__(self, config: BacktestConfig = None):
        """
        Initialize backtest engine.
        
        Args:
            config: Backtest configuration
        """
        self.config = config or BacktestConfig()
        self.config.validate()
        
        # State
        self.state = BacktestState()
        
        # Cost calculator
        if self.config.use_bist_cost_model:
            self.cost_calculator = BISTCostCalculator()
        else:
            self.cost_calculator = None
        
        # Current data reference
        self._data: Optional[pd.DataFrame] = None
        self._symbol: str = ""
        
        logger.info(f"BacktestEngine initialized with {self.config.initial_capital:.0f} TRY capital")
    
    def run(
        self,
        data: pd.DataFrame,
        strategy: BaseStrategy,
        symbol: str,
        timeframe: str = "1d",
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None,
        benchmark_data: Optional[pd.DataFrame] = None
    ) -> BacktestResult:
        """
        Run backtest.
        
        Args:
            data: OHLCV DataFrame with DatetimeIndex
            strategy: Strategy instance
            symbol: Trading symbol
            timeframe: Data timeframe
            start_date: Backtest start date
            end_date: Backtest end date
            benchmark_data: Benchmark data for comparison
            
        Returns:
            BacktestResult with all metrics and trade history
        """
        start_time = time_module.time()
        
        logger.info(f"Starting backtest for {symbol} with strategy '{strategy.name}'")
        
        # Validate and prepare data
        data = self._prepare_data(data, start_date, end_date)
        
        if len(data) < strategy.warmup_period + 10:
            raise InsufficientDataError(
                symbol=symbol,
                available_bars=len(data),
                required_bars=strategy.warmup_period + 10,
                timeframe=timeframe
            )
        
        # Store references
        self._data = data
        self._symbol = symbol
        
        # Reset state
        self.state.reset(self.config.initial_capital)
        
        # Initialize strategy
        strategy.initialize()
        
        # Main backtest loop
        self._run_backtest_loop(data, strategy, symbol)
        
        # Close any remaining positions
        self._close_all_positions(
            data.iloc[-1],
            data.index[-1],
            ExitReason.END_OF_BACKTEST
        )
        
        # Calculate results
        result = self._calculate_results(
            symbol=symbol,
            timeframe=timeframe,
            start_date=data.index[strategy.warmup_period],
            end_date=data.index[-1],
            benchmark_data=benchmark_data
        )
        
        result.execution_time_seconds = time_module.time() - start_time
        
        logger.info(
            f"Backtest complete: {result.total_trades} trades, "
            f"Return: {result.total_return_pct:.2%}, "
            f"Sharpe: {result.sharpe_ratio:.2f}, "
            f"in {result.execution_time_seconds:.2f}s"
        )
        
        return result
    
    def _prepare_data(
        self,
        data: pd.DataFrame,
        start_date: Optional[datetime],
        end_date: Optional[datetime]
    ) -> pd.DataFrame:
        """Validate and prepare data."""
        # Check required columns
        required = ['Open', 'High', 'Low', 'Close', 'Volume']
        missing = [col for col in required if col not in data.columns]
        if missing:
            raise InvalidDataError(
                symbol=self._symbol,
                issue=f"Missing columns: {missing}"
            )
        
        # Ensure datetime index
        if not isinstance(data.index, pd.DatetimeIndex):
            raise InvalidDataError(
                symbol=self._symbol,
                issue="Data must have DatetimeIndex"
            )
        
        # Sort by date
        data = data.sort_index()
        
        # Filter by date range
        if start_date:
            data = data[data.index >= start_date]
        if end_date:
            data = data[data.index <= end_date]
        
        # Check for NaN in critical columns
        nan_count = data[['Open', 'High', 'Low', 'Close']].isna().sum().sum()
        if nan_count > 0:
            logger.warning(f"Data contains {nan_count} NaN values, forward-filling")
            data = data.fillna(method='ffill')
        
        return data
    
    def _run_backtest_loop(
        self,
        data: pd.DataFrame,
        strategy: BaseStrategy,
        symbol: str
    ) -> None:
        """Main backtest loop."""
        warmup = strategy.warmup_period
        
        for i in range(warmup, len(data)):
            self.state.current_bar_index = i
            current_bar = data.iloc[i]
            current_time = data.index[i]
            self.state.current_timestamp = current_time
            
            # Historical data up to current bar
            historical = data.iloc[:i + 1]
            
            # 1. Update positions with current prices
            self._update_positions(current_bar)
            
            # 2. Check stop loss / take profit
            self._check_exits(current_bar, current_time)
            
            # 3. Strategy callback
            strategy.on_bar(current_bar, i)
            
            # 4. Get signal from strategy
            try:
                signal = strategy.generate_signal(historical)
            except Exception as e:
                logger.error(f"Strategy error at bar {i}: {e}")
                signal = Signal.no_action()
            
            # 5. Execute signal
            if signal.signal_type != SignalType.NO_ACTION:
                self._process_signal(signal, current_bar, current_time, symbol, strategy)
            
            # 6. Record equity
            self.state.record_equity(current_time)
            
            # 7. Check drawdown limit
            if self.config.max_drawdown_limit:
                if self.state.current_drawdown >= self.config.max_drawdown_limit:
                    logger.warning(
                        f"Max drawdown limit reached: {self.state.current_drawdown:.2%}"
                    )
                    self._close_all_positions(
                        current_bar, current_time, ExitReason.RISK_LIMIT
                    )
                    break
    
    def _update_positions(self, bar: pd.Series) -> None:
        """Update all positions with current prices."""
        for symbol, position in self.state.positions.items():
            position.update_price(
                price=bar['Close'],
                high=bar['High'],
                low=bar['Low'],
                timestamp=self.state.current_timestamp
            )
            position.increment_bars()
    
    def _check_exits(self, bar: pd.Series, timestamp: datetime) -> None:
        """Check and execute stop loss / take profit exits."""
        positions_to_close = []
        
        for symbol, position in self.state.positions.items():
            triggered, reason, exit_price = position.should_stop_out(
                high=bar['High'],
                low=bar['Low']
            )
            
            if triggered:
                positions_to_close.append((symbol, exit_price, reason))
        
        for symbol, exit_price, reason in positions_to_close:
            self._close_position(symbol, exit_price, timestamp, reason)
    
    def _process_signal(
        self,
        signal: Signal,
        bar: pd.Series,
        timestamp: datetime,
        symbol: str,
        strategy: BaseStrategy
    ) -> None:
        """Process trading signal."""
        # Exit signals
        if signal.signal_type == SignalType.EXIT_ALL:
            self._close_all_positions(bar, timestamp, ExitReason.SIGNAL)
            return
        
        if signal.signal_type == SignalType.EXIT_LONG:
            if symbol in self.state.positions:
                pos = self.state.positions[symbol]
                if pos.is_long:
                    self._close_position(symbol, bar['Close'], timestamp, ExitReason.SIGNAL)
            return
        
        if signal.signal_type == SignalType.EXIT_SHORT:
            if symbol in self.state.positions:
                pos = self.state.positions[symbol]
                if pos.is_short:
                    self._close_position(symbol, bar['Close'], timestamp, ExitReason.SIGNAL)
            return
        
        # Entry signals
        if signal.signal_type == SignalType.ENTRY_LONG:
            # Check if already in position
            if symbol in self.state.positions:
                return
            
            # Check max positions
            if self.state.num_positions >= self.config.max_positions:
                logger.debug(f"Max positions reached ({self.config.max_positions}), skipping entry")
                return
            
            # Open long
            self._open_position(
                symbol=symbol,
                side=PositionSide.LONG,
                entry_price=signal.entry_price or bar['Close'],
                stop_loss=signal.stop_loss,
                take_profit=signal.take_profit,
                position_size_pct=signal.position_size,
                timestamp=timestamp,
                strategy=strategy,
                signal=signal
            )
        
        elif signal.signal_type == SignalType.ENTRY_SHORT:
            if not self.config.allow_shorting:
                logger.debug("Short selling not allowed, skipping")
                return
            
            if symbol in self.state.positions:
                return
            
            if self.state.num_positions >= self.config.max_positions:
                return
            
            self._open_position(
                symbol=symbol,
                side=PositionSide.SHORT,
                entry_price=signal.entry_price or bar['Close'],
                stop_loss=signal.stop_loss,
                take_profit=signal.take_profit,
                position_size_pct=signal.position_size,
                timestamp=timestamp,
                strategy=strategy,
                signal=signal
            )
    
    def _open_position(
        self,
        symbol: str,
        side: PositionSide,
        entry_price: float,
        stop_loss: Optional[float],
        take_profit: Optional[float],
        position_size_pct: float,
        timestamp: datetime,
        strategy: BaseStrategy,
        signal: Signal
    ) -> None:
        """Open a new position."""
        # Calculate position size
        position_value = self.state.cash * min(
            position_size_pct,
            self.config.max_position_size
        )
        
        # Check minimum
        if position_value < self.config.min_position_value:
            logger.debug(f"Position too small: {position_value:.2f} < {self.config.min_position_value}")
            return
        
        # Calculate costs
        if self.cost_calculator:
            costs = self.cost_calculator.calculate_total_costs(
                price=entry_price,
                quantity=int(position_value / entry_price),
                side=OrderSide.BUY if side == PositionSide.LONG else OrderSide.SELL,
                symbol=symbol
            )
            effective_price = costs['effective_price']
            commission = costs['total_commission']
            slippage = costs['total_slippage']
        else:
            slippage_amount = entry_price * self.config.slippage_rate
            if side == PositionSide.LONG:
                effective_price = entry_price + slippage_amount
            else:
                effective_price = entry_price - slippage_amount
            commission = position_value * self.config.commission_rate
            slippage = slippage_amount * int(position_value / entry_price)
        
        # Calculate quantity
        quantity = int(position_value / effective_price)
        if quantity < self.config.lot_size:
            logger.debug(f"Quantity too small: {quantity}")
            return
        
        # Check funds
        total_cost = (quantity * effective_price) + commission
        if total_cost > self.state.cash:
            logger.debug(f"Insufficient funds: need {total_cost:.2f}, have {self.state.cash:.2f}")
            return
        
        # Deduct cash
        self.state.cash -= total_cost
        
        # Create position
        if side == PositionSide.LONG:
            position = create_long_position(
                symbol=symbol,
                quantity=quantity,
                entry_price=effective_price,
                commission=commission,
                stop_loss=stop_loss,
                take_profit=take_profit,
                strategy_id=strategy.name
            )
        else:
            position = create_short_position(
                symbol=symbol,
                quantity=quantity,
                entry_price=effective_price,
                commission=commission,
                stop_loss=stop_loss,
                take_profit=take_profit,
                strategy_id=strategy.name
            )
        
        position.opened_at = timestamp
        self.state.positions[symbol] = position
        
        # Notify strategy
        strategy.on_trade_open({
            "symbol": symbol,
            "direction": "long" if side == PositionSide.LONG else "short",
            "quantity": quantity,
            "entry_price": effective_price,
            "stop_loss": stop_loss,
            "take_profit": take_profit
        })
        
        if self.config.log_trades:
            sl_str = f"{stop_loss:.2f}" if stop_loss else "N/A"
            tp_str = f"{take_profit:.2f}" if take_profit else "N/A"
            logger.info(
                f"OPEN {side.value.upper()} {quantity} {symbol} @ {effective_price:.2f} "
                f"SL:{sl_str} TP:{tp_str}"
            )
    
    def _close_position(
        self,
        symbol: str,
        exit_price: float,
        timestamp: datetime,
        reason: ExitReason
    ) -> None:
        """Close an existing position."""
        if symbol not in self.state.positions:
            return
        
        position = self.state.positions[symbol]
        
        # Calculate exit costs
        if self.cost_calculator:
            costs = self.cost_calculator.calculate_total_costs(
                price=exit_price,
                quantity=position.quantity,
                side=OrderSide.SELL if position.is_long else OrderSide.BUY,
                symbol=symbol
            )
            effective_price = costs['effective_price']
            exit_commission = costs['total_commission']
            exit_slippage = costs['total_slippage']
        else:
            slippage_amount = exit_price * self.config.slippage_rate
            if position.is_long:
                effective_price = exit_price - slippage_amount
            else:
                effective_price = exit_price + slippage_amount
            exit_commission = position.quantity * effective_price * self.config.commission_rate
            exit_slippage = slippage_amount * position.quantity
        
        # Calculate P&L
        if position.is_long:
            gross_pnl = (effective_price - position.avg_entry_price) * position.quantity
        else:
            gross_pnl = (position.avg_entry_price - effective_price) * position.quantity
        
        total_commission = position.total_entry_commission + exit_commission
        net_pnl = gross_pnl - exit_commission
        
        # Return cash
        exit_value = position.quantity * effective_price
        self.state.cash += exit_value - exit_commission
        
        # Create trade record
        trade = Trade(
            symbol=symbol,
            direction=TradeDirection.LONG if position.is_long else TradeDirection.SHORT,
            quantity=position.quantity,
            entry_price=position.avg_entry_price,
            exit_price=effective_price,
            entry_time=position.opened_at,
            exit_time=timestamp,
            exit_reason=reason,
            entry_commission=position.total_entry_commission,
            exit_commission=exit_commission,
            entry_slippage=0,  # Already included in entry price
            exit_slippage=exit_slippage,
            initial_stop_loss=position.stop_loss,
            initial_take_profit=position.take_profit,
            highest_price=position.highest_price,
            lowest_price=position.lowest_price,
            max_favorable_excursion=position.max_favorable_excursion,
            max_adverse_excursion=position.max_adverse_excursion,
            bars_held=position.bars_held,
            strategy_id=position.strategy_id,
            position_id=position.position_id
        )
        
        # Update state
        self.state.trades.append(trade)
        self.state.total_trades += 1
        if trade.is_winner:
            self.state.winning_trades += 1
        else:
            self.state.losing_trades += 1
        
        # Remove position
        del self.state.positions[symbol]
        
        if self.config.log_trades:
            pnl_str = f"+{net_pnl:.2f}" if net_pnl >= 0 else f"{net_pnl:.2f}"
            logger.info(
                f"CLOSE {symbol} @ {effective_price:.2f} "
                f"P&L: {pnl_str} ({trade.pnl_pct:+.2%}) - {reason.value}"
            )
    
    def _close_all_positions(
        self,
        bar: pd.Series,
        timestamp: datetime,
        reason: ExitReason
    ) -> None:
        """Close all open positions."""
        for symbol in list(self.state.positions.keys()):
            self._close_position(symbol, bar['Close'], timestamp, reason)
    
    def _calculate_results(
        self,
        symbol: str,
        timeframe: str,
        start_date: datetime,
        end_date: datetime,
        benchmark_data: Optional[pd.DataFrame]
    ) -> BacktestResult:
        """Calculate comprehensive backtest results."""
        # Align equity history with timestamps
        min_len = min(len(self.state.equity_history), len(self.state.timestamp_history))
        equity_series = pd.Series(
            self.state.equity_history[:min_len],
            index=self.state.timestamp_history[:min_len]
        )
        
        if len(equity_series) < 2:
            # Not enough data
            return BacktestResult(
                config=self.config,
                symbol=symbol,
                timeframe=timeframe,
                start_date=start_date,
                end_date=end_date,
                trades=self.state.trades
            )
        
        # Returns
        returns = equity_series.pct_change().dropna()
        
        total_return = equity_series.iloc[-1] - self.config.initial_capital
        total_return_pct = total_return / self.config.initial_capital
        
        # Annualized return
        trading_days = len(returns)
        years = trading_days / 252 if trading_days > 0 else 1
        annualized_return = (1 + total_return_pct) ** (1 / years) - 1 if years > 0 else 0
        
        # Volatility
        volatility = returns.std() * np.sqrt(252) if len(returns) > 1 else 0
        
        # Drawdown
        rolling_max = equity_series.expanding().max()
        drawdown = (equity_series - rolling_max) / rolling_max
        max_drawdown = abs(drawdown.min()) if len(drawdown) > 0 else 0
        
        # Risk-adjusted metrics
        risk_free_rate = 0.05  # 5% annual risk-free rate
        excess_returns = returns - risk_free_rate / 252
        
        sharpe_ratio = (
            excess_returns.mean() / returns.std() * np.sqrt(252)
            if returns.std() > 0 else 0
        )
        
        # Sortino ratio
        downside_returns = returns[returns < 0]
        downside_std = downside_returns.std() * np.sqrt(252) if len(downside_returns) > 0 else 1
        sortino_ratio = (annualized_return - risk_free_rate) / downside_std if downside_std > 0 else 0
        
        # Calmar ratio
        calmar_ratio = annualized_return / max_drawdown if max_drawdown > 0 else 0
        
        # Trade statistics
        trades = self.state.trades
        total_trades = len(trades)
        winning_trades = len(trades.winners)
        losing_trades = len(trades.losers)
        win_rate = trades.win_rate
        
        # P&L statistics
        profit_factor = trades.profit_factor
        avg_trade_pnl = trades.expectancy
        avg_winner = trades.avg_winner
        avg_loser = trades.avg_loser
        
        largest_winner = max([t.net_pnl for t in trades.winners]) if trades.winners else 0
        largest_loser = min([t.net_pnl for t in trades.losers]) if trades.losers else 0
        
        # Trade timing
        avg_bars = np.mean([t.bars_held for t in trades]) if trades else 0
        avg_duration = np.mean([t.holding_hours for t in trades]) if trades else 0
        
        # Exposure
        time_in_market = sum(1 for e in self.state.equity_history if e != self.state.equity_history[0]) / len(self.state.equity_history) if self.state.equity_history else 0
        
        # Monthly returns
        if len(equity_series) > 0:
            monthly_returns = returns.resample('M').apply(lambda x: (1 + x).prod() - 1)
        else:
            monthly_returns = pd.Series()
        
        return BacktestResult(
            config=self.config,
            symbol=symbol,
            timeframe=timeframe,
            start_date=start_date,
            end_date=end_date,
            total_return=total_return,
            total_return_pct=total_return_pct,
            annualized_return=annualized_return,
            volatility=volatility,
            max_drawdown=max_drawdown,
            sharpe_ratio=sharpe_ratio,
            sortino_ratio=sortino_ratio,
            calmar_ratio=calmar_ratio,
            total_trades=total_trades,
            winning_trades=winning_trades,
            losing_trades=losing_trades,
            win_rate=win_rate,
            profit_factor=profit_factor,
            avg_trade_pnl=avg_trade_pnl,
            avg_winner=avg_winner,
            avg_loser=avg_loser,
            largest_winner=largest_winner,
            largest_loser=largest_loser,
            avg_bars_in_trade=avg_bars,
            avg_trade_duration=avg_duration,
            time_in_market=time_in_market,
            equity_curve=equity_series,
            drawdown_curve=drawdown,
            monthly_returns=monthly_returns,
            trades=trades
        )
