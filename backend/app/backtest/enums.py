"""
AlphaTerminal Pro - Backtest Enumerations
=========================================

Comprehensive enum definitions for backtesting operations.

Includes:
- Order types and status
- Position and trade states
- Time-related enums
- Market and session types
- BIST-specific enums

Author: AlphaTerminal Team
Version: 1.0.0
"""

from enum import Enum, IntEnum, auto
from typing import List


# =============================================================================
# ORDER ENUMS
# =============================================================================

class OrderType(str, Enum):
    """
    Order type enumeration.
    
    Defines the different types of orders that can be placed.
    """
    MARKET = "market"           # Execute at current market price
    LIMIT = "limit"             # Execute at specified price or better
    STOP = "stop"               # Trigger market order when stop price hit
    STOP_LIMIT = "stop_limit"   # Trigger limit order when stop price hit
    
    @classmethod
    def requires_price(cls, order_type: "OrderType") -> bool:
        """Check if order type requires a price."""
        return order_type in [cls.LIMIT, cls.STOP_LIMIT]
    
    @classmethod
    def requires_stop_price(cls, order_type: "OrderType") -> bool:
        """Check if order type requires a stop price."""
        return order_type in [cls.STOP, cls.STOP_LIMIT]


class OrderSide(str, Enum):
    """
    Order side enumeration.
    
    Defines whether an order is a buy or sell.
    """
    BUY = "buy"
    SELL = "sell"
    
    @property
    def opposite(self) -> "OrderSide":
        """Get the opposite side."""
        return OrderSide.SELL if self == OrderSide.BUY else OrderSide.BUY
    
    @property
    def sign(self) -> int:
        """Get the sign for calculations (+1 for buy, -1 for sell)."""
        return 1 if self == OrderSide.BUY else -1


class OrderStatus(str, Enum):
    """
    Order status enumeration.
    
    Tracks the lifecycle of an order.
    """
    PENDING = "pending"                     # Order created, not yet submitted
    SUBMITTED = "submitted"                 # Order submitted to market
    ACCEPTED = "accepted"                   # Order accepted by market
    PARTIALLY_FILLED = "partially_filled"   # Order partially executed
    FILLED = "filled"                       # Order fully executed
    CANCELLED = "cancelled"                 # Order cancelled
    REJECTED = "rejected"                   # Order rejected by market
    EXPIRED = "expired"                     # Order expired (time limit reached)
    
    @property
    def is_active(self) -> bool:
        """Check if order is still active (can be filled or cancelled)."""
        return self in [
            OrderStatus.PENDING,
            OrderStatus.SUBMITTED,
            OrderStatus.ACCEPTED,
            OrderStatus.PARTIALLY_FILLED
        ]
    
    @property
    def is_terminal(self) -> bool:
        """Check if order is in a terminal state."""
        return self in [
            OrderStatus.FILLED,
            OrderStatus.CANCELLED,
            OrderStatus.REJECTED,
            OrderStatus.EXPIRED
        ]


class OrderTimeInForce(str, Enum):
    """
    Time-in-force enumeration.
    
    Defines how long an order remains active.
    """
    GTC = "gtc"     # Good Till Cancelled
    DAY = "day"     # Day order (expires at market close)
    IOC = "ioc"     # Immediate Or Cancel
    FOK = "fok"     # Fill Or Kill
    GTD = "gtd"     # Good Till Date
    
    @classmethod
    def default(cls) -> "OrderTimeInForce":
        """Get the default time-in-force."""
        return cls.DAY


# =============================================================================
# POSITION & TRADE ENUMS
# =============================================================================

class PositionSide(str, Enum):
    """
    Position side enumeration.
    
    Defines the direction of a position.
    """
    LONG = "long"       # Positive position (bought)
    SHORT = "short"     # Negative position (sold short)
    FLAT = "flat"       # No position
    
    @classmethod
    def from_quantity(cls, quantity: int) -> "PositionSide":
        """Determine position side from quantity."""
        if quantity > 0:
            return cls.LONG
        elif quantity < 0:
            return cls.SHORT
        return cls.FLAT


class TradeDirection(str, Enum):
    """
    Trade direction enumeration.
    
    Defines the direction of a completed trade.
    """
    LONG = "long"
    SHORT = "short"


class ExitReason(str, Enum):
    """
    Exit reason enumeration.
    
    Defines why a position was closed.
    """
    STOP_LOSS = "stop_loss"             # Stop loss triggered
    TAKE_PROFIT = "take_profit"         # Take profit triggered
    TRAILING_STOP = "trailing_stop"     # Trailing stop triggered
    SIGNAL = "signal"                   # Exit signal from strategy
    SIGNAL_REVERSAL = "signal_reversal" # Opposite signal received
    TIME_STOP = "time_stop"             # Time-based exit
    END_OF_BACKTEST = "end_of_backtest" # Backtest period ended
    MANUAL = "manual"                   # Manual exit
    MARGIN_CALL = "margin_call"         # Margin requirement not met
    RISK_LIMIT = "risk_limit"           # Risk limit exceeded
    
    @property
    def is_forced(self) -> bool:
        """Check if exit was forced (not strategy-driven)."""
        return self in [
            ExitReason.END_OF_BACKTEST,
            ExitReason.MARGIN_CALL,
            ExitReason.RISK_LIMIT
        ]


# =============================================================================
# SIGNAL ENUMS
# =============================================================================

class SignalType(str, Enum):
    """
    Signal type enumeration.
    
    Defines the type of trading signal.
    """
    ENTRY_LONG = "entry_long"       # Enter long position
    ENTRY_SHORT = "entry_short"     # Enter short position
    EXIT_LONG = "exit_long"         # Exit long position
    EXIT_SHORT = "exit_short"       # Exit short position
    EXIT_ALL = "exit_all"           # Exit all positions
    SCALE_IN = "scale_in"           # Add to position
    SCALE_OUT = "scale_out"         # Reduce position
    NO_ACTION = "no_action"         # No action required
    
    @property
    def is_entry(self) -> bool:
        """Check if signal is an entry signal."""
        return self in [SignalType.ENTRY_LONG, SignalType.ENTRY_SHORT, SignalType.SCALE_IN]
    
    @property
    def is_exit(self) -> bool:
        """Check if signal is an exit signal."""
        return self in [
            SignalType.EXIT_LONG,
            SignalType.EXIT_SHORT,
            SignalType.EXIT_ALL,
            SignalType.SCALE_OUT
        ]


class SignalStrength(str, Enum):
    """
    Signal strength enumeration.
    
    Indicates the confidence level of a signal.
    """
    VERY_WEAK = "very_weak"     # 0-20%
    WEAK = "weak"               # 20-40%
    MODERATE = "moderate"       # 40-60%
    STRONG = "strong"           # 60-80%
    VERY_STRONG = "very_strong" # 80-100%
    
    @classmethod
    def from_score(cls, score: float) -> "SignalStrength":
        """Convert numeric score (0-1) to signal strength."""
        if score < 0.2:
            return cls.VERY_WEAK
        elif score < 0.4:
            return cls.WEAK
        elif score < 0.6:
            return cls.MODERATE
        elif score < 0.8:
            return cls.STRONG
        return cls.VERY_STRONG
    
    @property
    def min_score(self) -> float:
        """Get minimum score for this strength level."""
        mapping = {
            SignalStrength.VERY_WEAK: 0.0,
            SignalStrength.WEAK: 0.2,
            SignalStrength.MODERATE: 0.4,
            SignalStrength.STRONG: 0.6,
            SignalStrength.VERY_STRONG: 0.8
        }
        return mapping[self]


# =============================================================================
# TIME & MARKET ENUMS
# =============================================================================

class Timeframe(str, Enum):
    """
    Timeframe enumeration.
    
    Defines the time interval for bars/candles.
    """
    M1 = "1m"       # 1 minute
    M5 = "5m"       # 5 minutes
    M15 = "15m"     # 15 minutes
    M30 = "30m"     # 30 minutes
    H1 = "1h"       # 1 hour
    H4 = "4h"       # 4 hours
    D1 = "1d"       # 1 day
    W1 = "1w"       # 1 week
    MN1 = "1M"      # 1 month
    
    @property
    def minutes(self) -> int:
        """Get the timeframe in minutes."""
        mapping = {
            Timeframe.M1: 1,
            Timeframe.M5: 5,
            Timeframe.M15: 15,
            Timeframe.M30: 30,
            Timeframe.H1: 60,
            Timeframe.H4: 240,
            Timeframe.D1: 1440,
            Timeframe.W1: 10080,
            Timeframe.MN1: 43200
        }
        return mapping[self]
    
    @property
    def bars_per_day(self) -> float:
        """Get approximate number of bars per trading day."""
        # BIST: 10:00-18:00 = 8 hours = 480 minutes
        return 480 / self.minutes if self.minutes <= 480 else 1 / (self.minutes / 1440)


class MarketSession(str, Enum):
    """
    Market session enumeration.
    
    Defines trading sessions.
    """
    PRE_MARKET = "pre_market"       # Before market open
    OPENING = "opening"             # Opening auction
    CONTINUOUS = "continuous"       # Continuous trading
    CLOSING = "closing"             # Closing auction
    AFTER_HOURS = "after_hours"     # After market close
    CLOSED = "closed"               # Market closed


class DayOfWeek(IntEnum):
    """
    Day of week enumeration.
    
    Monday = 0, Sunday = 6 (ISO standard).
    """
    MONDAY = 0
    TUESDAY = 1
    WEDNESDAY = 2
    THURSDAY = 3
    FRIDAY = 4
    SATURDAY = 5
    SUNDAY = 6
    
    @classmethod
    def trading_days(cls) -> List["DayOfWeek"]:
        """Get list of trading days (Mon-Fri for BIST)."""
        return [cls.MONDAY, cls.TUESDAY, cls.WEDNESDAY, cls.THURSDAY, cls.FRIDAY]
    
    @classmethod
    def is_trading_day(cls, day: int) -> bool:
        """Check if a day is a trading day."""
        return day in [d.value for d in cls.trading_days()]


# =============================================================================
# BIST-SPECIFIC ENUMS
# =============================================================================

class BISTMarket(str, Enum):
    """
    BIST market enumeration.
    
    Defines the different markets in Borsa Istanbul.
    """
    BIST_STARS = "bist_stars"           # BIST Yıldız Pazar
    BIST_MAIN = "bist_main"             # BIST Ana Pazar
    BIST_SUB = "bist_sub"               # BIST Alt Pazar
    BIST_EQUITY = "bist_equity"         # Pay Piyasası
    BIST_DEBT = "bist_debt"             # Borçlanma Araçları
    BIST_DERIVATIVES = "bist_derivatives" # Vadeli İşlem ve Opsiyon


class BISTOrderType(str, Enum):
    """
    BIST-specific order types.
    """
    NORMAL = "normal"                       # Normal emir
    SPECIAL_LIMIT = "special_limit"         # Özel limit emir
    MID_POINT = "mid_point"                 # Orta fiyatlı emir
    IMBALANCE = "imbalance"                 # Dengesizlik emri


class SettlementType(str, Enum):
    """
    Settlement type enumeration.
    
    BIST uses T+2 settlement for equities.
    """
    T_PLUS_0 = "T+0"    # Same day settlement
    T_PLUS_1 = "T+1"    # Next day settlement
    T_PLUS_2 = "T+2"    # Two days settlement (BIST standard)
    T_PLUS_3 = "T+3"    # Three days settlement
    
    @classmethod
    def bist_equity_default(cls) -> "SettlementType":
        """Get BIST equity default settlement."""
        return cls.T_PLUS_2


# =============================================================================
# BACKTEST MODE ENUMS
# =============================================================================

class BacktestMode(str, Enum):
    """
    Backtest execution mode.
    """
    VECTORIZED = "vectorized"       # Fast vectorized backtest
    EVENT_DRIVEN = "event_driven"   # Detailed event-driven backtest
    TICK_BY_TICK = "tick_by_tick"   # Most detailed tick-level backtest


class FillMode(str, Enum):
    """
    Order fill mode enumeration.
    
    Defines when and at what price orders are filled.
    """
    CLOSE = "close"             # Fill at bar's close price
    OPEN = "open"               # Fill at bar's open price
    NEXT_OPEN = "next_open"     # Fill at next bar's open (most realistic)
    NEXT_CLOSE = "next_close"   # Fill at next bar's close
    VWAP = "vwap"               # Fill at estimated VWAP
    WORST = "worst"             # Fill at worst price (conservative)
    
    @classmethod
    def default(cls) -> "FillMode":
        """Get the default fill mode (most realistic)."""
        return cls.NEXT_OPEN


class SlippageModel(str, Enum):
    """
    Slippage model enumeration.
    """
    NONE = "none"               # No slippage
    FIXED = "fixed"             # Fixed percentage
    VARIABLE = "variable"       # Variable based on volume
    VOLATILITY = "volatility"   # Based on volatility
    MARKET_IMPACT = "market_impact"  # Full market impact model


class CommissionModel(str, Enum):
    """
    Commission model enumeration.
    """
    NONE = "none"               # No commission
    FIXED = "fixed"             # Fixed amount per trade
    PERCENTAGE = "percentage"   # Percentage of trade value
    TIERED = "tiered"           # Tiered based on volume
    BIST_STANDARD = "bist_standard"  # BIST standard rates


# =============================================================================
# METRICS & ANALYSIS ENUMS
# =============================================================================

class MetricCategory(str, Enum):
    """
    Metric category enumeration.
    """
    RETURN = "return"           # Return metrics
    RISK = "risk"               # Risk metrics
    RISK_ADJUSTED = "risk_adjusted"  # Risk-adjusted metrics
    TRADE = "trade"             # Trade statistics
    EXPOSURE = "exposure"       # Exposure metrics
    BENCHMARK = "benchmark"     # Benchmark comparison


class RiskMetricType(str, Enum):
    """
    Risk metric type enumeration.
    """
    MAX_DRAWDOWN = "max_drawdown"
    VAR = "var"                     # Value at Risk
    CVAR = "cvar"                   # Conditional VaR
    VOLATILITY = "volatility"
    DOWNSIDE_DEVIATION = "downside_deviation"
    ULCER_INDEX = "ulcer_index"


class ReturnMetricType(str, Enum):
    """
    Return metric type enumeration.
    """
    TOTAL_RETURN = "total_return"
    ANNUALIZED_RETURN = "annualized_return"
    MONTHLY_RETURN = "monthly_return"
    DAILY_RETURN = "daily_return"
    ROLLING_RETURN = "rolling_return"


# =============================================================================
# PORTFOLIO ENUMS
# =============================================================================

class PortfolioAllocation(str, Enum):
    """
    Portfolio allocation method enumeration.
    """
    EQUAL_WEIGHT = "equal_weight"           # Equal weight per position
    RISK_PARITY = "risk_parity"             # Risk parity allocation
    MAX_SHARPE = "max_sharpe"               # Maximum Sharpe ratio
    MIN_VARIANCE = "min_variance"           # Minimum variance
    FIXED_FRACTIONAL = "fixed_fractional"   # Fixed fractional
    KELLY = "kelly"                         # Kelly criterion


class RebalanceFrequency(str, Enum):
    """
    Portfolio rebalancing frequency.
    """
    NEVER = "never"
    DAILY = "daily"
    WEEKLY = "weekly"
    MONTHLY = "monthly"
    QUARTERLY = "quarterly"
    ANNUALLY = "annually"
    ON_THRESHOLD = "on_threshold"   # Rebalance when drift exceeds threshold


class LiquidityTier(str, Enum):
    """
    Stock liquidity classification.
    """
    VERY_HIGH = "very_high"     # BIST30 stocks
    HIGH = "high"               # BIST50 stocks
    MEDIUM = "medium"           # BIST100 stocks
    LOW = "low"                 # Other stocks
    VERY_LOW = "very_low"       # Illiquid stocks
