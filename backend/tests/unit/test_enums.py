"""
AlphaTerminal Pro - Enum Tests
==============================

Unit tests for backtest enumerations.

Author: AlphaTerminal Team
"""

import pytest
from app.backtest.enums import (
    OrderType, OrderSide, OrderStatus,
    PositionSide, TradeDirection, ExitReason,
    SignalType, FillMode, Timeframe,
    BacktestMode, LiquidityTier
)


class TestOrderEnums:
    """Test order-related enums."""
    
    def test_order_type_values(self):
        """Test OrderType enum values."""
        assert OrderType.MARKET.value == "market"
        assert OrderType.LIMIT.value == "limit"
        assert OrderType.STOP.value == "stop"
        assert OrderType.STOP_LIMIT.value == "stop_limit"
    
    def test_order_side_values(self):
        """Test OrderSide enum values."""
        assert OrderSide.BUY.value == "buy"
        assert OrderSide.SELL.value == "sell"
    
    def test_order_side_opposite(self):
        """Test OrderSide.opposite property."""
        assert OrderSide.BUY.opposite == OrderSide.SELL
        assert OrderSide.SELL.opposite == OrderSide.BUY
    
    def test_order_side_sign(self):
        """Test OrderSide.sign property."""
        assert OrderSide.BUY.sign == 1
        assert OrderSide.SELL.sign == -1
    
    def test_order_status_lifecycle(self):
        """Test OrderStatus enum values."""
        assert OrderStatus.PENDING.value == "pending"
        assert OrderStatus.SUBMITTED.value == "submitted"
        assert OrderStatus.FILLED.value == "filled"
        assert OrderStatus.CANCELLED.value == "cancelled"
        assert OrderStatus.REJECTED.value == "rejected"
    
    def test_order_status_is_final(self):
        """Test OrderStatus.is_final property."""
        assert not OrderStatus.PENDING.is_final
        assert not OrderStatus.SUBMITTED.is_final
        assert OrderStatus.FILLED.is_final
        assert OrderStatus.CANCELLED.is_final
        assert OrderStatus.REJECTED.is_final


class TestPositionEnums:
    """Test position-related enums."""
    
    def test_position_side_values(self):
        """Test PositionSide enum values."""
        assert PositionSide.LONG.value == "long"
        assert PositionSide.SHORT.value == "short"
        assert PositionSide.FLAT.value == "flat"
    
    def test_position_side_opposite(self):
        """Test PositionSide.opposite property."""
        assert PositionSide.LONG.opposite == PositionSide.SHORT
        assert PositionSide.SHORT.opposite == PositionSide.LONG
        assert PositionSide.FLAT.opposite == PositionSide.FLAT
    
    def test_trade_direction_values(self):
        """Test TradeDirection enum values."""
        assert TradeDirection.LONG.value == "long"
        assert TradeDirection.SHORT.value == "short"


class TestSignalEnums:
    """Test signal-related enums."""
    
    def test_signal_type_values(self):
        """Test SignalType enum values."""
        assert SignalType.NO_ACTION.value == "no_action"
        assert SignalType.ENTRY_LONG.value == "entry_long"
        assert SignalType.ENTRY_SHORT.value == "entry_short"
        assert SignalType.EXIT_LONG.value == "exit_long"
        assert SignalType.EXIT_SHORT.value == "exit_short"
        assert SignalType.EXIT_ALL.value == "exit_all"
    
    def test_signal_type_is_entry(self):
        """Test SignalType.is_entry property."""
        assert SignalType.ENTRY_LONG.is_entry
        assert SignalType.ENTRY_SHORT.is_entry
        assert not SignalType.EXIT_LONG.is_entry
        assert not SignalType.EXIT_SHORT.is_entry
        assert not SignalType.NO_ACTION.is_entry
    
    def test_signal_type_is_exit(self):
        """Test SignalType.is_exit property."""
        assert SignalType.EXIT_LONG.is_exit
        assert SignalType.EXIT_SHORT.is_exit
        assert SignalType.EXIT_ALL.is_exit
        assert not SignalType.ENTRY_LONG.is_exit
        assert not SignalType.NO_ACTION.is_exit


class TestExitReasonEnum:
    """Test ExitReason enum."""
    
    def test_exit_reason_values(self):
        """Test ExitReason enum values."""
        assert ExitReason.STOP_LOSS.value == "stop_loss"
        assert ExitReason.TAKE_PROFIT.value == "take_profit"
        assert ExitReason.TRAILING_STOP.value == "trailing_stop"
        assert ExitReason.SIGNAL.value == "signal"
        assert ExitReason.TIME_STOP.value == "time_stop"
        assert ExitReason.END_OF_BACKTEST.value == "end_of_backtest"


class TestTimeframeEnum:
    """Test Timeframe enum."""
    
    def test_timeframe_values(self):
        """Test Timeframe enum values."""
        assert Timeframe.M1.value == "1m"
        assert Timeframe.M5.value == "5m"
        assert Timeframe.M15.value == "15m"
        assert Timeframe.H1.value == "1h"
        assert Timeframe.D1.value == "1d"
        assert Timeframe.W1.value == "1w"
    
    def test_timeframe_minutes(self):
        """Test Timeframe.minutes property."""
        assert Timeframe.M1.minutes == 1
        assert Timeframe.M5.minutes == 5
        assert Timeframe.M15.minutes == 15
        assert Timeframe.H1.minutes == 60
        assert Timeframe.H4.minutes == 240
        assert Timeframe.D1.minutes == 1440
        assert Timeframe.W1.minutes == 10080
    
    def test_timeframe_is_intraday(self):
        """Test Timeframe.is_intraday property."""
        assert Timeframe.M1.is_intraday
        assert Timeframe.M5.is_intraday
        assert Timeframe.H1.is_intraday
        assert Timeframe.H4.is_intraday
        assert not Timeframe.D1.is_intraday
        assert not Timeframe.W1.is_intraday


class TestFillModeEnum:
    """Test FillMode enum."""
    
    def test_fill_mode_values(self):
        """Test FillMode enum values."""
        assert FillMode.CLOSE.value == "close"
        assert FillMode.OPEN.value == "open"
        assert FillMode.NEXT_OPEN.value == "next_open"
        assert FillMode.NEXT_CLOSE.value == "next_close"


class TestLiquidityTierEnum:
    """Test LiquidityTier enum."""
    
    def test_liquidity_tier_values(self):
        """Test LiquidityTier enum values."""
        assert LiquidityTier.VERY_HIGH.value == "very_high"
        assert LiquidityTier.HIGH.value == "high"
        assert LiquidityTier.MEDIUM.value == "medium"
        assert LiquidityTier.LOW.value == "low"
        assert LiquidityTier.VERY_LOW.value == "very_low"


class TestEnumStringConversion:
    """Test enum string conversion."""
    
    def test_enum_from_string(self):
        """Test creating enum from string value."""
        assert OrderType("market") == OrderType.MARKET
        assert OrderSide("buy") == OrderSide.BUY
        assert SignalType("entry_long") == SignalType.ENTRY_LONG
    
    def test_enum_to_string(self):
        """Test converting enum to string."""
        assert str(OrderType.MARKET) == "OrderType.MARKET"
        assert OrderType.MARKET.value == "market"
