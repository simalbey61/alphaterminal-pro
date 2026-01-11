"""
AlphaTerminal Pro - Model Tests
===============================

Unit tests for backtest data models (Order, Position, Trade).

Author: AlphaTerminal Team
"""

import pytest
from datetime import datetime, timedelta
from app.backtest.models import (
    Order, Position, Trade, TradeList,
    create_market_order, create_limit_order,
    create_long_position, create_short_position
)
from app.backtest.enums import (
    OrderType, OrderSide, OrderStatus,
    PositionSide, TradeDirection, ExitReason
)


class TestOrderModel:
    """Test Order model."""
    
    def test_create_market_order(self):
        """Test market order creation."""
        order = create_market_order(
            symbol="THYAO",
            side=OrderSide.BUY,
            quantity=100
        )
        
        assert order.symbol == "THYAO"
        assert order.order_type == OrderType.MARKET
        assert order.side == OrderSide.BUY
        assert order.quantity == 100
        assert order.status == OrderStatus.PENDING
        assert order.order_id is not None
    
    def test_create_limit_order(self):
        """Test limit order creation."""
        order = create_limit_order(
            symbol="GARAN",
            side=OrderSide.SELL,
            quantity=200,
            price=55.50
        )
        
        assert order.symbol == "GARAN"
        assert order.order_type == OrderType.LIMIT
        assert order.side == OrderSide.SELL
        assert order.quantity == 200
        assert order.limit_price == 55.50
    
    def test_order_fill(self):
        """Test order fill."""
        order = create_market_order("THYAO", OrderSide.BUY, 100)
        
        order.fill(
            fill_price=285.50,
            fill_quantity=100,
            commission=1.50,
            slippage=0.50
        )
        
        assert order.status == OrderStatus.FILLED
        assert order.filled_quantity == 100
        assert order.avg_fill_price == 285.50
        assert order.is_fully_filled
    
    def test_order_partial_fill(self):
        """Test partial order fill."""
        order = create_market_order("THYAO", OrderSide.BUY, 100)
        
        # First partial fill
        order.fill(fill_price=285.50, fill_quantity=50, commission=0.75)
        
        assert order.status == OrderStatus.PARTIAL
        assert order.filled_quantity == 50
        assert not order.is_fully_filled
        
        # Complete fill
        order.fill(fill_price=286.00, fill_quantity=50, commission=0.75)
        
        assert order.status == OrderStatus.FILLED
        assert order.filled_quantity == 100
        assert order.is_fully_filled
        # Average price: (50*285.50 + 50*286.00) / 100 = 285.75
        assert order.avg_fill_price == pytest.approx(285.75)
    
    def test_order_cancel(self):
        """Test order cancellation."""
        order = create_market_order("THYAO", OrderSide.BUY, 100)
        
        order.cancel(reason="User requested")
        
        assert order.status == OrderStatus.CANCELLED
        assert order.cancel_reason == "User requested"
    
    def test_order_reject(self):
        """Test order rejection."""
        order = create_market_order("THYAO", OrderSide.BUY, 100)
        
        order.reject(reason="Insufficient funds")
        
        assert order.status == OrderStatus.REJECTED
        assert order.reject_reason == "Insufficient funds"
    
    def test_order_to_dict(self):
        """Test order serialization."""
        order = create_market_order("THYAO", OrderSide.BUY, 100)
        order_dict = order.to_dict()
        
        assert order_dict["symbol"] == "THYAO"
        assert order_dict["order_type"] == "market"
        assert order_dict["side"] == "buy"
        assert order_dict["quantity"] == 100


class TestPositionModel:
    """Test Position model."""
    
    def test_create_long_position(self):
        """Test long position creation."""
        position = create_long_position(
            symbol="THYAO",
            quantity=100,
            entry_price=285.50,
            commission=1.50,
            stop_loss=275.00,
            take_profit=300.00
        )
        
        assert position.symbol == "THYAO"
        assert position.side == PositionSide.LONG
        assert position.quantity == 100
        assert position.avg_entry_price == 285.50
        assert position.stop_loss == 275.00
        assert position.take_profit == 300.00
        assert position.is_long
        assert not position.is_short
    
    def test_create_short_position(self):
        """Test short position creation."""
        position = create_short_position(
            symbol="GARAN",
            quantity=200,
            entry_price=50.00,
            commission=1.00,
            stop_loss=55.00,
            take_profit=45.00
        )
        
        assert position.symbol == "GARAN"
        assert position.side == PositionSide.SHORT
        assert position.quantity == 200
        assert position.is_short
        assert not position.is_long
    
    def test_position_unrealized_pnl_long(self):
        """Test unrealized P&L calculation for long."""
        position = create_long_position(
            symbol="THYAO",
            quantity=100,
            entry_price=100.00,
            commission=1.00
        )
        
        # Update price to 110 (10% gain)
        position.update_price(110.00)
        
        assert position.unrealized_pnl == pytest.approx(1000.00)  # (110-100)*100
        assert position.unrealized_pnl_pct == pytest.approx(0.10)  # 10%
    
    def test_position_unrealized_pnl_short(self):
        """Test unrealized P&L calculation for short."""
        position = create_short_position(
            symbol="GARAN",
            quantity=100,
            entry_price=100.00,
            commission=1.00
        )
        
        # Update price to 90 (profit for short)
        position.update_price(90.00)
        
        assert position.unrealized_pnl == pytest.approx(1000.00)  # (100-90)*100
        assert position.unrealized_pnl_pct == pytest.approx(0.10)
    
    def test_position_stop_loss_hit_long(self):
        """Test stop loss detection for long position."""
        position = create_long_position(
            symbol="THYAO",
            quantity=100,
            entry_price=100.00,
            commission=1.00,
            stop_loss=95.00
        )
        
        # Price above stop - no trigger
        triggered, reason, price = position.should_stop_out(high=102, low=97)
        assert not triggered
        
        # Price hits stop
        triggered, reason, price = position.should_stop_out(high=100, low=94)
        assert triggered
        assert reason == ExitReason.STOP_LOSS
        assert price == 95.00
    
    def test_position_take_profit_hit_long(self):
        """Test take profit detection for long position."""
        position = create_long_position(
            symbol="THYAO",
            quantity=100,
            entry_price=100.00,
            commission=1.00,
            take_profit=110.00
        )
        
        # Price hits take profit
        triggered, reason, price = position.should_stop_out(high=112, low=105)
        assert triggered
        assert reason == ExitReason.TAKE_PROFIT
        assert price == 110.00
    
    def test_position_mfe_mae_tracking(self):
        """Test MFE/MAE tracking."""
        position = create_long_position(
            symbol="THYAO",
            quantity=100,
            entry_price=100.00,
            commission=1.00
        )
        
        # Price moves in favor
        position.update_price(110.00, high=115.00, low=100.00)
        assert position.max_favorable_excursion == pytest.approx(0.15)  # +15%
        
        # Price moves against
        position.update_price(95.00, high=110.00, low=92.00)
        assert position.max_adverse_excursion == pytest.approx(0.08)  # -8%
    
    def test_position_bars_held(self):
        """Test bars held counter."""
        position = create_long_position(
            symbol="THYAO",
            quantity=100,
            entry_price=100.00,
            commission=1.00
        )
        
        assert position.bars_held == 0
        
        position.increment_bars()
        assert position.bars_held == 1
        
        for _ in range(5):
            position.increment_bars()
        assert position.bars_held == 6


class TestTradeModel:
    """Test Trade model."""
    
    def test_trade_creation(self):
        """Test trade creation."""
        trade = Trade(
            symbol="THYAO",
            direction=TradeDirection.LONG,
            quantity=100,
            entry_price=100.00,
            exit_price=110.00,
            entry_time=datetime(2023, 1, 5, 10, 0),
            exit_time=datetime(2023, 1, 10, 15, 0),
            exit_reason=ExitReason.TAKE_PROFIT,
            entry_commission=1.00,
            exit_commission=1.10
        )
        
        assert trade.symbol == "THYAO"
        assert trade.direction == TradeDirection.LONG
        assert trade.quantity == 100
        assert trade.trade_id is not None
    
    def test_trade_pnl_calculation(self):
        """Test trade P&L calculation."""
        trade = Trade(
            symbol="THYAO",
            direction=TradeDirection.LONG,
            quantity=100,
            entry_price=100.00,
            exit_price=110.00,
            entry_time=datetime(2023, 1, 5, 10, 0),
            exit_time=datetime(2023, 1, 10, 15, 0),
            exit_reason=ExitReason.TAKE_PROFIT,
            entry_commission=1.00,
            exit_commission=1.10
        )
        
        # Gross: (110-100)*100 = 1000
        assert trade.gross_pnl == pytest.approx(1000.00)
        
        # Net: 1000 - 1.00 - 1.10 = 997.90
        assert trade.net_pnl == pytest.approx(997.90)
        
        # Percentage: 997.90 / (100*100) = 9.979%
        assert trade.pnl_pct == pytest.approx(0.09979)
    
    def test_trade_short_pnl(self):
        """Test short trade P&L calculation."""
        trade = Trade(
            symbol="GARAN",
            direction=TradeDirection.SHORT,
            quantity=100,
            entry_price=100.00,
            exit_price=90.00,  # Profitable short
            entry_time=datetime(2023, 1, 5, 10, 0),
            exit_time=datetime(2023, 1, 10, 15, 0),
            exit_reason=ExitReason.TAKE_PROFIT,
            entry_commission=1.00,
            exit_commission=0.90
        )
        
        # Gross: (100-90)*100 = 1000
        assert trade.gross_pnl == pytest.approx(1000.00)
    
    def test_trade_r_multiple(self):
        """Test R-multiple calculation."""
        trade = Trade(
            symbol="THYAO",
            direction=TradeDirection.LONG,
            quantity=100,
            entry_price=100.00,
            exit_price=110.00,
            entry_time=datetime(2023, 1, 5, 10, 0),
            exit_time=datetime(2023, 1, 10, 15, 0),
            exit_reason=ExitReason.TAKE_PROFIT,
            initial_stop_loss=95.00  # 5% risk
        )
        
        # Risk: 100 - 95 = 5 per share
        # Gain: 110 - 100 = 10 per share
        # R = 10 / 5 = 2.0
        assert trade.r_multiple == pytest.approx(2.0)
    
    def test_trade_holding_period(self):
        """Test holding period calculation."""
        trade = Trade(
            symbol="THYAO",
            direction=TradeDirection.LONG,
            quantity=100,
            entry_price=100.00,
            exit_price=110.00,
            entry_time=datetime(2023, 1, 5, 10, 0),
            exit_time=datetime(2023, 1, 10, 15, 0),
            exit_reason=ExitReason.TAKE_PROFIT
        )
        
        # 5 days and 5 hours = 125 hours
        assert trade.holding_hours == pytest.approx(125.0)
        assert trade.holding_days == pytest.approx(5.208, rel=0.01)
    
    def test_trade_is_winner_loser(self):
        """Test winner/loser classification."""
        winner = Trade(
            symbol="THYAO",
            direction=TradeDirection.LONG,
            quantity=100,
            entry_price=100.00,
            exit_price=110.00,
            entry_time=datetime(2023, 1, 5),
            exit_time=datetime(2023, 1, 10),
            exit_reason=ExitReason.TAKE_PROFIT
        )
        
        loser = Trade(
            symbol="GARAN",
            direction=TradeDirection.LONG,
            quantity=100,
            entry_price=100.00,
            exit_price=90.00,
            entry_time=datetime(2023, 1, 5),
            exit_time=datetime(2023, 1, 10),
            exit_reason=ExitReason.STOP_LOSS
        )
        
        assert winner.is_winner
        assert not winner.is_loser
        assert loser.is_loser
        assert not loser.is_winner


class TestTradeList:
    """Test TradeList model."""
    
    def test_tradelist_append(self, sample_trades):
        """Test trade list append."""
        assert len(sample_trades) == 3
    
    def test_tradelist_winners_losers(self, sample_trades):
        """Test winners/losers filtering."""
        winners = sample_trades.winners
        losers = sample_trades.losers
        
        assert len(winners) == 1
        assert len(losers) == 1
    
    def test_tradelist_win_rate(self, sample_trades):
        """Test win rate calculation."""
        # 1 winner, 1 loser, 1 breakeven = 33.33% win rate
        assert sample_trades.win_rate == pytest.approx(1/3)
    
    def test_tradelist_profit_factor(self, sample_trades):
        """Test profit factor calculation."""
        # Gross profit / Gross loss
        winners = sample_trades.winners
        losers = sample_trades.losers
        
        if losers:
            expected = sum(t.net_pnl for t in winners) / abs(sum(t.net_pnl for t in losers))
            assert sample_trades.profit_factor == pytest.approx(expected)
    
    def test_tradelist_total_pnl(self, sample_trades):
        """Test total P&L calculation."""
        expected = sum(t.net_pnl for t in sample_trades)
        assert sample_trades.total_pnl == pytest.approx(expected)
    
    def test_tradelist_expectancy(self, sample_trades):
        """Test expectancy (average P&L per trade)."""
        expected = sample_trades.total_pnl / len(sample_trades)
        assert sample_trades.expectancy == pytest.approx(expected)
    
    def test_tradelist_by_symbol(self, sample_trades):
        """Test filtering by symbol."""
        thyao_trades = sample_trades.by_symbol("THYAO")
        assert len(thyao_trades) == 1
        assert thyao_trades[0].symbol == "THYAO"
    
    def test_tradelist_to_dataframe(self, sample_trades):
        """Test conversion to DataFrame."""
        df = sample_trades.to_dataframe()
        
        assert len(df) == 3
        assert 'symbol' in df.columns
        assert 'net_pnl' in df.columns
        assert 'pnl_pct' in df.columns
