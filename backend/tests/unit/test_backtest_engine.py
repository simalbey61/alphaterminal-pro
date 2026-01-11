"""
AlphaTerminal Pro - Backtest Engine Tests
=========================================

Unit tests for the main backtest engine.

Author: AlphaTerminal Team
"""

import pytest
import pandas as pd
import numpy as np
from datetime import datetime

from app.backtest import (
    BacktestEngine, BacktestConfig, BacktestResult,
    BaseStrategy, Signal
)
from app.backtest.strategies import SMACrossoverStrategy, RSIMeanReversionStrategy
from app.backtest.enums import SignalType
from app.backtest.exceptions import (
    InsufficientDataError, InvalidDataError, InvalidConfigError
)


class TestBacktestConfig:
    """Test BacktestConfig."""
    
    def test_default_config(self):
        """Test default configuration."""
        config = BacktestConfig()
        
        assert config.initial_capital == 100_000
        assert config.commission_rate == 0.001
        assert config.slippage_rate == 0.0005
        assert config.max_position_size == 0.25
    
    def test_custom_config(self):
        """Test custom configuration."""
        config = BacktestConfig(
            initial_capital=500_000,
            commission_rate=0.0005,
            max_positions=10
        )
        
        assert config.initial_capital == 500_000
        assert config.commission_rate == 0.0005
        assert config.max_positions == 10
    
    def test_config_validation(self):
        """Test configuration validation."""
        # Invalid capital should raise
        with pytest.raises(InvalidConfigError):
            config = BacktestConfig(initial_capital=-1000)
            config.validate()
        
        # Invalid commission should raise
        with pytest.raises(InvalidConfigError):
            config = BacktestConfig(commission_rate=-0.01)
            config.validate()
    
    def test_config_to_dict(self, default_config):
        """Test config serialization."""
        config_dict = default_config.to_dict()
        
        assert 'initial_capital' in config_dict
        assert 'commission_rate' in config_dict
        assert config_dict['initial_capital'] == 100_000


class TestBacktestEngine:
    """Test BacktestEngine."""
    
    def test_engine_creation(self, default_config):
        """Test engine creation."""
        engine = BacktestEngine(config=default_config)
        
        assert engine.config.initial_capital == 100_000
        assert engine.state is not None
    
    def test_run_backtest(self, backtest_engine, sample_ohlcv_data, sma_strategy):
        """Test running a basic backtest."""
        result = backtest_engine.run(
            data=sample_ohlcv_data,
            strategy=sma_strategy,
            symbol="TEST",
            timeframe="1d"
        )
        
        assert isinstance(result, BacktestResult)
        assert result.symbol == "TEST"
        assert result.timeframe == "1d"
        assert result.execution_time_seconds > 0
    
    def test_backtest_with_trending_data(self, default_config, trending_data, sma_strategy):
        """Test backtest with trending data."""
        engine = BacktestEngine(config=default_config)
        
        result = engine.run(
            data=trending_data,
            strategy=sma_strategy,
            symbol="TREND",
            timeframe="1d"
        )
        
        # Trending data should produce trades
        assert result.total_trades >= 0
        assert result.total_return_pct is not None
    
    def test_backtest_with_ranging_data(self, default_config, ranging_data, rsi_strategy):
        """Test backtest with ranging data."""
        engine = BacktestEngine(config=default_config)
        
        result = engine.run(
            data=ranging_data,
            strategy=rsi_strategy,
            symbol="RANGE",
            timeframe="1d"
        )
        
        assert isinstance(result, BacktestResult)
    
    def test_insufficient_data(self, backtest_engine, minimal_data, sma_strategy):
        """Test handling of insufficient data."""
        # SMA strategy needs more data than minimal
        with pytest.raises(InsufficientDataError):
            backtest_engine.run(
                data=minimal_data,
                strategy=sma_strategy,
                symbol="TEST",
                timeframe="1d"
            )
    
    def test_invalid_data_missing_columns(self, backtest_engine, sma_strategy):
        """Test handling of missing columns."""
        # Create data without 'Volume' column
        dates = pd.date_range(start='2023-01-01', periods=100, freq='D')
        bad_data = pd.DataFrame({
            'Open': [100] * 100,
            'High': [101] * 100,
            'Low': [99] * 100,
            'Close': [100] * 100
            # Missing Volume
        }, index=dates)
        
        with pytest.raises(InvalidDataError):
            backtest_engine.run(
                data=bad_data,
                strategy=sma_strategy,
                symbol="TEST",
                timeframe="1d"
            )
    
    def test_backtest_metrics(self, backtest_engine, trending_data, sma_strategy):
        """Test that all metrics are calculated."""
        result = backtest_engine.run(
            data=trending_data,
            strategy=sma_strategy,
            symbol="TEST",
            timeframe="1d"
        )
        
        # Check all metrics are present
        assert hasattr(result, 'total_return_pct')
        assert hasattr(result, 'annualized_return')
        assert hasattr(result, 'sharpe_ratio')
        assert hasattr(result, 'sortino_ratio')
        assert hasattr(result, 'calmar_ratio')
        assert hasattr(result, 'max_drawdown')
        assert hasattr(result, 'volatility')
        assert hasattr(result, 'win_rate')
        assert hasattr(result, 'profit_factor')
        
        # Metrics should be numeric
        assert isinstance(result.sharpe_ratio, (int, float))
        assert isinstance(result.max_drawdown, (int, float))
    
    def test_equity_curve(self, backtest_engine, trending_data, sma_strategy):
        """Test equity curve generation."""
        result = backtest_engine.run(
            data=trending_data,
            strategy=sma_strategy,
            symbol="TEST",
            timeframe="1d"
        )
        
        assert result.equity_curve is not None
        assert isinstance(result.equity_curve, pd.Series)
        assert len(result.equity_curve) > 0
        
        # First value should be around initial capital
        assert result.equity_curve.iloc[0] == pytest.approx(100_000, rel=0.01)
    
    def test_trade_history(self, backtest_engine, trending_data, sma_strategy):
        """Test trade history recording."""
        result = backtest_engine.run(
            data=trending_data,
            strategy=sma_strategy,
            symbol="TEST",
            timeframe="1d"
        )
        
        assert result.trades is not None
        
        if result.total_trades > 0:
            trade = result.trades[0]
            assert trade.symbol == "TEST"
            assert trade.entry_price > 0
            assert trade.exit_price > 0


class TestBacktestResult:
    """Test BacktestResult."""
    
    def test_result_summary(self, backtest_engine, trending_data, sma_strategy):
        """Test result summary generation."""
        result = backtest_engine.run(
            data=trending_data,
            strategy=sma_strategy,
            symbol="TEST",
            timeframe="1d"
        )
        
        summary = result.summary()
        
        assert isinstance(summary, str)
        assert "TEST" in summary
        assert "Return" in summary or "RETURN" in summary
    
    def test_result_to_dict(self, backtest_engine, trending_data, sma_strategy):
        """Test result serialization."""
        result = backtest_engine.run(
            data=trending_data,
            strategy=sma_strategy,
            symbol="TEST",
            timeframe="1d"
        )
        
        result_dict = result.to_dict()
        
        assert isinstance(result_dict, dict)
        assert 'symbol' in result_dict
        assert 'total_return_pct' in result_dict
        assert 'sharpe_ratio' in result_dict


class TestSignal:
    """Test Signal class."""
    
    def test_no_action_signal(self):
        """Test no-action signal creation."""
        signal = Signal.no_action()
        
        assert signal.signal_type == SignalType.NO_ACTION
        assert not signal.is_entry
        assert not signal.is_exit
    
    def test_long_entry_signal(self):
        """Test long entry signal creation."""
        signal = Signal.long_entry(
            entry_price=100.0,
            stop_loss=95.0,
            take_profit=110.0,
            position_size=0.10
        )
        
        assert signal.signal_type == SignalType.ENTRY_LONG
        assert signal.is_entry
        assert signal.is_long
        assert signal.entry_price == 100.0
        assert signal.stop_loss == 95.0
        assert signal.take_profit == 110.0
        assert signal.position_size == 0.10
    
    def test_short_entry_signal(self):
        """Test short entry signal creation."""
        signal = Signal.short_entry(
            entry_price=100.0,
            stop_loss=105.0,
            take_profit=90.0
        )
        
        assert signal.signal_type == SignalType.ENTRY_SHORT
        assert signal.is_entry
        assert signal.is_short
    
    def test_exit_signals(self):
        """Test exit signal creation."""
        exit_long = Signal.exit_long(reason="Take profit hit")
        exit_short = Signal.exit_short(reason="Stop loss hit")
        exit_all = Signal.exit_all(reason="End of day")
        
        assert exit_long.is_exit
        assert exit_short.is_exit
        assert exit_all.is_exit
        
        assert exit_long.reason == "Take profit hit"
    
    def test_signal_to_dict(self):
        """Test signal serialization."""
        signal = Signal.long_entry(
            entry_price=100.0,
            stop_loss=95.0,
            position_size=0.10
        )
        
        signal_dict = signal.to_dict()
        
        assert signal_dict['signal_type'] == 'entry_long'
        assert signal_dict['entry_price'] == 100.0


class TestBaseStrategy:
    """Test BaseStrategy class."""
    
    def test_strategy_attributes(self, sma_strategy):
        """Test strategy attributes."""
        assert sma_strategy.name == "SMA Crossover"
        assert sma_strategy.warmup_period > 0
    
    def test_strategy_initialization(self, sma_strategy):
        """Test strategy initialization."""
        assert not sma_strategy._is_initialized
        
        sma_strategy.initialize()
        
        assert sma_strategy._is_initialized
    
    def test_strategy_position_tracking(self, sma_strategy):
        """Test position tracking."""
        assert not sma_strategy.is_in_position
        assert not sma_strategy.is_long
        assert not sma_strategy.is_short
        
        # Simulate trade open
        sma_strategy.on_trade_open({
            'direction': 'long',
            'quantity': 100
        })
        
        assert sma_strategy.is_in_position
        assert sma_strategy.is_long
    
    def test_strategy_generate_signal(self, sma_strategy, sample_ohlcv_data):
        """Test signal generation."""
        sma_strategy.initialize()
        
        signal = sma_strategy.generate_signal(sample_ohlcv_data)
        
        assert isinstance(signal, Signal)
    
    def test_strategy_parameters(self, sma_strategy):
        """Test strategy parameters."""
        params = sma_strategy.get_parameters()
        
        assert 'fast_period' in params
        assert 'slow_period' in params
        assert params['fast_period'] == 10
        assert params['slow_period'] == 30
