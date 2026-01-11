"""
AlphaTerminal Pro - Backtest Integration Tests
==============================================

Integration tests for full backtest workflows.

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
from app.backtest.strategies import (
    SMACrossoverStrategy,
    RSIMeanReversionStrategy
)
from app.backtest.models import Trade, TradeList
from app.backtest.enums import TradeDirection, ExitReason
from app.backtest.metrics import calculate_all_metrics


class TestFullBacktestWorkflow:
    """Test complete backtest workflow."""
    
    def test_sma_strategy_full_workflow(self, trending_data, default_config):
        """Test SMA strategy from start to finish."""
        strategy = SMACrossoverStrategy(
            fast_period=10,
            slow_period=30,
            atr_multiplier=2.0,
            risk_reward=2.0
        )
        
        engine = BacktestEngine(config=default_config)
        
        result = engine.run(
            data=trending_data,
            strategy=strategy,
            symbol="THYAO",
            timeframe="1d"
        )
        
        assert isinstance(result, BacktestResult)
        assert result.symbol == "THYAO"
        assert result.timeframe == "1d"
        assert result.total_return_pct is not None
        assert result.sharpe_ratio is not None
        assert result.max_drawdown is not None
        assert result.equity_curve is not None
        assert len(result.equity_curve) > 0
    
    def test_rsi_strategy_full_workflow(self, ranging_data, default_config):
        """Test RSI strategy from start to finish."""
        strategy = RSIMeanReversionStrategy(
            rsi_period=14,
            oversold=30,
            overbought=70,
            use_trend_filter=False
        )
        
        engine = BacktestEngine(config=default_config)
        
        result = engine.run(
            data=ranging_data,
            strategy=strategy,
            symbol="GARAN",
            timeframe="1d"
        )
        
        assert isinstance(result, BacktestResult)
        assert result.symbol == "GARAN"
    
    def test_multiple_strategies_comparison(self, sample_ohlcv_data, default_config):
        """Test running multiple strategies on same data."""
        strategies = [
            SMACrossoverStrategy(fast_period=5, slow_period=20),
            SMACrossoverStrategy(fast_period=10, slow_period=30),
            SMACrossoverStrategy(fast_period=20, slow_period=50),
        ]
        
        results = []
        engine = BacktestEngine(config=default_config)
        
        for strategy in strategies:
            result = engine.run(
                data=sample_ohlcv_data.copy(),
                strategy=strategy,
                symbol="TEST",
                timeframe="1d"
            )
            results.append(result)
        
        assert len(results) == 3
        for result in results:
            assert result.sharpe_ratio is not None
            assert result.max_drawdown >= 0


class TestConfigurationVariations:
    """Test different configuration scenarios."""
    
    def test_conservative_config(self, sample_ohlcv_data, conservative_config, sma_strategy):
        """Test with conservative configuration."""
        engine = BacktestEngine(config=conservative_config)
        
        result = engine.run(
            data=sample_ohlcv_data,
            strategy=sma_strategy,
            symbol="TEST",
            timeframe="1d"
        )
        
        assert result is not None
        assert conservative_config.max_position_size == 0.10
    
    def test_aggressive_config(self, sample_ohlcv_data, aggressive_config, sma_strategy):
        """Test with aggressive configuration."""
        engine = BacktestEngine(config=aggressive_config)
        
        result = engine.run(
            data=sample_ohlcv_data,
            strategy=sma_strategy,
            symbol="TEST",
            timeframe="1d"
        )
        
        assert result is not None
        assert aggressive_config.max_position_size == 0.50


class TestEdgeCases:
    """Test edge cases and error handling."""
    
    def test_no_trades_generated(self, default_config):
        """Test when strategy generates no trades."""
        class NoTradeStrategy(BaseStrategy):
            name = "No Trade Strategy"
            warmup_period = 10
            
            def generate_signal(self, data):
                return Signal.no_action()
        
        dates = pd.date_range(start='2023-01-01', periods=100, freq='D')
        data = pd.DataFrame({
            'Open': [100] * 100,
            'High': [101] * 100,
            'Low': [99] * 100,
            'Close': [100] * 100,
            'Volume': [1000000] * 100
        }, index=dates)
        
        engine = BacktestEngine(config=default_config)
        result = engine.run(data, NoTradeStrategy(), "TEST", "1d")
        
        assert result.total_trades == 0
        assert result.total_return_pct == pytest.approx(0, abs=0.001)
    
    def test_volatile_market(self, volatile_data, default_config, sma_strategy):
        """Test in highly volatile market conditions."""
        engine = BacktestEngine(config=default_config)
        
        result = engine.run(
            data=volatile_data,
            strategy=sma_strategy,
            symbol="VOLATILE",
            timeframe="1d"
        )
        
        assert result is not None
        assert result.max_drawdown >= 0


class TestMetricsIntegration:
    """Test metrics integration with backtest results."""
    
    def test_metrics_consistency(self, trending_data, default_config, sma_strategy):
        """Test that metrics are consistent."""
        engine = BacktestEngine(config=default_config)
        
        result = engine.run(
            data=trending_data,
            strategy=sma_strategy,
            symbol="TEST",
            timeframe="1d"
        )
        
        assert 0 <= result.max_drawdown <= 1
        assert 0 <= result.win_rate <= 1
    
    def test_trade_statistics_match(self, trending_data, default_config, sma_strategy):
        """Test that trade statistics match trade list."""
        engine = BacktestEngine(config=default_config)
        
        result = engine.run(
            data=trending_data,
            strategy=sma_strategy,
            symbol="TEST",
            timeframe="1d"
        )
        
        assert result.total_trades == len(result.trades)
        assert result.winning_trades + result.losing_trades <= result.total_trades


class TestDataIntegrity:
    """Test data handling and integrity."""
    
    def test_data_not_modified(self, sample_ohlcv_data, default_config, sma_strategy):
        """Test that original data is not modified."""
        original_data = sample_ohlcv_data.copy()
        
        engine = BacktestEngine(config=default_config)
        engine.run(
            data=sample_ohlcv_data,
            strategy=sma_strategy,
            symbol="TEST",
            timeframe="1d"
        )
        
        pd.testing.assert_frame_equal(sample_ohlcv_data, original_data)
    
    def test_date_range_filtering(self, sample_ohlcv_data, default_config, sma_strategy):
        """Test date range filtering."""
        engine = BacktestEngine(config=default_config)
        
        start_date = sample_ohlcv_data.index[50]
        end_date = sample_ohlcv_data.index[150]
        
        result = engine.run(
            data=sample_ohlcv_data,
            strategy=sma_strategy,
            symbol="TEST",
            timeframe="1d",
            start_date=start_date,
            end_date=end_date
        )
        
        assert result.start_date >= start_date
        assert result.end_date <= end_date
