"""
AlphaTerminal Pro - Test Configuration
======================================

Pytest fixtures and configuration for all tests.

Author: AlphaTerminal Team
Version: 1.0.0
"""

import sys
import os
from datetime import datetime, timedelta
from typing import Generator

import pytest
import pandas as pd
import numpy as np

# Add project root to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))


# =============================================================================
# DATA FIXTURES
# =============================================================================

@pytest.fixture
def sample_ohlcv_data() -> pd.DataFrame:
    """Create sample OHLCV data for testing."""
    np.random.seed(42)
    periods = 252  # 1 year of daily data
    
    dates = pd.date_range(start='2023-01-01', periods=periods, freq='D')
    
    # Generate realistic price series
    returns = np.random.normal(0.0005, 0.02, periods)
    close = 100 * np.cumprod(1 + returns)
    
    data = pd.DataFrame({
        'Open': close * (1 + np.random.uniform(-0.01, 0.01, periods)),
        'High': close * (1 + np.abs(np.random.normal(0, 0.015, periods))),
        'Low': close * (1 - np.abs(np.random.normal(0, 0.015, periods))),
        'Close': close,
        'Volume': np.random.randint(100000, 2000000, periods)
    }, index=dates)
    
    # Fix OHLC consistency
    data['High'] = data[['Open', 'Close', 'High']].max(axis=1)
    data['Low'] = data[['Open', 'Close', 'Low']].min(axis=1)
    
    return data


@pytest.fixture
def trending_data() -> pd.DataFrame:
    """Create uptrending data for testing trend-following strategies."""
    np.random.seed(42)
    periods = 300
    
    dates = pd.date_range(start='2023-01-01', periods=periods, freq='D')
    
    # Strong uptrend with noise
    trend = np.linspace(0, 50, periods)  # 50% gain over period
    noise = np.random.normal(0, 2, periods)
    close = 100 + trend + np.cumsum(noise * 0.1)
    
    data = pd.DataFrame({
        'Open': np.roll(close, 1),
        'High': close * 1.01,
        'Low': close * 0.99,
        'Close': close,
        'Volume': np.random.randint(500000, 1500000, periods)
    }, index=dates)
    
    data['Open'].iloc[0] = 100
    data['High'] = data[['Open', 'Close', 'High']].max(axis=1)
    data['Low'] = data[['Open', 'Close', 'Low']].min(axis=1)
    
    return data


@pytest.fixture
def ranging_data() -> pd.DataFrame:
    """Create ranging/sideways data for testing mean reversion strategies."""
    np.random.seed(42)
    periods = 300
    
    dates = pd.date_range(start='2023-01-01', periods=periods, freq='D')
    
    # Mean-reverting price around 100
    prices = [100]
    for _ in range(periods - 1):
        shock = np.random.normal(0, 1.5)
        reversion = 0.1 * (100 - prices[-1])
        prices.append(prices[-1] + shock + reversion)
    
    close = np.array(prices)
    
    data = pd.DataFrame({
        'Open': np.roll(close, 1),
        'High': close * (1 + np.abs(np.random.normal(0, 0.01, periods))),
        'Low': close * (1 - np.abs(np.random.normal(0, 0.01, periods))),
        'Close': close,
        'Volume': np.random.randint(500000, 1500000, periods)
    }, index=dates)
    
    data['Open'].iloc[0] = 100
    
    return data


@pytest.fixture
def volatile_data() -> pd.DataFrame:
    """Create highly volatile data for stress testing."""
    np.random.seed(42)
    periods = 200
    
    dates = pd.date_range(start='2023-01-01', periods=periods, freq='D')
    
    # High volatility returns
    returns = np.random.normal(0, 0.05, periods)  # 5% daily volatility
    close = 100 * np.cumprod(1 + returns)
    
    data = pd.DataFrame({
        'Open': close * (1 + np.random.uniform(-0.02, 0.02, periods)),
        'High': close * (1 + np.abs(np.random.normal(0, 0.03, periods))),
        'Low': close * (1 - np.abs(np.random.normal(0, 0.03, periods))),
        'Close': close,
        'Volume': np.random.randint(1000000, 5000000, periods)
    }, index=dates)
    
    data['High'] = data[['Open', 'Close', 'High']].max(axis=1)
    data['Low'] = data[['Open', 'Close', 'Low']].min(axis=1)
    
    return data


@pytest.fixture
def minimal_data() -> pd.DataFrame:
    """Create minimal data for edge case testing."""
    dates = pd.date_range(start='2023-01-01', periods=10, freq='D')
    
    return pd.DataFrame({
        'Open': [100, 101, 102, 101, 100, 99, 100, 101, 102, 103],
        'High': [101, 102, 103, 102, 101, 100, 101, 102, 103, 104],
        'Low': [99, 100, 101, 100, 99, 98, 99, 100, 101, 102],
        'Close': [100.5, 101.5, 102.5, 101.5, 100.5, 99.5, 100.5, 101.5, 102.5, 103.5],
        'Volume': [100000] * 10
    }, index=dates)


# =============================================================================
# CONFIG FIXTURES
# =============================================================================

@pytest.fixture
def default_config():
    """Default backtest configuration."""
    from app.backtest import BacktestConfig
    
    return BacktestConfig(
        initial_capital=100_000,
        commission_rate=0.001,
        slippage_rate=0.0005,
        max_position_size=0.20,
        max_positions=5,
        risk_per_trade=0.02,
        allow_shorting=False,
        log_trades=False
    )


@pytest.fixture
def aggressive_config():
    """Aggressive configuration for stress testing."""
    from app.backtest import BacktestConfig
    
    return BacktestConfig(
        initial_capital=50_000,
        commission_rate=0.002,
        slippage_rate=0.001,
        max_position_size=0.50,
        max_positions=3,
        risk_per_trade=0.05,
        allow_shorting=True,
        log_trades=False
    )


@pytest.fixture
def conservative_config():
    """Conservative configuration."""
    from app.backtest import BacktestConfig
    
    return BacktestConfig(
        initial_capital=500_000,
        commission_rate=0.0005,
        slippage_rate=0.0002,
        max_position_size=0.10,
        max_positions=10,
        risk_per_trade=0.01,
        allow_shorting=False,
        log_trades=False
    )


# =============================================================================
# STRATEGY FIXTURES
# =============================================================================

@pytest.fixture
def sma_strategy():
    """SMA Crossover strategy."""
    from app.backtest.strategies import SMACrossoverStrategy
    
    return SMACrossoverStrategy(
        fast_period=10,
        slow_period=30,
        atr_period=14,
        atr_multiplier=2.0,
        risk_reward=2.0,
        position_size=0.10
    )


@pytest.fixture
def rsi_strategy():
    """RSI Mean Reversion strategy."""
    from app.backtest.strategies import RSIMeanReversionStrategy
    
    return RSIMeanReversionStrategy(
        rsi_period=14,
        oversold=30,
        overbought=70,
        sma_period=50,
        use_trend_filter=True,
        atr_period=14,
        atr_stop_mult=2.0,
        atr_tp_mult=3.0,
        position_size=0.10
    )


# =============================================================================
# ENGINE FIXTURES
# =============================================================================

@pytest.fixture
def backtest_engine(default_config):
    """Create backtest engine with default config."""
    from app.backtest import BacktestEngine
    
    return BacktestEngine(config=default_config)


# =============================================================================
# TRADE FIXTURES
# =============================================================================

@pytest.fixture
def sample_trades():
    """Create sample trades for testing."""
    from app.backtest.models import Trade, TradeList
    from app.backtest.enums import TradeDirection, ExitReason
    
    trades = TradeList()
    
    # Winning trade
    trades.append(Trade(
        symbol="THYAO",
        direction=TradeDirection.LONG,
        quantity=100,
        entry_price=100.0,
        exit_price=110.0,
        entry_time=datetime(2023, 1, 5, 10, 0),
        exit_time=datetime(2023, 1, 10, 15, 0),
        exit_reason=ExitReason.TAKE_PROFIT,
        entry_commission=5.0,
        exit_commission=5.5,
        initial_stop_loss=95.0,
        initial_take_profit=110.0
    ))
    
    # Losing trade
    trades.append(Trade(
        symbol="GARAN",
        direction=TradeDirection.LONG,
        quantity=200,
        entry_price=50.0,
        exit_price=47.0,
        entry_time=datetime(2023, 1, 15, 10, 0),
        exit_time=datetime(2023, 1, 18, 11, 0),
        exit_reason=ExitReason.STOP_LOSS,
        entry_commission=5.0,
        exit_commission=4.7,
        initial_stop_loss=47.0
    ))
    
    # Breakeven trade
    trades.append(Trade(
        symbol="ASELS",
        direction=TradeDirection.LONG,
        quantity=50,
        entry_price=200.0,
        exit_price=200.5,
        entry_time=datetime(2023, 2, 1, 10, 0),
        exit_time=datetime(2023, 2, 5, 14, 0),
        exit_reason=ExitReason.SIGNAL,
        entry_commission=5.0,
        exit_commission=5.0
    ))
    
    return trades


# =============================================================================
# UTILITY FIXTURES
# =============================================================================

@pytest.fixture
def mock_logger(mocker):
    """Mock logger for testing log outputs."""
    return mocker.patch('app.backtest.engine.core.logger')


# =============================================================================
# CLEANUP
# =============================================================================

@pytest.fixture(autouse=True)
def reset_random_seed():
    """Reset random seed before each test for reproducibility."""
    np.random.seed(42)
    yield
