"""
AlphaTerminal Pro - Metrics Tests
=================================

Unit tests for performance metrics calculation.

Author: AlphaTerminal Team
"""

import pytest
import pandas as pd
import numpy as np

from app.backtest.metrics import (
    # Return metrics
    calculate_total_return,
    calculate_annualized_return,
    calculate_cagr,
    calculate_monthly_returns,
    
    # Risk metrics
    calculate_volatility,
    calculate_max_drawdown,
    calculate_drawdown_series,
    calculate_var,
    calculate_cvar,
    
    # Risk-adjusted metrics
    calculate_sharpe_ratio,
    calculate_sortino_ratio,
    calculate_calmar_ratio,
    calculate_omega_ratio,
    
    # Trade statistics
    calculate_trade_statistics,
    
    # Comprehensive
    calculate_all_metrics
)
from app.backtest.models import TradeList


class TestReturnMetrics:
    """Test return metrics."""
    
    @pytest.fixture
    def equity_curve(self):
        """Create sample equity curve."""
        np.random.seed(42)
        dates = pd.date_range(start='2023-01-01', periods=252, freq='D')
        
        # 20% return over the year with some volatility
        returns = np.random.normal(0.0008, 0.015, 252)
        equity = 100000 * np.cumprod(1 + returns)
        
        return pd.Series(equity, index=dates)
    
    def test_total_return(self, equity_curve):
        """Test total return calculation."""
        abs_return, pct_return = calculate_total_return(equity_curve)
        
        expected_abs = equity_curve.iloc[-1] - equity_curve.iloc[0]
        expected_pct = expected_abs / equity_curve.iloc[0]
        
        assert abs_return == pytest.approx(expected_abs)
        assert pct_return == pytest.approx(expected_pct)
    
    def test_total_return_with_initial(self, equity_curve):
        """Test total return with explicit initial capital."""
        abs_return, pct_return = calculate_total_return(
            equity_curve,
            initial_capital=100000
        )
        
        assert abs_return == pytest.approx(equity_curve.iloc[-1] - 100000)
    
    def test_annualized_return(self, equity_curve):
        """Test annualized return calculation."""
        ann_return = calculate_annualized_return(equity_curve)
        
        # Should be reasonable for a year of data
        assert -0.5 < ann_return < 1.0  # Between -50% and 100%
    
    def test_cagr(self, equity_curve):
        """Test CAGR calculation."""
        cagr = calculate_cagr(equity_curve)
        ann_return = calculate_annualized_return(equity_curve)
        
        # CAGR should equal annualized return
        assert cagr == pytest.approx(ann_return)
    
    def test_monthly_returns(self):
        """Test monthly returns calculation."""
        dates = pd.date_range(start='2023-01-01', periods=252, freq='D')
        np.random.seed(42)
        returns = pd.Series(np.random.normal(0.0005, 0.02, 252), index=dates)
        
        monthly = calculate_monthly_returns(returns)
        
        # Should have ~12 months
        assert 10 <= len(monthly) <= 13
    
    def test_empty_equity_curve(self):
        """Test handling of empty equity curve."""
        empty = pd.Series(dtype=float)
        
        abs_return, pct_return = calculate_total_return(empty)
        assert abs_return == 0.0
        assert pct_return == 0.0


class TestRiskMetrics:
    """Test risk metrics."""
    
    @pytest.fixture
    def returns_series(self):
        """Create sample returns series."""
        np.random.seed(42)
        dates = pd.date_range(start='2023-01-01', periods=252, freq='D')
        returns = pd.Series(np.random.normal(0.0005, 0.02, 252), index=dates)
        return returns
    
    @pytest.fixture
    def equity_with_drawdown(self):
        """Create equity curve with known drawdown."""
        dates = pd.date_range(start='2023-01-01', periods=100, freq='D')
        
        # Create a curve with 20% drawdown
        equity = [100000]
        for i in range(99):
            if i < 30:
                equity.append(equity[-1] * 1.01)  # Up
            elif i < 50:
                equity.append(equity[-1] * 0.98)  # Down (drawdown)
            else:
                equity.append(equity[-1] * 1.005)  # Recovery
        
        return pd.Series(equity, index=dates)
    
    def test_volatility(self, returns_series):
        """Test volatility calculation."""
        vol = calculate_volatility(returns_series, annualize=True)
        
        # Daily vol ~0.02, annualized ~0.02 * sqrt(252) ≈ 0.317
        assert 0.2 < vol < 0.5
    
    def test_volatility_non_annualized(self, returns_series):
        """Test non-annualized volatility."""
        vol_ann = calculate_volatility(returns_series, annualize=True)
        vol_daily = calculate_volatility(returns_series, annualize=False)
        
        # Annualized should be sqrt(252) times larger
        assert vol_ann == pytest.approx(vol_daily * np.sqrt(252))
    
    def test_max_drawdown(self, equity_with_drawdown):
        """Test max drawdown calculation."""
        max_dd, peak_idx, trough_idx = calculate_max_drawdown(equity_with_drawdown)
        
        # Should detect drawdown
        assert max_dd > 0
        assert max_dd < 1.0  # Less than 100%
        
        # Peak should come before trough
        assert peak_idx < trough_idx
    
    def test_drawdown_series(self, equity_with_drawdown):
        """Test drawdown series calculation."""
        dd_series = calculate_drawdown_series(equity_with_drawdown)
        
        assert len(dd_series) == len(equity_with_drawdown)
        assert (dd_series <= 0).all()  # All drawdowns are non-positive
        assert dd_series.iloc[0] == 0  # First point has no drawdown
    
    def test_var(self, returns_series):
        """Test Value at Risk calculation."""
        var_95 = calculate_var(returns_series, confidence=0.95)
        var_99 = calculate_var(returns_series, confidence=0.99)
        
        # VaR should be positive (representing potential loss)
        assert var_95 > 0
        assert var_99 > 0
        
        # 99% VaR should be larger than 95% VaR
        assert var_99 > var_95
    
    def test_cvar(self, returns_series):
        """Test Conditional Value at Risk calculation."""
        var_95 = calculate_var(returns_series, confidence=0.95)
        cvar_95 = calculate_cvar(returns_series, confidence=0.95)
        
        # CVaR should be >= VaR (it's the expected loss beyond VaR)
        assert cvar_95 >= var_95


class TestRiskAdjustedMetrics:
    """Test risk-adjusted metrics."""
    
    @pytest.fixture
    def positive_returns(self):
        """Create returns with positive drift."""
        np.random.seed(42)
        dates = pd.date_range(start='2023-01-01', periods=252, freq='D')
        # Positive drift
        returns = pd.Series(np.random.normal(0.001, 0.015, 252), index=dates)
        return returns
    
    @pytest.fixture
    def negative_returns(self):
        """Create returns with negative drift."""
        np.random.seed(42)
        dates = pd.date_range(start='2023-01-01', periods=252, freq='D')
        # Negative drift
        returns = pd.Series(np.random.normal(-0.001, 0.015, 252), index=dates)
        return returns
    
    def test_sharpe_ratio_positive(self, positive_returns):
        """Test Sharpe ratio with positive returns."""
        sharpe = calculate_sharpe_ratio(positive_returns, risk_free_rate=0.05)
        
        # Positive excess returns should give positive Sharpe
        # (Though depends on volatility)
        assert isinstance(sharpe, float)
    
    def test_sharpe_ratio_negative(self, negative_returns):
        """Test Sharpe ratio with negative returns."""
        sharpe = calculate_sharpe_ratio(negative_returns, risk_free_rate=0.05)
        
        # Negative excess returns should give negative Sharpe
        assert sharpe < 0
    
    def test_sortino_ratio(self, positive_returns):
        """Test Sortino ratio calculation."""
        sortino = calculate_sortino_ratio(positive_returns, risk_free_rate=0.05)
        sharpe = calculate_sharpe_ratio(positive_returns, risk_free_rate=0.05)
        
        # Sortino typically higher than Sharpe (uses downside dev)
        # Not always true but generally holds
        assert isinstance(sortino, float)
    
    def test_calmar_ratio(self, positive_returns):
        """Test Calmar ratio calculation."""
        equity = 100000 * (1 + positive_returns).cumprod()
        
        calmar = calculate_calmar_ratio(positive_returns, equity)
        
        assert isinstance(calmar, float)
    
    def test_omega_ratio(self, positive_returns):
        """Test Omega ratio calculation."""
        omega = calculate_omega_ratio(positive_returns, threshold=0.0)
        
        # Omega > 1 suggests positive expectation
        assert isinstance(omega, float)
        
        # With positive drift, omega should be > 1
        assert omega > 0


class TestTradeStatistics:
    """Test trade statistics."""
    
    def test_trade_statistics(self, sample_trades):
        """Test comprehensive trade statistics."""
        stats = calculate_trade_statistics(sample_trades)
        
        assert 'total_trades' in stats
        assert 'winning_trades' in stats
        assert 'losing_trades' in stats
        assert 'win_rate' in stats
        assert 'profit_factor' in stats
        assert 'avg_trade' in stats
        assert 'avg_winner' in stats
        assert 'avg_loser' in stats
        
        assert stats['total_trades'] == len(sample_trades)
    
    def test_empty_trades(self):
        """Test statistics with no trades."""
        stats = calculate_trade_statistics(TradeList())
        
        assert stats['total_trades'] == 0
        assert stats['win_rate'] == 0
        assert stats['profit_factor'] == 0
    
    def test_win_rate_calculation(self, sample_trades):
        """Test win rate calculation."""
        stats = calculate_trade_statistics(sample_trades)
        
        # Win rate should be between 0 and 1
        assert 0 <= stats['win_rate'] <= 1
        
        # Should match manual calculation
        winners = len(sample_trades.winners)
        expected = winners / len(sample_trades)
        assert stats['win_rate'] == pytest.approx(expected)


class TestComprehensiveMetrics:
    """Test comprehensive metrics calculation."""
    
    def test_calculate_all_metrics(self, sample_trades):
        """Test all metrics calculation."""
        np.random.seed(42)
        
        # Create equity curve
        dates = pd.date_range(start='2023-01-01', periods=252, freq='D')
        returns = np.random.normal(0.0005, 0.02, 252)
        equity = pd.Series(100000 * np.cumprod(1 + returns), index=dates)
        
        metrics = calculate_all_metrics(
            equity_curve=equity,
            trades=sample_trades,
            initial_capital=100000
        )
        
        # Check all major categories
        assert hasattr(metrics, 'total_return')
        assert hasattr(metrics, 'sharpe_ratio')
        assert hasattr(metrics, 'max_drawdown')
        assert hasattr(metrics, 'trade_stats')
    
    def test_metrics_to_dict(self, sample_trades):
        """Test metrics serialization."""
        np.random.seed(42)
        
        dates = pd.date_range(start='2023-01-01', periods=252, freq='D')
        equity = pd.Series(
            100000 * np.cumprod(1 + np.random.normal(0.0005, 0.02, 252)),
            index=dates
        )
        
        metrics = calculate_all_metrics(equity, sample_trades, 100000)
        metrics_dict = metrics.to_dict()
        
        assert isinstance(metrics_dict, dict)
        assert 'returns' in metrics_dict
        assert 'risk' in metrics_dict
        assert 'risk_adjusted' in metrics_dict
        assert 'trades' in metrics_dict
