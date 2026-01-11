"""
AlphaTerminal Pro - Performance Metrics
=======================================

Performance and risk metrics calculation.

Exports:
    - Return metrics
    - Risk metrics
    - Risk-adjusted metrics
    - Trade statistics
    - PerformanceMetrics calculator
"""

from app.backtest.metrics.performance import (
    # Return metrics
    calculate_total_return,
    calculate_annualized_return,
    calculate_cagr,
    calculate_rolling_returns,
    calculate_monthly_returns,
    calculate_yearly_returns,
    
    # Risk metrics
    calculate_volatility,
    calculate_downside_deviation,
    calculate_max_drawdown,
    calculate_drawdown_series,
    calculate_drawdown_duration,
    calculate_var,
    calculate_cvar,
    calculate_ulcer_index,
    
    # Risk-adjusted metrics
    calculate_sharpe_ratio,
    calculate_sortino_ratio,
    calculate_calmar_ratio,
    calculate_omega_ratio,
    calculate_information_ratio,
    
    # Trade statistics
    calculate_trade_statistics,
    
    # Comprehensive calculator
    PerformanceMetrics,
    calculate_all_metrics
)

__all__ = [
    # Returns
    "calculate_total_return",
    "calculate_annualized_return",
    "calculate_cagr",
    "calculate_rolling_returns",
    "calculate_monthly_returns",
    "calculate_yearly_returns",
    
    # Risk
    "calculate_volatility",
    "calculate_downside_deviation",
    "calculate_max_drawdown",
    "calculate_drawdown_series",
    "calculate_drawdown_duration",
    "calculate_var",
    "calculate_cvar",
    "calculate_ulcer_index",
    
    # Risk-adjusted
    "calculate_sharpe_ratio",
    "calculate_sortino_ratio",
    "calculate_calmar_ratio",
    "calculate_omega_ratio",
    "calculate_information_ratio",
    
    # Trades
    "calculate_trade_statistics",
    
    # All-in-one
    "PerformanceMetrics",
    "calculate_all_metrics"
]
