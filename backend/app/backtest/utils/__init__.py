"""
AlphaTerminal Pro - Backtest Utilities
======================================

Helper functions and utilities.
"""

from app.backtest.utils.helpers import (
    # Validation
    validate_ohlcv_data,
    clean_ohlcv_data,
    
    # Data generation
    generate_random_ohlcv,
    generate_trending_data,
    generate_ranging_data,
    
    # Formatting
    format_currency,
    format_percentage,
    format_ratio,
    format_backtest_summary,
    
    # Analysis
    analyze_trades_by_day_of_week,
    analyze_trades_by_month,
    calculate_max_consecutive_losses,
    calculate_recovery_time
)

__all__ = [
    "validate_ohlcv_data",
    "clean_ohlcv_data",
    "generate_random_ohlcv",
    "generate_trending_data",
    "generate_ranging_data",
    "format_currency",
    "format_percentage",
    "format_ratio",
    "format_backtest_summary",
    "analyze_trades_by_day_of_week",
    "analyze_trades_by_month",
    "calculate_max_consecutive_losses",
    "calculate_recovery_time"
]
