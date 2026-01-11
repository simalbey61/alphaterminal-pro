"""
AlphaTerminal Pro - Backtest Utilities
======================================

Helper functions and utilities for backtesting.

Author: AlphaTerminal Team
Version: 1.0.0
"""

from __future__ import annotations

import logging
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Tuple, Union

import pandas as pd
import numpy as np

logger = logging.getLogger(__name__)


# =============================================================================
# DATA VALIDATION
# =============================================================================

def validate_ohlcv_data(
    data: pd.DataFrame,
    symbol: str = "Unknown"
) -> Tuple[bool, List[str]]:
    """
    Validate OHLCV data for backtesting.
    
    Args:
        data: DataFrame to validate
        symbol: Symbol name for error messages
        
    Returns:
        Tuple of (is_valid, list_of_issues)
    """
    issues = []
    
    # Check required columns
    required_columns = ['Open', 'High', 'Low', 'Close', 'Volume']
    missing = [col for col in required_columns if col not in data.columns]
    if missing:
        issues.append(f"Missing columns: {missing}")
    
    # Check index type
    if not isinstance(data.index, pd.DatetimeIndex):
        issues.append("Index must be DatetimeIndex")
    
    # Check for NaN values
    for col in required_columns:
        if col in data.columns:
            nan_count = data[col].isna().sum()
            if nan_count > 0:
                issues.append(f"{col} has {nan_count} NaN values")
    
    # Check OHLC consistency
    if all(col in data.columns for col in ['Open', 'High', 'Low', 'Close']):
        # High should be highest
        invalid_high = (data['High'] < data[['Open', 'Close']].max(axis=1)).sum()
        if invalid_high > 0:
            issues.append(f"{invalid_high} bars where High is not highest")
        
        # Low should be lowest
        invalid_low = (data['Low'] > data[['Open', 'Close']].min(axis=1)).sum()
        if invalid_low > 0:
            issues.append(f"{invalid_low} bars where Low is not lowest")
        
        # No negative prices
        negative_count = (data[['Open', 'High', 'Low', 'Close']] < 0).sum().sum()
        if negative_count > 0:
            issues.append(f"{negative_count} negative price values")
    
    # Check for duplicates
    duplicate_count = data.index.duplicated().sum()
    if duplicate_count > 0:
        issues.append(f"{duplicate_count} duplicate timestamps")
    
    # Check for monotonic index
    if not data.index.is_monotonic_increasing:
        issues.append("Index is not monotonically increasing (not sorted)")
    
    is_valid = len(issues) == 0
    
    if not is_valid:
        logger.warning(f"Data validation issues for {symbol}: {issues}")
    
    return is_valid, issues


def clean_ohlcv_data(
    data: pd.DataFrame,
    fill_method: str = 'ffill'
) -> pd.DataFrame:
    """
    Clean and fix OHLCV data issues.
    
    Args:
        data: DataFrame to clean
        fill_method: Method to fill NaN values ('ffill', 'bfill', 'interpolate')
        
    Returns:
        Cleaned DataFrame
    """
    data = data.copy()
    
    # Sort by index
    data = data.sort_index()
    
    # Remove duplicates (keep last)
    data = data[~data.index.duplicated(keep='last')]
    
    # Fill NaN values
    if fill_method == 'ffill':
        data = data.fillna(method='ffill')
    elif fill_method == 'bfill':
        data = data.fillna(method='bfill')
    elif fill_method == 'interpolate':
        data = data.interpolate(method='linear')
    
    # Fix OHLC consistency
    # High should be >= max(Open, Close)
    data['High'] = data[['High', 'Open', 'Close']].max(axis=1)
    # Low should be <= min(Open, Close)
    data['Low'] = data[['Low', 'Open', 'Close']].min(axis=1)
    
    # Ensure positive prices
    for col in ['Open', 'High', 'Low', 'Close']:
        if col in data.columns:
            data[col] = data[col].clip(lower=0.01)
    
    # Ensure non-negative volume
    if 'Volume' in data.columns:
        data['Volume'] = data['Volume'].clip(lower=0)
    
    return data


# =============================================================================
# DATA GENERATION (FOR TESTING)
# =============================================================================

def generate_random_ohlcv(
    start_date: datetime,
    periods: int = 252,
    freq: str = 'D',
    initial_price: float = 100.0,
    volatility: float = 0.02,
    trend: float = 0.0001,
    volume_base: int = 1_000_000
) -> pd.DataFrame:
    """
    Generate random OHLCV data for testing.
    
    Args:
        start_date: Start date for data
        periods: Number of periods
        freq: Frequency ('D' for daily, 'H' for hourly)
        initial_price: Starting price
        volatility: Daily volatility (standard deviation)
        trend: Daily trend (drift)
        volume_base: Base volume
        
    Returns:
        DataFrame with OHLCV data
    """
    np.random.seed(42)  # For reproducibility
    
    # Generate index
    index = pd.date_range(start=start_date, periods=periods, freq=freq)
    
    # Generate returns with trend and volatility
    returns = np.random.normal(trend, volatility, periods)
    
    # Generate price series (geometric brownian motion)
    price_multipliers = np.exp(returns)
    close_prices = initial_price * np.cumprod(price_multipliers)
    
    # Generate OHLC from close
    data = pd.DataFrame(index=index)
    
    # Close is the main series
    data['Close'] = close_prices
    
    # Open is previous close (with small gap)
    data['Open'] = data['Close'].shift(1) * (1 + np.random.normal(0, volatility/4, periods))
    data['Open'].iloc[0] = initial_price
    
    # High and Low
    intraday_range = np.abs(np.random.normal(0, volatility * 1.5, periods))
    data['High'] = data[['Open', 'Close']].max(axis=1) * (1 + intraday_range)
    data['Low'] = data[['Open', 'Close']].min(axis=1) * (1 - intraday_range)
    
    # Volume (with some randomness)
    data['Volume'] = (volume_base * (1 + np.random.uniform(-0.5, 0.5, periods))).astype(int)
    
    return data


def generate_trending_data(
    start_date: datetime,
    periods: int = 252,
    initial_price: float = 100.0,
    trend_strength: float = 0.001,
    noise_level: float = 0.015
) -> pd.DataFrame:
    """
    Generate trending market data (for strategy testing).
    
    Args:
        start_date: Start date
        periods: Number of periods
        initial_price: Starting price
        trend_strength: Daily trend (positive = uptrend)
        noise_level: Random noise level
        
    Returns:
        DataFrame with OHLCV data
    """
    return generate_random_ohlcv(
        start_date=start_date,
        periods=periods,
        initial_price=initial_price,
        volatility=noise_level,
        trend=trend_strength
    )


def generate_ranging_data(
    start_date: datetime,
    periods: int = 252,
    initial_price: float = 100.0,
    range_size: float = 0.10,
    mean_reversion: float = 0.1
) -> pd.DataFrame:
    """
    Generate ranging/mean-reverting market data.
    
    Args:
        start_date: Start date
        periods: Number of periods
        initial_price: Starting price
        range_size: Size of the range (% from center)
        mean_reversion: Mean reversion strength
        
    Returns:
        DataFrame with OHLCV data
    """
    np.random.seed(42)
    
    index = pd.date_range(start=start_date, periods=periods, freq='D')
    
    # Generate mean-reverting series
    prices = [initial_price]
    center = initial_price
    
    for _ in range(periods - 1):
        # Random shock
        shock = np.random.normal(0, initial_price * 0.01)
        
        # Mean reversion
        reversion = mean_reversion * (center - prices[-1])
        
        new_price = prices[-1] + shock + reversion
        
        # Keep within range
        upper = center * (1 + range_size)
        lower = center * (1 - range_size)
        new_price = np.clip(new_price, lower, upper)
        
        prices.append(new_price)
    
    close_prices = np.array(prices)
    
    data = pd.DataFrame(index=index)
    data['Close'] = close_prices
    data['Open'] = data['Close'].shift(1)
    data['Open'].iloc[0] = initial_price
    
    volatility = 0.01
    data['High'] = data[['Open', 'Close']].max(axis=1) * (1 + np.abs(np.random.normal(0, volatility, periods)))
    data['Low'] = data[['Open', 'Close']].min(axis=1) * (1 - np.abs(np.random.normal(0, volatility, periods)))
    data['Volume'] = np.random.randint(500000, 1500000, periods)
    
    return data


# =============================================================================
# RESULT FORMATTING
# =============================================================================

def format_currency(value: float, currency: str = "TRY") -> str:
    """Format value as currency string."""
    return f"{value:,.2f} {currency}"


def format_percentage(value: float, decimals: int = 2) -> str:
    """Format value as percentage string."""
    return f"{value * 100:.{decimals}f}%"


def format_ratio(value: float, decimals: int = 2) -> str:
    """Format value as ratio."""
    return f"{value:.{decimals}f}"


def format_backtest_summary(result: Any) -> str:
    """
    Format backtest result as summary string.
    
    Args:
        result: BacktestResult object
        
    Returns:
        Formatted summary string
    """
    return f"""
╔══════════════════════════════════════════════════════════════════╗
║                    BACKTEST SONUÇLARI                            ║
╠══════════════════════════════════════════════════════════════════╣
║  Sembol: {result.symbol:<12}  Zaman Dilimi: {result.timeframe:<8}              ║
║  Dönem: {result.start_date.strftime('%Y-%m-%d')} - {result.end_date.strftime('%Y-%m-%d')}                    ║
╠══════════════════════════════════════════════════════════════════╣
║  GETİRİ                                                          ║
║    Toplam Getiri:    {format_percentage(result.total_return_pct):>12}                         ║
║    Yıllık Getiri:    {format_percentage(result.annualized_return):>12}                         ║
╠══════════════════════════════════════════════════════════════════╣
║  RİSK                                                            ║
║    Maks Düşüş:       {format_percentage(result.max_drawdown):>12}                         ║
║    Volatilite:       {format_percentage(result.volatility):>12}                         ║
╠══════════════════════════════════════════════════════════════════╣
║  RİSK-AYARLI                                                     ║
║    Sharpe Oranı:     {format_ratio(result.sharpe_ratio):>12}                         ║
║    Sortino Oranı:    {format_ratio(result.sortino_ratio):>12}                         ║
║    Calmar Oranı:     {format_ratio(result.calmar_ratio):>12}                         ║
╠══════════════════════════════════════════════════════════════════╣
║  İŞLEMLER                                                        ║
║    Toplam:           {result.total_trades:>12}                              ║
║    Kazanma Oranı:    {format_percentage(result.win_rate):>12}                         ║
║    Kar Faktörü:      {format_ratio(result.profit_factor):>12}                         ║
║    Ort. İşlem:       {format_currency(result.avg_trade_pnl):>12}                      ║
╚══════════════════════════════════════════════════════════════════╝
"""


# =============================================================================
# ANALYSIS HELPERS
# =============================================================================

def analyze_trades_by_day_of_week(trades: List) -> pd.DataFrame:
    """
    Analyze trade performance by day of week.
    
    Args:
        trades: List of Trade objects
        
    Returns:
        DataFrame with day-of-week analysis
    """
    if not trades:
        return pd.DataFrame()
    
    data = []
    for trade in trades:
        data.append({
            'day': trade.entry_time.strftime('%A'),
            'day_num': trade.entry_time.weekday(),
            'pnl': trade.net_pnl,
            'is_winner': trade.is_winner
        })
    
    df = pd.DataFrame(data)
    
    summary = df.groupby(['day_num', 'day']).agg({
        'pnl': ['count', 'sum', 'mean'],
        'is_winner': 'mean'
    }).round(2)
    
    summary.columns = ['trades', 'total_pnl', 'avg_pnl', 'win_rate']
    summary = summary.sort_index(level=0)
    
    return summary.reset_index(level=0, drop=True)


def analyze_trades_by_month(trades: List) -> pd.DataFrame:
    """
    Analyze trade performance by month.
    
    Args:
        trades: List of Trade objects
        
    Returns:
        DataFrame with monthly analysis
    """
    if not trades:
        return pd.DataFrame()
    
    data = []
    for trade in trades:
        data.append({
            'month': trade.entry_time.strftime('%Y-%m'),
            'pnl': trade.net_pnl,
            'is_winner': trade.is_winner
        })
    
    df = pd.DataFrame(data)
    
    summary = df.groupby('month').agg({
        'pnl': ['count', 'sum', 'mean'],
        'is_winner': 'mean'
    }).round(2)
    
    summary.columns = ['trades', 'total_pnl', 'avg_pnl', 'win_rate']
    
    return summary


def calculate_max_consecutive_losses(trades: List) -> int:
    """Calculate maximum consecutive losses."""
    if not trades:
        return 0
    
    max_losses = 0
    current_losses = 0
    
    for trade in trades:
        if trade.is_loser:
            current_losses += 1
            max_losses = max(max_losses, current_losses)
        else:
            current_losses = 0
    
    return max_losses


def calculate_recovery_time(equity_curve: pd.Series) -> Optional[int]:
    """
    Calculate time to recover from max drawdown.
    
    Args:
        equity_curve: Equity time series
        
    Returns:
        Number of bars to recover, or None if not recovered
    """
    if len(equity_curve) < 2:
        return None
    
    rolling_max = equity_curve.expanding().max()
    drawdown = (equity_curve - rolling_max) / rolling_max
    
    # Find max drawdown point
    max_dd_idx = drawdown.idxmin()
    
    # Find recovery point (equity >= previous high)
    recovery_slice = equity_curve[max_dd_idx:]
    peak_at_dd = rolling_max[max_dd_idx]
    
    recovered = recovery_slice >= peak_at_dd
    
    if recovered.any():
        recovery_idx = recovered.idxmax()
        
        # Calculate bars between
        start_pos = equity_curve.index.get_loc(max_dd_idx)
        end_pos = equity_curve.index.get_loc(recovery_idx)
        
        return end_pos - start_pos
    
    return None  # Never recovered
