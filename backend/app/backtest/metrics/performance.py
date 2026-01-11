"""
AlphaTerminal Pro - Performance Metrics
=======================================

Comprehensive performance metrics calculation.

Metrics Categories:
- Return metrics (total, annualized, rolling)
- Risk metrics (volatility, drawdown, VaR)
- Risk-adjusted metrics (Sharpe, Sortino, Calmar)
- Trade statistics (win rate, profit factor, expectancy)

Author: AlphaTerminal Team
Version: 1.0.0
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd
import numpy as np
from scipy import stats

from app.backtest.models.trade import Trade, TradeList
from app.backtest.exceptions import CalculationError, InsufficientTradesError

logger = logging.getLogger(__name__)


# =============================================================================
# RETURN METRICS
# =============================================================================

def calculate_total_return(
    equity_curve: pd.Series,
    initial_capital: Optional[float] = None
) -> Tuple[float, float]:
    """
    Calculate total return.
    
    Args:
        equity_curve: Equity time series
        initial_capital: Starting capital
        
    Returns:
        Tuple of (absolute return, percentage return)
    """
    if len(equity_curve) < 2:
        return 0.0, 0.0
    
    initial = initial_capital or equity_curve.iloc[0]
    final = equity_curve.iloc[-1]
    
    absolute_return = final - initial
    pct_return = absolute_return / initial if initial > 0 else 0.0
    
    return absolute_return, pct_return


def calculate_annualized_return(
    equity_curve: pd.Series,
    trading_days_per_year: int = 252
) -> float:
    """
    Calculate annualized return.
    
    Args:
        equity_curve: Equity time series
        trading_days_per_year: Trading days per year (252 for stocks)
        
    Returns:
        Annualized return as decimal
    """
    if len(equity_curve) < 2:
        return 0.0
    
    total_days = len(equity_curve)
    total_return = equity_curve.iloc[-1] / equity_curve.iloc[0] - 1
    
    years = total_days / trading_days_per_year
    if years <= 0:
        return 0.0
    
    return (1 + total_return) ** (1 / years) - 1


def calculate_cagr(
    equity_curve: pd.Series,
    trading_days_per_year: int = 252
) -> float:
    """
    Calculate Compound Annual Growth Rate.
    
    Same as annualized return but commonly used term.
    """
    return calculate_annualized_return(equity_curve, trading_days_per_year)


def calculate_rolling_returns(
    equity_curve: pd.Series,
    window: int = 252
) -> pd.Series:
    """
    Calculate rolling returns.
    
    Args:
        equity_curve: Equity time series
        window: Rolling window size
        
    Returns:
        Rolling return series
    """
    return equity_curve.pct_change(window).dropna()


def calculate_monthly_returns(returns: pd.Series) -> pd.Series:
    """
    Calculate monthly returns from daily returns.
    
    Args:
        returns: Daily returns series
        
    Returns:
        Monthly returns series
    """
    if len(returns) == 0:
        return pd.Series()
    
    # Compound daily returns to monthly
    monthly = returns.resample('M').apply(lambda x: (1 + x).prod() - 1)
    return monthly


def calculate_yearly_returns(returns: pd.Series) -> pd.Series:
    """
    Calculate yearly returns from daily returns.
    
    Args:
        returns: Daily returns series
        
    Returns:
        Yearly returns series
    """
    if len(returns) == 0:
        return pd.Series()
    
    yearly = returns.resample('Y').apply(lambda x: (1 + x).prod() - 1)
    return yearly


# =============================================================================
# RISK METRICS
# =============================================================================

def calculate_volatility(
    returns: pd.Series,
    annualize: bool = True,
    trading_days: int = 252
) -> float:
    """
    Calculate return volatility (standard deviation).
    
    Args:
        returns: Returns series
        annualize: Whether to annualize
        trading_days: Trading days per year
        
    Returns:
        Volatility
    """
    if len(returns) < 2:
        return 0.0
    
    vol = returns.std()
    
    if annualize:
        vol *= np.sqrt(trading_days)
    
    return vol


def calculate_downside_deviation(
    returns: pd.Series,
    mar: float = 0.0,
    annualize: bool = True,
    trading_days: int = 252
) -> float:
    """
    Calculate downside deviation (semi-deviation).
    
    Args:
        returns: Returns series
        mar: Minimum acceptable return (default 0)
        annualize: Whether to annualize
        trading_days: Trading days per year
        
    Returns:
        Downside deviation
    """
    if len(returns) < 2:
        return 0.0
    
    downside = returns[returns < mar]
    
    if len(downside) < 2:
        return 0.0
    
    dd = np.sqrt(np.mean(downside ** 2))
    
    if annualize:
        dd *= np.sqrt(trading_days)
    
    return dd


def calculate_max_drawdown(equity_curve: pd.Series) -> Tuple[float, int, int]:
    """
    Calculate maximum drawdown.
    
    Args:
        equity_curve: Equity time series
        
    Returns:
        Tuple of (max_drawdown, peak_idx, trough_idx)
    """
    if len(equity_curve) < 2:
        return 0.0, 0, 0
    
    rolling_max = equity_curve.expanding().max()
    drawdown = (equity_curve - rolling_max) / rolling_max
    
    max_dd = abs(drawdown.min())
    
    # Find peak and trough indices
    trough_idx = drawdown.idxmin()
    # Peak is the max before the trough
    peak_idx = equity_curve[:trough_idx].idxmax()
    
    return max_dd, peak_idx, trough_idx


def calculate_drawdown_series(equity_curve: pd.Series) -> pd.Series:
    """
    Calculate drawdown series.
    
    Args:
        equity_curve: Equity time series
        
    Returns:
        Drawdown series (negative values)
    """
    rolling_max = equity_curve.expanding().max()
    drawdown = (equity_curve - rolling_max) / rolling_max
    return drawdown


def calculate_drawdown_duration(equity_curve: pd.Series) -> Tuple[int, pd.Series]:
    """
    Calculate drawdown duration.
    
    Args:
        equity_curve: Equity time series
        
    Returns:
        Tuple of (max duration, duration series)
    """
    drawdown = calculate_drawdown_series(equity_curve)
    
    # Track duration of each drawdown period
    is_in_drawdown = drawdown < 0
    
    # Calculate consecutive drawdown periods
    duration = pd.Series(0, index=drawdown.index)
    current_duration = 0
    
    for i in range(len(is_in_drawdown)):
        if is_in_drawdown.iloc[i]:
            current_duration += 1
        else:
            current_duration = 0
        duration.iloc[i] = current_duration
    
    max_duration = duration.max()
    
    return max_duration, duration


def calculate_var(
    returns: pd.Series,
    confidence: float = 0.95,
    method: str = "historical"
) -> float:
    """
    Calculate Value at Risk.
    
    Args:
        returns: Returns series
        confidence: Confidence level (e.g., 0.95)
        method: "historical" or "parametric"
        
    Returns:
        VaR (positive number representing potential loss)
    """
    if len(returns) < 10:
        return 0.0
    
    if method == "historical":
        var = np.percentile(returns, (1 - confidence) * 100)
    elif method == "parametric":
        mean = returns.mean()
        std = returns.std()
        var = stats.norm.ppf(1 - confidence, mean, std)
    else:
        raise ValueError(f"Unknown VaR method: {method}")
    
    return abs(var)


def calculate_cvar(
    returns: pd.Series,
    confidence: float = 0.95
) -> float:
    """
    Calculate Conditional Value at Risk (Expected Shortfall).
    
    Args:
        returns: Returns series
        confidence: Confidence level
        
    Returns:
        CVaR (positive number)
    """
    if len(returns) < 10:
        return 0.0
    
    var = calculate_var(returns, confidence)
    
    # CVaR is the mean of returns below VaR
    tail_returns = returns[returns <= -var]
    
    if len(tail_returns) == 0:
        return var
    
    return abs(tail_returns.mean())


def calculate_ulcer_index(equity_curve: pd.Series) -> float:
    """
    Calculate Ulcer Index (measure of downside risk).
    
    Lower is better. Measures depth and duration of drawdowns.
    
    Args:
        equity_curve: Equity time series
        
    Returns:
        Ulcer Index
    """
    if len(equity_curve) < 2:
        return 0.0
    
    drawdown = calculate_drawdown_series(equity_curve)
    
    # Ulcer Index = sqrt(mean of squared drawdowns)
    squared_dd = drawdown ** 2
    
    return np.sqrt(squared_dd.mean())


# =============================================================================
# RISK-ADJUSTED METRICS
# =============================================================================

def calculate_sharpe_ratio(
    returns: pd.Series,
    risk_free_rate: float = 0.05,
    trading_days: int = 252
) -> float:
    """
    Calculate Sharpe Ratio.
    
    Args:
        returns: Daily returns series
        risk_free_rate: Annual risk-free rate
        trading_days: Trading days per year
        
    Returns:
        Sharpe Ratio
    """
    if len(returns) < 2:
        return 0.0
    
    daily_rf = risk_free_rate / trading_days
    excess_returns = returns - daily_rf
    
    if excess_returns.std() == 0:
        return 0.0
    
    sharpe = excess_returns.mean() / excess_returns.std()
    
    # Annualize
    return sharpe * np.sqrt(trading_days)


def calculate_sortino_ratio(
    returns: pd.Series,
    risk_free_rate: float = 0.05,
    mar: float = 0.0,
    trading_days: int = 252
) -> float:
    """
    Calculate Sortino Ratio.
    
    Uses downside deviation instead of total volatility.
    
    Args:
        returns: Daily returns series
        risk_free_rate: Annual risk-free rate
        mar: Minimum acceptable return
        trading_days: Trading days per year
        
    Returns:
        Sortino Ratio
    """
    if len(returns) < 2:
        return 0.0
    
    # Annualized return
    ann_return = (1 + returns.mean()) ** trading_days - 1
    
    # Downside deviation
    downside_std = calculate_downside_deviation(returns, mar, annualize=True, trading_days=trading_days)
    
    if downside_std == 0:
        return 0.0
    
    return (ann_return - risk_free_rate) / downside_std


def calculate_calmar_ratio(
    returns: pd.Series,
    equity_curve: pd.Series,
    trading_days: int = 252
) -> float:
    """
    Calculate Calmar Ratio.
    
    Annualized return / Max drawdown.
    
    Args:
        returns: Returns series
        equity_curve: Equity time series
        trading_days: Trading days per year
        
    Returns:
        Calmar Ratio
    """
    ann_return = calculate_annualized_return(equity_curve, trading_days)
    max_dd, _, _ = calculate_max_drawdown(equity_curve)
    
    if max_dd == 0:
        return 0.0
    
    return ann_return / max_dd


def calculate_omega_ratio(
    returns: pd.Series,
    threshold: float = 0.0
) -> float:
    """
    Calculate Omega Ratio.
    
    Sum of returns above threshold / Sum of returns below threshold.
    
    Args:
        returns: Returns series
        threshold: Return threshold
        
    Returns:
        Omega Ratio
    """
    if len(returns) < 2:
        return 0.0
    
    above = returns[returns > threshold].sum()
    below = abs(returns[returns < threshold].sum())
    
    if below == 0:
        return float('inf') if above > 0 else 0.0
    
    return above / below


def calculate_information_ratio(
    returns: pd.Series,
    benchmark_returns: pd.Series,
    trading_days: int = 252
) -> float:
    """
    Calculate Information Ratio.
    
    Active return / Tracking error.
    
    Args:
        returns: Strategy returns
        benchmark_returns: Benchmark returns
        trading_days: Trading days per year
        
    Returns:
        Information Ratio
    """
    if len(returns) < 2 or len(benchmark_returns) < 2:
        return 0.0
    
    # Align series
    aligned_returns, aligned_benchmark = returns.align(benchmark_returns, join='inner')
    
    if len(aligned_returns) < 2:
        return 0.0
    
    # Active return (excess over benchmark)
    active_returns = aligned_returns - aligned_benchmark
    
    tracking_error = active_returns.std() * np.sqrt(trading_days)
    
    if tracking_error == 0:
        return 0.0
    
    return active_returns.mean() * trading_days / tracking_error


# =============================================================================
# TRADE STATISTICS
# =============================================================================

def calculate_trade_statistics(trades: TradeList) -> Dict[str, Any]:
    """
    Calculate comprehensive trade statistics.
    
    Args:
        trades: List of trades
        
    Returns:
        Dictionary with trade statistics
    """
    if len(trades) == 0:
        return _empty_trade_stats()
    
    # Basic counts
    total = len(trades)
    winners = trades.winners
    losers = trades.losers
    
    num_winners = len(winners)
    num_losers = len(losers)
    
    # Win rate
    win_rate = num_winners / total if total > 0 else 0
    
    # P&L
    total_pnl = trades.total_pnl
    gross_profit = sum(t.net_pnl for t in winners)
    gross_loss = abs(sum(t.net_pnl for t in losers))
    
    # Averages
    avg_trade = total_pnl / total if total > 0 else 0
    avg_winner = gross_profit / num_winners if num_winners > 0 else 0
    avg_loser = gross_loss / num_losers if num_losers > 0 else 0
    
    # Extremes
    largest_winner = max([t.net_pnl for t in winners]) if winners else 0
    largest_loser = abs(min([t.net_pnl for t in losers])) if losers else 0
    
    # Profit factor
    profit_factor = gross_profit / gross_loss if gross_loss > 0 else float('inf') if gross_profit > 0 else 0
    
    # Expectancy (average profit per dollar risked)
    avg_risk = np.mean([t.risk_amount for t in trades if t.risk_amount]) if any(t.risk_amount for t in trades) else None
    expectancy_r = np.mean([t.r_multiple for t in trades if t.r_multiple is not None]) if any(t.r_multiple is not None for t in trades) else None
    
    # Consecutive wins/losses
    max_consecutive_wins, max_consecutive_losses = _calculate_consecutive_streaks(trades)
    
    # Duration
    avg_duration_hours = np.mean([t.holding_hours for t in trades])
    avg_bars = np.mean([t.bars_held for t in trades])
    
    # Win/loss by direction
    long_trades = [t for t in trades if t.direction == TradeDirection.LONG]
    short_trades = [t for t in trades if t.direction == TradeDirection.SHORT]
    
    long_win_rate = len([t for t in long_trades if t.is_winner]) / len(long_trades) if long_trades else 0
    short_win_rate = len([t for t in short_trades if t.is_winner]) / len(short_trades) if short_trades else 0
    
    # Edge calculation
    edge = (win_rate * avg_winner) - ((1 - win_rate) * avg_loser) if avg_loser > 0 else avg_winner * win_rate
    
    return {
        # Counts
        "total_trades": total,
        "winning_trades": num_winners,
        "losing_trades": num_losers,
        "breakeven_trades": total - num_winners - num_losers,
        
        # Rates
        "win_rate": win_rate,
        "loss_rate": 1 - win_rate,
        
        # P&L
        "total_pnl": total_pnl,
        "gross_profit": gross_profit,
        "gross_loss": gross_loss,
        "net_profit": total_pnl,
        
        # Averages
        "avg_trade": avg_trade,
        "avg_winner": avg_winner,
        "avg_loser": avg_loser,
        "avg_win_loss_ratio": avg_winner / avg_loser if avg_loser > 0 else float('inf'),
        
        # Extremes
        "largest_winner": largest_winner,
        "largest_loser": largest_loser,
        
        # Risk metrics
        "profit_factor": profit_factor,
        "expectancy": avg_trade,
        "expectancy_r": expectancy_r,
        "edge": edge,
        
        # Streaks
        "max_consecutive_wins": max_consecutive_wins,
        "max_consecutive_losses": max_consecutive_losses,
        
        # Duration
        "avg_duration_hours": avg_duration_hours,
        "avg_bars_in_trade": avg_bars,
        
        # By direction
        "long_trades": len(long_trades),
        "short_trades": len(short_trades),
        "long_win_rate": long_win_rate,
        "short_win_rate": short_win_rate
    }


def _empty_trade_stats() -> Dict[str, Any]:
    """Return empty trade statistics."""
    return {
        "total_trades": 0,
        "winning_trades": 0,
        "losing_trades": 0,
        "breakeven_trades": 0,
        "win_rate": 0,
        "loss_rate": 0,
        "total_pnl": 0,
        "gross_profit": 0,
        "gross_loss": 0,
        "net_profit": 0,
        "avg_trade": 0,
        "avg_winner": 0,
        "avg_loser": 0,
        "avg_win_loss_ratio": 0,
        "largest_winner": 0,
        "largest_loser": 0,
        "profit_factor": 0,
        "expectancy": 0,
        "expectancy_r": None,
        "edge": 0,
        "max_consecutive_wins": 0,
        "max_consecutive_losses": 0,
        "avg_duration_hours": 0,
        "avg_bars_in_trade": 0,
        "long_trades": 0,
        "short_trades": 0,
        "long_win_rate": 0,
        "short_win_rate": 0
    }


def _calculate_consecutive_streaks(trades: TradeList) -> Tuple[int, int]:
    """Calculate max consecutive wins and losses."""
    if len(trades) == 0:
        return 0, 0
    
    max_wins = 0
    max_losses = 0
    current_wins = 0
    current_losses = 0
    
    for trade in trades:
        if trade.is_winner:
            current_wins += 1
            current_losses = 0
            max_wins = max(max_wins, current_wins)
        elif trade.is_loser:
            current_losses += 1
            current_wins = 0
            max_losses = max(max_losses, current_losses)
        else:
            current_wins = 0
            current_losses = 0
    
    return max_wins, max_losses


# Need to import TradeDirection
from app.backtest.enums import TradeDirection


# =============================================================================
# COMPREHENSIVE METRICS CALCULATOR
# =============================================================================

@dataclass
class PerformanceMetrics:
    """All performance metrics in one place."""
    
    # Returns
    total_return: float = 0.0
    total_return_pct: float = 0.0
    annualized_return: float = 0.0
    cagr: float = 0.0
    
    # Risk
    volatility: float = 0.0
    max_drawdown: float = 0.0
    max_drawdown_duration: int = 0
    downside_deviation: float = 0.0
    var_95: float = 0.0
    cvar_95: float = 0.0
    ulcer_index: float = 0.0
    
    # Risk-adjusted
    sharpe_ratio: float = 0.0
    sortino_ratio: float = 0.0
    calmar_ratio: float = 0.0
    omega_ratio: float = 0.0
    
    # Trade statistics
    trade_stats: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "returns": {
                "total_return": round(self.total_return, 2),
                "total_return_pct": round(self.total_return_pct, 4),
                "annualized_return": round(self.annualized_return, 4),
                "cagr": round(self.cagr, 4)
            },
            "risk": {
                "volatility": round(self.volatility, 4),
                "max_drawdown": round(self.max_drawdown, 4),
                "max_drawdown_duration": self.max_drawdown_duration,
                "downside_deviation": round(self.downside_deviation, 4),
                "var_95": round(self.var_95, 4),
                "cvar_95": round(self.cvar_95, 4),
                "ulcer_index": round(self.ulcer_index, 4)
            },
            "risk_adjusted": {
                "sharpe_ratio": round(self.sharpe_ratio, 2),
                "sortino_ratio": round(self.sortino_ratio, 2),
                "calmar_ratio": round(self.calmar_ratio, 2),
                "omega_ratio": round(self.omega_ratio, 2)
            },
            "trades": self.trade_stats
        }


def calculate_all_metrics(
    equity_curve: pd.Series,
    trades: TradeList,
    initial_capital: float,
    risk_free_rate: float = 0.05,
    trading_days: int = 252
) -> PerformanceMetrics:
    """
    Calculate all performance metrics.
    
    Args:
        equity_curve: Equity time series
        trades: List of trades
        initial_capital: Starting capital
        risk_free_rate: Annual risk-free rate
        trading_days: Trading days per year
        
    Returns:
        PerformanceMetrics with all calculations
    """
    returns = equity_curve.pct_change().dropna()
    
    # Returns
    abs_return, pct_return = calculate_total_return(equity_curve, initial_capital)
    ann_return = calculate_annualized_return(equity_curve, trading_days)
    cagr = calculate_cagr(equity_curve, trading_days)
    
    # Risk
    vol = calculate_volatility(returns, True, trading_days)
    max_dd, _, _ = calculate_max_drawdown(equity_curve)
    max_dd_duration, _ = calculate_drawdown_duration(equity_curve)
    downside_dev = calculate_downside_deviation(returns, 0, True, trading_days)
    var_95 = calculate_var(returns, 0.95)
    cvar_95 = calculate_cvar(returns, 0.95)
    ulcer = calculate_ulcer_index(equity_curve)
    
    # Risk-adjusted
    sharpe = calculate_sharpe_ratio(returns, risk_free_rate, trading_days)
    sortino = calculate_sortino_ratio(returns, risk_free_rate, 0, trading_days)
    calmar = calculate_calmar_ratio(returns, equity_curve, trading_days)
    omega = calculate_omega_ratio(returns, 0)
    
    # Trades
    trade_stats = calculate_trade_statistics(trades)
    
    return PerformanceMetrics(
        total_return=abs_return,
        total_return_pct=pct_return,
        annualized_return=ann_return,
        cagr=cagr,
        volatility=vol,
        max_drawdown=max_dd,
        max_drawdown_duration=int(max_dd_duration),
        downside_deviation=downside_dev,
        var_95=var_95,
        cvar_95=cvar_95,
        ulcer_index=ulcer,
        sharpe_ratio=sharpe,
        sortino_ratio=sortino,
        calmar_ratio=calmar,
        omega_ratio=omega,
        trade_stats=trade_stats
    )
