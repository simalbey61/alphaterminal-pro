"""
AlphaTerminal Pro - Visualizations
==================================

Chart generation for trading reports.

Author: AlphaTerminal Team
Version: 1.0.0
"""

import logging
import io
import base64
from datetime import datetime
from typing import Optional, Dict, List, Any, Tuple, Union
from dataclasses import dataclass
import numpy as np
import pandas as pd

from app.reporting.types import ChartType, ChartConfig


logger = logging.getLogger(__name__)


# =============================================================================
# COLOR SCHEMES
# =============================================================================

COLORS = {
    "primary": "#2962FF",
    "secondary": "#00BCD4",
    "success": "#4CAF50",
    "danger": "#F44336",
    "warning": "#FF9800",
    "info": "#2196F3",
    "dark": "#263238",
    "light": "#ECEFF1",
    
    # Trading specific
    "bullish": "#26A69A",
    "bearish": "#EF5350",
    "neutral": "#78909C",
    
    # Chart colors
    "equity": "#2962FF",
    "benchmark": "#78909C",
    "drawdown": "#EF5350",
    "positive": "#4CAF50",
    "negative": "#F44336",
}

DARK_THEME = {
    "background": "#1E1E1E",
    "text": "#FFFFFF",
    "grid": "#333333",
    "axis": "#666666",
}

LIGHT_THEME = {
    "background": "#FFFFFF",
    "text": "#333333",
    "grid": "#EEEEEE",
    "axis": "#999999",
}


# =============================================================================
# CHART GENERATOR
# =============================================================================

class ChartGenerator:
    """
    Generate various chart types for trading reports.
    
    Uses matplotlib for static charts and can export to
    PNG, SVG, or base64 encoded images.
    """
    
    def __init__(self, dark_mode: bool = False):
        """
        Initialize chart generator.
        
        Args:
            dark_mode: Use dark theme
        """
        self.dark_mode = dark_mode
        self.theme = DARK_THEME if dark_mode else LIGHT_THEME
        
        # Set matplotlib backend
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        self.plt = plt
        
        # Apply theme
        self._apply_theme()
    
    def _apply_theme(self):
        """Apply color theme to matplotlib."""
        if self.dark_mode:
            self.plt.style.use('dark_background')
        else:
            self.plt.style.use('seaborn-v0_8-whitegrid')
    
    def _create_figure(
        self,
        config: ChartConfig
    ) -> Tuple[Any, Any]:
        """Create matplotlib figure and axis."""
        fig, ax = self.plt.subplots(
            figsize=(config.width / 100, config.height / 100),
            dpi=100
        )
        
        fig.patch.set_facecolor(self.theme["background"])
        ax.set_facecolor(self.theme["background"])
        
        ax.set_title(config.title, fontsize=14, color=self.theme["text"])
        
        if config.subtitle:
            ax.set_title(
                config.subtitle,
                fontsize=10,
                color=self.theme["axis"],
                loc='left'
            )
        
        if config.x_label:
            ax.set_xlabel(config.x_label, color=self.theme["text"])
        
        if config.y_label:
            ax.set_ylabel(config.y_label, color=self.theme["text"])
        
        ax.tick_params(colors=self.theme["text"])
        
        if config.show_grid:
            ax.grid(True, color=self.theme["grid"], alpha=0.5)
        
        return fig, ax
    
    def _export_figure(
        self,
        fig,
        format: str = "png"
    ) -> Union[bytes, str]:
        """Export figure to bytes or base64."""
        buffer = io.BytesIO()
        
        actual_format = "png" if format == "base64" else format
        
        fig.savefig(
            buffer,
            format=actual_format,
            bbox_inches='tight',
            facecolor=self.theme["background"],
            edgecolor='none'
        )
        buffer.seek(0)
        
        self.plt.close(fig)
        
        if format == "base64":
            return base64.b64encode(buffer.getvalue()).decode()
        
        return buffer.getvalue()
    
    # =========================================================================
    # EQUITY CURVE
    # =========================================================================
    
    def equity_curve(
        self,
        equity: Union[pd.Series, np.ndarray],
        benchmark: Optional[Union[pd.Series, np.ndarray]] = None,
        config: Optional[ChartConfig] = None,
        format: str = "png"
    ) -> Union[bytes, str]:
        """
        Generate equity curve chart.
        
        Args:
            equity: Equity values over time
            benchmark: Optional benchmark for comparison
            config: Chart configuration
            format: Output format (png, svg, base64)
            
        Returns:
            Chart as bytes or base64 string
        """
        config = config or ChartConfig(
            chart_type=ChartType.EQUITY_CURVE,
            title="Equity Curve",
            y_label="Portfolio Value"
        )
        
        fig, ax = self._create_figure(config)
        
        # Plot equity
        x = range(len(equity))
        ax.plot(x, equity, color=COLORS["equity"], linewidth=2, label="Strategy")
        
        # Plot benchmark
        if benchmark is not None:
            ax.plot(x, benchmark, color=COLORS["benchmark"], 
                   linewidth=1.5, linestyle='--', label="Benchmark")
        
        # Fill positive/negative areas
        if isinstance(equity, pd.Series):
            initial = equity.iloc[0]
        else:
            initial = equity[0]
        
        ax.axhline(y=initial, color=self.theme["axis"], linestyle=':', alpha=0.5)
        ax.fill_between(x, initial, equity, 
                       where=(np.array(equity) >= initial),
                       color=COLORS["positive"], alpha=0.1)
        ax.fill_between(x, initial, equity,
                       where=(np.array(equity) < initial),
                       color=COLORS["negative"], alpha=0.1)
        
        if config.show_legend:
            ax.legend(loc='upper left', framealpha=0.9)
        
        return self._export_figure(fig, format)
    
    # =========================================================================
    # DRAWDOWN CHART
    # =========================================================================
    
    def drawdown_chart(
        self,
        equity: Union[pd.Series, np.ndarray],
        config: Optional[ChartConfig] = None,
        format: str = "png"
    ) -> Union[bytes, str]:
        """
        Generate drawdown chart.
        
        Args:
            equity: Equity values over time
            config: Chart configuration
            format: Output format
            
        Returns:
            Chart as bytes or base64 string
        """
        config = config or ChartConfig(
            chart_type=ChartType.DRAWDOWN,
            title="Drawdown",
            y_label="Drawdown %"
        )
        
        # Calculate drawdown
        equity = np.array(equity)
        running_max = np.maximum.accumulate(equity)
        drawdown = (equity - running_max) / running_max * 100
        
        fig, ax = self._create_figure(config)
        
        x = range(len(drawdown))
        ax.fill_between(x, 0, drawdown, color=COLORS["drawdown"], alpha=0.5)
        ax.plot(x, drawdown, color=COLORS["danger"], linewidth=1)
        
        # Mark max drawdown
        max_dd_idx = np.argmin(drawdown)
        ax.scatter([max_dd_idx], [drawdown[max_dd_idx]], 
                  color=COLORS["danger"], s=50, zorder=5)
        ax.annotate(f'Max DD: {drawdown[max_dd_idx]:.1f}%',
                   xy=(max_dd_idx, drawdown[max_dd_idx]),
                   xytext=(10, -10), textcoords='offset points',
                   fontsize=9, color=self.theme["text"])
        
        ax.axhline(y=0, color=self.theme["axis"], linewidth=0.5)
        
        return self._export_figure(fig, format)
    
    # =========================================================================
    # RETURNS DISTRIBUTION
    # =========================================================================
    
    def returns_distribution(
        self,
        returns: Union[pd.Series, np.ndarray],
        config: Optional[ChartConfig] = None,
        format: str = "png"
    ) -> Union[bytes, str]:
        """
        Generate returns distribution histogram.
        
        Args:
            returns: Return values
            config: Chart configuration
            format: Output format
            
        Returns:
            Chart as bytes or base64 string
        """
        config = config or ChartConfig(
            chart_type=ChartType.HISTOGRAM,
            title="Returns Distribution",
            x_label="Return %",
            y_label="Frequency"
        )
        
        fig, ax = self._create_figure(config)
        
        returns = np.array(returns) * 100  # Convert to percentage
        
        # Calculate histogram
        n, bins, patches = ax.hist(returns, bins=50, edgecolor='white', alpha=0.7)
        
        # Color positive/negative bins
        for i, patch in enumerate(patches):
            if bins[i] < 0:
                patch.set_facecolor(COLORS["negative"])
            else:
                patch.set_facecolor(COLORS["positive"])
        
        # Add mean and std lines
        mean = np.mean(returns)
        std = np.std(returns)
        
        ax.axvline(x=mean, color=COLORS["primary"], linestyle='--', 
                  linewidth=2, label=f'Mean: {mean:.2f}%')
        ax.axvline(x=mean + std, color=self.theme["axis"], linestyle=':', 
                  alpha=0.7, label=f'Std: {std:.2f}%')
        ax.axvline(x=mean - std, color=self.theme["axis"], linestyle=':', alpha=0.7)
        
        if config.show_legend:
            ax.legend(loc='upper right', framealpha=0.9)
        
        return self._export_figure(fig, format)
    
    # =========================================================================
    # MONTHLY RETURNS HEATMAP
    # =========================================================================
    
    def monthly_returns_heatmap(
        self,
        returns: pd.Series,
        config: Optional[ChartConfig] = None,
        format: str = "png"
    ) -> Union[bytes, str]:
        """
        Generate monthly returns heatmap.
        
        Args:
            returns: Daily returns with datetime index
            config: Chart configuration
            format: Output format
            
        Returns:
            Chart as bytes or base64 string
        """
        config = config or ChartConfig(
            chart_type=ChartType.HEATMAP,
            title="Monthly Returns (%)",
            width=1400,
            height=500
        )
        
        # Resample to monthly
        monthly = (1 + returns).resample('ME').prod() - 1
        
        # Create pivot table
        monthly_df = pd.DataFrame({
            'year': monthly.index.year,
            'month': monthly.index.month,
            'return': monthly.values * 100
        })
        
        pivot = monthly_df.pivot(index='year', columns='month', values='return')
        pivot.columns = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun',
                        'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
        
        fig, ax = self._create_figure(config)
        
        # Create heatmap
        import matplotlib.colors as mcolors
        
        vmax = max(abs(pivot.min().min()), abs(pivot.max().max()))
        norm = mcolors.TwoSlopeNorm(vmin=-vmax, vcenter=0, vmax=vmax)
        
        cmap = mcolors.LinearSegmentedColormap.from_list(
            'returns', [COLORS["negative"], '#FFFFFF', COLORS["positive"]]
        )
        
        im = ax.imshow(pivot.values, cmap=cmap, norm=norm, aspect='auto')
        
        # Labels
        ax.set_xticks(range(len(pivot.columns)))
        ax.set_xticklabels(pivot.columns, fontsize=9)
        ax.set_yticks(range(len(pivot.index)))
        ax.set_yticklabels(pivot.index, fontsize=9)
        
        # Add text annotations
        for i in range(len(pivot.index)):
            for j in range(len(pivot.columns)):
                val = pivot.iloc[i, j]
                if not pd.isna(val):
                    color = 'white' if abs(val) > vmax * 0.5 else self.theme["text"]
                    ax.text(j, i, f'{val:.1f}', ha='center', va='center',
                           fontsize=8, color=color)
        
        # Colorbar
        cbar = fig.colorbar(im, ax=ax, shrink=0.8)
        cbar.ax.tick_params(colors=self.theme["text"])
        
        return self._export_figure(fig, format)
    
    # =========================================================================
    # TRADE ANALYSIS
    # =========================================================================
    
    def trade_analysis(
        self,
        trades_df: pd.DataFrame,
        config: Optional[ChartConfig] = None,
        format: str = "png"
    ) -> Union[bytes, str]:
        """
        Generate trade analysis chart.
        
        Shows win/loss distribution and trade PnL scatter.
        
        Args:
            trades_df: DataFrame with trade data
            config: Chart configuration
            format: Output format
            
        Returns:
            Chart as bytes or base64 string
        """
        config = config or ChartConfig(
            chart_type=ChartType.SCATTER,
            title="Trade Analysis",
            width=1400,
            height=500
        )
        
        fig, axes = self.plt.subplots(1, 3, figsize=(14, 5))
        fig.patch.set_facecolor(self.theme["background"])
        
        # 1. Win/Loss pie chart
        ax1 = axes[0]
        ax1.set_facecolor(self.theme["background"])
        
        if 'pnl' in trades_df.columns:
            wins = (trades_df['pnl'] > 0).sum()
            losses = (trades_df['pnl'] <= 0).sum()
        else:
            wins = losses = 0
        
        ax1.pie([wins, losses], labels=['Wins', 'Losses'],
               colors=[COLORS["positive"], COLORS["negative"]],
               autopct='%1.1f%%', startangle=90)
        ax1.set_title('Win/Loss Ratio', color=self.theme["text"])
        
        # 2. PnL distribution
        ax2 = axes[1]
        ax2.set_facecolor(self.theme["background"])
        
        if 'pnl' in trades_df.columns:
            pnl = trades_df['pnl'].values
            colors = [COLORS["positive"] if p > 0 else COLORS["negative"] for p in pnl]
            ax2.bar(range(len(pnl)), pnl, color=colors, alpha=0.7)
        
        ax2.axhline(y=0, color=self.theme["axis"], linewidth=0.5)
        ax2.set_title('Trade PnL', color=self.theme["text"])
        ax2.set_xlabel('Trade #', color=self.theme["text"])
        ax2.set_ylabel('PnL', color=self.theme["text"])
        ax2.tick_params(colors=self.theme["text"])
        
        # 3. Holding period vs PnL
        ax3 = axes[2]
        ax3.set_facecolor(self.theme["background"])
        
        if 'pnl' in trades_df.columns and 'holding_hours' in trades_df.columns:
            colors = [COLORS["positive"] if p > 0 else COLORS["negative"] 
                     for p in trades_df['pnl']]
            ax3.scatter(trades_df['holding_hours'], trades_df['pnl'],
                       c=colors, alpha=0.6, s=30)
        
        ax3.axhline(y=0, color=self.theme["axis"], linewidth=0.5)
        ax3.set_title('Holding Period vs PnL', color=self.theme["text"])
        ax3.set_xlabel('Hours', color=self.theme["text"])
        ax3.set_ylabel('PnL', color=self.theme["text"])
        ax3.tick_params(colors=self.theme["text"])
        
        self.plt.tight_layout()
        
        return self._export_figure(fig, format)
    
    # =========================================================================
    # METRICS SUMMARY
    # =========================================================================
    
    def metrics_summary(
        self,
        metrics: Dict[str, float],
        config: Optional[ChartConfig] = None,
        format: str = "png"
    ) -> Union[bytes, str]:
        """
        Generate metrics summary visualization.
        
        Args:
            metrics: Dict of metric name to value
            config: Chart configuration
            format: Output format
            
        Returns:
            Chart as bytes or base64 string
        """
        config = config or ChartConfig(
            chart_type=ChartType.BAR,
            title="Performance Metrics",
            width=1200,
            height=400
        )
        
        fig, ax = self._create_figure(config)
        
        # Select key metrics
        key_metrics = {
            'Total Return': metrics.get('total_return_pct', 0),
            'Sharpe Ratio': metrics.get('sharpe_ratio', 0),
            'Max Drawdown': metrics.get('max_drawdown', 0) * -100,
            'Win Rate': metrics.get('win_rate', 0) * 100,
            'Profit Factor': metrics.get('profit_factor', 0),
        }
        
        x = range(len(key_metrics))
        values = list(key_metrics.values())
        
        colors = [COLORS["positive"] if v >= 0 else COLORS["negative"] for v in values]
        
        bars = ax.bar(x, values, color=colors, alpha=0.8)
        
        ax.set_xticks(x)
        ax.set_xticklabels(list(key_metrics.keys()), rotation=45, ha='right')
        ax.axhline(y=0, color=self.theme["axis"], linewidth=0.5)
        
        # Add value labels
        for bar, val in zip(bars, values):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{val:.2f}', ha='center', va='bottom' if height >= 0 else 'top',
                   fontsize=10, color=self.theme["text"])
        
        self.plt.tight_layout()
        
        return self._export_figure(fig, format)


# =============================================================================
# ALL EXPORTS
# =============================================================================

__all__ = [
    "COLORS",
    "DARK_THEME",
    "LIGHT_THEME",
    "ChartGenerator",
]
