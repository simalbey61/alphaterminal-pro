"""
AlphaTerminal Pro - Report Generators
=====================================

Generate various report types in multiple formats.

Author: AlphaTerminal Team
Version: 1.0.0
"""

import logging
import uuid
import io
from datetime import datetime
from typing import Optional, Dict, List, Any, Union
from dataclasses import dataclass, field
from pathlib import Path
import json

from app.reporting.types import (
    ReportType, ReportFormat, ReportMetadata,
    ReportSection, MetricsSection, TableSection
)
from app.reporting.visualizations.charts import ChartGenerator


logger = logging.getLogger(__name__)


# =============================================================================
# REPORT DATA
# =============================================================================

@dataclass
class BacktestReportData:
    """Data for backtest report."""
    
    # Identification
    symbol: str
    interval: str
    strategy_name: str
    
    # Time range
    start_date: datetime
    end_date: datetime
    total_bars: int
    
    # Performance metrics
    initial_capital: float
    final_equity: float
    total_return: float
    total_return_pct: float
    
    # Risk metrics
    max_drawdown: float
    sharpe_ratio: float
    sortino_ratio: float
    calmar_ratio: float
    volatility: float
    
    # Trade statistics
    total_trades: int
    winning_trades: int
    losing_trades: int
    win_rate: float
    profit_factor: float
    avg_trade_pnl: float
    avg_winner: float
    avg_loser: float
    largest_winner: float
    largest_loser: float
    avg_holding_hours: float
    
    # Data series
    equity_curve: Optional[List[float]] = None
    returns: Optional[List[float]] = None
    trades: Optional[List[Dict]] = None
    monthly_returns: Optional[Dict[str, float]] = None
    
    # Config
    strategy_params: Dict[str, Any] = field(default_factory=dict)


# =============================================================================
# BASE REPORT GENERATOR
# =============================================================================

class BaseReportGenerator:
    """Base class for report generators."""
    
    report_type: ReportType = ReportType.BACKTEST
    
    def __init__(self, dark_mode: bool = False):
        """
        Initialize report generator.
        
        Args:
            dark_mode: Use dark theme for charts
        """
        self.dark_mode = dark_mode
        self.chart_gen = ChartGenerator(dark_mode=dark_mode)
        self.sections: List[ReportSection] = []
    
    def add_section(self, section: ReportSection):
        """Add section to report."""
        self.sections.append(section)
    
    def generate(self, format: ReportFormat = ReportFormat.HTML) -> bytes:
        """Generate report in specified format."""
        if format == ReportFormat.HTML:
            return self._generate_html()
        elif format == ReportFormat.MARKDOWN:
            return self._generate_markdown()
        elif format == ReportFormat.JSON:
            return self._generate_json()
        else:
            raise ValueError(f"Unsupported format: {format}")
    
    def _generate_html(self) -> bytes:
        """Generate HTML report."""
        raise NotImplementedError
    
    def _generate_markdown(self) -> bytes:
        """Generate Markdown report."""
        raise NotImplementedError
    
    def _generate_json(self) -> bytes:
        """Generate JSON report."""
        raise NotImplementedError


# =============================================================================
# BACKTEST REPORT GENERATOR
# =============================================================================

class BacktestReportGenerator(BaseReportGenerator):
    """
    Generate comprehensive backtest reports.
    
    Includes:
    - Performance summary
    - Equity curve
    - Drawdown analysis
    - Trade statistics
    - Monthly returns
    - Risk metrics
    """
    
    report_type = ReportType.BACKTEST
    
    def __init__(self, data: BacktestReportData, dark_mode: bool = False):
        """
        Initialize backtest report generator.
        
        Args:
            data: Backtest report data
            dark_mode: Use dark theme
        """
        super().__init__(dark_mode)
        self.data = data
        
        # Generate report ID
        self.report_id = str(uuid.uuid4())[:8]
        
        # Create metadata
        self.metadata = ReportMetadata(
            report_id=self.report_id,
            report_type=ReportType.BACKTEST,
            title=f"Backtest Report - {data.symbol}",
            description=f"Strategy: {data.strategy_name}",
            symbol=data.symbol,
            interval=data.interval,
            start_date=data.start_date,
            end_date=data.end_date,
        )
    
    def _generate_html(self) -> bytes:
        """Generate HTML backtest report."""
        
        # Generate charts
        equity_chart = ""
        if self.data.equity_curve:
            equity_b64 = self.chart_gen.equity_curve(
                self.data.equity_curve, format="base64"
            )
            equity_chart = f'<img src="data:image/png;base64,{equity_b64}" class="chart">'
        
        drawdown_chart = ""
        if self.data.equity_curve:
            dd_b64 = self.chart_gen.drawdown_chart(
                self.data.equity_curve, format="base64"
            )
            drawdown_chart = f'<img src="data:image/png;base64,{dd_b64}" class="chart">'
        
        returns_chart = ""
        if self.data.returns:
            ret_b64 = self.chart_gen.returns_distribution(
                self.data.returns, format="base64"
            )
            returns_chart = f'<img src="data:image/png;base64,{ret_b64}" class="chart">'
        
        # Build HTML
        html = f"""
<!DOCTYPE html>
<html>
<head>
    <meta charset="UTF-8">
    <title>{self.metadata.title}</title>
    <style>
        {self._get_css()}
    </style>
</head>
<body>
    <div class="container">
        <header>
            <h1>📊 {self.metadata.title}</h1>
            <p class="subtitle">{self.data.strategy_name} | {self.data.interval} | {self.data.start_date.strftime('%Y-%m-%d')} to {self.data.end_date.strftime('%Y-%m-%d')}</p>
        </header>
        
        <section class="summary">
            <h2>Performance Summary</h2>
            <div class="metrics-grid">
                {self._generate_metric_card("Total Return", f"{self.data.total_return_pct:.2f}%", self.data.total_return_pct >= 0)}
                {self._generate_metric_card("Sharpe Ratio", f"{self.data.sharpe_ratio:.2f}", self.data.sharpe_ratio >= 0)}
                {self._generate_metric_card("Max Drawdown", f"{self.data.max_drawdown*100:.2f}%", False)}
                {self._generate_metric_card("Win Rate", f"{self.data.win_rate*100:.1f}%", self.data.win_rate >= 0.5)}
                {self._generate_metric_card("Profit Factor", f"{self.data.profit_factor:.2f}", self.data.profit_factor >= 1)}
                {self._generate_metric_card("Total Trades", f"{self.data.total_trades}", True)}
            </div>
        </section>
        
        <section class="charts">
            <h2>Equity Curve</h2>
            {equity_chart}
            
            <h2>Drawdown</h2>
            {drawdown_chart}
            
            <h2>Returns Distribution</h2>
            {returns_chart}
        </section>
        
        <section class="statistics">
            <h2>Trade Statistics</h2>
            <table>
                <tr><th>Metric</th><th>Value</th></tr>
                <tr><td>Total Trades</td><td>{self.data.total_trades}</td></tr>
                <tr><td>Winning Trades</td><td>{self.data.winning_trades}</td></tr>
                <tr><td>Losing Trades</td><td>{self.data.losing_trades}</td></tr>
                <tr><td>Win Rate</td><td>{self.data.win_rate*100:.2f}%</td></tr>
                <tr><td>Average Trade</td><td>${self.data.avg_trade_pnl:.2f}</td></tr>
                <tr><td>Average Winner</td><td>${self.data.avg_winner:.2f}</td></tr>
                <tr><td>Average Loser</td><td>${self.data.avg_loser:.2f}</td></tr>
                <tr><td>Largest Winner</td><td>${self.data.largest_winner:.2f}</td></tr>
                <tr><td>Largest Loser</td><td>${self.data.largest_loser:.2f}</td></tr>
                <tr><td>Avg Holding Time</td><td>{self.data.avg_holding_hours:.1f} hours</td></tr>
            </table>
        </section>
        
        <section class="risk">
            <h2>Risk Metrics</h2>
            <table>
                <tr><th>Metric</th><th>Value</th></tr>
                <tr><td>Sharpe Ratio</td><td>{self.data.sharpe_ratio:.3f}</td></tr>
                <tr><td>Sortino Ratio</td><td>{self.data.sortino_ratio:.3f}</td></tr>
                <tr><td>Calmar Ratio</td><td>{self.data.calmar_ratio:.3f}</td></tr>
                <tr><td>Max Drawdown</td><td>{self.data.max_drawdown*100:.2f}%</td></tr>
                <tr><td>Volatility (Ann.)</td><td>{self.data.volatility*100:.2f}%</td></tr>
            </table>
        </section>
        
        <section class="config">
            <h2>Strategy Configuration</h2>
            <pre>{json.dumps(self.data.strategy_params, indent=2)}</pre>
        </section>
        
        <footer>
            <p>Generated by AlphaTerminal Pro | {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
            <p>Report ID: {self.report_id}</p>
        </footer>
    </div>
</body>
</html>
        """
        
        return html.encode('utf-8')
    
    def _generate_metric_card(
        self,
        label: str,
        value: str,
        positive: bool
    ) -> str:
        """Generate metric card HTML."""
        color_class = "positive" if positive else "negative"
        return f"""
        <div class="metric-card {color_class}">
            <div class="metric-label">{label}</div>
            <div class="metric-value">{value}</div>
        </div>
        """
    
    def _get_css(self) -> str:
        """Get CSS styles."""
        bg_color = "#1E1E1E" if self.dark_mode else "#FFFFFF"
        text_color = "#FFFFFF" if self.dark_mode else "#333333"
        card_bg = "#2D2D2D" if self.dark_mode else "#F5F5F5"
        
        return f"""
        * {{ box-sizing: border-box; margin: 0; padding: 0; }}
        body {{ 
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Oxygen, Ubuntu, sans-serif;
            background: {bg_color};
            color: {text_color};
            line-height: 1.6;
        }}
        .container {{ max-width: 1200px; margin: 0 auto; padding: 20px; }}
        header {{ text-align: center; margin-bottom: 30px; padding: 20px; }}
        h1 {{ font-size: 2em; margin-bottom: 10px; }}
        .subtitle {{ color: #888; font-size: 1.1em; }}
        h2 {{ margin: 30px 0 20px; padding-bottom: 10px; border-bottom: 2px solid #333; }}
        
        .metrics-grid {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(180px, 1fr));
            gap: 15px;
            margin: 20px 0;
        }}
        .metric-card {{
            background: {card_bg};
            padding: 20px;
            border-radius: 10px;
            text-align: center;
        }}
        .metric-card.positive {{ border-left: 4px solid #4CAF50; }}
        .metric-card.negative {{ border-left: 4px solid #F44336; }}
        .metric-label {{ font-size: 0.85em; color: #888; margin-bottom: 5px; }}
        .metric-value {{ font-size: 1.5em; font-weight: bold; }}
        
        .chart {{ width: 100%; max-width: 100%; margin: 20px 0; border-radius: 8px; }}
        
        table {{
            width: 100%;
            border-collapse: collapse;
            margin: 20px 0;
            background: {card_bg};
            border-radius: 8px;
            overflow: hidden;
        }}
        th, td {{ padding: 12px 15px; text-align: left; border-bottom: 1px solid #444; }}
        th {{ background: #333; font-weight: 600; }}
        tr:hover {{ background: rgba(255,255,255,0.05); }}
        
        pre {{
            background: #1a1a1a;
            padding: 15px;
            border-radius: 8px;
            overflow-x: auto;
            font-size: 0.9em;
        }}
        
        footer {{
            margin-top: 50px;
            padding: 20px;
            text-align: center;
            color: #666;
            border-top: 1px solid #333;
        }}
        """
    
    def _generate_markdown(self) -> bytes:
        """Generate Markdown backtest report."""
        md = f"""# 📊 {self.metadata.title}

**Strategy:** {self.data.strategy_name}  
**Symbol:** {self.data.symbol} | **Interval:** {self.data.interval}  
**Period:** {self.data.start_date.strftime('%Y-%m-%d')} to {self.data.end_date.strftime('%Y-%m-%d')}

---

## Performance Summary

| Metric | Value |
|--------|-------|
| Total Return | {self.data.total_return_pct:.2f}% |
| Sharpe Ratio | {self.data.sharpe_ratio:.2f} |
| Max Drawdown | {self.data.max_drawdown*100:.2f}% |
| Win Rate | {self.data.win_rate*100:.1f}% |
| Profit Factor | {self.data.profit_factor:.2f} |
| Total Trades | {self.data.total_trades} |

---

## Trade Statistics

| Metric | Value |
|--------|-------|
| Total Trades | {self.data.total_trades} |
| Winning Trades | {self.data.winning_trades} |
| Losing Trades | {self.data.losing_trades} |
| Win Rate | {self.data.win_rate*100:.2f}% |
| Average Trade | ${self.data.avg_trade_pnl:.2f} |
| Average Winner | ${self.data.avg_winner:.2f} |
| Average Loser | ${self.data.avg_loser:.2f} |
| Largest Winner | ${self.data.largest_winner:.2f} |
| Largest Loser | ${self.data.largest_loser:.2f} |

---

## Risk Metrics

| Metric | Value |
|--------|-------|
| Sharpe Ratio | {self.data.sharpe_ratio:.3f} |
| Sortino Ratio | {self.data.sortino_ratio:.3f} |
| Calmar Ratio | {self.data.calmar_ratio:.3f} |
| Max Drawdown | {self.data.max_drawdown*100:.2f}% |
| Volatility | {self.data.volatility*100:.2f}% |

---

## Strategy Configuration

```json
{json.dumps(self.data.strategy_params, indent=2)}
```

---

*Generated by AlphaTerminal Pro | {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}*  
*Report ID: {self.report_id}*
"""
        
        return md.encode('utf-8')
    
    def _generate_json(self) -> bytes:
        """Generate JSON backtest report."""
        data = {
            "report_id": self.report_id,
            "report_type": "backtest",
            "generated_at": datetime.now().isoformat(),
            
            "metadata": {
                "symbol": self.data.symbol,
                "interval": self.data.interval,
                "strategy_name": self.data.strategy_name,
                "start_date": self.data.start_date.isoformat(),
                "end_date": self.data.end_date.isoformat(),
                "total_bars": self.data.total_bars,
            },
            
            "performance": {
                "initial_capital": self.data.initial_capital,
                "final_equity": self.data.final_equity,
                "total_return": self.data.total_return,
                "total_return_pct": self.data.total_return_pct,
            },
            
            "risk": {
                "max_drawdown": self.data.max_drawdown,
                "sharpe_ratio": self.data.sharpe_ratio,
                "sortino_ratio": self.data.sortino_ratio,
                "calmar_ratio": self.data.calmar_ratio,
                "volatility": self.data.volatility,
            },
            
            "trades": {
                "total_trades": self.data.total_trades,
                "winning_trades": self.data.winning_trades,
                "losing_trades": self.data.losing_trades,
                "win_rate": self.data.win_rate,
                "profit_factor": self.data.profit_factor,
                "avg_trade_pnl": self.data.avg_trade_pnl,
                "avg_winner": self.data.avg_winner,
                "avg_loser": self.data.avg_loser,
                "largest_winner": self.data.largest_winner,
                "largest_loser": self.data.largest_loser,
                "avg_holding_hours": self.data.avg_holding_hours,
            },
            
            "strategy_params": self.data.strategy_params,
            
            "data": {
                "equity_curve": self.data.equity_curve,
                "trades": self.data.trades,
                "monthly_returns": self.data.monthly_returns,
            }
        }
        
        return json.dumps(data, indent=2).encode('utf-8')


# =============================================================================
# SIGNAL REPORT GENERATOR
# =============================================================================

class SignalReportGenerator(BaseReportGenerator):
    """Generate trading signal reports."""
    
    report_type = ReportType.SIGNAL
    
    def __init__(
        self,
        signals: List[Dict[str, Any]],
        symbol: str,
        dark_mode: bool = False
    ):
        """
        Initialize signal report generator.
        
        Args:
            signals: List of trading signals
            symbol: Trading symbol
            dark_mode: Use dark theme
        """
        super().__init__(dark_mode)
        self.signals = signals
        self.symbol = symbol
        
        self.report_id = str(uuid.uuid4())[:8]
        self.metadata = ReportMetadata(
            report_id=self.report_id,
            report_type=ReportType.SIGNAL,
            title=f"Signal Report - {symbol}",
            symbol=symbol,
        )
    
    def _generate_html(self) -> bytes:
        """Generate HTML signal report."""
        
        signal_rows = ""
        for s in self.signals:
            signal_type = s.get('type', 'unknown')
            color = "#4CAF50" if signal_type in ['buy', 'long'] else "#F44336"
            
            signal_rows += f"""
            <tr>
                <td>{s.get('timestamp', '')}</td>
                <td style="color: {color}; font-weight: bold;">{signal_type.upper()}</td>
                <td>{s.get('price', 0):.2f}</td>
                <td>{s.get('target', '-')}</td>
                <td>{s.get('stop_loss', '-')}</td>
                <td>{s.get('confidence', 0)*100:.0f}%</td>
                <td>{s.get('strategy', '-')}</td>
            </tr>
            """
        
        html = f"""
<!DOCTYPE html>
<html>
<head>
    <meta charset="UTF-8">
    <title>{self.metadata.title}</title>
    <style>
        body {{ font-family: sans-serif; background: #1E1E1E; color: #FFF; padding: 20px; }}
        h1 {{ text-align: center; }}
        table {{ width: 100%; border-collapse: collapse; margin-top: 20px; }}
        th, td {{ padding: 12px; border: 1px solid #333; }}
        th {{ background: #333; }}
    </style>
</head>
<body>
    <h1>📈 {self.metadata.title}</h1>
    <p style="text-align: center;">Generated: {datetime.now().strftime('%Y-%m-%d %H:%M')}</p>
    
    <table>
        <tr>
            <th>Time</th>
            <th>Signal</th>
            <th>Price</th>
            <th>Target</th>
            <th>Stop Loss</th>
            <th>Confidence</th>
            <th>Strategy</th>
        </tr>
        {signal_rows}
    </table>
</body>
</html>
        """
        
        return html.encode('utf-8')
    
    def _generate_markdown(self) -> bytes:
        """Generate Markdown signal report."""
        md = f"# 📈 {self.metadata.title}\n\n"
        md += f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M')}\n\n"
        md += "| Time | Signal | Price | Target | Stop Loss | Confidence | Strategy |\n"
        md += "|------|--------|-------|--------|-----------|------------|----------|\n"
        
        for s in self.signals:
            md += f"| {s.get('timestamp', '')} | {s.get('type', '').upper()} | {s.get('price', 0):.2f} | {s.get('target', '-')} | {s.get('stop_loss', '-')} | {s.get('confidence', 0)*100:.0f}% | {s.get('strategy', '-')} |\n"
        
        return md.encode('utf-8')
    
    def _generate_json(self) -> bytes:
        """Generate JSON signal report."""
        data = {
            "report_id": self.report_id,
            "symbol": self.symbol,
            "generated_at": datetime.now().isoformat(),
            "signals": self.signals,
        }
        return json.dumps(data, indent=2).encode('utf-8')


# =============================================================================
# ALL EXPORTS
# =============================================================================

__all__ = [
    "BacktestReportData",
    "BaseReportGenerator",
    "BacktestReportGenerator",
    "SignalReportGenerator",
]
