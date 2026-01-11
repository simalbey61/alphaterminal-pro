"""
AlphaTerminal Pro - Reporting Enums and Types
=============================================

Enumerations and data types for reporting system.

Author: AlphaTerminal Team
Version: 1.0.0
"""

from enum import Enum, auto
from typing import Optional, Dict, List, Any
from dataclasses import dataclass, field
from datetime import datetime


# =============================================================================
# ENUMS
# =============================================================================

class ReportType(str, Enum):
    """Types of reports."""
    
    BACKTEST = "backtest"
    STRATEGY = "strategy"
    DAILY = "daily"
    WEEKLY = "weekly"
    MONTHLY = "monthly"
    PERFORMANCE = "performance"
    RISK = "risk"
    ML_TRAINING = "ml_training"
    SIGNAL = "signal"


class ReportFormat(str, Enum):
    """Report output formats."""
    
    PDF = "pdf"
    HTML = "html"
    MARKDOWN = "markdown"
    JSON = "json"
    EXCEL = "excel"


class ChartType(str, Enum):
    """Chart visualization types."""
    
    LINE = "line"
    CANDLESTICK = "candlestick"
    BAR = "bar"
    AREA = "area"
    SCATTER = "scatter"
    PIE = "pie"
    HEATMAP = "heatmap"
    HISTOGRAM = "histogram"
    DRAWDOWN = "drawdown"
    EQUITY_CURVE = "equity_curve"
    RETURNS_DISTRIBUTION = "returns_distribution"
    MONTHLY_RETURNS = "monthly_returns"
    UNDERWATER = "underwater"


class NotificationChannel(str, Enum):
    """Notification channels."""
    
    TELEGRAM = "telegram"
    EMAIL = "email"
    WEBHOOK = "webhook"
    SLACK = "slack"
    DISCORD = "discord"


class NotificationPriority(str, Enum):
    """Notification priority levels."""
    
    LOW = "low"
    NORMAL = "normal"
    HIGH = "high"
    CRITICAL = "critical"


class SignalType(str, Enum):
    """Trading signal types."""
    
    BUY = "buy"
    SELL = "sell"
    HOLD = "hold"
    CLOSE = "close"
    STOP_LOSS = "stop_loss"
    TAKE_PROFIT = "take_profit"


# =============================================================================
# DATA CLASSES
# =============================================================================

@dataclass
class ReportMetadata:
    """Metadata for a report."""
    
    report_id: str
    report_type: ReportType
    title: str
    description: str = ""
    
    created_at: datetime = field(default_factory=datetime.now)
    created_by: str = "system"
    
    symbol: Optional[str] = None
    interval: Optional[str] = None
    start_date: Optional[datetime] = None
    end_date: Optional[datetime] = None
    
    tags: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ChartConfig:
    """Configuration for a chart."""
    
    chart_type: ChartType
    title: str
    subtitle: str = ""
    
    width: int = 1200
    height: int = 600
    
    show_legend: bool = True
    show_grid: bool = True
    dark_mode: bool = False
    
    colors: Optional[List[str]] = None
    x_label: str = ""
    y_label: str = ""
    
    annotations: List[Dict[str, Any]] = field(default_factory=list)


@dataclass
class TradingSignal:
    """Trading signal for notifications."""
    
    signal_id: str
    signal_type: SignalType
    symbol: str
    
    price: float
    target_price: Optional[float] = None
    stop_loss: Optional[float] = None
    
    confidence: float = 0.5
    strategy_name: str = ""
    timeframe: str = ""
    
    created_at: datetime = field(default_factory=datetime.now)
    expires_at: Optional[datetime] = None
    
    indicators: Dict[str, float] = field(default_factory=dict)
    notes: str = ""
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "signal_id": self.signal_id,
            "signal_type": self.signal_type.value,
            "symbol": self.symbol,
            "price": self.price,
            "target_price": self.target_price,
            "stop_loss": self.stop_loss,
            "confidence": self.confidence,
            "strategy_name": self.strategy_name,
            "timeframe": self.timeframe,
            "created_at": self.created_at.isoformat(),
            "expires_at": self.expires_at.isoformat() if self.expires_at else None,
            "indicators": self.indicators,
            "notes": self.notes,
        }


@dataclass
class NotificationMessage:
    """Notification message."""
    
    message_id: str
    channel: NotificationChannel
    priority: NotificationPriority
    
    title: str
    body: str
    
    symbol: Optional[str] = None
    signal: Optional[TradingSignal] = None
    
    created_at: datetime = field(default_factory=datetime.now)
    sent_at: Optional[datetime] = None
    delivered: bool = False
    
    metadata: Dict[str, Any] = field(default_factory=dict)


# =============================================================================
# REPORT SECTIONS
# =============================================================================

@dataclass
class ReportSection:
    """A section within a report."""
    
    section_id: str
    title: str
    content_type: str  # "text", "table", "chart", "metrics"
    content: Any
    order: int = 0


@dataclass
class MetricsSection:
    """Metrics section data."""
    
    metrics: Dict[str, float]
    title: str = "Performance Metrics"
    columns: int = 3  # Display in N columns


@dataclass
class TableSection:
    """Table section data."""
    
    headers: List[str]
    rows: List[List[Any]]
    title: str = ""
    highlight_rows: List[int] = field(default_factory=list)


# =============================================================================
# ALL EXPORTS
# =============================================================================

__all__ = [
    # Enums
    "ReportType",
    "ReportFormat",
    "ChartType",
    "NotificationChannel",
    "NotificationPriority",
    "SignalType",
    
    # Data classes
    "ReportMetadata",
    "ChartConfig",
    "TradingSignal",
    "NotificationMessage",
    "ReportSection",
    "MetricsSection",
    "TableSection",
]
