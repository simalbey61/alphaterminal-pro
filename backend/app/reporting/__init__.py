"""
AlphaTerminal Pro - Reporting System
====================================

Enterprise-grade reporting, visualization, and notifications.

Features:
- Multiple report formats (HTML, Markdown, JSON, PDF)
- Professional chart generation (equity curves, drawdowns, heatmaps)
- Telegram notifications for signals and alerts
- Customizable templates
- Dark/light themes

Quick Start:
    from app.reporting import (
        BacktestReportGenerator, BacktestReportData,
        TelegramBot, NotificationService
    )
    
    # Generate backtest report
    data = BacktestReportData(...)
    generator = BacktestReportGenerator(data)
    html_report = generator.generate(ReportFormat.HTML)
    
    # Send Telegram notification
    bot = TelegramBot(token="...", chat_id="...")
    await bot.send_signal(signal)

Author: AlphaTerminal Team
Version: 1.0.0
"""

# Types
from app.reporting.types import (
    ReportType,
    ReportFormat,
    ChartType,
    NotificationChannel,
    NotificationPriority,
    SignalType,
    ReportMetadata,
    ChartConfig,
    TradingSignal,
    NotificationMessage,
    ReportSection,
    MetricsSection,
    TableSection,
)

# Visualizations
from app.reporting.visualizations.charts import (
    COLORS,
    DARK_THEME,
    LIGHT_THEME,
    ChartGenerator,
)

# Generators
from app.reporting.generators.report_generator import (
    BacktestReportData,
    BaseReportGenerator,
    BacktestReportGenerator,
    SignalReportGenerator,
)

# Notifications
from app.reporting.notifications.telegram import (
    MessageFormatter,
    TelegramBot,
    NotificationService,
)


__all__ = [
    # Types
    "ReportType",
    "ReportFormat",
    "ChartType",
    "NotificationChannel",
    "NotificationPriority",
    "SignalType",
    "ReportMetadata",
    "ChartConfig",
    "TradingSignal",
    "NotificationMessage",
    "ReportSection",
    "MetricsSection",
    "TableSection",
    
    # Visualizations
    "COLORS",
    "DARK_THEME",
    "LIGHT_THEME",
    "ChartGenerator",
    
    # Generators
    "BacktestReportData",
    "BaseReportGenerator",
    "BacktestReportGenerator",
    "SignalReportGenerator",
    
    # Notifications
    "MessageFormatter",
    "TelegramBot",
    "NotificationService",
]

__version__ = "1.0.0"
