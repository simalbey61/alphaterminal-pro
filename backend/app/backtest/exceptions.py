"""
AlphaTerminal Pro - Backtest Exception Classes
===============================================

Kurumsal seviye exception hierarchy for backtesting operations.

Exception Hierarchy:
    BacktestError (base)
    ├── ConfigurationError
    │   ├── InvalidConfigError
    │   └── MissingConfigError
    ├── DataError
    │   ├── InsufficientDataError
    │   ├── InvalidDataError
    │   ├── DataGapError
    │   └── DataTypeError
    ├── ExecutionError
    │   ├── OrderRejectedError
    │   ├── InsufficientFundsError
    │   ├── PositionError
    │   └── FillError
    ├── StrategyError
    │   ├── StrategyInitError
    │   ├── SignalGenerationError
    │   └── InvalidSignalError
    ├── MetricsError
    │   ├── CalculationError
    │   └── InsufficientTradesError
    └── ReportError
        ├── ReportGenerationError
        └── VisualizationError

Author: AlphaTerminal Team
Version: 1.0.0
"""

from typing import Any, Dict, List, Optional
from datetime import datetime


class BacktestError(Exception):
    """
    Base exception for all backtest-related errors.
    
    Attributes:
        message: Human-readable error message
        error_code: Unique error code for programmatic handling
        details: Additional context about the error
        timestamp: When the error occurred
        recoverable: Whether the error can be recovered from
    """
    
    def __init__(
        self,
        message: str,
        error_code: str = "BT_ERROR",
        details: Optional[Dict[str, Any]] = None,
        recoverable: bool = False
    ):
        self.message = message
        self.error_code = error_code
        self.details = details or {}
        self.timestamp = datetime.utcnow()
        self.recoverable = recoverable
        super().__init__(self.message)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert exception to dictionary for logging/serialization."""
        return {
            "error_type": self.__class__.__name__,
            "error_code": self.error_code,
            "message": self.message,
            "details": self.details,
            "timestamp": self.timestamp.isoformat(),
            "recoverable": self.recoverable
        }
    
    def __str__(self) -> str:
        return f"[{self.error_code}] {self.message}"
    
    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(message='{self.message}', error_code='{self.error_code}')"


# =============================================================================
# CONFIGURATION ERRORS
# =============================================================================

class ConfigurationError(BacktestError):
    """Base exception for configuration-related errors."""
    
    def __init__(self, message: str, **kwargs):
        super().__init__(
            message=message,
            error_code=kwargs.pop("error_code", "BT_CONFIG_ERROR"),
            **kwargs
        )


class InvalidConfigError(ConfigurationError):
    """Raised when configuration values are invalid."""
    
    def __init__(
        self,
        param_name: str,
        param_value: Any,
        expected: str,
        message: Optional[str] = None
    ):
        self.param_name = param_name
        self.param_value = param_value
        self.expected = expected
        
        msg = message or f"Invalid configuration for '{param_name}': got {param_value}, expected {expected}"
        super().__init__(
            message=msg,
            error_code="BT_INVALID_CONFIG",
            details={
                "param_name": param_name,
                "param_value": str(param_value),
                "expected": expected
            }
        )


class MissingConfigError(ConfigurationError):
    """Raised when required configuration is missing."""
    
    def __init__(self, missing_params: List[str]):
        self.missing_params = missing_params
        super().__init__(
            message=f"Missing required configuration parameters: {', '.join(missing_params)}",
            error_code="BT_MISSING_CONFIG",
            details={"missing_params": missing_params}
        )


# =============================================================================
# DATA ERRORS
# =============================================================================

class DataError(BacktestError):
    """Base exception for data-related errors."""
    
    def __init__(self, message: str, symbol: Optional[str] = None, **kwargs):
        self.symbol = symbol
        details = kwargs.pop("details", {})
        if symbol:
            details["symbol"] = symbol
        super().__init__(
            message=message,
            error_code=kwargs.pop("error_code", "BT_DATA_ERROR"),
            details=details,
            **kwargs
        )


class InsufficientDataError(DataError):
    """Raised when there's not enough data for backtesting."""
    
    def __init__(
        self,
        symbol: str,
        available_bars: int,
        required_bars: int,
        timeframe: Optional[str] = None
    ):
        self.available_bars = available_bars
        self.required_bars = required_bars
        self.timeframe = timeframe
        
        super().__init__(
            message=f"Insufficient data for {symbol}: {available_bars} bars available, {required_bars} required",
            symbol=symbol,
            error_code="BT_INSUFFICIENT_DATA",
            details={
                "available_bars": available_bars,
                "required_bars": required_bars,
                "timeframe": timeframe
            },
            recoverable=False
        )


class InvalidDataError(DataError):
    """Raised when data format or content is invalid."""
    
    def __init__(
        self,
        symbol: str,
        issue: str,
        row_index: Optional[int] = None,
        column: Optional[str] = None
    ):
        self.issue = issue
        self.row_index = row_index
        self.column = column
        
        location = ""
        if row_index is not None:
            location = f" at row {row_index}"
        if column:
            location += f" column '{column}'"
        
        super().__init__(
            message=f"Invalid data for {symbol}{location}: {issue}",
            symbol=symbol,
            error_code="BT_INVALID_DATA",
            details={
                "issue": issue,
                "row_index": row_index,
                "column": column
            }
        )


class DataGapError(DataError):
    """Raised when there are gaps in the data."""
    
    def __init__(
        self,
        symbol: str,
        gap_start: datetime,
        gap_end: datetime,
        expected_bars: int,
        actual_bars: int
    ):
        self.gap_start = gap_start
        self.gap_end = gap_end
        self.expected_bars = expected_bars
        self.actual_bars = actual_bars
        
        super().__init__(
            message=f"Data gap detected for {symbol} between {gap_start} and {gap_end}",
            symbol=symbol,
            error_code="BT_DATA_GAP",
            details={
                "gap_start": gap_start.isoformat(),
                "gap_end": gap_end.isoformat(),
                "expected_bars": expected_bars,
                "actual_bars": actual_bars,
                "missing_bars": expected_bars - actual_bars
            },
            recoverable=True  # Can potentially be handled by filling gaps
        )


class DataTypeError(DataError):
    """Raised when data types are incorrect."""
    
    def __init__(
        self,
        symbol: str,
        column: str,
        expected_type: str,
        actual_type: str
    ):
        super().__init__(
            message=f"Invalid data type for {symbol}.{column}: expected {expected_type}, got {actual_type}",
            symbol=symbol,
            error_code="BT_DATA_TYPE",
            details={
                "column": column,
                "expected_type": expected_type,
                "actual_type": actual_type
            }
        )


# =============================================================================
# EXECUTION ERRORS
# =============================================================================

class ExecutionError(BacktestError):
    """Base exception for order execution errors."""
    
    def __init__(
        self,
        message: str,
        order_id: Optional[str] = None,
        symbol: Optional[str] = None,
        **kwargs
    ):
        self.order_id = order_id
        self.symbol = symbol
        details = kwargs.pop("details", {})
        if order_id:
            details["order_id"] = order_id
        if symbol:
            details["symbol"] = symbol
        super().__init__(
            message=message,
            error_code=kwargs.pop("error_code", "BT_EXECUTION_ERROR"),
            details=details,
            **kwargs
        )


class OrderRejectedError(ExecutionError):
    """Raised when an order is rejected."""
    
    def __init__(
        self,
        order_id: str,
        symbol: str,
        reason: str,
        order_details: Optional[Dict] = None
    ):
        self.reason = reason
        self.order_details = order_details or {}
        
        super().__init__(
            message=f"Order {order_id} for {symbol} rejected: {reason}",
            order_id=order_id,
            symbol=symbol,
            error_code="BT_ORDER_REJECTED",
            details={
                "reason": reason,
                "order_details": order_details
            }
        )


class InsufficientFundsError(ExecutionError):
    """Raised when there are insufficient funds for an order."""
    
    def __init__(
        self,
        order_id: str,
        symbol: str,
        required_amount: float,
        available_amount: float
    ):
        self.required_amount = required_amount
        self.available_amount = available_amount
        
        super().__init__(
            message=f"Insufficient funds for order {order_id}: need {required_amount:.2f}, have {available_amount:.2f}",
            order_id=order_id,
            symbol=symbol,
            error_code="BT_INSUFFICIENT_FUNDS",
            details={
                "required_amount": required_amount,
                "available_amount": available_amount,
                "shortfall": required_amount - available_amount
            }
        )


class PositionError(ExecutionError):
    """Raised when there's an error with position management."""
    
    def __init__(
        self,
        symbol: str,
        issue: str,
        current_position: Optional[int] = None,
        requested_action: Optional[str] = None
    ):
        self.issue = issue
        self.current_position = current_position
        self.requested_action = requested_action
        
        super().__init__(
            message=f"Position error for {symbol}: {issue}",
            symbol=symbol,
            error_code="BT_POSITION_ERROR",
            details={
                "issue": issue,
                "current_position": current_position,
                "requested_action": requested_action
            }
        )


class FillError(ExecutionError):
    """Raised when order cannot be filled."""
    
    def __init__(
        self,
        order_id: str,
        symbol: str,
        reason: str,
        partial_fill: Optional[int] = None
    ):
        self.reason = reason
        self.partial_fill = partial_fill
        
        super().__init__(
            message=f"Cannot fill order {order_id} for {symbol}: {reason}",
            order_id=order_id,
            symbol=symbol,
            error_code="BT_FILL_ERROR",
            details={
                "reason": reason,
                "partial_fill": partial_fill
            },
            recoverable=True  # Partial fills may be acceptable
        )


# =============================================================================
# STRATEGY ERRORS
# =============================================================================

class StrategyError(BacktestError):
    """Base exception for strategy-related errors."""
    
    def __init__(
        self,
        message: str,
        strategy_name: Optional[str] = None,
        **kwargs
    ):
        self.strategy_name = strategy_name
        details = kwargs.pop("details", {})
        if strategy_name:
            details["strategy_name"] = strategy_name
        super().__init__(
            message=message,
            error_code=kwargs.pop("error_code", "BT_STRATEGY_ERROR"),
            details=details,
            **kwargs
        )


class StrategyInitError(StrategyError):
    """Raised when strategy initialization fails."""
    
    def __init__(
        self,
        strategy_name: str,
        reason: str,
        missing_params: Optional[List[str]] = None
    ):
        self.reason = reason
        self.missing_params = missing_params
        
        super().__init__(
            message=f"Failed to initialize strategy '{strategy_name}': {reason}",
            strategy_name=strategy_name,
            error_code="BT_STRATEGY_INIT",
            details={
                "reason": reason,
                "missing_params": missing_params
            }
        )


class SignalGenerationError(StrategyError):
    """Raised when signal generation fails."""
    
    def __init__(
        self,
        strategy_name: str,
        bar_index: int,
        reason: str,
        exception: Optional[Exception] = None
    ):
        self.bar_index = bar_index
        self.reason = reason
        self.original_exception = exception
        
        super().__init__(
            message=f"Strategy '{strategy_name}' failed to generate signal at bar {bar_index}: {reason}",
            strategy_name=strategy_name,
            error_code="BT_SIGNAL_GEN",
            details={
                "bar_index": bar_index,
                "reason": reason,
                "original_exception": str(exception) if exception else None
            },
            recoverable=True  # Can skip this bar and continue
        )


class InvalidSignalError(StrategyError):
    """Raised when a signal has invalid values."""
    
    def __init__(
        self,
        strategy_name: str,
        signal_type: str,
        issue: str,
        signal_data: Optional[Dict] = None
    ):
        self.signal_type = signal_type
        self.issue = issue
        self.signal_data = signal_data
        
        super().__init__(
            message=f"Invalid {signal_type} signal from '{strategy_name}': {issue}",
            strategy_name=strategy_name,
            error_code="BT_INVALID_SIGNAL",
            details={
                "signal_type": signal_type,
                "issue": issue,
                "signal_data": signal_data
            }
        )


# =============================================================================
# METRICS ERRORS
# =============================================================================

class MetricsError(BacktestError):
    """Base exception for metrics calculation errors."""
    
    def __init__(self, message: str, **kwargs):
        super().__init__(
            message=message,
            error_code=kwargs.pop("error_code", "BT_METRICS_ERROR"),
            **kwargs
        )


class CalculationError(MetricsError):
    """Raised when a metric calculation fails."""
    
    def __init__(
        self,
        metric_name: str,
        reason: str,
        input_data: Optional[Dict] = None
    ):
        self.metric_name = metric_name
        self.reason = reason
        self.input_data = input_data
        
        super().__init__(
            message=f"Failed to calculate {metric_name}: {reason}",
            error_code="BT_CALC_ERROR",
            details={
                "metric_name": metric_name,
                "reason": reason,
                "input_data_summary": str(input_data)[:200] if input_data else None
            }
        )


class InsufficientTradesError(MetricsError):
    """Raised when there aren't enough trades for meaningful metrics."""
    
    def __init__(
        self,
        trade_count: int,
        minimum_required: int,
        metric_name: Optional[str] = None
    ):
        self.trade_count = trade_count
        self.minimum_required = minimum_required
        self.metric_name = metric_name
        
        metric_str = f" for {metric_name}" if metric_name else ""
        super().__init__(
            message=f"Insufficient trades{metric_str}: {trade_count} trades, minimum {minimum_required} required",
            error_code="BT_INSUFFICIENT_TRADES",
            details={
                "trade_count": trade_count,
                "minimum_required": minimum_required,
                "metric_name": metric_name
            }
        )


# =============================================================================
# REPORT ERRORS
# =============================================================================

class ReportError(BacktestError):
    """Base exception for report generation errors."""
    
    def __init__(self, message: str, **kwargs):
        super().__init__(
            message=message,
            error_code=kwargs.pop("error_code", "BT_REPORT_ERROR"),
            **kwargs
        )


class ReportGenerationError(ReportError):
    """Raised when report generation fails."""
    
    def __init__(
        self,
        report_type: str,
        reason: str,
        output_path: Optional[str] = None
    ):
        self.report_type = report_type
        self.reason = reason
        self.output_path = output_path
        
        super().__init__(
            message=f"Failed to generate {report_type} report: {reason}",
            error_code="BT_REPORT_GEN",
            details={
                "report_type": report_type,
                "reason": reason,
                "output_path": output_path
            }
        )


class VisualizationError(ReportError):
    """Raised when visualization generation fails."""
    
    def __init__(
        self,
        chart_type: str,
        reason: str
    ):
        self.chart_type = chart_type
        self.reason = reason
        
        super().__init__(
            message=f"Failed to generate {chart_type} chart: {reason}",
            error_code="BT_VIZ_ERROR",
            details={
                "chart_type": chart_type,
                "reason": reason
            }
        )


# =============================================================================
# UTILITY FUNCTIONS
# =============================================================================

def format_exception_chain(exc: Exception) -> str:
    """
    Format exception chain for logging.
    
    Args:
        exc: The exception to format
        
    Returns:
        Formatted string with exception chain
    """
    messages = []
    current = exc
    
    while current is not None:
        if isinstance(current, BacktestError):
            messages.append(f"[{current.error_code}] {current.message}")
        else:
            messages.append(f"[{type(current).__name__}] {str(current)}")
        current = current.__cause__
    
    return " -> ".join(messages)


def is_recoverable(exc: Exception) -> bool:
    """
    Check if an exception is recoverable.
    
    Args:
        exc: The exception to check
        
    Returns:
        True if the exception is recoverable
    """
    if isinstance(exc, BacktestError):
        return exc.recoverable
    return False
