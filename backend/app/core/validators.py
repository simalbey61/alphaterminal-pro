"""
AlphaTerminal Pro - Input Validators
====================================

Kurumsal seviye input validation.

Provides:
- DataFrame validation
- Numeric value validation
- Configuration validation
- Type checking decorators

Author: AlphaTerminal Team
Version: 1.0.0
"""

from __future__ import annotations

import logging
from functools import wraps
from typing import Any, Callable, Dict, List, Optional, Set, Tuple, Type, Union
from datetime import datetime, date
import inspect

import pandas as pd
import numpy as np

from app.backtest.exceptions import (
    InvalidDataError,
    InsufficientDataError,
    InvalidConfigError,
    MissingConfigError
)

logger = logging.getLogger(__name__)


# =============================================================================
# DATAFRAME VALIDATORS
# =============================================================================

class DataFrameValidator:
    """
    Comprehensive DataFrame validation.
    
    Example:
        ```python
        validator = DataFrameValidator()
        validator.validate_ohlcv(df, symbol="THYAO", min_rows=100)
        ```
    """
    
    OHLCV_COLUMNS = {'Open', 'High', 'Low', 'Close', 'Volume'}
    PRICE_COLUMNS = {'Open', 'High', 'Low', 'Close'}
    
    @classmethod
    def validate_ohlcv(
        cls,
        df: pd.DataFrame,
        symbol: str = "Unknown",
        min_rows: int = 1,
        max_nan_pct: float = 0.05,
        check_consistency: bool = True,
        raise_on_error: bool = True
    ) -> Tuple[bool, List[str]]:
        """
        Validate OHLCV DataFrame.
        
        Args:
            df: DataFrame to validate
            symbol: Symbol name for error messages
            min_rows: Minimum required rows
            max_nan_pct: Maximum allowed NaN percentage
            check_consistency: Check OHLC price consistency
            raise_on_error: Raise exception on first error
            
        Returns:
            Tuple of (is_valid, list_of_issues)
        """
        issues = []
        
        # Check if DataFrame
        if not isinstance(df, pd.DataFrame):
            issues.append(f"Expected DataFrame, got {type(df).__name__}")
            if raise_on_error:
                raise InvalidDataError(symbol=symbol, issue=issues[0])
            return False, issues
        
        # Check empty
        if df.empty:
            issues.append("DataFrame is empty")
            if raise_on_error:
                raise InsufficientDataError(
                    symbol=symbol,
                    available_bars=0,
                    required_bars=min_rows
                )
            return False, issues
        
        # Check minimum rows
        if len(df) < min_rows:
            issues.append(f"Insufficient rows: {len(df)} < {min_rows}")
            if raise_on_error:
                raise InsufficientDataError(
                    symbol=symbol,
                    available_bars=len(df),
                    required_bars=min_rows
                )
        
        # Check required columns
        missing_cols = cls.OHLCV_COLUMNS - set(df.columns)
        if missing_cols:
            issues.append(f"Missing columns: {missing_cols}")
            if raise_on_error:
                raise InvalidDataError(
                    symbol=symbol,
                    issue=f"Missing columns: {missing_cols}"
                )
        
        # Check index type
        if not isinstance(df.index, pd.DatetimeIndex):
            issues.append("Index must be DatetimeIndex")
            if raise_on_error:
                raise InvalidDataError(
                    symbol=symbol,
                    issue="Index must be DatetimeIndex"
                )
        
        # Check NaN values
        for col in cls.OHLCV_COLUMNS:
            if col in df.columns:
                nan_count = df[col].isna().sum()
                nan_pct = nan_count / len(df)
                if nan_pct > max_nan_pct:
                    issues.append(f"{col}: {nan_pct:.1%} NaN values (max {max_nan_pct:.1%})")
        
        # Check price consistency
        if check_consistency and cls.PRICE_COLUMNS.issubset(df.columns):
            # High should be >= Open, Close
            invalid_high = (df['High'] < df[['Open', 'Close']].max(axis=1)).sum()
            if invalid_high > 0:
                issues.append(f"{invalid_high} rows where High < max(Open, Close)")
            
            # Low should be <= Open, Close
            invalid_low = (df['Low'] > df[['Open', 'Close']].min(axis=1)).sum()
            if invalid_low > 0:
                issues.append(f"{invalid_low} rows where Low > min(Open, Close)")
            
            # No negative prices
            for col in cls.PRICE_COLUMNS:
                neg_count = (df[col] < 0).sum()
                if neg_count > 0:
                    issues.append(f"{col}: {neg_count} negative values")
        
        # Check for duplicates
        dup_count = df.index.duplicated().sum()
        if dup_count > 0:
            issues.append(f"{dup_count} duplicate timestamps")
        
        is_valid = len(issues) == 0
        
        if not is_valid:
            logger.warning(f"DataFrame validation issues for {symbol}: {issues}")
        
        return is_valid, issues
    
    @classmethod
    def validate_returns(
        cls,
        returns: pd.Series,
        max_abs_return: float = 1.0,
        raise_on_error: bool = True
    ) -> Tuple[bool, List[str]]:
        """
        Validate returns series.
        
        Args:
            returns: Returns series
            max_abs_return: Maximum absolute return (flag outliers)
            raise_on_error: Raise exception on error
            
        Returns:
            Tuple of (is_valid, list_of_issues)
        """
        issues = []
        
        if not isinstance(returns, pd.Series):
            issues.append(f"Expected Series, got {type(returns).__name__}")
            if raise_on_error:
                raise InvalidDataError(symbol="returns", issue=issues[0])
            return False, issues
        
        if len(returns) == 0:
            issues.append("Returns series is empty")
            return False, issues
        
        # Check for NaN
        nan_count = returns.isna().sum()
        if nan_count > 0:
            issues.append(f"{nan_count} NaN values in returns")
        
        # Check for infinite values
        inf_count = np.isinf(returns).sum()
        if inf_count > 0:
            issues.append(f"{inf_count} infinite values in returns")
        
        # Check for extreme returns
        extreme_count = (returns.abs() > max_abs_return).sum()
        if extreme_count > 0:
            issues.append(f"{extreme_count} extreme returns (>{max_abs_return:.0%})")
        
        return len(issues) == 0, issues


# =============================================================================
# NUMERIC VALIDATORS
# =============================================================================

class NumericValidator:
    """
    Numeric value validation.
    """
    
    @staticmethod
    def validate_positive(
        value: float,
        name: str,
        allow_zero: bool = False
    ) -> float:
        """Validate that value is positive."""
        if value is None:
            raise InvalidConfigError(
                param_name=name,
                param_value=value,
                expected="positive number"
            )
        
        if allow_zero:
            if value < 0:
                raise InvalidConfigError(
                    param_name=name,
                    param_value=value,
                    expected="non-negative number"
                )
        else:
            if value <= 0:
                raise InvalidConfigError(
                    param_name=name,
                    param_value=value,
                    expected="positive number"
                )
        
        return value
    
    @staticmethod
    def validate_range(
        value: float,
        name: str,
        min_val: Optional[float] = None,
        max_val: Optional[float] = None,
        inclusive: bool = True
    ) -> float:
        """Validate that value is within range."""
        if value is None:
            raise InvalidConfigError(
                param_name=name,
                param_value=value,
                expected=f"number in range [{min_val}, {max_val}]"
            )
        
        if min_val is not None:
            if inclusive:
                if value < min_val:
                    raise InvalidConfigError(
                        param_name=name,
                        param_value=value,
                        expected=f">= {min_val}"
                    )
            else:
                if value <= min_val:
                    raise InvalidConfigError(
                        param_name=name,
                        param_value=value,
                        expected=f"> {min_val}"
                    )
        
        if max_val is not None:
            if inclusive:
                if value > max_val:
                    raise InvalidConfigError(
                        param_name=name,
                        param_value=value,
                        expected=f"<= {max_val}"
                    )
            else:
                if value >= max_val:
                    raise InvalidConfigError(
                        param_name=name,
                        param_value=value,
                        expected=f"< {max_val}"
                    )
        
        return value
    
    @staticmethod
    def validate_percentage(
        value: float,
        name: str,
        allow_zero: bool = True,
        allow_hundred: bool = True
    ) -> float:
        """Validate percentage (0-1 or 0-100)."""
        if value is None:
            raise InvalidConfigError(
                param_name=name,
                param_value=value,
                expected="percentage value"
            )
        
        # Normalize to 0-1 if > 1
        if value > 1:
            value = value / 100
        
        min_val = 0 if allow_zero else 0.0001
        max_val = 1 if allow_hundred else 0.9999
        
        return NumericValidator.validate_range(
            value, name, min_val, max_val
        )
    
    @staticmethod
    def validate_integer(
        value: Any,
        name: str,
        min_val: Optional[int] = None,
        max_val: Optional[int] = None
    ) -> int:
        """Validate integer value."""
        if value is None:
            raise InvalidConfigError(
                param_name=name,
                param_value=value,
                expected="integer"
            )
        
        try:
            int_value = int(value)
        except (ValueError, TypeError):
            raise InvalidConfigError(
                param_name=name,
                param_value=value,
                expected="integer"
            )
        
        if min_val is not None and int_value < min_val:
            raise InvalidConfigError(
                param_name=name,
                param_value=int_value,
                expected=f">= {min_val}"
            )
        
        if max_val is not None and int_value > max_val:
            raise InvalidConfigError(
                param_name=name,
                param_value=int_value,
                expected=f"<= {max_val}"
            )
        
        return int_value
    
    @staticmethod
    def is_finite(value: float) -> bool:
        """Check if value is finite (not NaN or infinite)."""
        return np.isfinite(value)
    
    @staticmethod
    def safe_divide(
        numerator: float,
        denominator: float,
        default: float = 0.0
    ) -> float:
        """Safe division with default for zero denominator."""
        if denominator == 0 or not np.isfinite(denominator):
            return default
        result = numerator / denominator
        return result if np.isfinite(result) else default


# =============================================================================
# CONFIG VALIDATORS
# =============================================================================

class ConfigValidator:
    """
    Configuration validation.
    """
    
    @staticmethod
    def validate_required_fields(
        config: Dict[str, Any],
        required: List[str]
    ) -> None:
        """Validate that all required fields are present."""
        missing = [field for field in required if field not in config]
        if missing:
            raise MissingConfigError(missing)
    
    @staticmethod
    def validate_field_types(
        config: Dict[str, Any],
        type_map: Dict[str, Type]
    ) -> None:
        """Validate field types."""
        for field, expected_type in type_map.items():
            if field in config:
                value = config[field]
                if not isinstance(value, expected_type):
                    raise InvalidConfigError(
                        param_name=field,
                        param_value=value,
                        expected=f"type {expected_type.__name__}"
                    )
    
    @staticmethod
    def validate_backtest_config(config: Any) -> None:
        """Validate BacktestConfig object."""
        from app.backtest.engine import BacktestConfig
        
        if not isinstance(config, BacktestConfig):
            raise InvalidConfigError(
                param_name="config",
                param_value=type(config).__name__,
                expected="BacktestConfig"
            )
        
        # Validate individual fields
        NumericValidator.validate_positive(config.initial_capital, "initial_capital")
        NumericValidator.validate_percentage(config.commission_rate, "commission_rate")
        NumericValidator.validate_percentage(config.slippage_rate, "slippage_rate")
        NumericValidator.validate_percentage(config.max_position_size, "max_position_size")
        NumericValidator.validate_integer(config.max_positions, "max_positions", min_val=1)
        NumericValidator.validate_percentage(config.risk_per_trade, "risk_per_trade")


# =============================================================================
# VALIDATION DECORATORS
# =============================================================================

def validate_dataframe(
    df_param: str = "data",
    min_rows: int = 1,
    required_columns: Optional[Set[str]] = None
):
    """
    Decorator to validate DataFrame parameter.
    
    Args:
        df_param: Name of DataFrame parameter
        min_rows: Minimum required rows
        required_columns: Required column names
        
    Example:
        ```python
        @validate_dataframe("data", min_rows=50, required_columns={'Close'})
        def my_function(data: pd.DataFrame):
            pass
        ```
    """
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        def wrapper(*args, **kwargs):
            # Get DataFrame from args or kwargs
            sig = inspect.signature(func)
            params = list(sig.parameters.keys())
            
            df = None
            if df_param in kwargs:
                df = kwargs[df_param]
            elif df_param in params:
                idx = params.index(df_param)
                if idx < len(args):
                    df = args[idx]
            
            if df is None:
                raise InvalidDataError(
                    symbol="Unknown",
                    issue=f"Missing required parameter: {df_param}"
                )
            
            # Validate
            if not isinstance(df, pd.DataFrame):
                raise InvalidDataError(
                    symbol="Unknown",
                    issue=f"Expected DataFrame, got {type(df).__name__}"
                )
            
            if len(df) < min_rows:
                raise InsufficientDataError(
                    symbol="Unknown",
                    available_bars=len(df),
                    required_bars=min_rows
                )
            
            if required_columns:
                missing = required_columns - set(df.columns)
                if missing:
                    raise InvalidDataError(
                        symbol="Unknown",
                        issue=f"Missing columns: {missing}"
                    )
            
            return func(*args, **kwargs)
        
        return wrapper
    return decorator


def validate_positive_params(*param_names: str):
    """
    Decorator to validate positive numeric parameters.
    
    Example:
        ```python
        @validate_positive_params("quantity", "price")
        def place_order(quantity: int, price: float):
            pass
        ```
    """
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        def wrapper(*args, **kwargs):
            sig = inspect.signature(func)
            params = list(sig.parameters.keys())
            
            for param_name in param_names:
                value = None
                if param_name in kwargs:
                    value = kwargs[param_name]
                elif param_name in params:
                    idx = params.index(param_name)
                    if idx < len(args):
                        value = args[idx]
                
                if value is not None and value <= 0:
                    raise InvalidConfigError(
                        param_name=param_name,
                        param_value=value,
                        expected="positive number"
                    )
            
            return func(*args, **kwargs)
        
        return wrapper
    return decorator


def validate_range_params(**param_ranges: Tuple[float, float]):
    """
    Decorator to validate parameter ranges.
    
    Example:
        ```python
        @validate_range_params(percentage=(0, 1), multiplier=(0.1, 10))
        def my_function(percentage: float, multiplier: float):
            pass
        ```
    """
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        def wrapper(*args, **kwargs):
            sig = inspect.signature(func)
            params = list(sig.parameters.keys())
            
            for param_name, (min_val, max_val) in param_ranges.items():
                value = None
                if param_name in kwargs:
                    value = kwargs[param_name]
                elif param_name in params:
                    idx = params.index(param_name)
                    if idx < len(args):
                        value = args[idx]
                
                if value is not None:
                    if value < min_val or value > max_val:
                        raise InvalidConfigError(
                            param_name=param_name,
                            param_value=value,
                            expected=f"between {min_val} and {max_val}"
                        )
            
            return func(*args, **kwargs)
        
        return wrapper
    return decorator


# =============================================================================
# CONVENIENCE FUNCTIONS
# =============================================================================

def validate_ohlcv(df: pd.DataFrame, symbol: str = "Unknown", **kwargs) -> bool:
    """
    Convenience function for OHLCV validation.
    
    Args:
        df: DataFrame to validate
        symbol: Symbol name
        **kwargs: Additional validation options
        
    Returns:
        True if valid
        
    Raises:
        InvalidDataError or InsufficientDataError if invalid
    """
    is_valid, issues = DataFrameValidator.validate_ohlcv(df, symbol, **kwargs)
    return is_valid


def validate_price(value: float, name: str = "price") -> float:
    """Validate price value (must be positive)."""
    return NumericValidator.validate_positive(value, name)


def validate_quantity(value: int, name: str = "quantity") -> int:
    """Validate quantity value (must be positive integer)."""
    return NumericValidator.validate_integer(value, name, min_val=1)


def validate_percentage(value: float, name: str = "percentage") -> float:
    """Validate percentage value (0-1)."""
    return NumericValidator.validate_percentage(value, name)
