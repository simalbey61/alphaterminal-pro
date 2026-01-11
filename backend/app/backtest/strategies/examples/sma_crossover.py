"""
AlphaTerminal Pro - SMA Crossover Strategy
==========================================

Basit ama etkili SMA Crossover stratejisi.

Mantık:
- Hızlı SMA, Yavaş SMA'yı yukarı keserse → LONG
- Hızlı SMA, Yavaş SMA'yı aşağı keserse → EXIT

Author: AlphaTerminal Team
Version: 1.0.0
"""

from __future__ import annotations

import logging
from typing import Optional

import pandas as pd
import numpy as np

from app.backtest.engine import BaseStrategy, Signal
from app.backtest.enums import SignalType

logger = logging.getLogger(__name__)


class SMACrossoverStrategy(BaseStrategy):
    """
    Simple Moving Average Crossover Strategy.
    
    Entry: Fast SMA crosses above Slow SMA
    Exit: Fast SMA crosses below Slow SMA OR stop loss/take profit hit
    
    Parameters:
        fast_period: Fast SMA period (default: 10)
        slow_period: Slow SMA period (default: 30)
        atr_period: ATR period for stop calculation (default: 14)
        atr_multiplier: Stop loss = Entry - (ATR * multiplier) (default: 2.0)
        risk_reward: Risk/reward ratio for take profit (default: 2.0)
        position_size: Position size as % of capital (default: 0.10)
    
    Example:
        ```python
        strategy = SMACrossoverStrategy(fast_period=10, slow_period=30)
        engine = BacktestEngine()
        result = engine.run(data, strategy, "THYAO")
        ```
    """
    
    name = "SMA Crossover"
    version = "1.0.0"
    description = "Simple Moving Average Crossover with ATR-based stops"
    
    def __init__(
        self,
        fast_period: int = 10,
        slow_period: int = 30,
        atr_period: int = 14,
        atr_multiplier: float = 2.0,
        risk_reward: float = 2.0,
        position_size: float = 0.10
    ):
        super().__init__()
        
        self.fast_period = fast_period
        self.slow_period = slow_period
        self.atr_period = atr_period
        self.atr_multiplier = atr_multiplier
        self.risk_reward = risk_reward
        self.position_size = position_size
        
        # Warmup needs the slowest indicator
        self.warmup_period = max(slow_period, atr_period) + 5
        
        # Store parameters
        self._parameters = {
            "fast_period": fast_period,
            "slow_period": slow_period,
            "atr_period": atr_period,
            "atr_multiplier": atr_multiplier,
            "risk_reward": risk_reward,
            "position_size": position_size
        }
    
    def generate_signal(self, data: pd.DataFrame) -> Signal:
        """
        Generate trading signal.
        
        Args:
            data: OHLCV DataFrame with history up to current bar
            
        Returns:
            Signal object
        """
        if len(data) < self.warmup_period:
            return Signal.no_action()
        
        # Calculate indicators
        close = data['Close']
        high = data['High']
        low = data['Low']
        
        fast_sma = close.rolling(self.fast_period).mean()
        slow_sma = close.rolling(self.slow_period).mean()
        
        # ATR for stop calculation
        tr = pd.concat([
            high - low,
            abs(high - close.shift(1)),
            abs(low - close.shift(1))
        ], axis=1).max(axis=1)
        atr = tr.rolling(self.atr_period).mean()
        
        # Current values
        current_fast = fast_sma.iloc[-1]
        current_slow = slow_sma.iloc[-1]
        prev_fast = fast_sma.iloc[-2]
        prev_slow = slow_sma.iloc[-2]
        current_atr = atr.iloc[-1]
        current_price = close.iloc[-1]
        
        # Check for NaN
        if pd.isna(current_fast) or pd.isna(current_slow) or pd.isna(current_atr):
            return Signal.no_action()
        
        # Exit signal: Fast crosses below Slow
        if self.is_long and prev_fast >= prev_slow and current_fast < current_slow:
            return Signal.exit_long(reason="SMA bearish cross")
        
        # Entry signal: Fast crosses above Slow
        if not self.is_in_position and prev_fast <= prev_slow and current_fast > current_slow:
            # Calculate stop loss and take profit
            stop_distance = current_atr * self.atr_multiplier
            stop_loss = current_price - stop_distance
            take_profit = current_price + (stop_distance * self.risk_reward)
            
            return Signal.long_entry(
                entry_price=current_price,
                stop_loss=stop_loss,
                take_profit=take_profit,
                position_size=self.position_size,
                strength=1.0,
                reason=f"SMA bullish cross (Fast:{current_fast:.2f} > Slow:{current_slow:.2f})"
            )
        
        return Signal.no_action()


class DualSMACrossoverStrategy(BaseStrategy):
    """
    Dual SMA Crossover with trend filter.
    
    Uses a third, longer SMA as trend filter.
    Only takes long signals when price is above trend SMA.
    
    Parameters:
        fast_period: Fast SMA period (default: 10)
        slow_period: Slow SMA period (default: 30)
        trend_period: Trend filter SMA period (default: 100)
    """
    
    name = "Dual SMA Crossover"
    version = "1.0.0"
    
    def __init__(
        self,
        fast_period: int = 10,
        slow_period: int = 30,
        trend_period: int = 100,
        atr_period: int = 14,
        atr_multiplier: float = 2.0,
        risk_reward: float = 2.0,
        position_size: float = 0.10
    ):
        super().__init__()
        
        self.fast_period = fast_period
        self.slow_period = slow_period
        self.trend_period = trend_period
        self.atr_period = atr_period
        self.atr_multiplier = atr_multiplier
        self.risk_reward = risk_reward
        self.position_size = position_size
        
        self.warmup_period = trend_period + 5
    
    def generate_signal(self, data: pd.DataFrame) -> Signal:
        """Generate signal with trend filter."""
        if len(data) < self.warmup_period:
            return Signal.no_action()
        
        close = data['Close']
        high = data['High']
        low = data['Low']
        
        fast_sma = close.rolling(self.fast_period).mean()
        slow_sma = close.rolling(self.slow_period).mean()
        trend_sma = close.rolling(self.trend_period).mean()
        
        # ATR
        tr = pd.concat([
            high - low,
            abs(high - close.shift(1)),
            abs(low - close.shift(1))
        ], axis=1).max(axis=1)
        atr = tr.rolling(self.atr_period).mean()
        
        current_fast = fast_sma.iloc[-1]
        current_slow = slow_sma.iloc[-1]
        current_trend = trend_sma.iloc[-1]
        prev_fast = fast_sma.iloc[-2]
        prev_slow = slow_sma.iloc[-2]
        current_atr = atr.iloc[-1]
        current_price = close.iloc[-1]
        
        if pd.isna(current_fast) or pd.isna(current_trend) or pd.isna(current_atr):
            return Signal.no_action()
        
        # Trend filter: only long above trend SMA
        is_uptrend = current_price > current_trend
        
        # Exit
        if self.is_long:
            # Exit on bearish cross OR price below trend
            if (prev_fast >= prev_slow and current_fast < current_slow) or not is_uptrend:
                return Signal.exit_long(reason="Exit signal")
        
        # Entry: bullish cross + uptrend
        if not self.is_in_position and is_uptrend:
            if prev_fast <= prev_slow and current_fast > current_slow:
                stop_distance = current_atr * self.atr_multiplier
                stop_loss = current_price - stop_distance
                take_profit = current_price + (stop_distance * self.risk_reward)
                
                return Signal.long_entry(
                    entry_price=current_price,
                    stop_loss=stop_loss,
                    take_profit=take_profit,
                    position_size=self.position_size,
                    reason="Bullish cross in uptrend"
                )
        
        return Signal.no_action()
