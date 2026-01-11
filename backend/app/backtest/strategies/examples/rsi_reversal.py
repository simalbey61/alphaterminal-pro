"""
AlphaTerminal Pro - RSI Mean Reversion Strategy
===============================================

RSI tabanlı mean reversion stratejisi.

Mantık:
- RSI oversold (< 30) → LONG (aşırı satım, geri dönüş beklentisi)
- RSI overbought (> 70) → EXIT
- Ek filtre: Fiyat SMA altında olmalı (dip buying)

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


class RSIMeanReversionStrategy(BaseStrategy):
    """
    RSI Mean Reversion Strategy.
    
    Entry: RSI < oversold level AND price below SMA (trend filter)
    Exit: RSI > overbought level OR stop loss/take profit
    
    Parameters:
        rsi_period: RSI calculation period (default: 14)
        oversold: Oversold threshold (default: 30)
        overbought: Overbought threshold (default: 70)
        sma_period: SMA for trend filter (default: 50)
        use_trend_filter: Use SMA trend filter (default: True)
        atr_period: ATR period for stops (default: 14)
        atr_stop_mult: ATR multiplier for stop loss (default: 2.0)
        atr_tp_mult: ATR multiplier for take profit (default: 3.0)
        position_size: Position size % (default: 0.10)
    
    Example:
        ```python
        strategy = RSIMeanReversionStrategy(
            rsi_period=14,
            oversold=30,
            overbought=70
        )
        result = engine.run(data, strategy, "THYAO")
        ```
    """
    
    name = "RSI Mean Reversion"
    version = "1.0.0"
    description = "RSI-based mean reversion with trend filter"
    
    def __init__(
        self,
        rsi_period: int = 14,
        oversold: float = 30,
        overbought: float = 70,
        sma_period: int = 50,
        use_trend_filter: bool = True,
        atr_period: int = 14,
        atr_stop_mult: float = 2.0,
        atr_tp_mult: float = 3.0,
        position_size: float = 0.10
    ):
        super().__init__()
        
        self.rsi_period = rsi_period
        self.oversold = oversold
        self.overbought = overbought
        self.sma_period = sma_period
        self.use_trend_filter = use_trend_filter
        self.atr_period = atr_period
        self.atr_stop_mult = atr_stop_mult
        self.atr_tp_mult = atr_tp_mult
        self.position_size = position_size
        
        # Warmup
        self.warmup_period = max(rsi_period, sma_period, atr_period) + 5
        
        self._parameters = {
            "rsi_period": rsi_period,
            "oversold": oversold,
            "overbought": overbought,
            "sma_period": sma_period,
            "use_trend_filter": use_trend_filter,
            "atr_stop_mult": atr_stop_mult,
            "atr_tp_mult": atr_tp_mult,
            "position_size": position_size
        }
    
    def _calculate_rsi(self, close: pd.Series) -> pd.Series:
        """Calculate RSI."""
        delta = close.diff()
        
        gain = delta.where(delta > 0, 0)
        loss = (-delta).where(delta < 0, 0)
        
        avg_gain = gain.rolling(self.rsi_period).mean()
        avg_loss = loss.rolling(self.rsi_period).mean()
        
        rs = avg_gain / avg_loss
        rsi = 100 - (100 / (1 + rs))
        
        return rsi
    
    def _calculate_atr(self, data: pd.DataFrame) -> pd.Series:
        """Calculate ATR."""
        high = data['High']
        low = data['Low']
        close = data['Close']
        
        tr = pd.concat([
            high - low,
            abs(high - close.shift(1)),
            abs(low - close.shift(1))
        ], axis=1).max(axis=1)
        
        return tr.rolling(self.atr_period).mean()
    
    def generate_signal(self, data: pd.DataFrame) -> Signal:
        """Generate trading signal."""
        if len(data) < self.warmup_period:
            return Signal.no_action()
        
        close = data['Close']
        
        # Calculate indicators
        rsi = self._calculate_rsi(close)
        sma = close.rolling(self.sma_period).mean()
        atr = self._calculate_atr(data)
        
        current_rsi = rsi.iloc[-1]
        prev_rsi = rsi.iloc[-2]
        current_sma = sma.iloc[-1]
        current_atr = atr.iloc[-1]
        current_price = close.iloc[-1]
        
        # Check for NaN
        if pd.isna(current_rsi) or pd.isna(current_sma) or pd.isna(current_atr):
            return Signal.no_action()
        
        # Exit signal: RSI overbought
        if self.is_long and current_rsi > self.overbought:
            return Signal.exit_long(reason=f"RSI overbought ({current_rsi:.1f})")
        
        # Entry conditions
        if not self.is_in_position:
            # RSI oversold
            is_oversold = current_rsi < self.oversold
            
            # Trend filter: price below SMA (we're buying the dip)
            # OR disabled
            trend_ok = not self.use_trend_filter or current_price < current_sma
            
            # RSI turning up (momentum confirmation)
            rsi_turning = current_rsi > prev_rsi
            
            if is_oversold and trend_ok and rsi_turning:
                stop_loss = current_price - (current_atr * self.atr_stop_mult)
                take_profit = current_price + (current_atr * self.atr_tp_mult)
                
                return Signal.long_entry(
                    entry_price=current_price,
                    stop_loss=stop_loss,
                    take_profit=take_profit,
                    position_size=self.position_size,
                    strength=min(1.0, (self.oversold - current_rsi) / 10),
                    reason=f"RSI oversold ({current_rsi:.1f}) and turning up"
                )
        
        return Signal.no_action()


class RSIExtremesStrategy(BaseStrategy):
    """
    RSI Extremes Strategy.
    
    Çok düşük RSI seviyelerinde alım yapar (< 20).
    Daha agresif mean reversion.
    
    Parameters:
        rsi_period: RSI period (default: 14)
        extreme_oversold: Extreme oversold level (default: 20)
        exit_level: Exit when RSI reaches (default: 50)
    """
    
    name = "RSI Extremes"
    version = "1.0.0"
    
    def __init__(
        self,
        rsi_period: int = 14,
        extreme_oversold: float = 20,
        exit_level: float = 50,
        atr_period: int = 14,
        atr_multiplier: float = 1.5,
        position_size: float = 0.10
    ):
        super().__init__()
        
        self.rsi_period = rsi_period
        self.extreme_oversold = extreme_oversold
        self.exit_level = exit_level
        self.atr_period = atr_period
        self.atr_multiplier = atr_multiplier
        self.position_size = position_size
        
        self.warmup_period = max(rsi_period, atr_period) + 5
    
    def _calculate_rsi(self, close: pd.Series) -> pd.Series:
        """Calculate RSI."""
        delta = close.diff()
        gain = delta.where(delta > 0, 0)
        loss = (-delta).where(delta < 0, 0)
        avg_gain = gain.rolling(self.rsi_period).mean()
        avg_loss = loss.rolling(self.rsi_period).mean()
        rs = avg_gain / avg_loss
        return 100 - (100 / (1 + rs))
    
    def generate_signal(self, data: pd.DataFrame) -> Signal:
        """Generate signal."""
        if len(data) < self.warmup_period:
            return Signal.no_action()
        
        close = data['Close']
        high = data['High']
        low = data['Low']
        
        rsi = self._calculate_rsi(close)
        
        # ATR
        tr = pd.concat([
            high - low,
            abs(high - close.shift(1)),
            abs(low - close.shift(1))
        ], axis=1).max(axis=1)
        atr = tr.rolling(self.atr_period).mean()
        
        current_rsi = rsi.iloc[-1]
        current_atr = atr.iloc[-1]
        current_price = close.iloc[-1]
        
        if pd.isna(current_rsi) or pd.isna(current_atr):
            return Signal.no_action()
        
        # Exit when RSI reaches target level
        if self.is_long and current_rsi >= self.exit_level:
            return Signal.exit_long(reason=f"RSI reached {self.exit_level}")
        
        # Entry at extreme oversold
        if not self.is_in_position and current_rsi < self.extreme_oversold:
            stop_loss = current_price - (current_atr * self.atr_multiplier)
            
            return Signal.long_entry(
                entry_price=current_price,
                stop_loss=stop_loss,
                take_profit=None,
                position_size=self.position_size,
                strength=min(1.0, (self.extreme_oversold - current_rsi) / 5),
                reason=f"Extreme RSI ({current_rsi:.1f})"
            )
        
        return Signal.no_action()
