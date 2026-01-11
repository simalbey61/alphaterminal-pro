"""
AlphaTerminal Pro - Feature Engineering
=======================================

Comprehensive feature engineering for ML models.

Author: AlphaTerminal Team
Version: 1.0.0
"""

import logging
from datetime import datetime
from typing import Optional, Dict, List, Any, Tuple, Union
from dataclasses import dataclass, field
import numpy as np
import pandas as pd
from abc import ABC, abstractmethod

from app.ml_pipeline.enums import FeatureCategory


logger = logging.getLogger(__name__)


# =============================================================================
# FEATURE DEFINITION
# =============================================================================

@dataclass
class FeatureDefinition:
    """Definition of a single feature."""
    name: str
    category: FeatureCategory
    description: str = ""
    parameters: Dict[str, Any] = field(default_factory=dict)
    lookback: int = 0
    is_stationary: bool = True
    requires_volume: bool = False
    
    def __hash__(self):
        return hash(self.name)


@dataclass
class FeatureSet:
    """Collection of features."""
    name: str
    features: List[FeatureDefinition]
    description: str = ""
    created_at: datetime = field(default_factory=datetime.now)
    
    @property
    def feature_names(self) -> List[str]:
        return [f.name for f in self.features]
    
    @property
    def max_lookback(self) -> int:
        return max(f.lookback for f in self.features) if self.features else 0


# =============================================================================
# ABSTRACT FEATURE CALCULATOR
# =============================================================================

class FeatureCalculator(ABC):
    """Abstract base for feature calculators."""
    
    category: FeatureCategory = FeatureCategory.PRICE
    
    @abstractmethod
    def calculate(
        self,
        df: pd.DataFrame,
        **params
    ) -> pd.DataFrame:
        """Calculate features and return DataFrame with new columns."""
        pass
    
    @abstractmethod
    def get_feature_names(self, **params) -> List[str]:
        """Get names of features this calculator produces."""
        pass
    
    def get_required_columns(self) -> List[str]:
        """Get required input columns."""
        return ['Open', 'High', 'Low', 'Close', 'Volume']


# =============================================================================
# PRICE FEATURES
# =============================================================================

class PriceFeatureCalculator(FeatureCalculator):
    """Calculate price-based features."""
    
    category = FeatureCategory.PRICE
    
    def calculate(self, df: pd.DataFrame, **params) -> pd.DataFrame:
        result = df.copy()
        
        # Returns
        for period in params.get('return_periods', [1, 2, 3, 5, 10, 20]):
            result[f'return_{period}d'] = df['Close'].pct_change(period)
        
        # Log returns
        result['log_return'] = np.log(df['Close'] / df['Close'].shift(1))
        
        # Price ratios
        result['hl_ratio'] = df['High'] / df['Low']
        result['co_ratio'] = df['Close'] / df['Open']
        result['oc_range'] = (df['Close'] - df['Open']) / df['Open']
        result['hl_range'] = (df['High'] - df['Low']) / df['Low']
        
        # Price position
        result['price_position'] = (df['Close'] - df['Low']) / (df['High'] - df['Low'] + 1e-10)
        
        # Gap
        result['gap'] = df['Open'] / df['Close'].shift(1) - 1
        
        # Body and wick sizes (for candlestick analysis)
        result['body_size'] = abs(df['Close'] - df['Open'])
        result['upper_wick'] = df['High'] - df[['Close', 'Open']].max(axis=1)
        result['lower_wick'] = df[['Close', 'Open']].min(axis=1) - df['Low']
        result['body_ratio'] = result['body_size'] / (df['High'] - df['Low'] + 1e-10)
        
        return result
    
    def get_feature_names(self, **params) -> List[str]:
        names = ['log_return', 'hl_ratio', 'co_ratio', 'oc_range', 'hl_range',
                 'price_position', 'gap', 'body_size', 'upper_wick', 
                 'lower_wick', 'body_ratio']
        
        for period in params.get('return_periods', [1, 2, 3, 5, 10, 20]):
            names.append(f'return_{period}d')
        
        return names


# =============================================================================
# TREND FEATURES
# =============================================================================

class TrendFeatureCalculator(FeatureCalculator):
    """Calculate trend-based features."""
    
    category = FeatureCategory.TREND
    
    def calculate(self, df: pd.DataFrame, **params) -> pd.DataFrame:
        result = df.copy()
        close = df['Close']
        
        # SMAs
        for period in params.get('sma_periods', [5, 10, 20, 50, 100, 200]):
            result[f'sma_{period}'] = close.rolling(period).mean()
            result[f'sma_{period}_slope'] = result[f'sma_{period}'].diff(5) / result[f'sma_{period}']
            result[f'price_to_sma_{period}'] = close / result[f'sma_{period}']
        
        # EMAs
        for period in params.get('ema_periods', [5, 10, 20, 50, 100]):
            result[f'ema_{period}'] = close.ewm(span=period, adjust=False).mean()
            result[f'price_to_ema_{period}'] = close / result[f'ema_{period}']
        
        # MACD
        macd_fast = params.get('macd_fast', 12)
        macd_slow = params.get('macd_slow', 26)
        macd_signal = params.get('macd_signal', 9)
        
        ema_fast = close.ewm(span=macd_fast, adjust=False).mean()
        ema_slow = close.ewm(span=macd_slow, adjust=False).mean()
        result['macd'] = ema_fast - ema_slow
        result['macd_signal'] = result['macd'].ewm(span=macd_signal, adjust=False).mean()
        result['macd_hist'] = result['macd'] - result['macd_signal']
        result['macd_hist_change'] = result['macd_hist'].diff()
        
        # ADX
        adx_period = params.get('adx_period', 14)
        result = self._calculate_adx(result, df, adx_period)
        
        # Trend direction
        result['trend_sma_20_50'] = np.where(
            result['sma_20'] > result['sma_50'], 1,
            np.where(result['sma_20'] < result['sma_50'], -1, 0)
        )
        
        result['trend_ema_10_20'] = np.where(
            result['ema_10'] > result['ema_20'], 1,
            np.where(result['ema_10'] < result['ema_20'], -1, 0)
        )
        
        # Higher highs / lower lows
        result['higher_high'] = (df['High'] > df['High'].shift(1)).astype(int)
        result['lower_low'] = (df['Low'] < df['Low'].shift(1)).astype(int)
        result['hh_count_5'] = result['higher_high'].rolling(5).sum()
        result['ll_count_5'] = result['lower_low'].rolling(5).sum()
        
        return result
    
    def _calculate_adx(
        self,
        result: pd.DataFrame,
        df: pd.DataFrame,
        period: int
    ) -> pd.DataFrame:
        """Calculate ADX indicator."""
        high = df['High']
        low = df['Low']
        close = df['Close']
        
        # True Range
        tr1 = high - low
        tr2 = abs(high - close.shift(1))
        tr3 = abs(low - close.shift(1))
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        
        # Directional Movement
        up_move = high - high.shift(1)
        down_move = low.shift(1) - low
        
        plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0)
        minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0)
        
        # Smoothed TR and DM
        atr = pd.Series(tr, index=df.index).rolling(period).mean()
        plus_di = 100 * pd.Series(plus_dm, index=df.index).rolling(period).mean() / atr
        minus_di = 100 * pd.Series(minus_dm, index=df.index).rolling(period).mean() / atr
        
        # ADX
        dx = 100 * abs(plus_di - minus_di) / (plus_di + minus_di + 1e-10)
        adx = dx.rolling(period).mean()
        
        result['adx'] = adx
        result['plus_di'] = plus_di
        result['minus_di'] = minus_di
        result['di_diff'] = plus_di - minus_di
        
        return result
    
    def get_feature_names(self, **params) -> List[str]:
        names = ['macd', 'macd_signal', 'macd_hist', 'macd_hist_change',
                 'adx', 'plus_di', 'minus_di', 'di_diff',
                 'trend_sma_20_50', 'trend_ema_10_20',
                 'higher_high', 'lower_low', 'hh_count_5', 'll_count_5']
        
        for period in params.get('sma_periods', [5, 10, 20, 50, 100, 200]):
            names.extend([f'sma_{period}', f'sma_{period}_slope', f'price_to_sma_{period}'])
        
        for period in params.get('ema_periods', [5, 10, 20, 50, 100]):
            names.extend([f'ema_{period}', f'price_to_ema_{period}'])
        
        return names


# =============================================================================
# MOMENTUM FEATURES
# =============================================================================

class MomentumFeatureCalculator(FeatureCalculator):
    """Calculate momentum-based features."""
    
    category = FeatureCategory.MOMENTUM
    
    def calculate(self, df: pd.DataFrame, **params) -> pd.DataFrame:
        result = df.copy()
        close = df['Close']
        high = df['High']
        low = df['Low']
        
        # RSI
        for period in params.get('rsi_periods', [7, 14, 21]):
            result[f'rsi_{period}'] = self._calculate_rsi(close, period)
        
        # Stochastic
        stoch_k = params.get('stoch_k', 14)
        stoch_d = params.get('stoch_d', 3)
        
        lowest_low = low.rolling(stoch_k).min()
        highest_high = high.rolling(stoch_k).max()
        result['stoch_k'] = 100 * (close - lowest_low) / (highest_high - lowest_low + 1e-10)
        result['stoch_d'] = result['stoch_k'].rolling(stoch_d).mean()
        result['stoch_diff'] = result['stoch_k'] - result['stoch_d']
        
        # Williams %R
        wr_period = params.get('williams_r_period', 14)
        highest = high.rolling(wr_period).max()
        lowest = low.rolling(wr_period).min()
        result['williams_r'] = -100 * (highest - close) / (highest - lowest + 1e-10)
        
        # CCI
        cci_period = params.get('cci_period', 20)
        tp = (high + low + close) / 3
        sma_tp = tp.rolling(cci_period).mean()
        mad = tp.rolling(cci_period).apply(lambda x: np.abs(x - x.mean()).mean())
        result['cci'] = (tp - sma_tp) / (0.015 * mad + 1e-10)
        
        # ROC
        for period in params.get('roc_periods', [5, 10, 20]):
            result[f'roc_{period}'] = 100 * (close - close.shift(period)) / close.shift(period)
        
        # Momentum
        for period in params.get('momentum_periods', [10, 20]):
            result[f'momentum_{period}'] = close - close.shift(period)
        
        # RSI divergence (price vs RSI)
        result['rsi_14_divergence'] = (
            (close.diff(5) > 0).astype(int) -
            (result['rsi_14'].diff(5) > 0).astype(int)
        )
        
        # Overbought/Oversold zones
        result['rsi_14_overbought'] = (result['rsi_14'] > 70).astype(int)
        result['rsi_14_oversold'] = (result['rsi_14'] < 30).astype(int)
        result['stoch_overbought'] = (result['stoch_k'] > 80).astype(int)
        result['stoch_oversold'] = (result['stoch_k'] < 20).astype(int)
        
        return result
    
    def _calculate_rsi(self, prices: pd.Series, period: int) -> pd.Series:
        """Calculate RSI."""
        delta = prices.diff()
        gain = delta.where(delta > 0, 0)
        loss = (-delta).where(delta < 0, 0)
        
        avg_gain = gain.rolling(period).mean()
        avg_loss = loss.rolling(period).mean()
        
        rs = avg_gain / (avg_loss + 1e-10)
        rsi = 100 - (100 / (1 + rs))
        
        return rsi
    
    def get_feature_names(self, **params) -> List[str]:
        names = ['stoch_k', 'stoch_d', 'stoch_diff', 'williams_r', 'cci',
                 'rsi_14_divergence', 'rsi_14_overbought', 'rsi_14_oversold',
                 'stoch_overbought', 'stoch_oversold']
        
        for period in params.get('rsi_periods', [7, 14, 21]):
            names.append(f'rsi_{period}')
        
        for period in params.get('roc_periods', [5, 10, 20]):
            names.append(f'roc_{period}')
        
        for period in params.get('momentum_periods', [10, 20]):
            names.append(f'momentum_{period}')
        
        return names


# =============================================================================
# VOLATILITY FEATURES
# =============================================================================

class VolatilityFeatureCalculator(FeatureCalculator):
    """Calculate volatility-based features."""
    
    category = FeatureCategory.VOLATILITY
    
    def calculate(self, df: pd.DataFrame, **params) -> pd.DataFrame:
        result = df.copy()
        close = df['Close']
        high = df['High']
        low = df['Low']
        
        # ATR
        for period in params.get('atr_periods', [7, 14, 21]):
            result[f'atr_{period}'] = self._calculate_atr(df, period)
            result[f'atr_{period}_pct'] = result[f'atr_{period}'] / close
        
        # Bollinger Bands
        bb_period = params.get('bb_period', 20)
        bb_std = params.get('bb_std', 2)
        
        sma = close.rolling(bb_period).mean()
        std = close.rolling(bb_period).std()
        
        result['bb_upper'] = sma + bb_std * std
        result['bb_lower'] = sma - bb_std * std
        result['bb_middle'] = sma
        result['bb_width'] = (result['bb_upper'] - result['bb_lower']) / sma
        result['bb_pct'] = (close - result['bb_lower']) / (result['bb_upper'] - result['bb_lower'] + 1e-10)
        
        # Keltner Channels
        kc_period = params.get('kc_period', 20)
        kc_mult = params.get('kc_mult', 2)
        
        ema_kc = close.ewm(span=kc_period, adjust=False).mean()
        atr_kc = self._calculate_atr(df, kc_period)
        
        result['kc_upper'] = ema_kc + kc_mult * atr_kc
        result['kc_lower'] = ema_kc - kc_mult * atr_kc
        result['kc_width'] = (result['kc_upper'] - result['kc_lower']) / ema_kc
        
        # Standard deviation
        for period in params.get('std_periods', [10, 20, 50]):
            result[f'std_{period}'] = close.rolling(period).std()
            result[f'std_{period}_pct'] = result[f'std_{period}'] / close
        
        # Historical volatility (annualized)
        log_returns = np.log(close / close.shift(1))
        result['hvol_20'] = log_returns.rolling(20).std() * np.sqrt(252)
        result['hvol_60'] = log_returns.rolling(60).std() * np.sqrt(252)
        
        # Volatility ratio
        result['vol_ratio'] = result['hvol_20'] / (result['hvol_60'] + 1e-10)
        
        # Donchian Channel
        dc_period = params.get('dc_period', 20)
        result['dc_upper'] = high.rolling(dc_period).max()
        result['dc_lower'] = low.rolling(dc_period).min()
        result['dc_width'] = (result['dc_upper'] - result['dc_lower']) / close
        
        # Squeeze (BB inside KC)
        result['squeeze'] = (
            (result['bb_lower'] > result['kc_lower']) & 
            (result['bb_upper'] < result['kc_upper'])
        ).astype(int)
        
        return result
    
    def _calculate_atr(self, df: pd.DataFrame, period: int) -> pd.Series:
        """Calculate ATR."""
        high = df['High']
        low = df['Low']
        close = df['Close']
        
        tr1 = high - low
        tr2 = abs(high - close.shift(1))
        tr3 = abs(low - close.shift(1))
        
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        atr = tr.rolling(period).mean()
        
        return atr
    
    def get_feature_names(self, **params) -> List[str]:
        names = ['bb_upper', 'bb_lower', 'bb_middle', 'bb_width', 'bb_pct',
                 'kc_upper', 'kc_lower', 'kc_width',
                 'hvol_20', 'hvol_60', 'vol_ratio',
                 'dc_upper', 'dc_lower', 'dc_width', 'squeeze']
        
        for period in params.get('atr_periods', [7, 14, 21]):
            names.extend([f'atr_{period}', f'atr_{period}_pct'])
        
        for period in params.get('std_periods', [10, 20, 50]):
            names.extend([f'std_{period}', f'std_{period}_pct'])
        
        return names


# =============================================================================
# VOLUME FEATURES
# =============================================================================

class VolumeFeatureCalculator(FeatureCalculator):
    """Calculate volume-based features."""
    
    category = FeatureCategory.VOLUME
    
    def calculate(self, df: pd.DataFrame, **params) -> pd.DataFrame:
        result = df.copy()
        close = df['Close']
        high = df['High']
        low = df['Low']
        volume = df['Volume']
        
        # Volume SMAs
        for period in params.get('vol_sma_periods', [5, 10, 20, 50]):
            result[f'volume_sma_{period}'] = volume.rolling(period).mean()
            result[f'volume_ratio_{period}'] = volume / result[f'volume_sma_{period}']
        
        # OBV
        obv = [0]
        for i in range(1, len(df)):
            if close.iloc[i] > close.iloc[i-1]:
                obv.append(obv[-1] + volume.iloc[i])
            elif close.iloc[i] < close.iloc[i-1]:
                obv.append(obv[-1] - volume.iloc[i])
            else:
                obv.append(obv[-1])
        result['obv'] = obv
        result['obv_sma_20'] = pd.Series(obv).rolling(20).mean().values
        result['obv_trend'] = np.sign(pd.Series(obv).diff(5)).values
        
        # VWAP (simplified - intraday would need reset)
        typical_price = (high + low + close) / 3
        result['vwap'] = (typical_price * volume).cumsum() / volume.cumsum()
        result['price_to_vwap'] = close / result['vwap']
        
        # A/D Line
        clv = ((close - low) - (high - close)) / (high - low + 1e-10)
        result['ad_line'] = (clv * volume).cumsum()
        
        # CMF (Chaikin Money Flow)
        cmf_period = params.get('cmf_period', 20)
        result['cmf'] = (clv * volume).rolling(cmf_period).sum() / volume.rolling(cmf_period).sum()
        
        # Force Index
        fi_period = params.get('fi_period', 13)
        result['force_index'] = (close.diff() * volume).ewm(span=fi_period, adjust=False).mean()
        
        # MFI (Money Flow Index)
        mfi_period = params.get('mfi_period', 14)
        result['mfi'] = self._calculate_mfi(df, mfi_period)
        
        # Volume price trend
        result['vpt'] = ((close.diff() / close.shift(1)) * volume).cumsum()
        
        # Relative volume
        result['relative_volume'] = volume / volume.rolling(20).mean()
        
        # Volume spike detection
        result['volume_spike'] = (volume > volume.rolling(20).mean() * 2).astype(int)
        
        return result
    
    def _calculate_mfi(self, df: pd.DataFrame, period: int) -> pd.Series:
        """Calculate Money Flow Index."""
        typical_price = (df['High'] + df['Low'] + df['Close']) / 3
        money_flow = typical_price * df['Volume']
        
        pos_flow = money_flow.where(typical_price > typical_price.shift(1), 0)
        neg_flow = money_flow.where(typical_price < typical_price.shift(1), 0)
        
        pos_sum = pos_flow.rolling(period).sum()
        neg_sum = neg_flow.rolling(period).sum()
        
        mfi = 100 - (100 / (1 + pos_sum / (neg_sum + 1e-10)))
        
        return mfi
    
    def get_feature_names(self, **params) -> List[str]:
        names = ['obv', 'obv_sma_20', 'obv_trend', 'vwap', 'price_to_vwap',
                 'ad_line', 'cmf', 'force_index', 'mfi', 'vpt',
                 'relative_volume', 'volume_spike']
        
        for period in params.get('vol_sma_periods', [5, 10, 20, 50]):
            names.extend([f'volume_sma_{period}', f'volume_ratio_{period}'])
        
        return names


# =============================================================================
# TEMPORAL FEATURES
# =============================================================================

class TemporalFeatureCalculator(FeatureCalculator):
    """Calculate time-based features."""
    
    category = FeatureCategory.TEMPORAL
    
    def calculate(self, df: pd.DataFrame, **params) -> pd.DataFrame:
        result = df.copy()
        
        # Ensure datetime index
        if not isinstance(df.index, pd.DatetimeIndex):
            return result
        
        # Basic time features
        result['day_of_week'] = df.index.dayofweek
        result['day_of_month'] = df.index.day
        result['week_of_year'] = df.index.isocalendar().week.values
        result['month'] = df.index.month
        result['quarter'] = df.index.quarter
        
        # Cyclical encoding
        result['day_of_week_sin'] = np.sin(2 * np.pi * result['day_of_week'] / 7)
        result['day_of_week_cos'] = np.cos(2 * np.pi * result['day_of_week'] / 7)
        result['month_sin'] = np.sin(2 * np.pi * result['month'] / 12)
        result['month_cos'] = np.cos(2 * np.pi * result['month'] / 12)
        
        # Is Monday/Friday
        result['is_monday'] = (df.index.dayofweek == 0).astype(int)
        result['is_friday'] = (df.index.dayofweek == 4).astype(int)
        
        # Month start/end
        result['is_month_start'] = (df.index.day <= 3).astype(int)
        result['is_month_end'] = (df.index.day >= 28).astype(int)
        
        # Quarter start/end
        result['is_quarter_start'] = ((df.index.month % 3 == 1) & (df.index.day <= 5)).astype(int)
        result['is_quarter_end'] = ((df.index.month % 3 == 0) & (df.index.day >= 25)).astype(int)
        
        return result
    
    def get_feature_names(self, **params) -> List[str]:
        return [
            'day_of_week', 'day_of_month', 'week_of_year', 'month', 'quarter',
            'day_of_week_sin', 'day_of_week_cos', 'month_sin', 'month_cos',
            'is_monday', 'is_friday', 'is_month_start', 'is_month_end',
            'is_quarter_start', 'is_quarter_end'
        ]


# =============================================================================
# FEATURE ENGINEER
# =============================================================================

class FeatureEngineer:
    """
    Main feature engineering class.
    
    Orchestrates multiple feature calculators to create
    comprehensive feature sets for ML models.
    """
    
    def __init__(self, params: Optional[Dict[str, Any]] = None):
        """
        Initialize feature engineer.
        
        Args:
            params: Global parameters for calculators
        """
        self.params = params or {}
        
        # Initialize calculators
        self.calculators = {
            'price': PriceFeatureCalculator(),
            'trend': TrendFeatureCalculator(),
            'momentum': MomentumFeatureCalculator(),
            'volatility': VolatilityFeatureCalculator(),
            'volume': VolumeFeatureCalculator(),
            'temporal': TemporalFeatureCalculator(),
        }
        
        self._computed_features: List[str] = []
    
    def calculate_all(
        self,
        df: pd.DataFrame,
        categories: Optional[List[str]] = None,
        params: Optional[Dict[str, Any]] = None
    ) -> pd.DataFrame:
        """
        Calculate all features.
        
        Args:
            df: OHLCV DataFrame
            categories: List of categories to include (None = all)
            params: Override parameters
            
        Returns:
            DataFrame with all features
        """
        calc_params = {**self.params, **(params or {})}
        categories = categories or list(self.calculators.keys())
        
        result = df.copy()
        self._computed_features = list(df.columns)
        
        for category in categories:
            if category in self.calculators:
                try:
                    calculator = self.calculators[category]
                    new_features = calculator.calculate(result, **calc_params)
                    
                    # Add only new columns
                    for col in new_features.columns:
                        if col not in result.columns:
                            result[col] = new_features[col]
                            self._computed_features.append(col)
                    
                    logger.debug(f"Calculated {category} features")
                    
                except Exception as e:
                    logger.error(f"Error calculating {category} features: {e}")
        
        return result
    
    def get_feature_names(self, categories: Optional[List[str]] = None) -> List[str]:
        """Get names of all features that will be calculated."""
        categories = categories or list(self.calculators.keys())
        names = []
        
        for category in categories:
            if category in self.calculators:
                names.extend(self.calculators[category].get_feature_names(**self.params))
        
        return names
    
    @property
    def computed_features(self) -> List[str]:
        """Get list of computed feature names."""
        return self._computed_features.copy()


# =============================================================================
# ALL EXPORTS
# =============================================================================

__all__ = [
    "FeatureDefinition",
    "FeatureSet",
    "FeatureCalculator",
    "PriceFeatureCalculator",
    "TrendFeatureCalculator",
    "MomentumFeatureCalculator",
    "VolatilityFeatureCalculator",
    "VolumeFeatureCalculator",
    "TemporalFeatureCalculator",
    "FeatureEngineer",
]
