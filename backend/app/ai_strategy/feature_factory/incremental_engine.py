"""
AlphaTerminal Pro - Incremental Feature Engine
==============================================

Polars tabanlı yüksek performanslı artımlı feature hesaplama.

Author: AlphaTerminal Team
Version: 1.0.0
"""

from __future__ import annotations

import logging
from typing import Optional, List, Dict, Any, Callable, Set
from datetime import datetime
from dataclasses import dataclass, field
from abc import ABC, abstractmethod
import hashlib

import numpy as np
import polars as pl

from app.ai_strategy.constants import (
    FeatureCategory,
    CalculationMode,
    TECHNICAL_FEATURES,
    SMC_FEATURES,
    ORDERFLOW_FEATURES,
    ALPHA_FEATURES,
    ALL_FEATURES,
)

logger = logging.getLogger(__name__)


@dataclass
class FeatureResult:
    """Feature hesaplama sonucu."""
    name: str
    value: float
    timestamp: datetime
    category: FeatureCategory
    window: int
    is_incremental: bool
    calculation_time_ms: float = 0.0


@dataclass
class FeatureBatch:
    """Toplu feature sonuçları."""
    symbol: str
    timeframe: str
    features: Dict[str, float]
    timestamp: datetime
    calculation_time_ms: float = 0.0
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "symbol": self.symbol,
            "timeframe": self.timeframe,
            "features": self.features,
            "timestamp": self.timestamp.isoformat(),
            "calculation_time_ms": self.calculation_time_ms,
        }


class BaseFeatureCalculator(ABC):
    """Feature hesaplayıcı temel sınıf."""
    
    @property
    @abstractmethod
    def name(self) -> str:
        """Feature adı."""
        pass
    
    @property
    @abstractmethod
    def category(self) -> FeatureCategory:
        """Feature kategorisi."""
        pass
    
    @property
    @abstractmethod
    def window(self) -> int:
        """Gereken minimum pencere boyutu."""
        pass
    
    @property
    def is_incremental(self) -> bool:
        """Artımlı hesaplama destekler mi?"""
        return False
    
    @abstractmethod
    def calculate(self, df: pl.DataFrame) -> float:
        """Tam hesaplama."""
        pass
    
    def calculate_incremental(
        self,
        df: pl.DataFrame,
        previous_value: Optional[float] = None
    ) -> float:
        """Artımlı hesaplama (varsayılan olarak tam hesaplamaya düşer)."""
        return self.calculate(df)


class IncrementalFeatureEngine:
    """
    Artımlı feature hesaplama motoru.
    
    Özellikler:
    - Polars ile yüksek performans
    - Artımlı hesaplama (delta-based)
    - Feature caching
    - Batch hesaplama
    
    Example:
        ```python
        engine = IncrementalFeatureEngine()
        
        # Tekil feature
        value = engine.calculate_feature("rsi_14", df)
        
        # Tüm features
        batch = engine.calculate_all(df, "THYAO", "4h")
        
        # Artımlı güncelleme
        updated = engine.update_incremental(df, previous_batch)
        ```
    """
    
    def __init__(
        self,
        enabled_features: Optional[Set[str]] = None,
        cache_enabled: bool = True,
    ):
        """
        Initialize feature engine.
        
        Args:
            enabled_features: Aktif feature listesi (None = tümü)
            cache_enabled: Önbellek aktif mi?
        """
        self.enabled_features = enabled_features or set(ALL_FEATURES.keys())
        self.cache_enabled = cache_enabled
        
        # Feature cache
        self._cache: Dict[str, Dict[str, float]] = {}
        self._cache_timestamps: Dict[str, datetime] = {}
        
        # Calculator registry
        self._calculators: Dict[str, BaseFeatureCalculator] = {}
        self._register_default_calculators()
    
    def _register_default_calculators(self) -> None:
        """Varsayılan hesaplayıcıları kaydet."""
        # Trend indicators
        self._calculators["sma_5"] = SMACalculator(5)
        self._calculators["sma_10"] = SMACalculator(10)
        self._calculators["sma_20"] = SMACalculator(20)
        self._calculators["sma_50"] = SMACalculator(50)
        self._calculators["sma_100"] = SMACalculator(100)
        self._calculators["sma_200"] = SMACalculator(200)
        
        self._calculators["ema_9"] = EMACalculator(9)
        self._calculators["ema_12"] = EMACalculator(12)
        self._calculators["ema_21"] = EMACalculator(21)
        self._calculators["ema_26"] = EMACalculator(26)
        self._calculators["ema_50"] = EMACalculator(50)
        
        # Momentum indicators
        self._calculators["rsi_7"] = RSICalculator(7)
        self._calculators["rsi_14"] = RSICalculator(14)
        self._calculators["rsi_21"] = RSICalculator(21)
        
        self._calculators["macd"] = MACDCalculator("macd")
        self._calculators["macd_signal"] = MACDCalculator("signal")
        self._calculators["macd_hist"] = MACDCalculator("histogram")
        
        self._calculators["stoch_k"] = StochasticCalculator("k")
        self._calculators["stoch_d"] = StochasticCalculator("d")
        
        self._calculators["cci_14"] = CCICalculator(14)
        self._calculators["mfi_14"] = MFICalculator(14)
        self._calculators["williams_r"] = WilliamsRCalculator(14)
        
        # Volatility indicators
        self._calculators["atr_7"] = ATRCalculator(7)
        self._calculators["atr_14"] = ATRCalculator(14)
        self._calculators["atr_21"] = ATRCalculator(21)
        
        self._calculators["bb_upper"] = BollingerCalculator("upper")
        self._calculators["bb_middle"] = BollingerCalculator("middle")
        self._calculators["bb_lower"] = BollingerCalculator("lower")
        self._calculators["bb_width"] = BollingerCalculator("width")
        self._calculators["bb_pct_b"] = BollingerCalculator("pct_b")
        
        self._calculators["historical_vol_20"] = HistoricalVolCalculator(20)
        
        # Volume indicators
        self._calculators["obv"] = OBVCalculator()
        self._calculators["volume_sma_20"] = VolumeSMACalculator(20)
        self._calculators["volume_ratio"] = VolumeRatioCalculator(20)
        self._calculators["vwap"] = VWAPCalculator()
        self._calculators["cmf"] = CMFCalculator(20)
        self._calculators["adl"] = ADLCalculator()
        
        # ADX
        self._calculators["adx_14"] = ADXCalculator(14)
        self._calculators["di_plus_14"] = DICalculator(14, "plus")
        self._calculators["di_minus_14"] = DICalculator(14, "minus")
        
        # Price action
        self._calculators["momentum_20"] = MomentumCalculator(20)
        self._calculators["momentum_60"] = MomentumCalculator(60)
    
    def calculate_feature(
        self,
        feature_name: str,
        df: pl.DataFrame,
        mode: CalculationMode = CalculationMode.FULL
    ) -> Optional[float]:
        """
        Tekil feature hesapla.
        
        Args:
            feature_name: Feature adı
            df: OHLCV DataFrame
            mode: Hesaplama modu
            
        Returns:
            Optional[float]: Feature değeri
        """
        if feature_name not in self._calculators:
            logger.warning(f"Unknown feature: {feature_name}")
            return None
        
        calculator = self._calculators[feature_name]
        
        if len(df) < calculator.window:
            return None
        
        try:
            if mode == CalculationMode.INCREMENTAL and calculator.is_incremental:
                cache_key = self._get_cache_key(df, feature_name)
                previous = self._cache.get(cache_key, {}).get(feature_name)
                value = calculator.calculate_incremental(df, previous)
            else:
                value = calculator.calculate(df)
            
            return value
            
        except Exception as e:
            logger.error(f"Error calculating {feature_name}: {e}")
            return None
    
    def calculate_all(
        self,
        df: pl.DataFrame,
        symbol: str,
        timeframe: str,
        mode: CalculationMode = CalculationMode.FULL
    ) -> FeatureBatch:
        """
        Tüm enabled feature'ları hesapla.
        
        Args:
            df: OHLCV DataFrame
            symbol: Hisse sembolü
            timeframe: Zaman dilimi
            mode: Hesaplama modu
            
        Returns:
            FeatureBatch: Tüm feature değerleri
        """
        import time
        start_time = time.time()
        
        features = {}
        
        for name in self.enabled_features:
            if name in self._calculators:
                value = self.calculate_feature(name, df, mode)
                if value is not None:
                    features[name] = value
        
        calculation_time = (time.time() - start_time) * 1000
        
        batch = FeatureBatch(
            symbol=symbol,
            timeframe=timeframe,
            features=features,
            timestamp=datetime.utcnow(),
            calculation_time_ms=calculation_time,
        )
        
        # Cache güncelle
        if self.cache_enabled:
            cache_key = self._get_cache_key(df, symbol)
            self._cache[cache_key] = features
            self._cache_timestamps[cache_key] = batch.timestamp
        
        logger.debug(
            f"Calculated {len(features)} features for {symbol} {timeframe} "
            f"in {calculation_time:.2f}ms"
        )
        
        return batch
    
    def update_incremental(
        self,
        df: pl.DataFrame,
        previous_batch: FeatureBatch
    ) -> FeatureBatch:
        """
        Artımlı güncelleme.
        
        Args:
            df: Güncel OHLCV DataFrame
            previous_batch: Önceki feature batch
            
        Returns:
            FeatureBatch: Güncellenmiş batch
        """
        import time
        start_time = time.time()
        
        features = previous_batch.features.copy()
        
        for name in self.enabled_features:
            if name not in self._calculators:
                continue
            
            calculator = self._calculators[name]
            
            if calculator.is_incremental:
                previous_value = previous_batch.features.get(name)
                value = calculator.calculate_incremental(df, previous_value)
            else:
                value = calculator.calculate(df)
            
            if value is not None:
                features[name] = value
        
        calculation_time = (time.time() - start_time) * 1000
        
        return FeatureBatch(
            symbol=previous_batch.symbol,
            timeframe=previous_batch.timeframe,
            features=features,
            timestamp=datetime.utcnow(),
            calculation_time_ms=calculation_time,
        )
    
    def _get_cache_key(self, df: pl.DataFrame, identifier: str) -> str:
        """Cache key oluştur."""
        if "timestamp" in df.columns and len(df) > 0:
            last_ts = str(df["timestamp"][-1])
        else:
            last_ts = str(len(df))
        return f"{identifier}:{last_ts}"
    
    def clear_cache(self) -> None:
        """Cache temizle."""
        self._cache.clear()
        self._cache_timestamps.clear()
    
    def get_feature_info(self, feature_name: str) -> Optional[Dict[str, Any]]:
        """Feature bilgisi al."""
        if feature_name not in self._calculators:
            return None
        
        calc = self._calculators[feature_name]
        return {
            "name": calc.name,
            "category": calc.category.value,
            "window": calc.window,
            "is_incremental": calc.is_incremental,
        }
    
    def list_features(self) -> List[str]:
        """Mevcut feature listesi."""
        return list(self._calculators.keys())


# =============================================================================
# CALCULATOR IMPLEMENTATIONS
# =============================================================================

class SMACalculator(BaseFeatureCalculator):
    """Simple Moving Average."""
    
    def __init__(self, period: int):
        self._period = period
    
    @property
    def name(self) -> str:
        return f"sma_{self._period}"
    
    @property
    def category(self) -> FeatureCategory:
        return FeatureCategory.TREND
    
    @property
    def window(self) -> int:
        return self._period
    
    @property
    def is_incremental(self) -> bool:
        return True
    
    def calculate(self, df: pl.DataFrame) -> float:
        return df["close"].tail(self._period).mean()
    
    def calculate_incremental(self, df: pl.DataFrame, previous: Optional[float]) -> float:
        if previous is None or len(df) < self._period + 1:
            return self.calculate(df)
        
        # SMA(t) = SMA(t-1) + (new - old) / period
        new_value = df["close"][-1]
        old_value = df["close"][-self._period - 1]
        return previous + (new_value - old_value) / self._period


class EMACalculator(BaseFeatureCalculator):
    """Exponential Moving Average."""
    
    def __init__(self, period: int):
        self._period = period
        self._multiplier = 2 / (period + 1)
    
    @property
    def name(self) -> str:
        return f"ema_{self._period}"
    
    @property
    def category(self) -> FeatureCategory:
        return FeatureCategory.TREND
    
    @property
    def window(self) -> int:
        return self._period
    
    @property
    def is_incremental(self) -> bool:
        return True
    
    def calculate(self, df: pl.DataFrame) -> float:
        close = df["close"].to_numpy()
        ema = close[0]
        for price in close[1:]:
            ema = price * self._multiplier + ema * (1 - self._multiplier)
        return ema
    
    def calculate_incremental(self, df: pl.DataFrame, previous: Optional[float]) -> float:
        if previous is None:
            return self.calculate(df)
        
        new_price = df["close"][-1]
        return new_price * self._multiplier + previous * (1 - self._multiplier)


class RSICalculator(BaseFeatureCalculator):
    """Relative Strength Index."""
    
    def __init__(self, period: int = 14):
        self._period = period
    
    @property
    def name(self) -> str:
        return f"rsi_{self._period}"
    
    @property
    def category(self) -> FeatureCategory:
        return FeatureCategory.MOMENTUM
    
    @property
    def window(self) -> int:
        return self._period + 1
    
    @property
    def is_incremental(self) -> bool:
        return True
    
    def calculate(self, df: pl.DataFrame) -> float:
        close = df["close"].to_numpy()
        deltas = np.diff(close)
        
        gains = np.where(deltas > 0, deltas, 0)
        losses = np.where(deltas < 0, -deltas, 0)
        
        avg_gain = np.mean(gains[-self._period:])
        avg_loss = np.mean(losses[-self._period:])
        
        if avg_loss == 0:
            return 100.0
        
        rs = avg_gain / avg_loss
        return 100 - (100 / (1 + rs))


class MACDCalculator(BaseFeatureCalculator):
    """MACD Calculator."""
    
    def __init__(self, component: str = "macd"):
        self._component = component  # macd, signal, histogram
        self._fast = 12
        self._slow = 26
        self._signal = 9
    
    @property
    def name(self) -> str:
        return f"macd_{self._component}" if self._component != "macd" else "macd"
    
    @property
    def category(self) -> FeatureCategory:
        return FeatureCategory.MOMENTUM
    
    @property
    def window(self) -> int:
        return self._slow + self._signal
    
    def calculate(self, df: pl.DataFrame) -> float:
        close = df["close"].to_numpy()
        
        ema_fast = self._calc_ema(close, self._fast)
        ema_slow = self._calc_ema(close, self._slow)
        macd_line = ema_fast - ema_slow
        
        if self._component == "macd":
            return macd_line
        
        # Signal line (EMA of MACD)
        # Simplified: just use recent MACD values
        signal_line = macd_line  # Would need history for proper signal
        
        if self._component == "signal":
            return signal_line
        
        # Histogram
        return macd_line - signal_line
    
    def _calc_ema(self, data: np.ndarray, period: int) -> float:
        multiplier = 2 / (period + 1)
        ema = data[0]
        for value in data[1:]:
            ema = value * multiplier + ema * (1 - multiplier)
        return ema


class StochasticCalculator(BaseFeatureCalculator):
    """Stochastic Oscillator."""
    
    def __init__(self, component: str = "k", period: int = 14):
        self._component = component
        self._period = period
    
    @property
    def name(self) -> str:
        return f"stoch_{self._component}"
    
    @property
    def category(self) -> FeatureCategory:
        return FeatureCategory.MOMENTUM
    
    @property
    def window(self) -> int:
        return self._period + 3
    
    def calculate(self, df: pl.DataFrame) -> float:
        high = df["high"].tail(self._period).max()
        low = df["low"].tail(self._period).min()
        close = df["close"][-1]
        
        if high == low:
            return 50.0
        
        k = ((close - low) / (high - low)) * 100
        
        if self._component == "k":
            return k
        
        # %D = 3-period SMA of %K (simplified)
        return k


class CCICalculator(BaseFeatureCalculator):
    """Commodity Channel Index."""
    
    def __init__(self, period: int = 14):
        self._period = period
    
    @property
    def name(self) -> str:
        return f"cci_{self._period}"
    
    @property
    def category(self) -> FeatureCategory:
        return FeatureCategory.MOMENTUM
    
    @property
    def window(self) -> int:
        return self._period
    
    def calculate(self, df: pl.DataFrame) -> float:
        tp = (df["high"] + df["low"] + df["close"]) / 3
        tp_sma = tp.tail(self._period).mean()
        mad = (tp.tail(self._period) - tp_sma).abs().mean()
        
        if mad == 0:
            return 0.0
        
        return (tp[-1] - tp_sma) / (0.015 * mad)


class MFICalculator(BaseFeatureCalculator):
    """Money Flow Index."""
    
    def __init__(self, period: int = 14):
        self._period = period
    
    @property
    def name(self) -> str:
        return f"mfi_{self._period}"
    
    @property
    def category(self) -> FeatureCategory:
        return FeatureCategory.MOMENTUM
    
    @property
    def window(self) -> int:
        return self._period + 1
    
    def calculate(self, df: pl.DataFrame) -> float:
        tp = ((df["high"] + df["low"] + df["close"]) / 3).to_numpy()
        volume = df["volume"].to_numpy()
        
        raw_mf = tp * volume
        
        pos_mf = 0.0
        neg_mf = 0.0
        
        for i in range(-self._period, 0):
            if tp[i] > tp[i-1]:
                pos_mf += raw_mf[i]
            else:
                neg_mf += raw_mf[i]
        
        if neg_mf == 0:
            return 100.0
        
        mfi = 100 - (100 / (1 + pos_mf / neg_mf))
        return mfi


class WilliamsRCalculator(BaseFeatureCalculator):
    """Williams %R."""
    
    def __init__(self, period: int = 14):
        self._period = period
    
    @property
    def name(self) -> str:
        return "williams_r"
    
    @property
    def category(self) -> FeatureCategory:
        return FeatureCategory.MOMENTUM
    
    @property
    def window(self) -> int:
        return self._period
    
    def calculate(self, df: pl.DataFrame) -> float:
        high = df["high"].tail(self._period).max()
        low = df["low"].tail(self._period).min()
        close = df["close"][-1]
        
        if high == low:
            return -50.0
        
        return ((high - close) / (high - low)) * -100


class ATRCalculator(BaseFeatureCalculator):
    """Average True Range."""
    
    def __init__(self, period: int = 14):
        self._period = period
    
    @property
    def name(self) -> str:
        return f"atr_{self._period}"
    
    @property
    def category(self) -> FeatureCategory:
        return FeatureCategory.VOLATILITY
    
    @property
    def window(self) -> int:
        return self._period + 1
    
    @property
    def is_incremental(self) -> bool:
        return True
    
    def calculate(self, df: pl.DataFrame) -> float:
        high = df["high"].to_numpy()
        low = df["low"].to_numpy()
        close = df["close"].to_numpy()
        
        tr = np.maximum(
            high[1:] - low[1:],
            np.maximum(
                np.abs(high[1:] - close[:-1]),
                np.abs(low[1:] - close[:-1])
            )
        )
        
        return np.mean(tr[-self._period:])


class BollingerCalculator(BaseFeatureCalculator):
    """Bollinger Bands."""
    
    def __init__(self, component: str = "middle", period: int = 20, std_dev: float = 2.0):
        self._component = component
        self._period = period
        self._std_dev = std_dev
    
    @property
    def name(self) -> str:
        return f"bb_{self._component}"
    
    @property
    def category(self) -> FeatureCategory:
        return FeatureCategory.VOLATILITY
    
    @property
    def window(self) -> int:
        return self._period
    
    @property
    def is_incremental(self) -> bool:
        return True
    
    def calculate(self, df: pl.DataFrame) -> float:
        close = df["close"].tail(self._period)
        middle = close.mean()
        std = close.std()
        
        if self._component == "middle":
            return middle
        elif self._component == "upper":
            return middle + self._std_dev * std
        elif self._component == "lower":
            return middle - self._std_dev * std
        elif self._component == "width":
            return (2 * self._std_dev * std) / middle if middle != 0 else 0
        elif self._component == "pct_b":
            upper = middle + self._std_dev * std
            lower = middle - self._std_dev * std
            if upper == lower:
                return 0.5
            return (df["close"][-1] - lower) / (upper - lower)
        
        return middle


class HistoricalVolCalculator(BaseFeatureCalculator):
    """Historical Volatility."""
    
    def __init__(self, period: int = 20):
        self._period = period
    
    @property
    def name(self) -> str:
        return f"historical_vol_{self._period}"
    
    @property
    def category(self) -> FeatureCategory:
        return FeatureCategory.VOLATILITY
    
    @property
    def window(self) -> int:
        return self._period + 1
    
    def calculate(self, df: pl.DataFrame) -> float:
        close = df["close"].to_numpy()
        returns = np.diff(np.log(close))
        return np.std(returns[-self._period:]) * np.sqrt(252) * 100


class OBVCalculator(BaseFeatureCalculator):
    """On Balance Volume."""
    
    @property
    def name(self) -> str:
        return "obv"
    
    @property
    def category(self) -> FeatureCategory:
        return FeatureCategory.VOLUME
    
    @property
    def window(self) -> int:
        return 2
    
    @property
    def is_incremental(self) -> bool:
        return True
    
    def calculate(self, df: pl.DataFrame) -> float:
        close = df["close"].to_numpy()
        volume = df["volume"].to_numpy()
        
        obv = 0.0
        for i in range(1, len(close)):
            if close[i] > close[i-1]:
                obv += volume[i]
            elif close[i] < close[i-1]:
                obv -= volume[i]
        
        return obv


class VolumeSMACalculator(BaseFeatureCalculator):
    """Volume SMA."""
    
    def __init__(self, period: int = 20):
        self._period = period
    
    @property
    def name(self) -> str:
        return f"volume_sma_{self._period}"
    
    @property
    def category(self) -> FeatureCategory:
        return FeatureCategory.VOLUME
    
    @property
    def window(self) -> int:
        return self._period
    
    def calculate(self, df: pl.DataFrame) -> float:
        return df["volume"].tail(self._period).mean()


class VolumeRatioCalculator(BaseFeatureCalculator):
    """Volume Ratio (current/average)."""
    
    def __init__(self, period: int = 20):
        self._period = period
    
    @property
    def name(self) -> str:
        return "volume_ratio"
    
    @property
    def category(self) -> FeatureCategory:
        return FeatureCategory.VOLUME
    
    @property
    def window(self) -> int:
        return self._period
    
    def calculate(self, df: pl.DataFrame) -> float:
        avg = df["volume"].tail(self._period).mean()
        if avg == 0:
            return 1.0
        return df["volume"][-1] / avg


class VWAPCalculator(BaseFeatureCalculator):
    """Volume Weighted Average Price."""
    
    @property
    def name(self) -> str:
        return "vwap"
    
    @property
    def category(self) -> FeatureCategory:
        return FeatureCategory.VOLUME
    
    @property
    def window(self) -> int:
        return 1
    
    def calculate(self, df: pl.DataFrame) -> float:
        tp = (df["high"] + df["low"] + df["close"]) / 3
        volume = df["volume"]
        
        total_volume = volume.sum()
        if total_volume == 0:
            return tp[-1]
        
        return (tp * volume).sum() / total_volume


class CMFCalculator(BaseFeatureCalculator):
    """Chaikin Money Flow."""
    
    def __init__(self, period: int = 20):
        self._period = period
    
    @property
    def name(self) -> str:
        return "cmf"
    
    @property
    def category(self) -> FeatureCategory:
        return FeatureCategory.VOLUME
    
    @property
    def window(self) -> int:
        return self._period
    
    def calculate(self, df: pl.DataFrame) -> float:
        high = df["high"].tail(self._period).to_numpy()
        low = df["low"].tail(self._period).to_numpy()
        close = df["close"].tail(self._period).to_numpy()
        volume = df["volume"].tail(self._period).to_numpy()
        
        hl_range = high - low
        hl_range = np.where(hl_range == 0, 1, hl_range)
        
        mf_multiplier = ((close - low) - (high - close)) / hl_range
        mf_volume = mf_multiplier * volume
        
        total_volume = np.sum(volume)
        if total_volume == 0:
            return 0.0
        
        return np.sum(mf_volume) / total_volume


class ADLCalculator(BaseFeatureCalculator):
    """Accumulation/Distribution Line."""
    
    @property
    def name(self) -> str:
        return "adl"
    
    @property
    def category(self) -> FeatureCategory:
        return FeatureCategory.VOLUME
    
    @property
    def window(self) -> int:
        return 1
    
    def calculate(self, df: pl.DataFrame) -> float:
        high = df["high"].to_numpy()
        low = df["low"].to_numpy()
        close = df["close"].to_numpy()
        volume = df["volume"].to_numpy()
        
        adl = 0.0
        for i in range(len(close)):
            hl_range = high[i] - low[i]
            if hl_range == 0:
                continue
            mf_multiplier = ((close[i] - low[i]) - (high[i] - close[i])) / hl_range
            adl += mf_multiplier * volume[i]
        
        return adl


class ADXCalculator(BaseFeatureCalculator):
    """Average Directional Index."""
    
    def __init__(self, period: int = 14):
        self._period = period
    
    @property
    def name(self) -> str:
        return f"adx_{self._period}"
    
    @property
    def category(self) -> FeatureCategory:
        return FeatureCategory.TREND
    
    @property
    def window(self) -> int:
        return self._period * 2
    
    def calculate(self, df: pl.DataFrame) -> float:
        # Simplified ADX calculation
        high = df["high"].to_numpy()
        low = df["low"].to_numpy()
        close = df["close"].to_numpy()
        
        plus_dm = np.maximum(high[1:] - high[:-1], 0)
        minus_dm = np.maximum(low[:-1] - low[1:], 0)
        
        tr = np.maximum(
            high[1:] - low[1:],
            np.maximum(
                np.abs(high[1:] - close[:-1]),
                np.abs(low[1:] - close[:-1])
            )
        )
        
        atr = np.mean(tr[-self._period:])
        if atr == 0:
            return 0.0
        
        plus_di = np.mean(plus_dm[-self._period:]) / atr * 100
        minus_di = np.mean(minus_dm[-self._period:]) / atr * 100
        
        di_sum = plus_di + minus_di
        if di_sum == 0:
            return 0.0
        
        dx = abs(plus_di - minus_di) / di_sum * 100
        return dx


class DICalculator(BaseFeatureCalculator):
    """Directional Indicator (+DI / -DI)."""
    
    def __init__(self, period: int = 14, direction: str = "plus"):
        self._period = period
        self._direction = direction
    
    @property
    def name(self) -> str:
        return f"di_{self._direction}_{self._period}"
    
    @property
    def category(self) -> FeatureCategory:
        return FeatureCategory.TREND
    
    @property
    def window(self) -> int:
        return self._period + 1
    
    def calculate(self, df: pl.DataFrame) -> float:
        high = df["high"].to_numpy()
        low = df["low"].to_numpy()
        close = df["close"].to_numpy()
        
        if self._direction == "plus":
            dm = np.maximum(high[1:] - high[:-1], 0)
        else:
            dm = np.maximum(low[:-1] - low[1:], 0)
        
        tr = np.maximum(
            high[1:] - low[1:],
            np.maximum(
                np.abs(high[1:] - close[:-1]),
                np.abs(low[1:] - close[:-1])
            )
        )
        
        atr = np.mean(tr[-self._period:])
        if atr == 0:
            return 0.0
        
        return np.mean(dm[-self._period:]) / atr * 100


class MomentumCalculator(BaseFeatureCalculator):
    """Price Momentum."""
    
    def __init__(self, period: int = 20):
        self._period = period
    
    @property
    def name(self) -> str:
        return f"momentum_{self._period}"
    
    @property
    def category(self) -> FeatureCategory:
        return FeatureCategory.ALPHA
    
    @property
    def window(self) -> int:
        return self._period + 1
    
    @property
    def is_incremental(self) -> bool:
        return True
    
    def calculate(self, df: pl.DataFrame) -> float:
        close = df["close"].to_numpy()
        return (close[-1] / close[-self._period - 1] - 1) * 100
