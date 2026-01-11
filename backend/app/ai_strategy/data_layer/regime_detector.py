"""
AlphaTerminal Pro - Regime Detector
====================================

Hidden Markov Model tabanlı piyasa rejimi tespiti.

Author: AlphaTerminal Team
Version: 1.0.0
"""

from __future__ import annotations

import logging
from typing import Optional, List, Dict, Any, Tuple
from datetime import datetime
from dataclasses import dataclass, field

import numpy as np
import polars as pl
from scipy import stats

from app.ai_strategy.constants import (
    TrendRegime,
    VolatilityRegime,
    LiquidityRegime,
    MarketPhase,
)

logger = logging.getLogger(__name__)


@dataclass
class RegimeState:
    """Mevcut rejim durumu."""
    trend: TrendRegime = TrendRegime.SIDEWAYS
    volatility: VolatilityRegime = VolatilityRegime.NORMAL
    liquidity: LiquidityRegime = LiquidityRegime.NORMAL
    market_phase: Optional[MarketPhase] = None
    trend_probabilities: Dict[str, float] = field(default_factory=dict)
    volatility_probabilities: Dict[str, float] = field(default_factory=dict)
    trend_strength: float = 0.0
    volatility_percentile: float = 50.0
    volume_ratio: float = 1.0
    trend_change_probability: float = 0.0
    volatility_change_probability: float = 0.0
    confidence: float = 0.0
    detected_at: datetime = field(default_factory=datetime.utcnow)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "trend": self.trend.value,
            "volatility": self.volatility.value,
            "liquidity": self.liquidity.value,
            "market_phase": self.market_phase.value if self.market_phase else None,
            "trend_strength": self.trend_strength,
            "volatility_percentile": self.volatility_percentile,
            "volume_ratio": self.volume_ratio,
            "trend_change_probability": self.trend_change_probability,
            "volatility_change_probability": self.volatility_change_probability,
            "confidence": self.confidence,
            "detected_at": self.detected_at.isoformat(),
        }
    
    @property
    def is_bullish(self) -> bool:
        return self.trend in [TrendRegime.STRONG_BULL, TrendRegime.BULL, TrendRegime.WEAK_BULL]
    
    @property
    def is_bearish(self) -> bool:
        return self.trend in [TrendRegime.STRONG_BEAR, TrendRegime.BEAR, TrendRegime.WEAK_BEAR]
    
    @property
    def is_high_volatility(self) -> bool:
        return self.volatility in [VolatilityRegime.HIGH, VolatilityRegime.EXTREME]


class RegimeDetector:
    """
    Hidden Markov Model tabanlı rejim tespiti.
    
    Example:
        ```python
        detector = RegimeDetector()
        state = detector.detect(df)
        print(f"Trend: {state.trend}, Volatility: {state.volatility}")
        ```
    """
    
    def __init__(
        self,
        trend_lookback: int = 60,
        vol_lookback: int = 20,
        volume_lookback: int = 20,
        hmm_states: int = 3,
    ):
        self.trend_lookback = trend_lookback
        self.vol_lookback = vol_lookback
        self.volume_lookback = volume_lookback
        self.hmm_states = hmm_states
        self._hmm_model = None
    
    def detect(self, df: pl.DataFrame, index_df: Optional[pl.DataFrame] = None) -> RegimeState:
        """Mevcut rejimi tespit et."""
        if len(df) < max(self.trend_lookback, self.vol_lookback):
            logger.warning("Insufficient data for regime detection")
            return RegimeState()
        
        state = RegimeState()
        state.trend, state.trend_strength = self._detect_trend_regime(df)
        state.volatility, state.volatility_percentile = self._detect_volatility_regime(df)
        state.liquidity, state.volume_ratio = self._detect_liquidity_regime(df)
        state.market_phase = self._detect_market_phase(df)
        state.trend_change_probability = self._calculate_trend_change_prob(df)
        state.volatility_change_probability = self._calculate_vol_change_prob(df)
        state.confidence = self._calculate_confidence(df, state)
        
        return state
    
    def _detect_trend_regime(self, df: pl.DataFrame) -> Tuple[TrendRegime, float]:
        """Trend rejimi tespit et."""
        close = df["close"].to_numpy()
        
        if len(close) < self.trend_lookback:
            return TrendRegime.SIDEWAYS, 0.0
        
        lookback_return = (close[-1] / close[-self.trend_lookback] - 1)
        annualized_return = lookback_return * (252 / self.trend_lookback)
        
        sma_20 = np.mean(close[-20:])
        sma_50 = np.mean(close[-50:]) if len(close) >= 50 else sma_20
        sma_200 = np.mean(close[-200:]) if len(close) >= 200 else sma_50
        
        price = close[-1]
        ma_alignment = sum([price > sma_20, price > sma_50, price > sma_200]) / 3
        
        x = np.arange(min(60, len(close)))
        y = close[-len(x):]
        _, _, r_value, _, _ = stats.linregress(x, y)
        r_squared = r_value ** 2
        
        trend_strength = 0.3 * ma_alignment + 0.3 * min(1, abs(annualized_return)) + 0.4 * r_squared
        
        if annualized_return > 0.20 and ma_alignment >= 0.8:
            regime = TrendRegime.STRONG_BULL
        elif annualized_return > 0.10:
            regime = TrendRegime.BULL
        elif annualized_return > 0.0:
            regime = TrendRegime.WEAK_BULL
        elif annualized_return > -0.10:
            regime = TrendRegime.SIDEWAYS if abs(annualized_return) < 0.05 else TrendRegime.WEAK_BEAR
        elif annualized_return > -0.20:
            regime = TrendRegime.BEAR
        else:
            regime = TrendRegime.STRONG_BEAR
        
        return regime, trend_strength
    
    def _detect_volatility_regime(self, df: pl.DataFrame) -> Tuple[VolatilityRegime, float]:
        """Volatilite rejimi tespit et."""
        close = df["close"].to_numpy()
        
        if len(close) < self.vol_lookback + 1:
            return VolatilityRegime.NORMAL, 50.0
        
        returns = np.diff(np.log(close))
        current_vol = np.std(returns[-self.vol_lookback:]) * np.sqrt(252) * 100
        
        lookback = min(252, len(returns))
        rolling_vols = []
        for i in range(self.vol_lookback, lookback):
            vol = np.std(returns[i-self.vol_lookback:i]) * np.sqrt(252) * 100
            rolling_vols.append(vol)
        
        percentile = stats.percentileofscore(rolling_vols, current_vol) if rolling_vols else 50.0
        
        if current_vol < 15:
            regime = VolatilityRegime.VERY_LOW
        elif current_vol < 20:
            regime = VolatilityRegime.LOW
        elif current_vol < 28:
            regime = VolatilityRegime.NORMAL
        elif current_vol < 35:
            regime = VolatilityRegime.ELEVATED
        elif current_vol < 45:
            regime = VolatilityRegime.HIGH
        else:
            regime = VolatilityRegime.EXTREME
        
        return regime, percentile
    
    def _detect_liquidity_regime(self, df: pl.DataFrame) -> Tuple[LiquidityRegime, float]:
        """Likidite rejimi tespit et."""
        if "volume" not in df.columns:
            return LiquidityRegime.NORMAL, 1.0
        
        volume = df["volume"].to_numpy()
        if len(volume) < self.volume_lookback:
            return LiquidityRegime.NORMAL, 1.0
        
        avg_volume = np.mean(volume[-self.volume_lookback:])
        current_volume = volume[-1]
        
        if avg_volume == 0:
            return LiquidityRegime.THIN, 0.0
        
        volume_ratio = current_volume / avg_volume
        
        if volume_ratio < 0.5:
            regime = LiquidityRegime.THIN
        elif volume_ratio < 0.8:
            regime = LiquidityRegime.LOW
        elif volume_ratio < 1.2:
            regime = LiquidityRegime.NORMAL
        elif volume_ratio < 2.0:
            regime = LiquidityRegime.HIGH
        else:
            regime = LiquidityRegime.EXTREME
        
        return regime, volume_ratio
    
    def _detect_market_phase(self, df: pl.DataFrame) -> Optional[MarketPhase]:
        """Wyckoff market phase tespit et."""
        close = df["close"].to_numpy()
        volume = df["volume"].to_numpy() if "volume" in df.columns else None
        
        if len(close) < 50:
            return None
        
        recent_close = close[-50:]
        recent_high = np.max(recent_close)
        recent_low = np.min(recent_close)
        current = close[-1]
        
        range_size = recent_high - recent_low
        if range_size == 0:
            return None
        
        position_in_range = (current - recent_low) / range_size
        
        sma_10 = np.mean(close[-10:])
        sma_30 = np.mean(close[-30:])
        trend_up = sma_10 > sma_30
        
        vol_trend = None
        if volume is not None and len(volume) >= 20:
            recent_vol = np.mean(volume[-10:])
            past_vol = np.mean(volume[-20:-10])
            vol_trend = "increasing" if recent_vol > past_vol * 1.1 else "decreasing"
        
        if position_in_range < 0.3:
            if vol_trend == "increasing" and not trend_up:
                return MarketPhase.ACCUMULATION
            elif trend_up:
                return MarketPhase.REACCUMULATION
        elif position_in_range > 0.7:
            if vol_trend == "increasing" and trend_up:
                return MarketPhase.DISTRIBUTION
            elif not trend_up:
                return MarketPhase.REDISTRIBUTION
        else:
            if trend_up:
                return MarketPhase.MARKUP
            else:
                return MarketPhase.MARKDOWN
        
        return None
    
    def _calculate_trend_change_prob(self, df: pl.DataFrame) -> float:
        """Trend değişim olasılığı hesapla."""
        close = df["close"].to_numpy()
        if len(close) < 20:
            return 0.0
        
        mom_5 = close[-1] / close[-5] - 1
        mom_20 = close[-1] / close[-20] - 1
        
        if (mom_5 > 0 and mom_20 < 0) or (mom_5 < 0 and mom_20 > 0):
            divergence_strength = abs(mom_5 - mom_20)
            return min(1.0, divergence_strength * 5)
        
        mom_10 = close[-1] / close[-10] - 1
        if abs(mom_5) < abs(mom_10) * 0.5:
            return 0.4
        
        return 0.1
    
    def _calculate_vol_change_prob(self, df: pl.DataFrame) -> float:
        """Volatilite değişim olasılığı hesapla."""
        close = df["close"].to_numpy()
        if len(close) < 30:
            return 0.0
        
        returns = np.diff(np.log(close))
        vol_5 = np.std(returns[-5:]) * np.sqrt(252)
        vol_20 = np.std(returns[-20:]) * np.sqrt(252)
        
        if vol_20 == 0:
            return 0.0
        
        vol_ratio = vol_5 / vol_20
        
        if vol_ratio > 1.5:
            return min(1.0, (vol_ratio - 1) * 0.5)
        elif vol_ratio < 0.6:
            return min(1.0, (1 - vol_ratio) * 0.5)
        
        return 0.1
    
    def _calculate_confidence(self, df: pl.DataFrame, state: RegimeState) -> float:
        """Rejim tespiti güven skoru hesapla."""
        close = df["close"].to_numpy()
        
        data_conf = min(1.0, len(close) / 200)
        
        x = np.arange(min(60, len(close)))
        y = close[-len(x):]
        _, _, r_value, _, _ = stats.linregress(x, y)
        trend_conf = r_value ** 2
        
        if len(close) >= 30:
            returns = np.diff(np.log(close[-30:]))
            vol_series = [np.std(returns[i:i+5]) for i in range(25)]
            vol_stability = 1 - (np.std(vol_series) / (np.mean(vol_series) + 1e-8))
            vol_conf = max(0, vol_stability)
        else:
            vol_conf = 0.5
        
        confidence = 0.3 * data_conf + 0.4 * trend_conf + 0.3 * vol_conf
        return min(1.0, max(0.0, confidence))
    
    def analyze_history(self, df: pl.DataFrame, lookback: int = 252) -> List[Dict[str, Any]]:
        """Geçmiş rejim analizi."""
        results = []
        min_window = max(self.trend_lookback, self.vol_lookback, 50)
        
        for i in range(min_window, min(lookback, len(df))):
            window_df = df[:i+1]
            state = self.detect(window_df)
            results.append({
                "index": i,
                "timestamp": df["timestamp"][i] if "timestamp" in df.columns else None,
                **state.to_dict()
            })
        
        return results
    
    def get_regime_transitions(self, history: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Rejim geçişlerini bul."""
        transitions = []
        
        for i in range(1, len(history)):
            prev, curr = history[i-1], history[i]
            
            if prev["trend"] != curr["trend"]:
                transitions.append({
                    "type": "trend",
                    "index": curr["index"],
                    "timestamp": curr["timestamp"],
                    "from": prev["trend"],
                    "to": curr["trend"],
                })
            
            if prev["volatility"] != curr["volatility"]:
                transitions.append({
                    "type": "volatility",
                    "index": curr["index"],
                    "timestamp": curr["timestamp"],
                    "from": prev["volatility"],
                    "to": curr["volatility"],
                })
        
        return transitions
    
    def get_regime_statistics(self, history: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Rejim istatistikleri."""
        if not history:
            return {}
        
        trend_counts = {}
        vol_counts = {}
        
        for h in history:
            trend = h["trend"]
            vol = h["volatility"]
            trend_counts[trend] = trend_counts.get(trend, 0) + 1
            vol_counts[vol] = vol_counts.get(vol, 0) + 1
        
        total = len(history)
        
        return {
            "total_periods": total,
            "trend_distribution": {k: v/total for k, v in trend_counts.items()},
            "volatility_distribution": {k: v/total for k, v in vol_counts.items()},
            "dominant_trend": max(trend_counts, key=trend_counts.get),
            "dominant_volatility": max(vol_counts, key=vol_counts.get),
        }
