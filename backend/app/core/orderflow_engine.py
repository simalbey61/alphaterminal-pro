"""
AlphaTerminal Pro - Order Flow Engine v4.2
==========================================

Kurumsal Seviye Order Flow Analiz Motoru

YENİ ÖZELLİKLER (v4.2):
- Whale Order Detection
- Order Flow Imbalance Analysis
- Time-Weighted Metrics
- Smart Money Divergence
- Absorption & Exhaustion Detection
- Advanced Volume Profile

Özellikler:
- Volume Delta Analysis
- Cumulative Volume Delta (CVD)
- VWAP & Deviation Bands
- Volume Profile (POC, VAH, VAL)
- Institutional Activity Detection
- Multi-session Analysis

Author: AlphaTerminal Team
Version: 4.2.0
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, field
from enum import Enum

from app.core.config import logger, ORDERFLOW_CONFIG


# ═══════════════════════════════════════════════════════════════════════════════
# ENUMS
# ═══════════════════════════════════════════════════════════════════════════════

class FlowDirection(Enum):
    """Order Flow yönü"""
    BULLISH = "BULLISH"
    BEARISH = "BEARISH"
    NEUTRAL = "NEUTRAL"


class FlowStrength(Enum):
    """Order Flow gücü"""
    STRONG = "STRONG"
    MODERATE = "MODERATE"
    WEAK = "WEAK"


class VolumeState(Enum):
    """Hacim durumu"""
    CLIMAX = "CLIMAX"
    SPIKE = "SPIKE"
    ABOVE_AVERAGE = "ABOVE_AVERAGE"
    AVERAGE = "AVERAGE"
    BELOW_AVERAGE = "BELOW_AVERAGE"
    DRY = "DRY"


# ═══════════════════════════════════════════════════════════════════════════════
# DATA CLASSES
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass
class VolumeProfile:
    """Volume Profile veri yapısı"""
    poc: float  # Point of Control
    vah: float  # Value Area High
    val: float  # Value Area Low
    high_volume_nodes: List[float]
    low_volume_nodes: List[float]
    volume_distribution: Dict[float, float] = field(default_factory=dict)
    poc_strength: float = 0.0
    value_area_percent: float = 0.70


@dataclass
class DeltaAnalysis:
    """Delta analiz sonucu"""
    current_delta: float
    cumulative_delta: float
    delta_percent: float
    delta_trend: FlowDirection
    delta_ma: float
    delta_divergence: bool
    divergence_type: str  # "BULLISH", "BEARISH", "NONE"
    absorption_detected: bool
    absorption_type: str  # "BUYING", "SELLING", "NONE"


@dataclass
class VWAPAnalysis:
    """VWAP analiz sonucu"""
    vwap: float
    upper_band_1: float
    upper_band_2: float
    upper_band_3: float
    lower_band_1: float
    lower_band_2: float
    lower_band_3: float
    price_position: str
    deviation_percent: float
    bands_width: float
    touch_count: int = 0


@dataclass
class WhaleActivity:
    """Whale (büyük oyuncu) aktivitesi"""
    detected: bool
    direction: FlowDirection
    volume_ratio: float
    body_ratio: float
    bar_index: int
    confidence: float
    timestamp: Optional[pd.Timestamp] = None


@dataclass
class ImbalanceAnalysis:
    """Order Flow Imbalance analizi"""
    buy_volume: float
    sell_volume: float
    imbalance_ratio: float
    imbalance_direction: FlowDirection
    stacked_imbalances: int
    significant: bool


@dataclass
class ExhaustionSignal:
    """Tükenme sinyali"""
    detected: bool
    exhaustion_type: str  # "BUYING", "SELLING", "NONE"
    confidence: float
    volume_spike: bool
    wick_ratio: float
    reversal_probability: float


@dataclass
class OrderFlowAnalysis:
    """Kapsamlı Order Flow analiz sonucu"""
    # Volume Analysis
    volume_state: VolumeState
    volume_trend: FlowDirection
    volume_ratio: float
    volume_ma: float
    volume_spike: bool
    volume_climax: bool
    
    # Delta Analysis
    delta: DeltaAnalysis
    
    # VWAP
    vwap: VWAPAnalysis
    
    # Volume Profile
    profile: Optional[VolumeProfile]
    
    # Absorption & Exhaustion
    buying_absorption: bool
    selling_absorption: bool
    buying_exhaustion: bool
    selling_exhaustion: bool
    exhaustion_signal: ExhaustionSignal
    
    # Institutional Activity
    institutional_buying: bool
    institutional_selling: bool
    smart_money_flow: FlowDirection
    
    # Whale Activity
    whale_activity: Optional[WhaleActivity]
    
    # Imbalance
    imbalance: ImbalanceAnalysis
    
    # Scores
    flow_score: float
    flow_bias: FlowDirection
    flow_strength: FlowStrength
    
    # Smart Money Divergence
    sm_divergence: bool
    sm_divergence_type: str


# ═══════════════════════════════════════════════════════════════════════════════
# ORDERFLOW ENGINE CLASS
# ═══════════════════════════════════════════════════════════════════════════════

class OrderFlowEngine:
    """
    Kurumsal Seviye Order Flow Analiz Motoru v4.2
    
    Özellikler:
    - Volume Delta Analysis
    - Cumulative Volume Delta (CVD)
    - VWAP & Deviation Bands
    - Volume Profile (POC, VAH, VAL)
    - Absorption & Exhaustion Detection
    - Institutional Activity Detection
    - Whale Order Detection
    - Order Flow Imbalance
    - Smart Money Divergence
    """
    
    def __init__(self, config=None):
        """
        Args:
            config: OrderFlowConfig instance
        """
        self.config = config or ORDERFLOW_CONFIG
    
    # ═══════════════════════════════════════════════════════════════════════════
    # VOLUME DELTA ANALYSIS
    # ═══════════════════════════════════════════════════════════════════════════
    
    def calculate_delta(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Bar-by-bar volume delta hesaplama
        
        Delta = Buy Volume - Sell Volume
        Buy Volume = Volume * (Close - Low) / (High - Low)
        Sell Volume = Volume * (High - Close) / (High - Low)
        
        Args:
            df: OHLCV DataFrame
            
        Returns:
            Delta sütunları eklenmiş DataFrame
        """
        df = df.copy()
        
        bar_range = df['High'] - df['Low']
        bar_range = bar_range.replace(0, np.nan).fillna(1)
        
        # Buying/Selling pressure
        buying_pressure = (df['Close'] - df['Low']) / bar_range
        selling_pressure = (df['High'] - df['Close']) / bar_range
        
        df['Buy_Volume'] = df['Volume'] * buying_pressure
        df['Sell_Volume'] = df['Volume'] * selling_pressure
        df['Delta'] = df['Buy_Volume'] - df['Sell_Volume']
        df['Delta_Percent'] = (df['Delta'] / df['Volume'] * 100).fillna(0)
        df['CVD'] = df['Delta'].cumsum()
        df['Delta_SMA'] = df['Delta'].rolling(self.config.delta_lookback).mean()
        df['CVD_SMA'] = df['CVD'].rolling(self.config.cvd_smoothing).mean()
        
        # Delta momentum
        df['Delta_Momentum'] = df['Delta'].rolling(5).mean() - df['Delta'].rolling(20).mean()
        
        return df
    
    def analyze_delta(self, df: pd.DataFrame) -> DeltaAnalysis:
        """
        Delta analizi
        
        Args:
            df: OHLCV DataFrame (delta sütunları ile)
            
        Returns:
            DeltaAnalysis dataclass
        """
        if 'Delta' not in df.columns:
            df = self.calculate_delta(df)
        
        current_delta = df['Delta'].iloc[-1]
        cumulative_delta = df['CVD'].iloc[-1]
        delta_percent = df['Delta_Percent'].iloc[-1]
        delta_ma = df['Delta_SMA'].iloc[-1] if not pd.isna(df['Delta_SMA'].iloc[-1]) else 0
        
        # Delta trend
        if current_delta > delta_ma * 1.2:
            delta_trend = FlowDirection.BULLISH
        elif current_delta < delta_ma * 0.8:
            delta_trend = FlowDirection.BEARISH
        else:
            delta_trend = FlowDirection.NEUTRAL
        
        # Divergence detection
        price_change = df['Close'].iloc[-5:].mean() - df['Close'].iloc[-10:-5].mean()
        cvd_change = df['CVD'].iloc[-5:].mean() - df['CVD'].iloc[-10:-5].mean()
        
        divergence = False
        divergence_type = "NONE"
        
        if price_change > 0 and cvd_change < 0:
            divergence = True
            divergence_type = "BEARISH"  # Fiyat yükseliyor ama CVD düşüyor
        elif price_change < 0 and cvd_change > 0:
            divergence = True
            divergence_type = "BULLISH"  # Fiyat düşüyor ama CVD yükseliyor
        
        # Absorption detection
        absorption, absorption_type = self._detect_absorption(df)
        
        return DeltaAnalysis(
            current_delta=current_delta,
            cumulative_delta=cumulative_delta,
            delta_percent=delta_percent,
            delta_trend=delta_trend,
            delta_ma=delta_ma,
            delta_divergence=divergence,
            divergence_type=divergence_type,
            absorption_detected=absorption,
            absorption_type=absorption_type
        )
    
    def _detect_absorption(self, df: pd.DataFrame) -> Tuple[bool, str]:
        """
        Absorption tespiti
        
        Absorption: Yüksek hacim ama düşük fiyat hareketi
        (Büyük oyuncular karşı tarafı absorbe ediyor)
        
        Args:
            df: OHLCV DataFrame
            
        Returns:
            (detected, type)
        """
        if len(df) < 20:
            return False, "NONE"
        
        last_bars = df.iloc[-5:]
        avg_volume = df['Volume'].rolling(20).mean().iloc[-1]
        avg_range = (df['High'] - df['Low']).rolling(20).mean().iloc[-1]
        
        for i in range(len(last_bars)):
            bar = last_bars.iloc[i]
            bar_range = bar['High'] - bar['Low']
            
            if (bar['Volume'] > avg_volume * self.config.absorption_volume_threshold and
                bar_range < avg_range * self.config.absorption_price_threshold):
                
                # Absorption yönünü belirle
                if bar['Close'] > bar['Open']:
                    return True, "BUYING"  # Alıcı absorption
                else:
                    return True, "SELLING"  # Satıcı absorption
        
        return False, "NONE"
    
    # ═══════════════════════════════════════════════════════════════════════════
    # VWAP ANALYSIS
    # ═══════════════════════════════════════════════════════════════════════════
    
    def calculate_vwap(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        VWAP ve standart sapma bantları hesaplama
        
        VWAP = Cumulative(TP * Volume) / Cumulative(Volume)
        TP = (High + Low + Close) / 3
        
        Args:
            df: OHLCV DataFrame
            
        Returns:
            VWAP sütunları eklenmiş DataFrame
        """
        df = df.copy()
        
        # Typical Price
        df['TP'] = (df['High'] + df['Low'] + df['Close']) / 3
        df['TP_Vol'] = df['TP'] * df['Volume']
        
        # Cumulative
        df['Cum_TP_Vol'] = df['TP_Vol'].cumsum()
        df['Cum_Vol'] = df['Volume'].cumsum()
        
        # VWAP
        df['VWAP'] = df['Cum_TP_Vol'] / df['Cum_Vol']
        
        # Standard Deviation Bands
        df['VWAP_Var'] = ((df['TP'] - df['VWAP']) ** 2 * df['Volume']).cumsum() / df['Cum_Vol']
        df['VWAP_Std'] = np.sqrt(df['VWAP_Var'])
        
        # Bands
        for i, mult in enumerate(self.config.vwap_std_bands, 1):
            df[f'VWAP_Upper_{i}'] = df['VWAP'] + df['VWAP_Std'] * mult
            df[f'VWAP_Lower_{i}'] = df['VWAP'] - df['VWAP_Std'] * mult
        
        return df
    
    def analyze_vwap(self, df: pd.DataFrame) -> VWAPAnalysis:
        """
        VWAP analizi
        
        Args:
            df: OHLCV DataFrame
            
        Returns:
            VWAPAnalysis dataclass
        """
        if 'VWAP' not in df.columns:
            df = self.calculate_vwap(df)
        
        current_price = df['Close'].iloc[-1]
        vwap = df['VWAP'].iloc[-1]
        
        # Get bands
        upper_1 = df.get('VWAP_Upper_1', pd.Series([vwap])).iloc[-1]
        upper_2 = df.get('VWAP_Upper_2', pd.Series([vwap])).iloc[-1]
        upper_3 = df.get('VWAP_Upper_3', pd.Series([vwap * 1.03])).iloc[-1]
        lower_1 = df.get('VWAP_Lower_1', pd.Series([vwap])).iloc[-1]
        lower_2 = df.get('VWAP_Lower_2', pd.Series([vwap])).iloc[-1]
        lower_3 = df.get('VWAP_Lower_3', pd.Series([vwap * 0.97])).iloc[-1]
        
        # Price position
        if current_price > upper_2:
            position = "AŞIRI PREMIUM (+2σ)"
        elif current_price > upper_1:
            position = "PREMIUM (+1σ)"
        elif current_price > vwap:
            position = "VWAP ÜSTÜ"
        elif current_price > lower_1:
            position = "VWAP ALTI"
        elif current_price > lower_2:
            position = "DISCOUNT (-1σ)"
        else:
            position = "AŞIRI DISCOUNT (-2σ)"
        
        # Deviation
        deviation = ((current_price - vwap) / vwap) * 100 if vwap != 0 else 0
        
        # Bands width
        bands_width = (upper_1 - lower_1) / vwap * 100 if vwap != 0 else 0
        
        # VWAP touch count (son 20 bar)
        touch_count = 0
        for i in range(-20, 0):
            if i + len(df) >= 0:
                bar_low = df['Low'].iloc[i]
                bar_high = df['High'].iloc[i]
                bar_vwap = df['VWAP'].iloc[i]
                if bar_low <= bar_vwap <= bar_high:
                    touch_count += 1
        
        return VWAPAnalysis(
            vwap=vwap,
            upper_band_1=upper_1,
            upper_band_2=upper_2,
            upper_band_3=upper_3,
            lower_band_1=lower_1,
            lower_band_2=lower_2,
            lower_band_3=lower_3,
            price_position=position,
            deviation_percent=deviation,
            bands_width=bands_width,
            touch_count=touch_count
        )
    
    # ═══════════════════════════════════════════════════════════════════════════
    # VOLUME PROFILE
    # ═══════════════════════════════════════════════════════════════════════════
    
    def calculate_volume_profile(
        self,
        df: pd.DataFrame,
        bins: int = None
    ) -> VolumeProfile:
        """
        Volume Profile hesaplama
        
        Args:
            df: OHLCV DataFrame
            bins: Fiyat bölme sayısı
            
        Returns:
            VolumeProfile dataclass
        """
        bins = bins or self.config.profile_bins
        
        price_min = df['Low'].min()
        price_max = df['High'].max()
        
        price_bins = np.linspace(price_min, price_max, bins + 1)
        volume_at_price = np.zeros(bins)
        
        # Her bar için volume dağılımı
        for i in range(len(df)):
            bar_low = df['Low'].iloc[i]
            bar_high = df['High'].iloc[i]
            bar_volume = df['Volume'].iloc[i]
            
            for j in range(bins):
                bin_low = price_bins[j]
                bin_high = price_bins[j + 1]
                
                # Bar ve bin kesişimi
                if bar_low <= bin_high and bar_high >= bin_low:
                    overlap = min(bar_high, bin_high) - max(bar_low, bin_low)
                    bar_range = bar_high - bar_low if bar_high != bar_low else 1
                    volume_at_price[j] += bar_volume * (overlap / bar_range)
        
        # POC (Point of Control) - en yüksek hacimli seviye
        poc_idx = np.argmax(volume_at_price)
        poc = (price_bins[poc_idx] + price_bins[poc_idx + 1]) / 2
        
        # Value Area (%70 hacim)
        total_volume = volume_at_price.sum()
        target_volume = total_volume * self.config.value_area_percent
        
        va_volume = volume_at_price[poc_idx]
        va_low_idx = poc_idx
        va_high_idx = poc_idx
        
        while va_volume < target_volume and (va_low_idx > 0 or va_high_idx < bins - 1):
            expand_low = va_low_idx > 0
            expand_high = va_high_idx < bins - 1
            
            low_vol = volume_at_price[va_low_idx - 1] if expand_low else 0
            high_vol = volume_at_price[va_high_idx + 1] if expand_high else 0
            
            if low_vol > high_vol and expand_low:
                va_low_idx -= 1
                va_volume += low_vol
            elif expand_high:
                va_high_idx += 1
                va_volume += high_vol
            elif expand_low:
                va_low_idx -= 1
                va_volume += low_vol
        
        vah = price_bins[min(va_high_idx + 1, bins)]
        val = price_bins[va_low_idx]
        
        # High/Low Volume Nodes
        avg_volume = volume_at_price.mean()
        high_vol_nodes = []
        low_vol_nodes = []
        
        for j in range(bins):
            price = (price_bins[j] + price_bins[j + 1]) / 2
            if volume_at_price[j] > avg_volume * 1.5:
                high_vol_nodes.append(price)
            elif volume_at_price[j] < avg_volume * 0.5:
                low_vol_nodes.append(price)
        
        # Volume distribution dict
        vol_dist = {}
        for j in range(bins):
            price = (price_bins[j] + price_bins[j + 1]) / 2
            vol_dist[price] = volume_at_price[j]
        
        # POC strength
        poc_strength = volume_at_price[poc_idx] / avg_volume if avg_volume > 0 else 1
        
        return VolumeProfile(
            poc=poc,
            vah=vah,
            val=val,
            high_volume_nodes=high_vol_nodes[:5],
            low_volume_nodes=low_vol_nodes[:5],
            volume_distribution=vol_dist,
            poc_strength=poc_strength,
            value_area_percent=self.config.value_area_percent
        )
    
    # ═══════════════════════════════════════════════════════════════════════════
    # EXHAUSTION DETECTION
    # ═══════════════════════════════════════════════════════════════════════════
    
    def detect_exhaustion(self, df: pd.DataFrame) -> Tuple[bool, bool, ExhaustionSignal]:
        """
        Buying/Selling exhaustion tespiti
        
        Exhaustion: Trend sonunda yüksek hacimli, uzun fitilli mum
        
        Args:
            df: OHLCV DataFrame
            
        Returns:
            (buying_exhaustion, selling_exhaustion, ExhaustionSignal)
        """
        if len(df) < 20:
            return False, False, ExhaustionSignal(
                detected=False, exhaustion_type="NONE",
                confidence=0, volume_spike=False,
                wick_ratio=0, reversal_probability=0
            )
        
        last_bar = df.iloc[-1]
        avg_volume = df['Volume'].rolling(20).mean().iloc[-1]
        avg_range = (df['High'] - df['Low']).rolling(20).mean().iloc[-1]
        
        bar_range = last_bar['High'] - last_bar['Low']
        body = abs(last_bar['Close'] - last_bar['Open'])
        upper_wick = last_bar['High'] - max(last_bar['Open'], last_bar['Close'])
        lower_wick = min(last_bar['Open'], last_bar['Close']) - last_bar['Low']
        
        # Wick ratio
        wick_ratio = max(upper_wick, lower_wick) / bar_range if bar_range > 0 else 0
        
        # Volume spike
        volume_spike = last_bar['Volume'] > avg_volume * 1.5
        
        # Trend direction (son 5 bar)
        trend_up = df['Close'].iloc[-5:].mean() > df['Close'].iloc[-10:-5].mean()
        trend_down = df['Close'].iloc[-5:].mean() < df['Close'].iloc[-10:-5].mean()
        
        buying_exhaustion = False
        selling_exhaustion = False
        exhaustion_type = "NONE"
        confidence = 0
        reversal_prob = 0
        
        # Buying Exhaustion: Yükseliş trendinde uzun üst fitil
        if (volume_spike and
            upper_wick > bar_range * self.config.exhaustion_wick_ratio and
            trend_up):
            buying_exhaustion = True
            exhaustion_type = "BUYING"
            confidence = min(wick_ratio * 100 + (last_bar['Volume'] / avg_volume - 1) * 30, 95)
            reversal_prob = confidence * 0.8
        
        # Selling Exhaustion: Düşüş trendinde uzun alt fitil
        if (volume_spike and
            lower_wick > bar_range * self.config.exhaustion_wick_ratio and
            trend_down):
            selling_exhaustion = True
            exhaustion_type = "SELLING"
            confidence = min(wick_ratio * 100 + (last_bar['Volume'] / avg_volume - 1) * 30, 95)
            reversal_prob = confidence * 0.8
        
        signal = ExhaustionSignal(
            detected=buying_exhaustion or selling_exhaustion,
            exhaustion_type=exhaustion_type,
            confidence=confidence,
            volume_spike=volume_spike,
            wick_ratio=wick_ratio,
            reversal_probability=reversal_prob
        )
        
        return buying_exhaustion, selling_exhaustion, signal
    
    # ═══════════════════════════════════════════════════════════════════════════
    # WHALE DETECTION
    # ═══════════════════════════════════════════════════════════════════════════
    
    def detect_whale_activity(self, df: pd.DataFrame) -> Optional[WhaleActivity]:
        """
        Whale (büyük oyuncu) aktivitesi tespiti
        
        Whale: Çok yüksek hacimli, geniş gövdeli bar
        
        Args:
            df: OHLCV DataFrame
            
        Returns:
            WhaleActivity veya None
        """
        if len(df) < self.config.whale_lookback:
            return None
        
        avg_volume = df['Volume'].rolling(self.config.whale_lookback).mean().iloc[-1]
        avg_body = abs(df['Close'] - df['Open']).rolling(self.config.whale_lookback).mean().iloc[-1]
        
        # Son 5 bar'ı kontrol et
        for i in range(-5, 0):
            bar = df.iloc[i]
            body = abs(bar['Close'] - bar['Open'])
            
            volume_ratio = bar['Volume'] / avg_volume if avg_volume > 0 else 1
            body_ratio = body / avg_body if avg_body > 0 else 1
            
            if (volume_ratio > self.config.whale_volume_threshold and
                body_ratio > self.config.whale_body_threshold):
                
                # Yön belirleme
                direction = FlowDirection.BULLISH if bar['Close'] > bar['Open'] else FlowDirection.BEARISH
                
                # Confidence
                confidence = min((volume_ratio - 2) * 25 + (body_ratio - 1) * 25, 95)
                
                return WhaleActivity(
                    detected=True,
                    direction=direction,
                    volume_ratio=volume_ratio,
                    body_ratio=body_ratio,
                    bar_index=len(df) + i,
                    confidence=confidence,
                    timestamp=df.index[i] if hasattr(df.index, '__getitem__') else None
                )
        
        return None
    
    # ═══════════════════════════════════════════════════════════════════════════
    # IMBALANCE ANALYSIS
    # ═══════════════════════════════════════════════════════════════════════════
    
    def analyze_imbalance(self, df: pd.DataFrame) -> ImbalanceAnalysis:
        """
        Order Flow Imbalance analizi
        
        Args:
            df: OHLCV DataFrame
            
        Returns:
            ImbalanceAnalysis dataclass
        """
        if 'Buy_Volume' not in df.columns:
            df = self.calculate_delta(df)
        
        # Son 10 bar toplam
        recent = df.iloc[-10:]
        buy_volume = recent['Buy_Volume'].sum()
        sell_volume = recent['Sell_Volume'].sum()
        
        total = buy_volume + sell_volume
        if total > 0:
            imbalance_ratio = (buy_volume - sell_volume) / total
        else:
            imbalance_ratio = 0
        
        # Direction
        if imbalance_ratio > self.config.imbalance_ratio_threshold:
            direction = FlowDirection.BULLISH
        elif imbalance_ratio < -self.config.imbalance_ratio_threshold:
            direction = FlowDirection.BEARISH
        else:
            direction = FlowDirection.NEUTRAL
        
        # Stacked imbalances (üst üste aynı yönde imbalance)
        stacked = 0
        for i in range(-5, 0):
            bar_buy = df['Buy_Volume'].iloc[i]
            bar_sell = df['Sell_Volume'].iloc[i]
            bar_total = bar_buy + bar_sell
            
            if bar_total > 0:
                bar_ratio = (bar_buy - bar_sell) / bar_total
                if direction == FlowDirection.BULLISH and bar_ratio > 0.3:
                    stacked += 1
                elif direction == FlowDirection.BEARISH and bar_ratio < -0.3:
                    stacked += 1
        
        # Significance
        significant = abs(imbalance_ratio) > self.config.imbalance_ratio_threshold and stacked >= 3
        
        return ImbalanceAnalysis(
            buy_volume=buy_volume,
            sell_volume=sell_volume,
            imbalance_ratio=imbalance_ratio,
            imbalance_direction=direction,
            stacked_imbalances=stacked,
            significant=significant
        )
    
    # ═══════════════════════════════════════════════════════════════════════════
    # INSTITUTIONAL ACTIVITY
    # ═══════════════════════════════════════════════════════════════════════════
    
    def detect_institutional_activity(
        self,
        df: pd.DataFrame
    ) -> Tuple[bool, bool, FlowDirection]:
        """
        Kurumsal aktivite tespiti
        
        Args:
            df: OHLCV DataFrame
            
        Returns:
            (institutional_buying, institutional_selling, smart_money_flow)
        """
        if len(df) < 20:
            return False, False, FlowDirection.NEUTRAL
        
        avg_volume = df['Volume'].rolling(20).mean().iloc[-1]
        avg_body = abs(df['Close'] - df['Open']).rolling(20).mean().iloc[-1]
        
        recent = df.iloc[-self.config.institutional_consecutive_bars:]
        
        bullish_bars = 0
        bearish_bars = 0
        high_vol_bullish = 0
        high_vol_bearish = 0
        
        for i in range(len(recent)):
            bar = recent.iloc[i]
            body = abs(bar['Close'] - bar['Open'])
            
            if bar['Close'] > bar['Open']:
                bullish_bars += 1
                if (bar['Volume'] > avg_volume * self.config.institutional_volume_mult and
                    body > avg_body * self.config.institutional_body_mult):
                    high_vol_bullish += 1
            else:
                bearish_bars += 1
                if (bar['Volume'] > avg_volume * self.config.institutional_volume_mult and
                    body > avg_body * self.config.institutional_body_mult):
                    high_vol_bearish += 1
        
        inst_buying = high_vol_bullish >= 2 and bullish_bars >= self.config.institutional_consecutive_bars
        inst_selling = high_vol_bearish >= 2 and bearish_bars >= self.config.institutional_consecutive_bars
        
        if inst_buying and not inst_selling:
            flow = FlowDirection.BULLISH
        elif inst_selling and not inst_buying:
            flow = FlowDirection.BEARISH
        else:
            flow = FlowDirection.NEUTRAL
        
        return inst_buying, inst_selling, flow
    
    # ═══════════════════════════════════════════════════════════════════════════
    # SMART MONEY DIVERGENCE
    # ═══════════════════════════════════════════════════════════════════════════
    
    def detect_smart_money_divergence(self, df: pd.DataFrame) -> Tuple[bool, str]:
        """
        Smart Money Divergence tespiti
        
        CVD ve fiyat arasındaki uyumsuzluk
        
        Args:
            df: OHLCV DataFrame
            
        Returns:
            (divergence_detected, divergence_type)
        """
        if 'CVD' not in df.columns:
            df = self.calculate_delta(df)
        
        if len(df) < 20:
            return False, "NONE"
        
        # Son 10 vs önceki 10 bar karşılaştırma
        recent_price = df['Close'].iloc[-10:].values
        prev_price = df['Close'].iloc[-20:-10].values
        
        recent_cvd = df['CVD'].iloc[-10:].values
        prev_cvd = df['CVD'].iloc[-20:-10].values
        
        # Price trend
        price_higher = recent_price[-1] > prev_price[-1]
        price_lower = recent_price[-1] < prev_price[-1]
        
        # CVD trend
        cvd_higher = recent_cvd[-1] > prev_cvd[-1]
        cvd_lower = recent_cvd[-1] < prev_cvd[-1]
        
        # New high/low with divergence
        price_new_high = recent_price[-1] > prev_price.max()
        price_new_low = recent_price[-1] < prev_price.min()
        cvd_no_new_high = recent_cvd[-1] < prev_cvd.max()
        cvd_no_new_low = recent_cvd[-1] > prev_cvd.min()
        
        # Bearish divergence: Fiyat yeni zirve ama CVD yapmıyor
        if price_new_high and cvd_no_new_high:
            return True, "BEARISH"
        
        # Bullish divergence: Fiyat yeni dip ama CVD yapmıyor
        if price_new_low and cvd_no_new_low:
            return True, "BULLISH"
        
        return False, "NONE"
    
    # ═══════════════════════════════════════════════════════════════════════════
    # VOLUME STATE & CLIMAX
    # ═══════════════════════════════════════════════════════════════════════════
    
    def analyze_volume_state(self, df: pd.DataFrame) -> Tuple[VolumeState, bool, bool]:
        """
        Hacim durumu analizi
        
        Args:
            df: OHLCV DataFrame
            
        Returns:
            (volume_state, spike, climax)
        """
        if len(df) < 20:
            return VolumeState.AVERAGE, False, False
        
        avg_volume = df['Volume'].rolling(self.config.volume_ma_period).mean().iloc[-1]
        current_volume = df['Volume'].iloc[-1]
        
        ratio = current_volume / avg_volume if avg_volume > 0 else 1
        
        # State
        if ratio > self.config.volume_climax_threshold:
            state = VolumeState.CLIMAX
        elif ratio > self.config.volume_spike_threshold:
            state = VolumeState.SPIKE
        elif ratio > 1.2:
            state = VolumeState.ABOVE_AVERAGE
        elif ratio > 0.8:
            state = VolumeState.AVERAGE
        elif ratio > 0.5:
            state = VolumeState.BELOW_AVERAGE
        else:
            state = VolumeState.DRY
        
        spike = ratio > self.config.volume_spike_threshold
        climax = ratio > self.config.volume_climax_threshold
        
        return state, spike, climax
    
    # ═══════════════════════════════════════════════════════════════════════════
    # FLOW SCORE CALCULATION
    # ═══════════════════════════════════════════════════════════════════════════
    
    def _calculate_flow_score(
        self,
        volume_trend: FlowDirection,
        volume_ratio: float,
        delta: DeltaAnalysis,
        vwap: VWAPAnalysis,
        imbalance: ImbalanceAnalysis,
        inst_buying: bool,
        inst_selling: bool,
        whale: Optional[WhaleActivity],
        exhaustion: ExhaustionSignal
    ) -> Tuple[float, FlowDirection, FlowStrength]:
        """
        Flow skoru hesapla
        
        Args:
            Tüm analiz sonuçları
            
        Returns:
            (score, bias, strength)
        """
        bullish_score = 0
        bearish_score = 0
        
        # Volume trend (15 puan)
        if volume_trend == FlowDirection.BULLISH:
            bullish_score += 15
        elif volume_trend == FlowDirection.BEARISH:
            bearish_score += 15
        
        # Volume ratio bonus (10 puan)
        if volume_ratio > 1.5:
            if delta.delta_trend == FlowDirection.BULLISH:
                bullish_score += 10
            else:
                bearish_score += 10
        
        # Delta (20 puan)
        if delta.delta_trend == FlowDirection.BULLISH:
            bullish_score += 20
        elif delta.delta_trend == FlowDirection.BEARISH:
            bearish_score += 20
        
        # VWAP position (15 puan)
        if "ÜSTÜ" in vwap.price_position or "PREMIUM" in vwap.price_position:
            bullish_score += 15
        elif "ALTI" in vwap.price_position or "DISCOUNT" in vwap.price_position:
            bearish_score += 15
        
        # Imbalance (15 puan)
        if imbalance.significant:
            if imbalance.imbalance_direction == FlowDirection.BULLISH:
                bullish_score += 15
            else:
                bearish_score += 15
        
        # Institutional (20 puan)
        if inst_buying:
            bullish_score += 20
        if inst_selling:
            bearish_score += 20
        
        # Whale (15 puan)
        if whale and whale.detected:
            if whale.direction == FlowDirection.BULLISH:
                bullish_score += 15
            else:
                bearish_score += 15
        
        # Exhaustion penalty (-15 puan)
        if exhaustion.detected:
            if exhaustion.exhaustion_type == "BUYING":
                bullish_score -= 15
            else:
                bearish_score -= 15
        
        # Divergence penalty (-10 puan)
        if delta.delta_divergence:
            if delta.divergence_type == "BEARISH":
                bullish_score -= 10
            else:
                bearish_score -= 10
        
        # Calculate final
        total_score = bullish_score - bearish_score
        
        if total_score > 25:
            bias = FlowDirection.BULLISH
            score = min(50 + total_score, 100)
        elif total_score < -25:
            bias = FlowDirection.BEARISH
            score = min(50 - total_score, 100)
        else:
            bias = FlowDirection.NEUTRAL
            score = 50 + abs(total_score)
        
        # Strength
        if abs(total_score) > 50:
            strength = FlowStrength.STRONG
        elif abs(total_score) > 25:
            strength = FlowStrength.MODERATE
        else:
            strength = FlowStrength.WEAK
        
        return score, bias, strength
    
    # ═══════════════════════════════════════════════════════════════════════════
    # ANA ANALİZ FONKSİYONU
    # ═══════════════════════════════════════════════════════════════════════════
    
    def analyze(self, df: pd.DataFrame) -> OrderFlowAnalysis:
        """
        Kapsamlı Order Flow analizi
        
        Args:
            df: OHLCV DataFrame
            
        Returns:
            OrderFlowAnalysis dataclass
        """
        if df is None or len(df) < 30:
            return self._empty_analysis()
        
        try:
            # Volume Analysis
            volume_state, volume_spike, volume_climax = self.analyze_volume_state(df)
            avg_volume = df['Volume'].rolling(self.config.volume_ma_period).mean().iloc[-1]
            current_volume = df['Volume'].iloc[-1]
            volume_ratio = current_volume / avg_volume if avg_volume > 0 else 1
            
            # Volume trend
            vol_sma_5 = df['Volume'].rolling(5).mean().iloc[-1]
            vol_sma_20 = df['Volume'].rolling(20).mean().iloc[-1]
            
            if vol_sma_5 > vol_sma_20 * 1.2:
                volume_trend = FlowDirection.BULLISH
            elif vol_sma_5 < vol_sma_20 * 0.8:
                volume_trend = FlowDirection.BEARISH
            else:
                volume_trend = FlowDirection.NEUTRAL
            
            # Delta Analysis
            df = self.calculate_delta(df)
            delta_analysis = self.analyze_delta(df)
            
            # VWAP Analysis
            df = self.calculate_vwap(df)
            vwap_analysis = self.analyze_vwap(df)
            
            # Volume Profile
            try:
                profile = self.calculate_volume_profile(df)
            except Exception:
                profile = None
            
            # Absorption
            buying_absorption = False
            selling_absorption = False
            
            if delta_analysis.absorption_detected:
                if delta_analysis.absorption_type == "BUYING":
                    buying_absorption = True
                elif delta_analysis.absorption_type == "SELLING":
                    selling_absorption = True
            
            # Exhaustion
            buying_exhaustion, selling_exhaustion, exhaustion_signal = self.detect_exhaustion(df)
            
            # Institutional Activity
            inst_buying, inst_selling, smart_flow = self.detect_institutional_activity(df)
            
            # Whale Activity
            whale_activity = self.detect_whale_activity(df)
            
            # Imbalance
            imbalance = self.analyze_imbalance(df)
            
            # Smart Money Divergence
            sm_div, sm_div_type = self.detect_smart_money_divergence(df)
            
            # Flow Score
            flow_score, flow_bias, flow_strength = self._calculate_flow_score(
                volume_trend, volume_ratio, delta_analysis, vwap_analysis,
                imbalance, inst_buying, inst_selling, whale_activity, exhaustion_signal
            )
            
            return OrderFlowAnalysis(
                volume_state=volume_state,
                volume_trend=volume_trend,
                volume_ratio=volume_ratio,
                volume_ma=avg_volume,
                volume_spike=volume_spike,
                volume_climax=volume_climax,
                delta=delta_analysis,
                vwap=vwap_analysis,
                profile=profile,
                buying_absorption=buying_absorption,
                selling_absorption=selling_absorption,
                buying_exhaustion=buying_exhaustion,
                selling_exhaustion=selling_exhaustion,
                exhaustion_signal=exhaustion_signal,
                institutional_buying=inst_buying,
                institutional_selling=inst_selling,
                smart_money_flow=smart_flow,
                whale_activity=whale_activity,
                imbalance=imbalance,
                flow_score=flow_score,
                flow_bias=flow_bias,
                flow_strength=flow_strength,
                sm_divergence=sm_div,
                sm_divergence_type=sm_div_type
            )
        
        except Exception as e:
            logger.error(f"❌ Order Flow Analysis Error: {e}")
            import traceback
            traceback.print_exc()
            return self._empty_analysis()
    
    def _empty_analysis(self) -> OrderFlowAnalysis:
        """Boş analiz sonucu"""
        return OrderFlowAnalysis(
            volume_state=VolumeState.AVERAGE,
            volume_trend=FlowDirection.NEUTRAL,
            volume_ratio=1.0,
            volume_ma=0,
            volume_spike=False,
            volume_climax=False,
            delta=DeltaAnalysis(0, 0, 0, FlowDirection.NEUTRAL, 0, False, "NONE", False, "NONE"),
            vwap=VWAPAnalysis(0, 0, 0, 0, 0, 0, 0, "HESAPLANAMADI", 0, 0),
            profile=None,
            buying_absorption=False,
            selling_absorption=False,
            buying_exhaustion=False,
            selling_exhaustion=False,
            exhaustion_signal=ExhaustionSignal(False, "NONE", 0, False, 0, 0),
            institutional_buying=False,
            institutional_selling=False,
            smart_money_flow=FlowDirection.NEUTRAL,
            whale_activity=None,
            imbalance=ImbalanceAnalysis(0, 0, 0, FlowDirection.NEUTRAL, 0, False),
            flow_score=50,
            flow_bias=FlowDirection.NEUTRAL,
            flow_strength=FlowStrength.WEAK,
            sm_divergence=False,
            sm_divergence_type="NONE"
        )


# ═══════════════════════════════════════════════════════════════════════════════
# TEST
# ═══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    print("OrderFlow Engine v4.2 - Test")
    print("=" * 60)
    
    # Test için dummy data
    import numpy as np
    
    dates = pd.date_range(start='2024-01-01', periods=100, freq='4H')
    np.random.seed(42)
    
    prices = 100 + np.cumsum(np.random.randn(100) * 2)
    
    df = pd.DataFrame({
        'Open': prices + np.random.randn(100) * 0.5,
        'High': prices + np.abs(np.random.randn(100)) * 2,
        'Low': prices - np.abs(np.random.randn(100)) * 2,
        'Close': prices,
        'Volume': np.random.randint(1000000, 5000000, 100)
    }, index=dates)
    
    df['High'] = df[['Open', 'High', 'Close']].max(axis=1)
    df['Low'] = df[['Open', 'Low', 'Close']].min(axis=1)
    
    engine = OrderFlowEngine()
    analysis = engine.analyze(df)
    
    print(f"\n📊 ORDERFLOW ANALİZ SONUCU")
    print("=" * 60)
    print(f"📈 Volume State: {analysis.volume_state.value}")
    print(f"📊 Volume Trend: {analysis.volume_trend.value}")
    print(f"📈 Volume Ratio: {analysis.volume_ratio:.2f}x")
    print(f"📊 Volume Spike: {analysis.volume_spike} | Climax: {analysis.volume_climax}")
    print(f"\n⚡ Delta: {analysis.delta.current_delta:.0f}")
    print(f"📊 CVD: {analysis.delta.cumulative_delta:.0f}")
    print(f"📈 Delta Trend: {analysis.delta.delta_trend.value}")
    print(f"⚠️ Divergence: {analysis.delta.delta_divergence} ({analysis.delta.divergence_type})")
    print(f"\n💰 VWAP: {analysis.vwap.vwap:.2f}")
    print(f"📍 Position: {analysis.vwap.price_position}")
    print(f"📏 Deviation: {analysis.vwap.deviation_percent:.2f}%")
    print(f"\n🐋 Whale Activity: {analysis.whale_activity is not None}")
    print(f"🏛️ Institutional Buy: {analysis.institutional_buying}")
    print(f"🏛️ Institutional Sell: {analysis.institutional_selling}")
    print(f"💹 Smart Money Flow: {analysis.smart_money_flow.value}")
    print(f"\n📊 Imbalance: {analysis.imbalance.imbalance_ratio:.2f}")
    print(f"🔥 Flow Score: {analysis.flow_score:.1f}/100")
    print(f"📍 Flow Bias: {analysis.flow_bias.value}")
    print(f"💪 Flow Strength: {analysis.flow_strength.value}")
