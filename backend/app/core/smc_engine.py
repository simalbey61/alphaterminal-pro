"""
AlphaTerminal Pro - Smart Money Concepts Engine v4.2
=====================================================

Kurumsal Seviye SMC Analiz Motoru

YENİ ÖZELLİKLER (v4.2):
- Multi-Timeframe Confluence Analysis
- Wyckoff Phase Detection
- Session-based Analysis (Asian, London, NY, BIST)
- Inducement Level Detection
- Institutional Order Block Detection
- Advanced Liquidity Mapping
- Kill Zone Detection

Özellikler:
- Gelişmiş Structure Analysis (BOS, CHoCH, MSS)
- Order Block Detection (OB, Breaker, Mitigation)
- Fair Value Gap Analysis
- Liquidity Mapping (EQH/EQL, Sweeps)
- Premium/Discount Zone Detection
- Multi-Timeframe Confluence

Author: AlphaTerminal Team
Version: 4.2.0
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple, Union
from dataclasses import dataclass, field
from datetime import datetime, time
from enum import Enum

from app.core.config import (
    logger, SMC_CONFIG, MarketStructure, ZoneType,
    WyckoffPhase, OrderBlockType, SessionType
)


# ═══════════════════════════════════════════════════════════════════════════════
# DATA CLASSES
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass
class SwingPoint:
    """Swing noktası veri yapısı"""
    index: int
    price: float
    type: str  # "HH", "HL", "LH", "LL", "HIGH", "LOW"
    strength: int  # Kaç bar onayladı
    timestamp: Optional[pd.Timestamp] = None
    volume: float = 0.0
    is_valid: bool = True


@dataclass
class OrderBlock:
    """Order Block veri yapısı"""
    top: float
    bottom: float
    index: int
    ob_type: str  # "BULLISH", "BEARISH", "BULLISH_BREAKER", "BEARISH_BREAKER"
    strength: float  # 0-100 skor
    mitigated: bool = False
    mitigation_level: float = 0.0
    volume_confirmation: bool = False
    timestamp: Optional[pd.Timestamp] = None
    imbalance_ratio: float = 0.0
    touch_count: int = 0
    age_bars: int = 0
    institutional: bool = False


@dataclass
class FVG:
    """Fair Value Gap veri yapısı"""
    top: float
    bottom: float
    index: int
    fvg_type: str  # "BULLISH", "BEARISH"
    size: float
    size_atr: float = 0.0
    filled: bool = False
    fill_percentage: float = 0.0
    timestamp: Optional[pd.Timestamp] = None


@dataclass
class LiquidityLevel:
    """Likidite seviyesi veri yapısı"""
    price: float
    level_type: str  # "EQH", "EQL", "SWING_HIGH", "SWING_LOW", "BSL", "SSL"
    strength: int  # Kaç kez test edildi
    swept: bool = False
    sweep_timestamp: Optional[pd.Timestamp] = None
    index: int = 0
    volume_at_level: float = 0.0
    touches: int = 1


@dataclass
class InducementLevel:
    """Inducement (tuzak) seviyesi veri yapısı"""
    price: float
    inducement_type: str  # "BULL_TRAP", "BEAR_TRAP"
    strength: float
    index: int
    triggered: bool = False
    timestamp: Optional[pd.Timestamp] = None


@dataclass 
class SessionAnalysis:
    """Session bazlı analiz sonucu"""
    session: SessionType
    high: float
    low: float
    open: float
    close: float
    volume: float
    range_atr: float
    direction: str  # "BULLISH", "BEARISH", "NEUTRAL"
    swept_high: bool = False
    swept_low: bool = False


@dataclass
class WyckoffAnalysis:
    """Wyckoff analiz sonucu"""
    phase: WyckoffPhase
    phase_label: str
    confidence: float
    volume_pattern: str
    price_pattern: str
    spring_detected: bool = False
    upthrust_detected: bool = False
    sos_detected: bool = False  # Sign of Strength
    sow_detected: bool = False  # Sign of Weakness


@dataclass
class MTFConfluence:
    """Multi-Timeframe Confluence sonucu"""
    timeframes_analyzed: List[str]
    overall_bias: str  # "BULLISH", "BEARISH", "NEUTRAL"
    confluence_score: float
    aligned_timeframes: int
    htf_structure: str  # Higher Timeframe structure
    ltf_structure: str  # Lower Timeframe structure
    htf_ob: Optional[OrderBlock] = None
    ltf_confirmation: bool = False


@dataclass
class SMCAnalysis:
    """Kapsamlı SMC analiz sonucu"""
    # Structure
    structure: MarketStructure
    structure_label: str
    bos_detected: bool
    choch_detected: bool
    mss_status: str
    trend_strength: str
    
    # Order Blocks
    order_blocks: List[OrderBlock]
    active_ob: Optional[OrderBlock]
    breaker_blocks: List[OrderBlock]
    institutional_obs: List[OrderBlock]
    
    # FVG
    fvgs: List[FVG]
    active_fvg: Optional[FVG]
    unfilled_fvgs: int
    
    # Liquidity
    liquidity_levels: List[LiquidityLevel]
    liquidity_high: float
    liquidity_low: float
    liquidity_sweep: str
    bsl_levels: List[float]  # Buy-side liquidity
    ssl_levels: List[float]  # Sell-side liquidity
    
    # Inducement
    inducement_levels: List[InducementLevel]
    active_inducement: Optional[InducementLevel]
    
    # Premium/Discount
    premium_discount: str
    equilibrium: float
    zone_type: ZoneType
    
    # Swing Points
    swing_points: List[SwingPoint]
    last_swing_high: Optional[SwingPoint]
    last_swing_low: Optional[SwingPoint]
    
    # Wyckoff
    wyckoff: Optional[WyckoffAnalysis]
    
    # Session
    session_analysis: Optional[SessionAnalysis]
    current_session: SessionType
    kill_zone_active: bool
    
    # MTF
    mtf_confluence: Optional[MTFConfluence]
    
    # Scores
    confluence_score: float
    institutional_score: float
    
    # Price Action
    price_action: str
    candle_pattern: str
    
    # Trade Setup
    bias: str  # "LONG", "SHORT", "NEUTRAL"
    entry_zone: Optional[Tuple[float, float]]
    invalidation: Optional[float]


# ═══════════════════════════════════════════════════════════════════════════════
# SMC ENGINE CLASS
# ═══════════════════════════════════════════════════════════════════════════════

class SMCEngine:
    """
    Kurumsal Seviye Smart Money Concepts Motoru v4.2
    
    Özellikler:
    - Gelişmiş Structure Analysis (BOS, CHoCH, MSS)
    - Order Block Detection (OB, Breaker, Mitigation, Institutional)
    - Fair Value Gap Analysis
    - Liquidity Mapping (EQH/EQL, BSL/SSL, Sweeps)
    - Premium/Discount Zone Detection
    - Multi-Timeframe Confluence
    - Wyckoff Phase Detection
    - Session-based Analysis (Kill Zones)
    - Inducement Level Detection
    """
    
    def __init__(self, config=None):
        """
        Args:
            config: SMCConfig instance (None ise default kullanılır)
        """
        self.config = config or SMC_CONFIG
        self._cache = {}
    
    # ═══════════════════════════════════════════════════════════════════════════
    # SWING POINT DETECTION
    # ═══════════════════════════════════════════════════════════════════════════
    
    def _detect_swing_points(
        self,
        df: pd.DataFrame,
        lookback: int = None,
        strength: int = None
    ) -> List[SwingPoint]:
        """
        Gelişmiş swing noktası tespiti
        
        Swing High: Sol ve sağda 'strength' kadar bar'dan yüksek
        Swing Low: Sol ve sağda 'strength' kadar bar'dan düşük
        
        Args:
            df: OHLCV DataFrame
            lookback: Geriye bakış periyodu
            strength: Swing gücü (kaç bar onaylamalı)
            
        Returns:
            SwingPoint listesi
        """
        lookback = lookback or self.config.swing_lookback
        strength = strength or self.config.swing_strength
        
        swing_points = []
        high = df['High'].values
        low = df['Low'].values
        volume = df['Volume'].values if 'Volume' in df.columns else np.zeros(len(df))
        
        for i in range(strength, len(df) - strength):
            # Swing High kontrolü
            is_swing_high = True
            for j in range(1, strength + 1):
                if high[i] <= high[i - j] or high[i] <= high[i + j]:
                    is_swing_high = False
                    break
            
            if is_swing_high:
                swing_points.append(SwingPoint(
                    index=i,
                    price=high[i],
                    type="HIGH",
                    strength=strength,
                    timestamp=df.index[i] if hasattr(df.index, '__getitem__') else None,
                    volume=volume[i],
                    is_valid=True
                ))
            
            # Swing Low kontrolü
            is_swing_low = True
            for j in range(1, strength + 1):
                if low[i] >= low[i - j] or low[i] >= low[i + j]:
                    is_swing_low = False
                    break
            
            if is_swing_low:
                swing_points.append(SwingPoint(
                    index=i,
                    price=low[i],
                    type="LOW",
                    strength=strength,
                    timestamp=df.index[i] if hasattr(df.index, '__getitem__') else None,
                    volume=volume[i],
                    is_valid=True
                ))
        
        return sorted(swing_points, key=lambda x: x.index)
    
    def _classify_swing_sequence(
        self,
        swing_points: List[SwingPoint]
    ) -> List[SwingPoint]:
        """
        Swing noktalarını HH, HL, LH, LL olarak sınıflandır
        
        Args:
            swing_points: Sınıflandırılmamış swing noktaları
            
        Returns:
            Sınıflandırılmış SwingPoint listesi
        """
        if len(swing_points) < 2:
            return swing_points
        
        classified = []
        last_high = None
        last_low = None
        
        for sp in swing_points:
            if sp.type == "HIGH":
                if last_high is None:
                    sp.type = "HH"  # İlk yüksek
                elif sp.price > last_high.price:
                    sp.type = "HH"  # Higher High
                else:
                    sp.type = "LH"  # Lower High
                last_high = sp
            else:  # LOW
                if last_low is None:
                    sp.type = "HL"  # İlk düşük
                elif sp.price > last_low.price:
                    sp.type = "HL"  # Higher Low
                else:
                    sp.type = "LL"  # Lower Low
                last_low = sp
            
            classified.append(sp)
        
        return classified
    
    # ═══════════════════════════════════════════════════════════════════════════
    # MARKET STRUCTURE ANALYSIS
    # ═══════════════════════════════════════════════════════════════════════════
    
    def _analyze_structure(
        self,
        df: pd.DataFrame,
        swing_points: List[SwingPoint]
    ) -> Tuple[MarketStructure, bool, bool, str, str]:
        """
        Market yapısı analizi
        
        Args:
            df: OHLCV DataFrame
            swing_points: Sınıflandırılmış swing noktaları
            
        Returns:
            (structure, bos_detected, choch_detected, label, trend_strength)
        """
        if len(swing_points) < 4:
            return MarketStructure.UNDEFINED, False, False, "VERİ YETERSİZ", "UNKNOWN"
        
        # Son 6 swing noktasını al
        recent_swings = swing_points[-6:] if len(swing_points) >= 6 else swing_points
        
        highs = [sp for sp in recent_swings if sp.type in ["HH", "LH"]]
        lows = [sp for sp in recent_swings if sp.type in ["HL", "LL"]]
        
        if len(highs) < 2 or len(lows) < 2:
            return MarketStructure.RANGING, False, False, "YATAY (RANGING)", "WEAK"
        
        # Son iki high ve low'u karşılaştır
        last_high_types = [sp.type for sp in highs[-2:]]
        last_low_types = [sp.type for sp in lows[-2:]]
        
        bos_detected = False
        choch_detected = False
        trend_strength = "MODERATE"
        
        close = df['Close'].iloc[-1]
        
        # BULLISH Structure: HH + HL serisi
        if last_high_types == ["HH", "HH"] and last_low_types == ["HL", "HL"]:
            structure = MarketStructure.BULLISH
            label = "BULLISH (YUKARI AKIŞ)"
            trend_strength = "STRONG"
            
            # BOS: Son swing high kırıldı mı?
            if close > highs[-1].price:
                bos_detected = True
        
        # BEARISH Structure: LH + LL serisi
        elif last_high_types == ["LH", "LH"] and last_low_types == ["LL", "LL"]:
            structure = MarketStructure.BEARISH
            label = "BEARISH (AŞAĞI AKIŞ)"
            trend_strength = "STRONG"
            
            # BOS: Son swing low kırıldı mı?
            if close < lows[-1].price:
                bos_detected = True
        
        # CHoCH: Trend değişimi - Bullish'ten Bearish'e
        elif len(last_high_types) >= 2 and len(last_low_types) >= 2:
            if last_high_types[-1] == "LH" and last_low_types[-2] == "HL":
                structure = MarketStructure.BEARISH
                label = "CHoCH - BEARISH DÖNÜŞ"
                choch_detected = True
                trend_strength = "MODERATE"
            
            # CHoCH: Bearish'ten Bullish'e
            elif last_high_types[-2] == "LH" and last_low_types[-1] == "HL":
                structure = MarketStructure.BULLISH
                label = "CHoCH - BULLISH DÖNÜŞ"
                choch_detected = True
                trend_strength = "MODERATE"
            else:
                structure = MarketStructure.RANGING
                label = "YATAY (RANGING)"
                trend_strength = "WEAK"
        else:
            structure = MarketStructure.RANGING
            label = "YATAY (RANGING)"
            trend_strength = "WEAK"
        
        return structure, bos_detected, choch_detected, label, trend_strength
    
    # ═══════════════════════════════════════════════════════════════════════════
    # ORDER BLOCK DETECTION
    # ═══════════════════════════════════════════════════════════════════════════
    
    def _detect_order_blocks(
        self,
        df: pd.DataFrame,
        swing_points: List[SwingPoint]
    ) -> List[OrderBlock]:
        """
        Gelişmiş Order Block tespiti
        
        OB Kriterleri:
        1. Güçlü momentum mumundan önceki ters yönlü mum
        2. Hacim onayı (ortalama üstü)
        3. Yapısal önemi (swing noktası yakını)
        4. Institutional imbalance
        
        Args:
            df: OHLCV DataFrame
            swing_points: Swing noktaları
            
        Returns:
            OrderBlock listesi
        """
        order_blocks = []
        
        high = df['High'].values
        low = df['Low'].values
        close = df['Close'].values
        open_ = df['Open'].values
        volume = df['Volume'].values if 'Volume' in df.columns else np.ones(len(df))
        
        lookback = min(self.config.ob_lookback, len(df) - 2)
        avg_body = np.mean(np.abs(close - open_))
        avg_volume = np.mean(volume)
        
        current_index = len(df) - 1
        
        for i in range(lookback, 1, -1):
            idx = len(df) - i
            
            if idx < 1 or idx >= len(df) - 1:
                continue
            
            current_body = abs(close[idx] - open_[idx])
            next_body = abs(close[idx + 1] - open_[idx + 1])
            
            # Bullish OB: Düşüş mumundan sonra güçlü yükseliş
            if (close[idx] < open_[idx] and  # Düşüş mumu (OB mumu)
                close[idx + 1] > open_[idx + 1] and  # Yükseliş mumu (momentum)
                next_body > avg_body * self.config.ob_body_multiplier and  # Güçlü hareket
                close[idx + 1] > high[idx]):  # Önceki mumu aştı
                
                # Hacim onayı
                vol_confirm = volume[idx + 1] > avg_volume
                
                # Mitigation kontrolü
                mitigated, mit_level = self._check_mitigation(
                    df, idx, low[idx], high[idx], "BULLISH"
                )
                
                # Imbalance ratio hesapla
                imbalance = self._calculate_imbalance(df, idx, "BULLISH")
                
                # Institutional check
                institutional = (
                    volume[idx + 1] > avg_volume * 2 and
                    next_body > avg_body * 2
                )
                
                # Güç skoru hesapla
                strength = self._calculate_ob_strength(
                    next_body / avg_body,
                    volume[idx + 1] / avg_volume if avg_volume > 0 else 1,
                    vol_confirm,
                    mitigated,
                    imbalance,
                    institutional
                )
                
                # Age (yaş) hesapla
                age_bars = current_index - idx
                
                if strength > self.config.ob_min_strength and not mitigated:
                    order_blocks.append(OrderBlock(
                        top=high[idx],
                        bottom=low[idx],
                        index=idx,
                        ob_type="BULLISH",
                        strength=strength,
                        mitigated=mitigated,
                        mitigation_level=mit_level,
                        volume_confirmation=vol_confirm,
                        timestamp=df.index[idx] if hasattr(df.index, '__getitem__') else None,
                        imbalance_ratio=imbalance,
                        touch_count=0,
                        age_bars=age_bars,
                        institutional=institutional
                    ))
            
            # Bearish OB: Yükseliş mumundan sonra güçlü düşüş
            elif (close[idx] > open_[idx] and  # Yükseliş mumu (OB mumu)
                  close[idx + 1] < open_[idx + 1] and  # Düşüş mumu (momentum)
                  next_body > avg_body * self.config.ob_body_multiplier and
                  close[idx + 1] < low[idx]):
                
                vol_confirm = volume[idx + 1] > avg_volume
                
                mitigated, mit_level = self._check_mitigation(
                    df, idx, low[idx], high[idx], "BEARISH"
                )
                
                imbalance = self._calculate_imbalance(df, idx, "BEARISH")
                
                institutional = (
                    volume[idx + 1] > avg_volume * 2 and
                    next_body > avg_body * 2
                )
                
                strength = self._calculate_ob_strength(
                    next_body / avg_body,
                    volume[idx + 1] / avg_volume if avg_volume > 0 else 1,
                    vol_confirm,
                    mitigated,
                    imbalance,
                    institutional
                )
                
                age_bars = current_index - idx
                
                if strength > self.config.ob_min_strength and not mitigated:
                    order_blocks.append(OrderBlock(
                        top=high[idx],
                        bottom=low[idx],
                        index=idx,
                        ob_type="BEARISH",
                        strength=strength,
                        mitigated=mitigated,
                        mitigation_level=mit_level,
                        volume_confirmation=vol_confirm,
                        timestamp=df.index[idx] if hasattr(df.index, '__getitem__') else None,
                        imbalance_ratio=imbalance,
                        touch_count=0,
                        age_bars=age_bars,
                        institutional=institutional
                    ))
        
        # En güçlü 10 OB'yi döndür (yaşa göre ağırlıklı)
        for ob in order_blocks:
            # Yaş cezası (her 20 bar için %5 azalma)
            age_penalty = max(0, 1 - (ob.age_bars / self.config.ob_max_age_bars))
            ob.strength *= age_penalty
        
        return sorted(order_blocks, key=lambda x: x.strength, reverse=True)[:10]
    
    def _check_mitigation(
        self,
        df: pd.DataFrame,
        ob_idx: int,
        ob_bottom: float,
        ob_top: float,
        ob_type: str
    ) -> Tuple[bool, float]:
        """
        OB mitigation kontrolü
        
        Args:
            df: OHLCV DataFrame
            ob_idx: OB indexi
            ob_bottom: OB alt seviyesi
            ob_top: OB üst seviyesi
            ob_type: OB tipi
            
        Returns:
            (mitigated, mitigation_level)
        """
        if ob_idx >= len(df) - 1:
            return False, 0.0
        
        subsequent = df.iloc[ob_idx + 1:]
        
        if ob_type == "BULLISH":
            # Fiyat OB'nin içine girdi mi?
            penetrations = subsequent[subsequent['Low'] <= ob_top]
            if len(penetrations) > 0:
                min_low = penetrations['Low'].min()
                mitigation = (ob_top - min_low) / (ob_top - ob_bottom) if ob_top != ob_bottom else 0
                return mitigation > self.config.ob_mitigation_threshold, mitigation
        else:
            penetrations = subsequent[subsequent['High'] >= ob_bottom]
            if len(penetrations) > 0:
                max_high = penetrations['High'].max()
                mitigation = (max_high - ob_bottom) / (ob_top - ob_bottom) if ob_top != ob_bottom else 0
                return mitigation > self.config.ob_mitigation_threshold, mitigation
        
        return False, 0.0
    
    def _calculate_imbalance(
        self,
        df: pd.DataFrame,
        idx: int,
        direction: str
    ) -> float:
        """
        Order Block imbalance oranı hesapla
        
        Args:
            df: OHLCV DataFrame
            idx: OB indexi
            direction: "BULLISH" veya "BEARISH"
            
        Returns:
            Imbalance ratio (0-1)
        """
        if idx < 1 or idx >= len(df) - 1:
            return 0.0
        
        # 3 mum arası gap kontrolü
        if direction == "BULLISH":
            # Mum 3'ün low'u ile Mum 1'in high'ı arasındaki gap
            if idx >= 2:
                gap = df['Low'].iloc[idx + 1] - df['High'].iloc[idx - 1]
                range_size = df['High'].iloc[idx + 1] - df['Low'].iloc[idx - 1]
                if range_size > 0:
                    return max(0, gap / range_size)
        else:
            if idx >= 2:
                gap = df['Low'].iloc[idx - 1] - df['High'].iloc[idx + 1]
                range_size = df['High'].iloc[idx - 1] - df['Low'].iloc[idx + 1]
                if range_size > 0:
                    return max(0, gap / range_size)
        
        return 0.0
    
    def _calculate_ob_strength(
        self,
        body_ratio: float,
        volume_ratio: float,
        vol_confirm: bool,
        mitigated: bool,
        imbalance: float,
        institutional: bool
    ) -> float:
        """
        OB güç skoru hesapla (0-100)
        
        Args:
            body_ratio: Gövde oranı
            volume_ratio: Hacim oranı
            vol_confirm: Hacim onayı var mı
            mitigated: Mitigation olmuş mu
            imbalance: Imbalance oranı
            institutional: Kurumsal OB mi
            
        Returns:
            Güç skoru (0-100)
        """
        score = 0
        
        # Gövde oranı (max 30 puan)
        score += min(body_ratio * 12, 30)
        
        # Hacim oranı (max 25 puan)
        score += min(volume_ratio * 10, 25)
        
        # Hacim onayı (15 puan)
        if vol_confirm:
            score += 15
        
        # Imbalance bonus (max 15 puan)
        score += imbalance * 15
        
        # Institutional bonus (15 puan)
        if institutional:
            score += 15
        
        # Mitigation cezası
        if mitigated:
            score *= 0.3
        
        return min(score, 100)
    
    def _detect_breaker_blocks(
        self,
        df: pd.DataFrame,
        order_blocks: List[OrderBlock]
    ) -> List[OrderBlock]:
        """
        Breaker Block tespiti
        
        Breaker: Kırılan OB artık ters yönde destek/direnç olur
        
        Args:
            df: OHLCV DataFrame
            order_blocks: Order Block listesi
            
        Returns:
            Breaker Block listesi
        """
        breaker_blocks = []
        close = df['Close'].iloc[-1]
        
        for ob in order_blocks:
            if ob.mitigated:
                # Bullish OB kırıldı -> Bearish Breaker
                if ob.ob_type == "BULLISH" and close < ob.bottom:
                    breaker = OrderBlock(
                        top=ob.top,
                        bottom=ob.bottom,
                        index=ob.index,
                        ob_type="BEARISH_BREAKER",
                        strength=ob.strength * 0.7,
                        mitigated=False,
                        timestamp=ob.timestamp,
                        imbalance_ratio=ob.imbalance_ratio,
                        institutional=ob.institutional
                    )
                    breaker_blocks.append(breaker)
                
                # Bearish OB kırıldı -> Bullish Breaker
                elif ob.ob_type == "BEARISH" and close > ob.top:
                    breaker = OrderBlock(
                        top=ob.top,
                        bottom=ob.bottom,
                        index=ob.index,
                        ob_type="BULLISH_BREAKER",
                        strength=ob.strength * 0.7,
                        mitigated=False,
                        timestamp=ob.timestamp,
                        imbalance_ratio=ob.imbalance_ratio,
                        institutional=ob.institutional
                    )
                    breaker_blocks.append(breaker)
        
        return breaker_blocks
    
    def _detect_institutional_obs(
        self,
        order_blocks: List[OrderBlock]
    ) -> List[OrderBlock]:
        """
        Kurumsal Order Block'ları filtrele
        
        Args:
            order_blocks: Tüm Order Block'lar
            
        Returns:
            Sadece institutional OB'ler
        """
        return [ob for ob in order_blocks if ob.institutional]
    
    # ═══════════════════════════════════════════════════════════════════════════
    # FAIR VALUE GAP (FVG) DETECTION
    # ═══════════════════════════════════════════════════════════════════════════
    
    def _detect_fvgs(self, df: pd.DataFrame) -> List[FVG]:
        """
        Fair Value Gap tespiti
        
        FVG: 3 mum arasındaki fiyat boşluğu (imbalance)
        
        Args:
            df: OHLCV DataFrame
            
        Returns:
            FVG listesi
        """
        fvgs = []
        
        high = df['High'].values
        low = df['Low'].values
        atr = df['ATR'].values if 'ATR' in df.columns else np.full(len(df), np.mean(high - low))
        
        lookback = min(self.config.fvg_lookback, len(df) - 2)
        
        for i in range(2, lookback):
            idx = len(df) - i
            
            if idx < 2:
                continue
            
            current_atr = atr[idx] if atr[idx] > 0 else 1.0
            
            # Bullish FVG: Mum 1'in high'ı ile Mum 3'ün low'u arasında boşluk
            if low[idx] > high[idx - 2]:
                gap_size = low[idx] - high[idx - 2]
                
                if gap_size > current_atr * self.config.fvg_min_size_atr:
                    # Fill kontrolü
                    filled, fill_pct = self._check_fvg_fill(
                        df, idx, high[idx - 2], low[idx], "BULLISH"
                    )
                    
                    fvgs.append(FVG(
                        top=low[idx],
                        bottom=high[idx - 2],
                        index=idx,
                        fvg_type="BULLISH",
                        size=gap_size,
                        size_atr=gap_size / current_atr,
                        filled=filled,
                        fill_percentage=fill_pct,
                        timestamp=df.index[idx] if hasattr(df.index, '__getitem__') else None
                    ))
            
            # Bearish FVG
            elif high[idx] < low[idx - 2]:
                gap_size = low[idx - 2] - high[idx]
                
                if gap_size > current_atr * self.config.fvg_min_size_atr:
                    filled, fill_pct = self._check_fvg_fill(
                        df, idx, high[idx], low[idx - 2], "BEARISH"
                    )
                    
                    fvgs.append(FVG(
                        top=low[idx - 2],
                        bottom=high[idx],
                        index=idx,
                        fvg_type="BEARISH",
                        size=gap_size,
                        size_atr=gap_size / current_atr,
                        filled=filled,
                        fill_percentage=fill_pct,
                        timestamp=df.index[idx] if hasattr(df.index, '__getitem__') else None
                    ))
        
        # Doldurulmamış FVG'leri öncelikle döndür
        return sorted(fvgs, key=lambda x: (x.filled, -x.size))[:10]
    
    def _check_fvg_fill(
        self,
        df: pd.DataFrame,
        fvg_idx: int,
        bottom: float,
        top: float,
        fvg_type: str
    ) -> Tuple[bool, float]:
        """
        FVG doldurulma kontrolü
        
        Args:
            df: OHLCV DataFrame
            fvg_idx: FVG indexi
            bottom: FVG alt seviyesi
            top: FVG üst seviyesi
            fvg_type: FVG tipi
            
        Returns:
            (filled, fill_percentage)
        """
        if fvg_idx >= len(df):
            return False, 0.0
        
        subsequent = df.iloc[fvg_idx:]
        
        if fvg_type == "BULLISH":
            # Fiyat FVG'nin içine girdi mi?
            penetrations = subsequent[subsequent['Low'] <= top]
            if len(penetrations) > 0:
                min_low = penetrations['Low'].min()
                fill_pct = (top - min_low) / (top - bottom) if top != bottom else 0
                return fill_pct >= 1.0, min(fill_pct, 1.0)
        else:
            penetrations = subsequent[subsequent['High'] >= bottom]
            if len(penetrations) > 0:
                max_high = penetrations['High'].max()
                fill_pct = (max_high - bottom) / (top - bottom) if top != bottom else 0
                return fill_pct >= 1.0, min(fill_pct, 1.0)
        
        return False, 0.0
    
    # ═══════════════════════════════════════════════════════════════════════════
    # LIQUIDITY ANALYSIS
    # ═══════════════════════════════════════════════════════════════════════════
    
    def _detect_liquidity_levels(
        self,
        df: pd.DataFrame,
        swing_points: List[SwingPoint]
    ) -> List[LiquidityLevel]:
        """
        Likidite seviyesi tespiti
        
        - Equal Highs/Lows (EQH/EQL)
        - Buy-Side Liquidity (BSL)
        - Sell-Side Liquidity (SSL)
        - Swing High/Low seviyeleri
        
        Args:
            df: OHLCV DataFrame
            swing_points: Swing noktaları
            
        Returns:
            LiquidityLevel listesi
        """
        liquidity_levels = []
        tolerance = self.config.equal_level_tolerance
        volume = df['Volume'].values if 'Volume' in df.columns else np.ones(len(df))
        
        # Swing High likiditeleri (BSL - Buy Side Liquidity)
        highs = [sp for sp in swing_points if sp.type in ["HH", "LH", "HIGH"]]
        
        for i, h1 in enumerate(highs):
            count = 1
            for h2 in highs[i + 1:]:
                if abs(h1.price - h2.price) / h1.price < tolerance:
                    count += 1
            
            level_type = "EQH" if count >= 2 else "SWING_HIGH"
            
            # Volume at level
            vol_at_level = volume[h1.index] if h1.index < len(volume) else 0
            
            liquidity_levels.append(LiquidityLevel(
                price=h1.price,
                level_type=level_type,
                strength=count,
                swept=False,
                index=h1.index,
                volume_at_level=vol_at_level,
                touches=count
            ))
        
        # Swing Low likiditeleri (SSL - Sell Side Liquidity)
        lows = [sp for sp in swing_points if sp.type in ["HL", "LL", "LOW"]]
        
        for i, l1 in enumerate(lows):
            count = 1
            for l2 in lows[i + 1:]:
                if abs(l1.price - l2.price) / l1.price < tolerance:
                    count += 1
            
            level_type = "EQL" if count >= 2 else "SWING_LOW"
            
            vol_at_level = volume[l1.index] if l1.index < len(volume) else 0
            
            liquidity_levels.append(LiquidityLevel(
                price=l1.price,
                level_type=level_type,
                strength=count,
                swept=False,
                index=l1.index,
                volume_at_level=vol_at_level,
                touches=count
            ))
        
        return liquidity_levels
    
    def _detect_liquidity_sweep(
        self,
        df: pd.DataFrame,
        liquidity_levels: List[LiquidityLevel]
    ) -> Tuple[str, List[LiquidityLevel]]:
        """
        Likidite süpürme tespiti
        
        Sweep: Fiyat seviyeyi kısa süreliğine aşıp geri döner
        
        Args:
            df: OHLCV DataFrame
            liquidity_levels: Likidite seviyeleri
            
        Returns:
            (sweep_type, updated_liquidity_levels)
        """
        if len(df) < 3:
            return "YOK", liquidity_levels
        
        last_candle = df.iloc[-1]
        prev_candle = df.iloc[-2]
        
        sweep_type = "YOK"
        
        for level in liquidity_levels:
            # Bullish Sweep (Ayı Tuzağı): Low seviyesinin altına inip üstünde kapanış
            if level.level_type in ["EQL", "SWING_LOW"]:
                if (last_candle['Low'] < level.price and
                    last_candle['Close'] > level.price and
                    prev_candle['Low'] >= level.price):
                    level.swept = True
                    level.sweep_timestamp = df.index[-1] if hasattr(df.index, '__getitem__') else None
                    sweep_type = "AYI TUZAĞI (BULLISH SWEEP)"
            
            # Bearish Sweep (Boğa Tuzağı): High seviyesinin üstüne çıkıp altında kapanış
            elif level.level_type in ["EQH", "SWING_HIGH"]:
                if (last_candle['High'] > level.price and
                    last_candle['Close'] < level.price and
                    prev_candle['High'] <= level.price):
                    level.swept = True
                    level.sweep_timestamp = df.index[-1] if hasattr(df.index, '__getitem__') else None
                    sweep_type = "BOĞA TUZAĞI (BEARISH SWEEP)"
        
        return sweep_type, liquidity_levels
    
    def _get_bsl_ssl_levels(
        self,
        liquidity_levels: List[LiquidityLevel]
    ) -> Tuple[List[float], List[float]]:
        """
        BSL ve SSL seviyelerini ayır
        
        Args:
            liquidity_levels: Tüm likidite seviyeleri
            
        Returns:
            (bsl_levels, ssl_levels)
        """
        bsl = [l.price for l in liquidity_levels if l.level_type in ["EQH", "SWING_HIGH"] and not l.swept]
        ssl = [l.price for l in liquidity_levels if l.level_type in ["EQL", "SWING_LOW"] and not l.swept]
        
        return sorted(bsl, reverse=True)[:5], sorted(ssl)[:5]
    
    # ═══════════════════════════════════════════════════════════════════════════
    # INDUCEMENT DETECTION
    # ═══════════════════════════════════════════════════════════════════════════
    
    def _detect_inducement_levels(
        self,
        df: pd.DataFrame,
        swing_points: List[SwingPoint],
        liquidity_levels: List[LiquidityLevel]
    ) -> List[InducementLevel]:
        """
        Inducement (tuzak) seviyesi tespiti
        
        Inducement: Smart Money'nin retail trader'ları tuzağa düşürmek için
        oluşturduğu sahte kırılma seviyeleri
        
        Args:
            df: OHLCV DataFrame
            swing_points: Swing noktaları
            liquidity_levels: Likidite seviyeleri
            
        Returns:
            InducementLevel listesi
        """
        inducements = []
        
        # Son 20 bar içinde sweep edilen seviyeleri ara
        recent_swings = [sp for sp in swing_points if sp.index > len(df) - 20]
        
        for liq in liquidity_levels:
            if liq.swept:
                # Sweep sonrası tersine dönüş var mı?
                if liq.level_type in ["EQH", "SWING_HIGH"]:
                    # High sweep sonrası düşüş = Bear Trap
                    inducements.append(InducementLevel(
                        price=liq.price,
                        inducement_type="BULL_TRAP",
                        strength=liq.strength * 10,
                        index=liq.index,
                        triggered=True,
                        timestamp=liq.sweep_timestamp
                    ))
                else:
                    # Low sweep sonrası yükseliş = Bull Trap
                    inducements.append(InducementLevel(
                        price=liq.price,
                        inducement_type="BEAR_TRAP",
                        strength=liq.strength * 10,
                        index=liq.index,
                        triggered=True,
                        timestamp=liq.sweep_timestamp
                    ))
        
        return inducements
    
    # ═══════════════════════════════════════════════════════════════════════════
    # PREMIUM/DISCOUNT ZONE
    # ═══════════════════════════════════════════════════════════════════════════
    
    def _calculate_premium_discount(
        self,
        df: pd.DataFrame
    ) -> Tuple[str, float, ZoneType]:
        """
        Premium/Discount bölgesi hesaplama
        
        Equilibrium: Range'in %50 noktası
        Premium: Equilibrium üstü (satış bölgesi)
        Discount: Equilibrium altı (alış bölgesi)
        
        Args:
            df: OHLCV DataFrame
            
        Returns:
            (zone_label, equilibrium, zone_type)
        """
        period = min(self.config.equilibrium_period, len(df))
        
        range_high = df['High'].iloc[-period:].max()
        range_low = df['Low'].iloc[-period:].min()
        equilibrium = (range_high + range_low) / 2
        
        current_price = df['Close'].iloc[-1]
        
        # Fibonacci seviyeleri
        fib_618 = range_low + (range_high - range_low) * 0.618
        fib_382 = range_low + (range_high - range_low) * 0.382
        
        if current_price > fib_618:
            zone = "PREMIUM (PAHALI - SATIŞ BÖLGESİ)"
            zone_type = ZoneType.PREMIUM
        elif current_price < fib_382:
            zone = "DISCOUNT (UCUZ - ALIŞ BÖLGESİ)"
            zone_type = ZoneType.DISCOUNT
        elif current_price > equilibrium:
            zone = "PREMIUM (DİKKATLİ OL)"
            zone_type = ZoneType.PREMIUM
        else:
            zone = "DISCOUNT (FIRSAT BÖLGESİ)"
            zone_type = ZoneType.DISCOUNT
        
        return zone, equilibrium, zone_type
    
    # ═══════════════════════════════════════════════════════════════════════════
    # WYCKOFF ANALYSIS
    # ═══════════════════════════════════════════════════════════════════════════
    
    def _analyze_wyckoff(
        self,
        df: pd.DataFrame,
        swing_points: List[SwingPoint],
        structure: MarketStructure
    ) -> WyckoffAnalysis:
        """
        Wyckoff faz analizi
        
        Fazlar:
        - Accumulation: Akıllı para biriktirme
        - Markup: Yükseliş trendi
        - Distribution: Akıllı para dağıtma
        - Markdown: Düşüş trendi
        
        Args:
            df: OHLCV DataFrame
            swing_points: Swing noktaları
            structure: Mevcut market yapısı
            
        Returns:
            WyckoffAnalysis
        """
        lookback = min(self.config.wyckoff_lookback, len(df))
        recent_df = df.iloc[-lookback:]
        
        volume = recent_df['Volume'].values if 'Volume' in recent_df.columns else np.ones(len(recent_df))
        close = recent_df['Close'].values
        high = recent_df['High'].values
        low = recent_df['Low'].values
        
        avg_volume = np.mean(volume)
        
        # Volume pattern analizi
        vol_trend = "INCREASING" if volume[-5:].mean() > volume[-20:-5].mean() else "DECREASING"
        
        # Price range analizi
        price_range = high.max() - low.min()
        recent_range = high[-10:].max() - low[-10:].min()
        range_contracting = recent_range < price_range * 0.5
        
        # Spring detection (SSL sweep sonrası yukarı)
        spring = False
        upthrust = False
        sos = False  # Sign of Strength
        sow = False  # Sign of Weakness
        
        # Son 10 bar'da spring/upthrust ara
        for i in range(-10, 0):
            if i + lookback >= 0 and i + lookback < len(recent_df):
                bar_low = low[i]
                bar_close = close[i]
                bar_high = high[i]
                
                # Spring: Yeni dip yapıp güçlü kapanış
                prev_low = low[:i].min() if i > -lookback else low[0]
                if bar_low < prev_low and bar_close > prev_low:
                    spring = True
                
                # Upthrust: Yeni zirve yapıp zayıf kapanış
                prev_high = high[:i].max() if i > -lookback else high[0]
                if bar_high > prev_high and bar_close < prev_high:
                    upthrust = True
        
        # SOS: Yüksek hacimli bullish bar
        if volume[-1] > avg_volume * 1.5 and close[-1] > close[-2]:
            sos = True
        
        # SOW: Yüksek hacimli bearish bar
        if volume[-1] > avg_volume * 1.5 and close[-1] < close[-2]:
            sow = True
        
        # Faz belirleme
        if structure == MarketStructure.RANGING:
            if vol_trend == "DECREASING" and range_contracting:
                if spring:
                    phase = WyckoffPhase.ACCUMULATION
                    phase_label = "BİRİKTİRME (Accumulation) - Spring Tespit"
                elif upthrust:
                    phase = WyckoffPhase.DISTRIBUTION
                    phase_label = "DAĞITIM (Distribution) - Upthrust Tespit"
                else:
                    phase = WyckoffPhase.ACCUMULATION
                    phase_label = "BİRİKTİRME/DAĞITIM BEKLENİYOR"
            else:
                phase = WyckoffPhase.ACCUMULATION
                phase_label = "YATAY FAZ"
        elif structure == MarketStructure.BULLISH:
            if sos:
                phase = WyckoffPhase.MARKUP
                phase_label = "MARKUP (Yükseliş) - SOS Tespit"
            else:
                phase = WyckoffPhase.MARKUP
                phase_label = "MARKUP (Yükseliş Trendi)"
        elif structure == MarketStructure.BEARISH:
            if sow:
                phase = WyckoffPhase.MARKDOWN
                phase_label = "MARKDOWN (Düşüş) - SOW Tespit"
            else:
                phase = WyckoffPhase.MARKDOWN
                phase_label = "MARKDOWN (Düşüş Trendi)"
        else:
            phase = WyckoffPhase.ACCUMULATION
            phase_label = "BELİRSİZ"
        
        # Confidence hesapla
        confidence = 50.0
        if spring or upthrust:
            confidence += 20
        if sos or sow:
            confidence += 15
        if range_contracting:
            confidence += 10
        
        return WyckoffAnalysis(
            phase=phase,
            phase_label=phase_label,
            confidence=min(confidence, 95),
            volume_pattern=vol_trend,
            price_pattern="CONTRACTING" if range_contracting else "EXPANDING",
            spring_detected=spring,
            upthrust_detected=upthrust,
            sos_detected=sos,
            sow_detected=sow
        )
    
    # ═══════════════════════════════════════════════════════════════════════════
    # SESSION ANALYSIS
    # ═══════════════════════════════════════════════════════════════════════════
    
    def _analyze_session(
        self,
        df: pd.DataFrame
    ) -> Tuple[SessionAnalysis, SessionType, bool]:
        """
        Session bazlı analiz
        
        Args:
            df: OHLCV DataFrame
            
        Returns:
            (session_analysis, current_session, kill_zone_active)
        """
        if not self.config.session_enabled:
            return None, SessionType.BIST, False
        
        # Son bar'ın saatini al
        if hasattr(df.index, 'hour'):
            current_hour = df.index[-1].hour
        else:
            current_hour = 12  # Default
        
        # Session belirle (UTC bazlı)
        if self.config.asian_session[0] <= current_hour < self.config.asian_session[1]:
            current_session = SessionType.ASIAN
        elif self.config.london_session[0] <= current_hour < self.config.london_session[1]:
            current_session = SessionType.LONDON
        elif self.config.ny_session[0] <= current_hour < self.config.ny_session[1]:
            current_session = SessionType.NEW_YORK
        else:
            current_session = SessionType.BIST
        
        # Kill zone kontrolü (session başlangıç/bitiş zamanları)
        kill_zone = False
        if current_hour in [self.config.london_session[0], self.config.ny_session[0]]:
            kill_zone = True
        
        # Session verisi (son 20 bar)
        session_bars = min(20, len(df))
        session_df = df.iloc[-session_bars:]
        
        session_high = session_df['High'].max()
        session_low = session_df['Low'].min()
        session_open = session_df['Open'].iloc[0]
        session_close = session_df['Close'].iloc[-1]
        session_volume = session_df['Volume'].sum() if 'Volume' in session_df.columns else 0
        
        # Range ATR
        atr = df['ATR'].iloc[-1] if 'ATR' in df.columns else (session_high - session_low)
        range_atr = (session_high - session_low) / atr if atr > 0 else 1
        
        # Direction
        if session_close > session_open * 1.002:
            direction = "BULLISH"
        elif session_close < session_open * 0.998:
            direction = "BEARISH"
        else:
            direction = "NEUTRAL"
        
        # Sweep kontrolü
        current_price = df['Close'].iloc[-1]
        swept_high = current_price < session_high and df['High'].iloc[-1] >= session_high
        swept_low = current_price > session_low and df['Low'].iloc[-1] <= session_low
        
        session_analysis = SessionAnalysis(
            session=current_session,
            high=session_high,
            low=session_low,
            open=session_open,
            close=session_close,
            volume=session_volume,
            range_atr=range_atr,
            direction=direction,
            swept_high=swept_high,
            swept_low=swept_low
        )
        
        return session_analysis, current_session, kill_zone
        
        # FVG (max 15)
        if active_fvg and not active_fvg.filled:
            score += 15
        
        # Liquidity Sweep (max 15)
        if "TUZAĞI" in sweep:
            score += 15
            inst_score += 10
        
        # Premium/Discount (max 10)
        if "DISCOUNT" in pd_zone and "FIRSAT" in pd_zone:
            score += 10
        elif "DISCOUNT" in pd_zone:
            score += 5
        
        # Wyckoff bonus (max 10)
        if wyckoff:
            if wyckoff.spring_detected or wyckoff.sos_detected:
                score += 10
                inst_score += 15
            elif wyckoff.upthrust_detected or wyckoff.sow_detected:
                score += 8
                inst_score += 15
        
        # Session bonus (max 5)
        if session:
            if session.swept_high or session.swept_low:
                score += 5
        
        # MTF bonus (max 10)
        if mtf:
            score += mtf.confluence_score * 0.1
        
        return min(score, 100), min(inst_score, 100)
    
    # ═══════════════════════════════════════════════════════════════════════════
    # CANDLE PATTERN DETECTION
    # ═══════════════════════════════════════════════════════════════════════════
    
    def _detect_candle_pattern(self, df: pd.DataFrame) -> str:
        """
        Son mum formasyonunu tespit et
        
        Args:
            df: OHLCV DataFrame
            
        Returns:
            Formasyon adı
        """
        if len(df) < 3:
            return "BELİRSİZ"
        
        last = df.iloc[-1]
        prev = df.iloc[-2]
        
        body = abs(last['Close'] - last['Open'])
        upper_wick = last['High'] - max(last['Open'], last['Close'])
        lower_wick = min(last['Open'], last['Close']) - last['Low']
        total_range = last['High'] - last['Low']
        
        if total_range == 0:
            return "DOJI"
        
        body_ratio = body / total_range
        
        # Doji
        if body_ratio < 0.1:
            if upper_wick > lower_wick * 2:
                return "GRAVESTONE DOJI"
            elif lower_wick > upper_wick * 2:
                return "DRAGONFLY DOJI"
            return "DOJI"
        
        # Hammer / Inverted Hammer
        if lower_wick > body * 2 and upper_wick < body * 0.5:
            if last['Close'] > last['Open']:
                return "HAMMER (Bullish)"
            return "HANGING MAN"
        
        if upper_wick > body * 2 and lower_wick < body * 0.5:
            if last['Close'] > last['Open']:
                return "INVERTED HAMMER"
            return "SHOOTING STAR (Bearish)"
        
        # Engulfing
        if last['Close'] > last['Open'] and prev['Close'] < prev['Open']:
            if last['Open'] <= prev['Close'] and last['Close'] >= prev['Open']:
                return "BULLISH ENGULFING"
        
        if last['Close'] < last['Open'] and prev['Close'] > prev['Open']:
            if last['Open'] >= prev['Close'] and last['Close'] <= prev['Open']:
                return "BEARISH ENGULFING"
        
        # Marubozu
        if body_ratio > 0.85:
            if last['Close'] > last['Open']:
                return "BULLISH MARUBOZU"
            return "BEARISH MARUBOZU"
        
        # Default
        if last['Close'] > last['Open']:
            return "BULLISH"
        return "BEARISH"
    
    def _determine_bias(
        self,
        structure: MarketStructure,
        active_ob: Optional[OrderBlock],
        sweep: str,
        pd_zone: str,
        wyckoff: Optional[WyckoffAnalysis]
    ) -> str:
        """
        Trade bias belirle
        
        Args:
            structure: Market yapısı
            active_ob: Aktif Order Block
            sweep: Sweep durumu
            pd_zone: Premium/Discount zone
            wyckoff: Wyckoff analizi
            
        Returns:
            "LONG", "SHORT", veya "NEUTRAL"
        """
        bullish_points = 0
        bearish_points = 0
        
        # Structure
        if structure == MarketStructure.BULLISH:
            bullish_points += 3
        elif structure == MarketStructure.BEARISH:
            bearish_points += 3
        
        # OB
        if active_ob:
            if active_ob.ob_type in ["BULLISH", "BULLISH_BREAKER"]:
                bullish_points += 2
            else:
                bearish_points += 2
        
        # Sweep
        if "AYI TUZAĞI" in sweep:
            bullish_points += 2
        elif "BOĞA TUZAĞI" in sweep:
            bearish_points += 2
        
        # Premium/Discount
        if "DISCOUNT" in pd_zone:
            bullish_points += 1
        elif "PREMIUM" in pd_zone:
            bearish_points += 1
        
        # Wyckoff
        if wyckoff:
            if wyckoff.spring_detected or wyckoff.sos_detected:
                bullish_points += 2
            if wyckoff.upthrust_detected or wyckoff.sow_detected:
                bearish_points += 2
        
        if bullish_points > bearish_points + 2:
            return "LONG"
        elif bearish_points > bullish_points + 2:
            return "SHORT"
        return "NEUTRAL"
    
    # ═══════════════════════════════════════════════════════════════════════════
    # ENTRY ZONE CALCULATION
    # ═══════════════════════════════════════════════════════════════════════════
    
    def _calculate_entry_zone(
        self,
        bias: str,
        active_ob: Optional[OrderBlock],
        active_fvg: Optional[FVG],
        current_price: float
    ) -> Tuple[Optional[Tuple[float, float]], Optional[float]]:
        """
        Giriş bölgesi ve invalidation seviyesi hesapla
        
        Args:
            bias: Trade yönü
            active_ob: Aktif Order Block
            active_fvg: Aktif FVG
            current_price: Güncel fiyat
            
        Returns:
            (entry_zone, invalidation)
        """
        if bias == "NEUTRAL":
            return None, None
        
        entry_zone = None
        invalidation = None
        
        if bias == "LONG":
            if active_ob and active_ob.ob_type in ["BULLISH", "BULLISH_BREAKER"]:
                entry_zone = (active_ob.bottom, active_ob.top)
                invalidation = active_ob.bottom * 0.995  # %0.5 buffer
            elif active_fvg and active_fvg.fvg_type == "BULLISH":
                entry_zone = (active_fvg.bottom, active_fvg.top)
                invalidation = active_fvg.bottom * 0.995
        
        elif bias == "SHORT":
            if active_ob and active_ob.ob_type in ["BEARISH", "BEARISH_BREAKER"]:
                entry_zone = (active_ob.bottom, active_ob.top)
                invalidation = active_ob.top * 1.005  # %0.5 buffer
            elif active_fvg and active_fvg.fvg_type == "BEARISH":
                entry_zone = (active_fvg.bottom, active_fvg.top)
                invalidation = active_fvg.top * 1.005
        
        return entry_zone, invalidation
    
    # ═══════════════════════════════════════════════════════════════════════════
    # ANA ANALİZ FONKSİYONU
    # ═══════════════════════════════════════════════════════════════════════════
    
    def analyze(
        self,
        df: pd.DataFrame,
        mtf_dataframes: Dict[str, pd.DataFrame] = None
    ) -> SMCAnalysis:
        """
        Kapsamlı SMC analizi
        
        Args:
            df: Ana OHLCV DataFrame
            mtf_dataframes: Multi-timeframe DataFrames (opsiyonel)
            
        Returns:
            SMCAnalysis dataclass
        """
        if df is None or len(df) < 30:
            return self._empty_analysis()
        
        try:
            # 1. Swing Point Detection
            swing_points = self._detect_swing_points(df)
            swing_points = self._classify_swing_sequence(swing_points)
            
            # 2. Market Structure Analysis
            structure, bos, choch, label, trend_strength = self._analyze_structure(df, swing_points)
            
            # 3. Order Block Detection
            order_blocks = self._detect_order_blocks(df, swing_points)
            breaker_blocks = self._detect_breaker_blocks(df, order_blocks)
            institutional_obs = self._detect_institutional_obs(order_blocks)
            
            # Aktif OB (fiyata en yakın, mitigated olmayan)
            active_ob = None
            current_price = df['Close'].iloc[-1]
            
            for ob in order_blocks:
                if not ob.mitigated:
                    if ob.ob_type in ["BULLISH", "BULLISH_BREAKER"] and current_price >= ob.bottom:
                        active_ob = ob
                        break
                    elif ob.ob_type in ["BEARISH", "BEARISH_BREAKER"] and current_price <= ob.top:
                        active_ob = ob
                        break
            
            # 4. FVG Detection
            fvgs = self._detect_fvgs(df)
            active_fvg = next((f for f in fvgs if not f.filled), None)
            unfilled_fvgs = len([f for f in fvgs if not f.filled])
            
            # 5. Liquidity Analysis
            liquidity_levels = self._detect_liquidity_levels(df, swing_points)
            sweep_type, liquidity_levels = self._detect_liquidity_sweep(df, liquidity_levels)
            bsl_levels, ssl_levels = self._get_bsl_ssl_levels(liquidity_levels)
            
            liq_high = max([l.price for l in liquidity_levels if l.level_type in ["EQH", "SWING_HIGH"]], default=0)
            liq_low = min([l.price for l in liquidity_levels if l.level_type in ["EQL", "SWING_LOW"]], default=float('inf'))
            if liq_low == float('inf'):
                liq_low = 0
            
            # 6. Inducement Detection
            inducement_levels = self._detect_inducement_levels(df, swing_points, liquidity_levels)
            active_inducement = next((i for i in inducement_levels if i.triggered), None)
            
            # 7. Premium/Discount
            pd_zone, equilibrium, zone_type = self._calculate_premium_discount(df)
            
            # 8. Wyckoff Analysis
            wyckoff = self._analyze_wyckoff(df, swing_points, structure)
            
            # 9. Session Analysis
            session_analysis, current_session, kill_zone = self._analyze_session(df)
            
            # 10. MTF Confluence
            mtf_confluence = None
            if mtf_dataframes:
                mtf_confluence = self.analyze_mtf_confluence(mtf_dataframes)
            
            # 11. Son swing noktaları
            last_swing_high = next((sp for sp in reversed(swing_points) if sp.type in ["HH", "LH"]), None)
            last_swing_low = next((sp for sp in reversed(swing_points) if sp.type in ["HL", "LL"]), None)
            
            # 12. Price Action
            if df['Close'].iloc[-1] > df['Open'].iloc[-1]:
                price_action = "ALICI BASKIN"
            else:
                price_action = "SATICI BASKIN"
            
            candle_pattern = self._detect_candle_pattern(df)
            
            # 13. Confluence Score
            confluence, inst_score = self._calculate_confluence_score(
                structure, bos, choch, active_ob, active_fvg,
                sweep_type, pd_zone, df, wyckoff, session_analysis, mtf_confluence
            )
            
            # 14. Bias & Entry Zone
            bias = self._determine_bias(structure, active_ob, sweep_type, pd_zone, wyckoff)
            entry_zone, invalidation = self._calculate_entry_zone(bias, active_ob, active_fvg, current_price)
            
            # MSS Status
            mss_status = "ONAYLANDI" if bos or choch else "BEKLEMEDE"
            
            return SMCAnalysis(
                structure=structure,
                structure_label=label,
                bos_detected=bos,
                choch_detected=choch,
                mss_status=mss_status,
                trend_strength=trend_strength,
                order_blocks=order_blocks,
                active_ob=active_ob,
                breaker_blocks=breaker_blocks,
                institutional_obs=institutional_obs,
                fvgs=fvgs,
                active_fvg=active_fvg,
                unfilled_fvgs=unfilled_fvgs,
                liquidity_levels=liquidity_levels,
                liquidity_high=liq_high,
                liquidity_low=liq_low,
                liquidity_sweep=sweep_type,
                bsl_levels=bsl_levels,
                ssl_levels=ssl_levels,
                inducement_levels=inducement_levels,
                active_inducement=active_inducement,
                premium_discount=pd_zone,
                equilibrium=equilibrium,
                zone_type=zone_type,
                swing_points=swing_points,
                last_swing_high=last_swing_high,
                last_swing_low=last_swing_low,
                wyckoff=wyckoff,
                session_analysis=session_analysis,
                current_session=current_session,
                kill_zone_active=kill_zone,
                mtf_confluence=mtf_confluence,
                confluence_score=confluence,
                institutional_score=inst_score,
                price_action=price_action,
                candle_pattern=candle_pattern,
                bias=bias,
                entry_zone=entry_zone,
                invalidation=invalidation
            )
        
        except Exception as e:
            logger.error(f"❌ SMC Analysis Error: {e}")
            import traceback
            traceback.print_exc()
            return self._empty_analysis()
    
    def _empty_analysis(self) -> SMCAnalysis:
        """Boş analiz sonucu"""
        return SMCAnalysis(
            structure=MarketStructure.UNDEFINED,
            structure_label="VERİ YETERSİZ",
            bos_detected=False,
            choch_detected=False,
            mss_status="BEKLEMEDE",
            trend_strength="UNKNOWN",
            order_blocks=[],
            active_ob=None,
            breaker_blocks=[],
            institutional_obs=[],
            fvgs=[],
            active_fvg=None,
            unfilled_fvgs=0,
            liquidity_levels=[],
            liquidity_high=0,
            liquidity_low=0,
            liquidity_sweep="YOK",
            bsl_levels=[],
            ssl_levels=[],
            inducement_levels=[],
            active_inducement=None,
            premium_discount="NÖTR",
            equilibrium=0,
            zone_type=ZoneType.EQUILIBRIUM,
            swing_points=[],
            last_swing_high=None,
            last_swing_low=None,
            wyckoff=None,
            session_analysis=None,
            current_session=SessionType.BIST,
            kill_zone_active=False,
            mtf_confluence=None,
            confluence_score=0,
            institutional_score=0,
            price_action="BELİRSİZ",
            candle_pattern="BELİRSİZ",
            bias="NEUTRAL",
            entry_zone=None,
            invalidation=None
        )
    
    # ═══════════════════════════════════════════════════════════════════════════
    # LEGACY COMPATIBILITY
    # ═══════════════════════════════════════════════════════════════════════════
    
    def analyze_structure(self, df: pd.DataFrame) -> Dict:
        """Eski API uyumluluğu için wrapper"""
        analysis = self.analyze(df)
        
        return {
            'structure': analysis.structure_label,
            'mss_status': analysis.mss_status,
            'liquidity_sweep': analysis.liquidity_sweep,
            'premium_discount': analysis.premium_discount,
            'fvg': f"{analysis.active_fvg.bottom:.2f}-{analysis.active_fvg.top:.2f}" if analysis.active_fvg else "TEMİZ",
            'ob': {
                'top': analysis.active_ob.top,
                'bottom': analysis.active_ob.bottom,
                'type': analysis.active_ob.ob_type,
                'strength': analysis.active_ob.strength
            } if analysis.active_ob else None,
            'liquidity_low': analysis.liquidity_low,
            'liquidity_high': analysis.liquidity_high,
            'price_action': analysis.price_action,
            'confluence_score': analysis.confluence_score,
            'bias': analysis.bias,
            'wyckoff_phase': analysis.wyckoff.phase_label if analysis.wyckoff else None
        }


# ═══════════════════════════════════════════════════════════════════════════════
# TEST
# ═══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    print("SMC Engine v4.2 - Test")
    print("=" * 60)
    
    # Test için dummy data oluştur
    import numpy as np
    
    dates = pd.date_range(start='2024-01-01', periods=100, freq='4H')
    np.random.seed(42)
    
    prices = 100 + np.cumsum(np.random.randn(100) * 2)
    
    df = pd.DataFrame({
        'Open': prices + np.random.randn(100) * 0.5,
        'High': prices + np.abs(np.random.randn(100)) * 2,
        'Low': prices - np.abs(np.random.randn(100)) * 2,
        'Close': prices,
        'Volume': np.random.randint(1000000, 5000000, 100),
        'ATR': np.ones(100) * 2
    }, index=dates)
    
    # OHLC tutarlılık
    df['High'] = df[['Open', 'High', 'Close']].max(axis=1)
    df['Low'] = df[['Open', 'Low', 'Close']].min(axis=1)
    
    engine = SMCEngine()
    analysis = engine.analyze(df)
    
    print(f"\n📊 SMC ANALİZ SONUCU")
    print("=" * 60)
    print(f"🌊 Yapı: {analysis.structure_label}")
    print(f"📈 Trend Gücü: {analysis.trend_strength}")
    print(f"📈 BOS: {analysis.bos_detected} | CHoCH: {analysis.choch_detected}")
    print(f"🎯 MSS: {analysis.mss_status}")
    print(f"📦 Order Blocks: {len(analysis.order_blocks)} (Aktif: {analysis.active_ob is not None})")
    print(f"📦 Institutional OBs: {len(analysis.institutional_obs)}")
    print(f"⚡ FVGs: {len(analysis.fvgs)} (Unfilled: {analysis.unfilled_fvgs})")
    print(f"💧 Sweep: {analysis.liquidity_sweep}")
    print(f"💰 Bölge: {analysis.premium_discount}")
    print(f"📊 Wyckoff: {analysis.wyckoff.phase_label if analysis.wyckoff else 'N/A'}")
    print(f"🕐 Session: {analysis.current_session.value}")
    print(f"🎯 Kill Zone: {analysis.kill_zone_active}")
    print(f"🔥 Confluence: {analysis.confluence_score:.1f}/100")
    print(f"🏛️ Institutional: {analysis.institutional_score:.1f}/100")
    print(f"📍 Bias: {analysis.bias}")
    print(f"🕯️ Candle: {analysis.candle_pattern}")
