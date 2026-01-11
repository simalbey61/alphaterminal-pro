"""
AlphaTerminal Pro - Signal Generator v4.2
==========================================

Çok motorlu sinyal üretim sistemi.
SMC, OrderFlow ve Alpha engine'lerini birleştirerek güçlü sinyaller üretir.

Author: AlphaTerminal Team
Version: 4.2.0
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum

from app.core.config import (
    logger, SIGNAL_CONFIG, SignalDirection, SignalStrength
)
from app.core.smc_engine import SMCEngine, SMCAnalysis
from app.core.orderflow_engine import OrderFlowEngine, OrderFlowAnalysis, FlowDirection
from app.core.alpha_engine import AlphaEngine, AlphaAnalysis, AlphaCategory
from app.core.risk_engine import RiskEngine, TradeSetup


# ═══════════════════════════════════════════════════════════════════════════════
# ENUMS
# ═══════════════════════════════════════════════════════════════════════════════

class SignalType(Enum):
    """Sinyal türü"""
    SMC_ENTRY = "SMC_ENTRY"
    ORDERFLOW_ENTRY = "ORDERFLOW_ENTRY"
    ALPHA_ENTRY = "ALPHA_ENTRY"
    CONFLUENCE_ENTRY = "CONFLUENCE_ENTRY"
    BREAKOUT = "BREAKOUT"
    PULLBACK = "PULLBACK"
    REVERSAL = "REVERSAL"
    CONTINUATION = "CONTINUATION"


class SignalStatus(Enum):
    """Sinyal durumu"""
    PENDING = "PENDING"
    ACTIVE = "ACTIVE"
    TRIGGERED = "TRIGGERED"
    EXPIRED = "EXPIRED"
    CANCELLED = "CANCELLED"


# ═══════════════════════════════════════════════════════════════════════════════
# DATA CLASSES
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass
class EngineScores:
    """Motor skorları"""
    smc_score: float
    orderflow_score: float
    alpha_score: float
    weighted_score: float
    agreement_level: float  # Motorların uyum seviyesi


@dataclass
class SignalConfidence:
    """Sinyal güven metrikleri"""
    overall: float  # 0-100
    smc_contribution: float
    orderflow_contribution: float
    alpha_contribution: float
    confluence_bonus: float
    timeframe_alignment: float
    historical_accuracy: float


@dataclass
class TradingSignal:
    """Trading sinyali"""
    symbol: str
    signal_type: SignalType
    direction: SignalDirection
    strength: SignalStrength
    
    # Fiyat bilgileri
    entry_price: float
    stop_loss: float
    take_profit_1: float
    take_profit_2: float
    take_profit_3: float
    
    # Skorlar
    scores: EngineScores
    confidence: SignalConfidence
    
    # Risk bilgileri
    risk_reward: float
    position_size_pct: float
    risk_amount: float
    
    # Meta
    timeframe: str
    signal_id: str
    status: SignalStatus
    created_at: datetime
    expires_at: Optional[datetime]
    
    # Açıklamalar
    smc_context: str
    orderflow_context: str
    alpha_context: str
    summary: str
    notes: List[str] = field(default_factory=list)


@dataclass
class SignalSummary:
    """Günlük sinyal özeti"""
    total_signals: int
    long_signals: int
    short_signals: int
    strong_signals: int
    average_confidence: float
    top_signals: List[TradingSignal]
    sector_distribution: Dict[str, int]
    timeframe_distribution: Dict[str, int]


# ═══════════════════════════════════════════════════════════════════════════════
# SIGNAL GENERATOR CLASS
# ═══════════════════════════════════════════════════════════════════════════════

class SignalGenerator:
    """
    Multi-Engine Sinyal Üretici v4.2
    
    Özellikler:
    - SMC, OrderFlow, Alpha motor entegrasyonu
    - Ağırlıklı skor hesaplama
    - Confluence detection
    - Multi-timeframe confirmation
    - Otomatik trade setup üretimi
    """
    
    def __init__(
        self,
        smc_engine: SMCEngine = None,
        orderflow_engine: OrderFlowEngine = None,
        alpha_engine: AlphaEngine = None,
        risk_engine: RiskEngine = None,
        config = None
    ):
        self.smc = smc_engine or SMCEngine()
        self.orderflow = orderflow_engine or OrderFlowEngine()
        self.alpha = alpha_engine or AlphaEngine()
        self.risk = risk_engine or RiskEngine()
        self.config = config or SIGNAL_CONFIG
        
        self.signals_history: List[TradingSignal] = []
        self._signal_counter = 0
    
    def _generate_signal_id(self, symbol: str) -> str:
        """Unique sinyal ID üret"""
        self._signal_counter += 1
        timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
        return f"SIG_{symbol}_{timestamp}_{self._signal_counter:04d}"
    
    # ═══════════════════════════════════════════════════════════════════════════
    # ENGINE ANALYSIS
    # ═══════════════════════════════════════════════════════════════════════════
    
    def _analyze_with_smc(
        self,
        df: pd.DataFrame,
        mtf_data: Dict[str, pd.DataFrame] = None
    ) -> Tuple[SMCAnalysis, float, str]:
        """
        SMC motoru ile analiz
        
        Returns:
            (analysis, score, context)
        """
        analysis = self.smc.analyze(df, mtf_data)
        
        # Score hesapla (0-100)
        score = analysis.confluence_score
        
        # Context oluştur
        context_parts = []
        
        # Structure
        context_parts.append(f"Yapı: {analysis.structure_label}")
        
        # BOS/CHoCH
        if analysis.bos_detected:
            context_parts.append("BOS teyitli")
        if analysis.choch_detected:
            context_parts.append("CHoCH tespit")
        
        # Order Block
        if analysis.active_ob:
            context_parts.append(f"OB: {analysis.active_ob.ob_type}")
        
        # FVG
        if analysis.active_fvg:
            context_parts.append(f"FVG: {analysis.active_fvg.fvg_type}")
        
        # Sweep
        if analysis.liquidity_sweep != "YOK":
            context_parts.append(f"Sweep: {analysis.liquidity_sweep}")
        
        # Wyckoff
        if analysis.wyckoff:
            context_parts.append(f"Wyckoff: {analysis.wyckoff.phase_label}")
        
        context = " | ".join(context_parts)
        
        return analysis, score, context
    
    def _analyze_with_orderflow(
        self,
        df: pd.DataFrame
    ) -> Tuple[OrderFlowAnalysis, float, str]:
        """
        OrderFlow motoru ile analiz
        
        Returns:
            (analysis, score, context)
        """
        analysis = self.orderflow.analyze(df)
        
        # Score hesapla
        score = analysis.flow_score
        
        # Context oluştur
        context_parts = []
        
        # Volume
        context_parts.append(f"Hacim: {analysis.volume_state.value}")
        
        # Delta
        context_parts.append(f"Delta: {analysis.delta.delta_trend.value}")
        
        # VWAP
        context_parts.append(f"VWAP: {analysis.vwap.price_position}")
        
        # Institutional
        if analysis.institutional_buying:
            context_parts.append("Kurumsal ALIŞ")
        if analysis.institutional_selling:
            context_parts.append("Kurumsal SATIŞ")
        
        # Whale
        if analysis.whale_activity and analysis.whale_activity.detected:
            context_parts.append(f"Whale: {analysis.whale_activity.direction.value}")
        
        # Exhaustion
        if analysis.exhaustion_signal.detected:
            context_parts.append(f"Tükenme: {analysis.exhaustion_signal.exhaustion_type}")
        
        context = " | ".join(context_parts)
        
        return analysis, score, context
    
    def _analyze_with_alpha(
        self,
        df: pd.DataFrame,
        df_index: pd.DataFrame,
        symbol: str = None
    ) -> Tuple[Optional[AlphaAnalysis], float, str]:
        """
        Alpha motoru ile analiz
        
        Returns:
            (analysis, score, context)
        """
        if df_index is None or len(df_index) < 30:
            return None, 50.0, "Endeks verisi yok"
        
        analysis = self.alpha.analyze(df, df_index, symbol)
        
        # Score hesapla
        score = analysis.alpha_score
        
        # Context oluştur
        context_parts = []
        
        context_parts.append(f"Alpha: {analysis.alpha_category.value}")
        context_parts.append(f"RS: {analysis.relative_strength.rs_slope}")
        context_parts.append(f"Momentum: {analysis.momentum_state.value}")
        context_parts.append(f"Sharpe: {analysis.ratios.sharpe_ratio}")
        
        context = " | ".join(context_parts)
        
        return analysis, score, context
    
    # ═══════════════════════════════════════════════════════════════════════════
    # SCORE COMBINATION
    # ═══════════════════════════════════════════════════════════════════════════
    
    def _calculate_engine_scores(
        self,
        smc_score: float,
        orderflow_score: float,
        alpha_score: float
    ) -> EngineScores:
        """Ağırlıklı skor hesapla"""
        
        weights = self.config.engine_weights
        
        weighted = (
            smc_score * weights['smc'] +
            orderflow_score * weights['orderflow'] +
            alpha_score * weights['alpha']
        )
        
        # Agreement level (motorların uyumu)
        scores = [smc_score, orderflow_score, alpha_score]
        std = np.std(scores)
        agreement = max(0, 100 - std * 2)
        
        return EngineScores(
            smc_score=round(smc_score, 2),
            orderflow_score=round(orderflow_score, 2),
            alpha_score=round(alpha_score, 2),
            weighted_score=round(weighted, 2),
            agreement_level=round(agreement, 2)
        )
    
    def _calculate_confidence(
        self,
        scores: EngineScores,
        smc: SMCAnalysis,
        orderflow: OrderFlowAnalysis,
        alpha: Optional[AlphaAnalysis]
    ) -> SignalConfidence:
        """Sinyal güvenini hesapla"""
        
        # Base confidence
        base = scores.weighted_score
        
        # Confluence bonus
        confluence_bonus = 0
        
        # SMC + OrderFlow uyumu
        smc_bullish = smc.bias == "LONG"
        of_bullish = orderflow.flow_bias == FlowDirection.BULLISH
        
        if smc_bullish == of_bullish:
            confluence_bonus += 10
        
        # Alpha uyumu
        if alpha:
            alpha_bullish = alpha.relative_strength.outperforming
            if smc_bullish == alpha_bullish:
                confluence_bonus += 5
        
        # Yüksek agreement bonus
        if scores.agreement_level > 80:
            confluence_bonus += 5
        
        # Contributions
        total_score = scores.smc_score + scores.orderflow_score + scores.alpha_score
        smc_contrib = (scores.smc_score / total_score * 100) if total_score > 0 else 33
        of_contrib = (scores.orderflow_score / total_score * 100) if total_score > 0 else 33
        alpha_contrib = (scores.alpha_score / total_score * 100) if total_score > 0 else 33
        
        overall = min(base + confluence_bonus, 100)
        
        return SignalConfidence(
            overall=round(overall, 2),
            smc_contribution=round(smc_contrib, 2),
            orderflow_contribution=round(of_contrib, 2),
            alpha_contribution=round(alpha_contrib, 2),
            confluence_bonus=round(confluence_bonus, 2),
            timeframe_alignment=0,  # MTF için
            historical_accuracy=0   # Backtest için
        )
    
    # ═══════════════════════════════════════════════════════════════════════════
    # SIGNAL GENERATION
    # ═══════════════════════════════════════════════════════════════════════════
    
    def _determine_direction(
        self,
        smc: SMCAnalysis,
        orderflow: OrderFlowAnalysis,
        alpha: Optional[AlphaAnalysis]
    ) -> Tuple[SignalDirection, bool]:
        """
        Sinyal yönünü belirle
        
        Returns:
            (direction, is_valid)
        """
        votes = {'LONG': 0, 'SHORT': 0, 'NEUTRAL': 0}
        
        # SMC vote
        if smc.bias == "LONG":
            votes['LONG'] += 2
        elif smc.bias == "SHORT":
            votes['SHORT'] += 2
        else:
            votes['NEUTRAL'] += 1
        
        # OrderFlow vote
        if orderflow.flow_bias == FlowDirection.BULLISH:
            votes['LONG'] += 1.5
        elif orderflow.flow_bias == FlowDirection.BEARISH:
            votes['SHORT'] += 1.5
        else:
            votes['NEUTRAL'] += 0.5
        
        # Alpha vote
        if alpha:
            if alpha.relative_strength.outperforming and alpha.momentum_state.value in ["ACCELERATING", "STRONG", "POSITIVE"]:
                votes['LONG'] += 1
            elif not alpha.relative_strength.outperforming and alpha.momentum_state.value in ["DECELERATING", "WEAK", "NEGATIVE"]:
                votes['SHORT'] += 1
        
        # Karar
        if votes['LONG'] > votes['SHORT'] and votes['LONG'] > votes['NEUTRAL']:
            if votes['LONG'] >= self.config.min_votes_for_signal:
                return SignalDirection.LONG, True
            return SignalDirection.LONG, False
        
        elif votes['SHORT'] > votes['LONG'] and votes['SHORT'] > votes['NEUTRAL']:
            if votes['SHORT'] >= self.config.min_votes_for_signal:
                return SignalDirection.SHORT, True
            return SignalDirection.SHORT, False
        
        return SignalDirection.NEUTRAL, False
    
    def _determine_strength(
        self,
        confidence: SignalConfidence,
        scores: EngineScores
    ) -> SignalStrength:
        """Sinyal gücünü belirle"""
        
        overall = confidence.overall
        
        if overall >= self.config.strong_signal_threshold:
            return SignalStrength.VERY_STRONG
        elif overall >= 70:
            return SignalStrength.STRONG
        elif overall >= 55:
            return SignalStrength.MODERATE
        elif overall >= 40:
            return SignalStrength.WEAK
        else:
            return SignalStrength.VERY_WEAK
    
    def _determine_signal_type(
        self,
        smc: SMCAnalysis,
        orderflow: OrderFlowAnalysis,
        confidence: SignalConfidence
    ) -> SignalType:
        """Sinyal türünü belirle"""
        
        # Confluence entry (en güçlü)
        if confidence.confluence_bonus >= 10:
            return SignalType.CONFLUENCE_ENTRY
        
        # SMC-based
        if smc.liquidity_sweep != "YOK":
            return SignalType.REVERSAL
        if smc.bos_detected:
            return SignalType.BREAKOUT
        if smc.active_ob:
            return SignalType.PULLBACK
        
        # OrderFlow-based
        if orderflow.exhaustion_signal.detected:
            return SignalType.REVERSAL
        
        # Default
        return SignalType.CONTINUATION
    
    def generate_signal(
        self,
        symbol: str,
        df: pd.DataFrame,
        df_index: pd.DataFrame = None,
        timeframe: str = "4h",
        mtf_data: Dict[str, pd.DataFrame] = None
    ) -> Optional[TradingSignal]:
        """
        Sinyal üret
        
        Args:
            symbol: Hisse kodu
            df: OHLCV DataFrame
            df_index: Endeks DataFrame
            timeframe: Zaman dilimi
            mtf_data: Multi-timeframe verisi
            
        Returns:
            TradingSignal veya None
        """
        if df is None or len(df) < 30:
            return None
        
        try:
            # Engine analizleri
            smc, smc_score, smc_context = self._analyze_with_smc(df, mtf_data)
            orderflow, of_score, of_context = self._analyze_with_orderflow(df)
            alpha, alpha_score, alpha_context = self._analyze_with_alpha(df, df_index, symbol)
            
            # Skorları birleştir
            scores = self._calculate_engine_scores(smc_score, of_score, alpha_score)
            
            # Minimum skor kontrolü
            if scores.weighted_score < self.config.min_signal_score:
                return None
            
            # Yön belirleme
            direction, is_valid = self._determine_direction(smc, orderflow, alpha)
            
            if not is_valid or direction == SignalDirection.NEUTRAL:
                return None
            
            # Güven hesapla
            confidence = self._calculate_confidence(scores, smc, orderflow, alpha)
            
            # Minimum güven kontrolü
            if confidence.overall < self.config.min_confidence:
                return None
            
            # Güç belirleme
            strength = self._determine_strength(confidence, scores)
            
            # Sinyal türü
            signal_type = self._determine_signal_type(smc, orderflow, confidence)
            
            # Trade setup
            current_price = df['Close'].iloc[-1]
            
            setup = self.risk.generate_trade_setup(
                symbol=symbol,
                df=df,
                direction="LONG" if direction == SignalDirection.LONG else "SHORT",
                entry_price=current_price,
                smc_data={
                    'ob': {
                        'bottom': smc.active_ob.bottom,
                        'top': smc.active_ob.top,
                        'type': smc.active_ob.ob_type
                    } if smc.active_ob else None,
                    'liquidity_high': smc.liquidity_high,
                    'liquidity_low': smc.liquidity_low,
                    'liquidity_sweep': smc.liquidity_sweep
                },
                confidence=confidence.overall
            )
            
            # Summary oluştur
            summary_parts = [
                f"{symbol} {direction.value}",
                f"Güç: {strength.value}",
                f"Güven: {confidence.overall:.0f}%"
            ]
            
            if smc.liquidity_sweep != "YOK":
                summary_parts.append(f"Sweep: {smc.liquidity_sweep}")
            if orderflow.institutional_buying:
                summary_parts.append("Kurumsal Alış")
            if orderflow.institutional_selling:
                summary_parts.append("Kurumsal Satış")
            
            summary = " | ".join(summary_parts)
            
            # Notes
            notes = []
            if smc.choch_detected:
                notes.append("⚠️ CHoCH tespit edildi - trend değişimi olabilir")
            if orderflow.exhaustion_signal.detected:
                notes.append("⚠️ Tükenme sinyali - dikkatli olun")
            if alpha and alpha.risk.max_drawdown > 15:
                notes.append(f"⚠️ Yüksek drawdown: {alpha.risk.max_drawdown:.1f}%")
            
            signal = TradingSignal(
                symbol=symbol,
                signal_type=signal_type,
                direction=direction,
                strength=strength,
                entry_price=setup.entry_price,
                stop_loss=setup.stop_loss.price,
                take_profit_1=setup.take_profit_levels[0].price if setup.take_profit_levels else 0,
                take_profit_2=setup.take_profit_levels[1].price if len(setup.take_profit_levels) > 1 else 0,
                take_profit_3=setup.take_profit_levels[2].price if len(setup.take_profit_levels) > 2 else 0,
                scores=scores,
                confidence=confidence,
                risk_reward=setup.risk_reward,
                position_size_pct=setup.position_size.risk_percent,
                risk_amount=setup.risk_amount,
                timeframe=timeframe,
                signal_id=self._generate_signal_id(symbol),
                status=SignalStatus.ACTIVE,
                created_at=datetime.now(),
                expires_at=None,
                smc_context=smc_context,
                orderflow_context=of_context,
                alpha_context=alpha_context,
                summary=summary,
                notes=notes
            )
            
            self.signals_history.append(signal)
            
            return signal
        
        except Exception as e:
            logger.error(f"❌ Signal Generation Error for {symbol}: {e}")
            import traceback
            traceback.print_exc()
            return None
    
    def generate_batch_signals(
        self,
        data_dict: Dict[str, pd.DataFrame],
        df_index: pd.DataFrame = None,
        timeframe: str = "4h",
        top_n: int = 10
    ) -> List[TradingSignal]:
        """
        Toplu sinyal üret
        
        Args:
            data_dict: {symbol: DataFrame}
            df_index: Endeks DataFrame
            timeframe: Zaman dilimi
            top_n: En iyi N sinyal
            
        Returns:
            Sıralanmış sinyal listesi
        """
        signals = []
        
        for symbol, df in data_dict.items():
            signal = self.generate_signal(symbol, df, df_index, timeframe)
            if signal:
                signals.append(signal)
        
        # Güvene göre sırala
        signals.sort(key=lambda s: s.confidence.overall, reverse=True)
        
        return signals[:top_n]
    
    def get_daily_summary(self) -> SignalSummary:
        """Günlük sinyal özeti"""
        
        today_signals = [
            s for s in self.signals_history
            if s.created_at.date() == datetime.now().date()
        ]
        
        long_signals = [s for s in today_signals if s.direction == SignalDirection.LONG]
        short_signals = [s for s in today_signals if s.direction == SignalDirection.SHORT]
        strong_signals = [s for s in today_signals if s.strength in [SignalStrength.STRONG, SignalStrength.VERY_STRONG]]
        
        avg_confidence = np.mean([s.confidence.overall for s in today_signals]) if today_signals else 0
        
        # Top signals
        top = sorted(today_signals, key=lambda s: s.confidence.overall, reverse=True)[:5]
        
        # Sector distribution
        sector_dist: Dict[str, int] = {}
        tf_dist: Dict[str, int] = {}
        
        for s in today_signals:
            tf_dist[s.timeframe] = tf_dist.get(s.timeframe, 0) + 1
        
        return SignalSummary(
            total_signals=len(today_signals),
            long_signals=len(long_signals),
            short_signals=len(short_signals),
            strong_signals=len(strong_signals),
            average_confidence=round(avg_confidence, 2),
            top_signals=top,
            sector_distribution=sector_dist,
            timeframe_distribution=tf_dist
        )


# ═══════════════════════════════════════════════════════════════════════════════
# TEST
# ═══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    print("Signal Generator v4.2 - Test")
    print("=" * 60)
    
    # Test data
    dates = pd.date_range(start='2024-01-01', periods=100, freq='4H')
    np.random.seed(42)
    
    prices = 100 + np.cumsum(np.random.randn(100) * 2)
    
    df = pd.DataFrame({
        'Open': prices * 0.99,
        'High': prices * 1.02,
        'Low': prices * 0.98,
        'Close': prices,
        'Volume': np.random.randint(1000000, 5000000, 100),
        'ATR': np.ones(100) * 2
    }, index=dates)
    
    df['High'] = df[['Open', 'High', 'Close']].max(axis=1)
    df['Low'] = df[['Open', 'Low', 'Close']].min(axis=1)
    
    df_index = pd.DataFrame({
        'Close': prices * 0.95
    }, index=dates)
    
    generator = SignalGenerator()
    signal = generator.generate_signal("THYAO", df, df_index, "4h")
    
    if signal:
        print(f"\n🚀 SİNYAL ÜRETİLDİ")
        print("=" * 60)
        print(f"📈 {signal.symbol} - {signal.direction.value}")
        print(f"💪 Güç: {signal.strength.value}")
        print(f"🔥 Güven: {signal.confidence.overall}%")
        print(f"\n💰 Entry: {signal.entry_price:.2f}")
        print(f"🛑 Stop: {signal.stop_loss:.2f}")
        print(f"🎯 TP1: {signal.take_profit_1:.2f}")
        print(f"📊 R:R: 1:{signal.risk_reward:.1f}")
        print(f"\n📝 Özet: {signal.summary}")
        print(f"\n🔍 SMC: {signal.smc_context}")
        print(f"📊 OF: {signal.orderflow_context}")
        print(f"📈 Alpha: {signal.alpha_context}")
    else:
        print("\n⚠️ Sinyal üretilemedi - kriterler karşılanmadı")
