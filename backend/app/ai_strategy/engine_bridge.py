"""
AlphaTerminal Pro - Engine Bridge
=================================

Mevcut analiz engine'lerini (SMC, OrderFlow, Alpha, Risk) 
AI Strategy sistemi ile entegre eden köprü modül.

Author: AlphaTerminal Team
Version: 1.0.0
"""

from __future__ import annotations

import logging
from typing import Optional, Dict, Any, List
from datetime import datetime
from dataclasses import dataclass, field

import numpy as np
import polars as pl

from app.ai_strategy.constants import (
    TrendRegime,
    VolatilityRegime,
    SignalType,
    FeatureCategory,
)

logger = logging.getLogger(__name__)


@dataclass
class EngineAnalysisResult:
    """Birleştirilmiş engine analiz sonucu."""
    symbol: str
    timeframe: str
    
    # SMC Analysis
    smc_score: float
    market_structure: str
    order_blocks: List[Dict]
    fair_value_gaps: List[Dict]
    liquidity_levels: List[Dict]
    smc_bias: str  # bullish, bearish, neutral
    
    # OrderFlow Analysis
    orderflow_score: float
    delta: float
    cvd_trend: str
    absorption_detected: bool
    flow_direction: str
    
    # Alpha Analysis
    alpha_score: float
    alpha_vs_index: float
    alpha_vs_sector: float
    sharpe_ratio: float
    momentum_score: float
    
    # Risk Analysis
    risk_score: float
    suggested_stop_loss: float
    suggested_take_profits: List[float]
    position_size_shares: int
    risk_reward_ratio: float
    
    # Combined
    total_score: float
    signal_strength: str
    suggested_action: str
    confidence: float
    
    # Key Factors
    bullish_factors: List[str]
    bearish_factors: List[str]
    
    analyzed_at: datetime = field(default_factory=datetime.utcnow)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "symbol": self.symbol,
            "timeframe": self.timeframe,
            "smc": {
                "score": self.smc_score,
                "structure": self.market_structure,
                "bias": self.smc_bias,
                "order_blocks": len(self.order_blocks),
                "fvgs": len(self.fair_value_gaps),
            },
            "orderflow": {
                "score": self.orderflow_score,
                "delta": self.delta,
                "cvd_trend": self.cvd_trend,
                "direction": self.flow_direction,
            },
            "alpha": {
                "score": self.alpha_score,
                "vs_index": self.alpha_vs_index,
                "sharpe": self.sharpe_ratio,
            },
            "risk": {
                "score": self.risk_score,
                "stop_loss": self.suggested_stop_loss,
                "position_size": self.position_size_shares,
                "risk_reward": self.risk_reward_ratio,
            },
            "combined": {
                "total_score": self.total_score,
                "signal_strength": self.signal_strength,
                "action": self.suggested_action,
                "confidence": self.confidence,
            },
            "factors": {
                "bullish": self.bullish_factors,
                "bearish": self.bearish_factors,
            },
            "analyzed_at": self.analyzed_at.isoformat(),
        }


class EngineBridge:
    """
    Mevcut engine'leri AI sistemiyle köprüleyen sınıf.
    
    Entegre edilen engine'ler:
    - SMC Engine (Smart Money Concepts)
    - OrderFlow Engine
    - Alpha Engine
    - Risk Engine
    - Data Engine / BIST Data Fetcher
    
    Example:
        ```python
        bridge = EngineBridge()
        
        # Tam analiz
        result = await bridge.full_analysis(
            symbol="THYAO",
            timeframe="4h",
            ohlcv_data=df,
            capital=100000
        )
        
        # Feature'ları AI sistemi için dönüştür
        features = bridge.extract_features_for_ai(result)
        ```
    """
    
    def __init__(self):
        """Initialize Engine Bridge."""
        # Engine instances (lazy loading)
        self._smc_engine = None
        self._orderflow_engine = None
        self._alpha_engine = None
        self._risk_engine = None
        self._data_engine = None
    
    @property
    def smc_engine(self):
        """SMC Engine (lazy load) - from core."""
        if self._smc_engine is None:
            try:
                from app.core.smc_engine import SMCEngine
                self._smc_engine = SMCEngine()
                logger.info("SMC Engine loaded from core")
            except ImportError:
                logger.warning("SMC Engine not available in core")
        return self._smc_engine
    
    @property
    def orderflow_engine(self):
        """OrderFlow Engine (lazy load) - from core."""
        if self._orderflow_engine is None:
            try:
                from app.core.orderflow_engine import OrderFlowEngine
                self._orderflow_engine = OrderFlowEngine()
                logger.info("OrderFlow Engine loaded from core")
            except ImportError:
                logger.warning("OrderFlow Engine not available in core")
        return self._orderflow_engine
    
    @property
    def alpha_engine(self):
        """Alpha Engine (lazy load) - from core."""
        if self._alpha_engine is None:
            try:
                from app.core.alpha_engine import AlphaEngine
                self._alpha_engine = AlphaEngine()
                logger.info("Alpha Engine loaded from core")
            except ImportError:
                logger.warning("Alpha Engine not available in core")
        return self._alpha_engine
    
    @property
    def risk_engine(self):
        """Risk Engine (lazy load) - from core."""
        if self._risk_engine is None:
            try:
                from app.core.risk_engine import RiskEngine
                self._risk_engine = RiskEngine()
                logger.info("Risk Engine loaded from core")
            except ImportError:
                logger.warning("Risk Engine not available in core")
        return self._risk_engine
    
    @property
    def data_engine(self):
        """Data Engine (lazy load) - from core."""
        if self._data_engine is None:
            try:
                from app.core.data_engine import DataEngine
                self._data_engine = DataEngine()
                logger.info("Data Engine loaded from core")
            except ImportError:
                logger.warning("Data Engine not available in core")
        return self._data_engine
    
    async def full_analysis(
        self,
        symbol: str,
        timeframe: str,
        ohlcv_data: pl.DataFrame,
        capital: float = 100000,
        max_risk_pct: float = 0.02,
        index_data: Optional[pl.DataFrame] = None,
        sector_data: Optional[pl.DataFrame] = None,
    ) -> EngineAnalysisResult:
        """
        Tüm engine'leri kullanarak tam analiz.
        
        Args:
            symbol: Hisse sembolü
            timeframe: Zaman dilimi
            ohlcv_data: OHLCV verisi
            capital: Trading sermayesi
            max_risk_pct: Max risk yüzdesi
            index_data: Endeks verisi (Alpha için)
            sector_data: Sektör verisi (Alpha için)
            
        Returns:
            EngineAnalysisResult: Birleştirilmiş analiz sonucu
        """
        # Convert Polars to numpy for engines
        close = ohlcv_data["close"].to_numpy()
        high = ohlcv_data["high"].to_numpy()
        low = ohlcv_data["low"].to_numpy()
        volume = ohlcv_data["volume"].to_numpy() if "volume" in ohlcv_data.columns else np.ones(len(close))
        
        current_price = close[-1]
        
        # SMC Analysis
        smc_result = await self._run_smc_analysis(high, low, close, volume)
        
        # OrderFlow Analysis
        orderflow_result = await self._run_orderflow_analysis(high, low, close, volume)
        
        # Alpha Analysis
        alpha_result = await self._run_alpha_analysis(close, index_data, sector_data)
        
        # Risk Analysis
        risk_result = await self._run_risk_analysis(
            current_price, smc_result.get("atr", current_price * 0.02),
            capital, max_risk_pct
        )
        
        # Combine scores
        total_score = self._calculate_combined_score(
            smc_result["score"],
            orderflow_result["score"],
            alpha_result["score"]
        )
        
        # Determine action
        signal_strength, suggested_action = self._determine_action(
            total_score, smc_result["bias"], orderflow_result["direction"]
        )
        
        # Extract factors
        bullish_factors, bearish_factors = self._extract_factors(
            smc_result, orderflow_result, alpha_result
        )
        
        # Calculate confidence
        confidence = self._calculate_confidence(
            total_score, len(bullish_factors), len(bearish_factors)
        )
        
        return EngineAnalysisResult(
            symbol=symbol,
            timeframe=timeframe,
            # SMC
            smc_score=smc_result["score"],
            market_structure=smc_result["structure"],
            order_blocks=smc_result["order_blocks"],
            fair_value_gaps=smc_result["fvgs"],
            liquidity_levels=smc_result["liquidity"],
            smc_bias=smc_result["bias"],
            # OrderFlow
            orderflow_score=orderflow_result["score"],
            delta=orderflow_result["delta"],
            cvd_trend=orderflow_result["cvd_trend"],
            absorption_detected=orderflow_result["absorption"],
            flow_direction=orderflow_result["direction"],
            # Alpha
            alpha_score=alpha_result["score"],
            alpha_vs_index=alpha_result["vs_index"],
            alpha_vs_sector=alpha_result["vs_sector"],
            sharpe_ratio=alpha_result["sharpe"],
            momentum_score=alpha_result["momentum"],
            # Risk
            risk_score=risk_result["score"],
            suggested_stop_loss=risk_result["stop_loss"],
            suggested_take_profits=risk_result["take_profits"],
            position_size_shares=risk_result["shares"],
            risk_reward_ratio=risk_result["risk_reward"],
            # Combined
            total_score=total_score,
            signal_strength=signal_strength,
            suggested_action=suggested_action,
            confidence=confidence,
            # Factors
            bullish_factors=bullish_factors,
            bearish_factors=bearish_factors,
        )
    
    def extract_features_for_ai(
        self,
        analysis_result: EngineAnalysisResult
    ) -> Dict[str, float]:
        """
        Engine analiz sonucunu AI sistemi için feature'lara dönüştür.
        
        Args:
            analysis_result: Engine analiz sonucu
            
        Returns:
            Dict[str, float]: Feature dictionary
        """
        features = {}
        
        # SMC Features
        features["smc_score"] = analysis_result.smc_score
        features["smc_bias_numeric"] = {
            "bullish": 1.0, "bearish": -1.0, "neutral": 0.0
        }.get(analysis_result.smc_bias, 0.0)
        features["ob_count"] = len(analysis_result.order_blocks)
        features["fvg_count"] = len(analysis_result.fair_value_gaps)
        features["liquidity_count"] = len(analysis_result.liquidity_levels)
        
        # OrderFlow Features
        features["orderflow_score"] = analysis_result.orderflow_score
        features["delta"] = analysis_result.delta
        features["cvd_trend_numeric"] = {
            "rising": 1.0, "falling": -1.0, "flat": 0.0
        }.get(analysis_result.cvd_trend, 0.0)
        features["absorption_detected"] = 1.0 if analysis_result.absorption_detected else 0.0
        features["flow_direction_numeric"] = {
            "buying": 1.0, "selling": -1.0, "neutral": 0.0
        }.get(analysis_result.flow_direction, 0.0)
        
        # Alpha Features
        features["alpha_score"] = analysis_result.alpha_score
        features["alpha_vs_index"] = analysis_result.alpha_vs_index
        features["alpha_vs_sector"] = analysis_result.alpha_vs_sector
        features["sharpe_ratio"] = analysis_result.sharpe_ratio
        features["momentum_score"] = analysis_result.momentum_score
        
        # Risk Features
        features["risk_score"] = analysis_result.risk_score
        features["risk_reward_ratio"] = analysis_result.risk_reward_ratio
        
        # Combined
        features["total_score"] = analysis_result.total_score
        features["confidence"] = analysis_result.confidence
        features["bullish_factor_count"] = len(analysis_result.bullish_factors)
        features["bearish_factor_count"] = len(analysis_result.bearish_factors)
        features["factor_balance"] = (
            len(analysis_result.bullish_factors) - len(analysis_result.bearish_factors)
        )
        
        return features
    
    async def _run_smc_analysis(
        self,
        high: np.ndarray,
        low: np.ndarray,
        close: np.ndarray,
        volume: np.ndarray,
    ) -> Dict[str, Any]:
        """SMC analizi çalıştır."""
        if self.smc_engine:
            try:
                result = self.smc_engine.analyze(high, low, close, volume)
                return result
            except Exception as e:
                logger.warning(f"SMC Engine error: {e}")
        
        # Placeholder result
        return self._placeholder_smc(high, low, close)
    
    async def _run_orderflow_analysis(
        self,
        high: np.ndarray,
        low: np.ndarray,
        close: np.ndarray,
        volume: np.ndarray,
    ) -> Dict[str, Any]:
        """OrderFlow analizi çalıştır."""
        if self.orderflow_engine:
            try:
                result = self.orderflow_engine.analyze(high, low, close, volume)
                return result
            except Exception as e:
                logger.warning(f"OrderFlow Engine error: {e}")
        
        # Placeholder result
        return self._placeholder_orderflow(high, low, close, volume)
    
    async def _run_alpha_analysis(
        self,
        close: np.ndarray,
        index_data: Optional[pl.DataFrame],
        sector_data: Optional[pl.DataFrame],
    ) -> Dict[str, Any]:
        """Alpha analizi çalıştır."""
        if self.alpha_engine:
            try:
                result = self.alpha_engine.analyze(close, index_data, sector_data)
                return result
            except Exception as e:
                logger.warning(f"Alpha Engine error: {e}")
        
        # Placeholder result
        return self._placeholder_alpha(close)
    
    async def _run_risk_analysis(
        self,
        current_price: float,
        atr: float,
        capital: float,
        max_risk_pct: float,
    ) -> Dict[str, Any]:
        """Risk analizi çalıştır."""
        if self.risk_engine:
            try:
                result = self.risk_engine.calculate(
                    current_price, atr, capital, max_risk_pct
                )
                return result
            except Exception as e:
                logger.warning(f"Risk Engine error: {e}")
        
        # Placeholder result
        return self._placeholder_risk(current_price, atr, capital, max_risk_pct)
    
    def _placeholder_smc(
        self,
        high: np.ndarray,
        low: np.ndarray,
        close: np.ndarray,
    ) -> Dict[str, Any]:
        """SMC placeholder sonucu."""
        # Simple trend detection
        sma_20 = np.mean(close[-20:]) if len(close) >= 20 else close[-1]
        sma_50 = np.mean(close[-50:]) if len(close) >= 50 else sma_20
        
        if close[-1] > sma_20 > sma_50:
            bias = "bullish"
            score = 70
        elif close[-1] < sma_20 < sma_50:
            bias = "bearish"
            score = 30
        else:
            bias = "neutral"
            score = 50
        
        return {
            "score": score,
            "structure": "bullish" if close[-1] > sma_50 else "bearish",
            "bias": bias,
            "order_blocks": [],
            "fvgs": [],
            "liquidity": [],
            "atr": np.mean(high[-14:] - low[-14:]) if len(high) >= 14 else (high[-1] - low[-1]),
        }
    
    def _placeholder_orderflow(
        self,
        high: np.ndarray,
        low: np.ndarray,
        close: np.ndarray,
        volume: np.ndarray,
    ) -> Dict[str, Any]:
        """OrderFlow placeholder sonucu."""
        # Simple delta approximation
        price_change = close[-1] - close[-2] if len(close) >= 2 else 0
        delta = volume[-1] * np.sign(price_change)
        
        # CVD trend
        if len(close) >= 10:
            recent_changes = np.diff(close[-10:])
            cvd_trend = "rising" if np.sum(recent_changes) > 0 else "falling"
        else:
            cvd_trend = "flat"
        
        # Direction
        if delta > volume[-1] * 0.3:
            direction = "buying"
            score = 70
        elif delta < -volume[-1] * 0.3:
            direction = "selling"
            score = 30
        else:
            direction = "neutral"
            score = 50
        
        return {
            "score": score,
            "delta": delta,
            "cvd_trend": cvd_trend,
            "absorption": False,
            "direction": direction,
        }
    
    def _placeholder_alpha(self, close: np.ndarray) -> Dict[str, Any]:
        """Alpha placeholder sonucu."""
        # Simple momentum
        if len(close) >= 20:
            returns = np.diff(np.log(close[-20:]))
            momentum = (close[-1] / close[-20] - 1) * 100
            sharpe = np.mean(returns) / (np.std(returns) + 1e-8) * np.sqrt(252)
        else:
            momentum = 0
            sharpe = 0
        
        score = 50 + min(25, max(-25, momentum))
        
        return {
            "score": score,
            "vs_index": momentum * 0.8,  # Placeholder
            "vs_sector": momentum * 0.9,  # Placeholder
            "sharpe": sharpe,
            "momentum": momentum,
        }
    
    def _placeholder_risk(
        self,
        current_price: float,
        atr: float,
        capital: float,
        max_risk_pct: float,
    ) -> Dict[str, Any]:
        """Risk placeholder sonucu."""
        stop_distance = atr * 1.5
        stop_loss = current_price - stop_distance
        
        risk_amount = capital * max_risk_pct
        shares = int(risk_amount / stop_distance)
        
        tp1 = current_price + stop_distance * 1.5
        tp2 = current_price + stop_distance * 2.5
        tp3 = current_price + stop_distance * 4.0
        
        return {
            "score": 50,
            "stop_loss": stop_loss,
            "take_profits": [tp1, tp2, tp3],
            "shares": shares,
            "risk_reward": 2.5,
        }
    
    def _calculate_combined_score(
        self,
        smc_score: float,
        orderflow_score: float,
        alpha_score: float,
    ) -> float:
        """Birleşik skor hesapla."""
        weights = {"smc": 0.35, "orderflow": 0.35, "alpha": 0.30}
        
        return (
            smc_score * weights["smc"] +
            orderflow_score * weights["orderflow"] +
            alpha_score * weights["alpha"]
        )
    
    def _determine_action(
        self,
        total_score: float,
        smc_bias: str,
        flow_direction: str,
    ) -> tuple[str, str]:
        """Aksiyon belirle."""
        # Signal strength
        if total_score >= 75:
            strength = "strong"
        elif total_score >= 60:
            strength = "moderate"
        elif total_score <= 25:
            strength = "strong"  # Strong bearish
        elif total_score <= 40:
            strength = "moderate"
        else:
            strength = "weak"
        
        # Action
        if total_score >= 70 and smc_bias == "bullish" and flow_direction == "buying":
            action = "STRONG_BUY"
        elif total_score >= 60:
            action = "BUY"
        elif total_score <= 30 and smc_bias == "bearish" and flow_direction == "selling":
            action = "STRONG_SELL"
        elif total_score <= 40:
            action = "SELL"
        else:
            action = "HOLD"
        
        return strength, action
    
    def _extract_factors(
        self,
        smc: Dict,
        orderflow: Dict,
        alpha: Dict,
    ) -> tuple[List[str], List[str]]:
        """Bullish/Bearish faktörleri çıkar."""
        bullish = []
        bearish = []
        
        # SMC factors
        if smc["bias"] == "bullish":
            bullish.append("SMC structure bullish")
        elif smc["bias"] == "bearish":
            bearish.append("SMC structure bearish")
        
        if smc["score"] >= 70:
            bullish.append("Strong SMC score")
        elif smc["score"] <= 30:
            bearish.append("Weak SMC score")
        
        # OrderFlow factors
        if orderflow["direction"] == "buying":
            bullish.append("Buying pressure detected")
        elif orderflow["direction"] == "selling":
            bearish.append("Selling pressure detected")
        
        if orderflow["cvd_trend"] == "rising":
            bullish.append("CVD trend rising")
        elif orderflow["cvd_trend"] == "falling":
            bearish.append("CVD trend falling")
        
        # Alpha factors
        if alpha["momentum"] > 5:
            bullish.append(f"Positive momentum: {alpha['momentum']:.1f}%")
        elif alpha["momentum"] < -5:
            bearish.append(f"Negative momentum: {alpha['momentum']:.1f}%")
        
        if alpha["sharpe"] > 1:
            bullish.append(f"Good risk-adjusted returns: Sharpe {alpha['sharpe']:.2f}")
        elif alpha["sharpe"] < 0:
            bearish.append(f"Poor risk-adjusted returns: Sharpe {alpha['sharpe']:.2f}")
        
        return bullish, bearish
    
    def _calculate_confidence(
        self,
        total_score: float,
        bullish_count: int,
        bearish_count: int,
    ) -> float:
        """Güven skoru hesapla."""
        # Score distance from neutral
        score_confidence = abs(total_score - 50) / 50
        
        # Factor alignment
        total_factors = bullish_count + bearish_count
        if total_factors > 0:
            alignment = abs(bullish_count - bearish_count) / total_factors
        else:
            alignment = 0
        
        confidence = score_confidence * 0.6 + alignment * 0.4
        return min(1.0, max(0.0, confidence))


# Singleton instance
engine_bridge = EngineBridge()
