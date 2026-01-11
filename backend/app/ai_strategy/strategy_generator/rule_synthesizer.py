"""
AlphaTerminal Pro - Rule Synthesizer
====================================

Decision tree kurallarından trading stratejisi oluşturma.

Author: AlphaTerminal Team
Version: 1.0.0
"""

from __future__ import annotations

import logging
from typing import Optional, List, Dict, Any, Tuple
from datetime import datetime
from dataclasses import dataclass, field
from uuid import uuid4
from decimal import Decimal

import numpy as np

from app.ai_strategy.constants import (
    StrategyType,
    DiscoveryMethod,
    StrategyLifecycle,
    SignalType,
)
from app.ai_strategy.pattern_discovery.tree_miner import DiscoveredPattern, TreeRule

logger = logging.getLogger(__name__)


@dataclass
class EntryCondition:
    """Giriş koşulu."""
    feature: str
    operator: str
    value: float
    weight: float = 1.0
    required: bool = True  # False = bonus condition
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "feature": self.feature,
            "operator": self.operator,
            "value": self.value,
            "weight": self.weight,
            "required": self.required,
        }
    
    def evaluate(self, features: Dict[str, float]) -> bool:
        """Koşulu değerlendir."""
        if self.feature not in features:
            return not self.required
        
        feat_val = features[self.feature]
        
        if self.operator == "<":
            return feat_val < self.value
        elif self.operator == ">":
            return feat_val > self.value
        elif self.operator == "<=":
            return feat_val <= self.value
        elif self.operator == ">=":
            return feat_val >= self.value
        elif self.operator == "==":
            return abs(feat_val - self.value) < 0.001
        elif self.operator == "between":
            # value = (low, high)
            if isinstance(self.value, (list, tuple)):
                return self.value[0] <= feat_val <= self.value[1]
        
        return False


@dataclass
class ExitCondition:
    """Çıkış koşulu."""
    exit_type: str  # stop_loss, take_profit, trailing_stop, time_based, signal
    value: float
    atr_multiplier: Optional[float] = None
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "exit_type": self.exit_type,
            "value": self.value,
            "atr_multiplier": self.atr_multiplier,
        }


@dataclass
class SynthesizedStrategy:
    """Sentezlenmiş strateji."""
    id: str
    name: str
    strategy_type: StrategyType
    discovery_method: DiscoveryMethod
    
    # Koşullar
    entry_conditions: List[EntryCondition]
    exit_conditions: List[ExitCondition]
    
    # Signal tipi
    signal_type: SignalType
    
    # Performans beklentisi
    expected_win_rate: float
    expected_profit_factor: float
    expected_avg_return: float
    
    # Meta
    source_pattern: Optional[DiscoveredPattern] = None
    confidence: float = 0.0
    sample_size: int = 0
    
    lifecycle: StrategyLifecycle = StrategyLifecycle.DISCOVERED
    created_at: datetime = field(default_factory=datetime.utcnow)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "name": self.name,
            "strategy_type": self.strategy_type.value,
            "discovery_method": self.discovery_method.value,
            "signal_type": self.signal_type.value,
            "entry_conditions": [c.to_dict() for c in self.entry_conditions],
            "exit_conditions": [c.to_dict() for c in self.exit_conditions],
            "expected_win_rate": self.expected_win_rate,
            "expected_profit_factor": self.expected_profit_factor,
            "expected_avg_return": self.expected_avg_return,
            "confidence": self.confidence,
            "sample_size": self.sample_size,
            "lifecycle": self.lifecycle.value,
            "created_at": self.created_at.isoformat(),
        }
    
    def evaluate_entry(self, features: Dict[str, float]) -> Tuple[bool, float]:
        """
        Giriş koşullarını değerlendir.
        
        Returns:
            Tuple[bool, float]: (Sinyal var mı?, Güç skoru)
        """
        required_met = 0
        required_total = 0
        bonus_score = 0.0
        
        for cond in self.entry_conditions:
            result = cond.evaluate(features)
            
            if cond.required:
                required_total += 1
                if result:
                    required_met += 1
            else:
                if result:
                    bonus_score += cond.weight
        
        # Tüm required koşullar karşılanmalı
        if required_total > 0 and required_met < required_total:
            return False, 0.0
        
        # Güç skoru = base (tüm required met) + bonus
        base_score = 1.0 if required_total == 0 else required_met / required_total
        strength = base_score * 0.7 + min(bonus_score, 0.3)
        
        return True, strength


class RuleSynthesizer:
    """
    Keşfedilen örüntülerden trading stratejisi sentezler.
    
    Example:
        ```python
        synthesizer = RuleSynthesizer()
        
        # Pattern'dan strateji oluştur
        strategy = synthesizer.synthesize_from_pattern(pattern)
        
        # Manuel strateji oluştur
        strategy = synthesizer.create_strategy(
            name="RSI Oversold Bounce",
            conditions=[("rsi_14", "<", 30), ("macd_hist", ">", 0)],
            signal_type=SignalType.LONG
        )
        ```
    """
    
    def __init__(
        self,
        default_stop_atr: float = 1.5,
        default_tp_atr: float = 2.5,
        min_confidence: float = 0.6,
    ):
        """
        Initialize Rule Synthesizer.
        
        Args:
            default_stop_atr: Varsayılan stop loss ATR çarpanı
            default_tp_atr: Varsayılan take profit ATR çarpanı
            min_confidence: Minimum güven eşiği
        """
        self.default_stop_atr = default_stop_atr
        self.default_tp_atr = default_tp_atr
        self.min_confidence = min_confidence
    
    def synthesize_from_pattern(
        self,
        pattern: DiscoveredPattern,
        signal_type: Optional[SignalType] = None,
    ) -> Optional[SynthesizedStrategy]:
        """
        Pattern'dan strateji sentezle.
        
        Args:
            pattern: Keşfedilen örüntü
            signal_type: Sinyal tipi (None = pattern'dan çıkar)
            
        Returns:
            Optional[SynthesizedStrategy]: Sentezlenen strateji
        """
        if pattern.confidence < self.min_confidence:
            logger.warning(f"Pattern confidence too low: {pattern.confidence}")
            return None
        
        # Entry conditions
        entry_conditions = []
        for rule in pattern.rules:
            entry_conditions.append(EntryCondition(
                feature=rule.feature,
                operator=rule.operator,
                value=rule.threshold,
                weight=1.0,
                required=True,
            ))
        
        # Signal type (pattern'dan çıkar veya parametre)
        if signal_type is None:
            signal_type = self._infer_signal_type(pattern)
        
        # Exit conditions
        exit_conditions = self._create_default_exits(signal_type)
        
        # Strateji adı oluştur
        name = self._generate_strategy_name(entry_conditions, signal_type)
        
        # Strategy type belirleme
        strategy_type = self._infer_strategy_type(entry_conditions)
        
        return SynthesizedStrategy(
            id=str(uuid4()),
            name=name,
            strategy_type=strategy_type,
            discovery_method=DiscoveryMethod.DECISION_TREE,
            entry_conditions=entry_conditions,
            exit_conditions=exit_conditions,
            signal_type=signal_type,
            expected_win_rate=pattern.win_rate,
            expected_profit_factor=self._estimate_profit_factor(pattern),
            expected_avg_return=pattern.avg_return,
            source_pattern=pattern,
            confidence=pattern.confidence,
            sample_size=pattern.sample_size,
            lifecycle=StrategyLifecycle.DISCOVERED,
        )
    
    def create_strategy(
        self,
        name: str,
        conditions: List[Tuple[str, str, float]],
        signal_type: SignalType,
        strategy_type: StrategyType = StrategyType.HYBRID,
        stop_atr: Optional[float] = None,
        tp_atr: Optional[float] = None,
    ) -> SynthesizedStrategy:
        """
        Manuel strateji oluştur.
        
        Args:
            name: Strateji adı
            conditions: Koşullar [(feature, operator, value), ...]
            signal_type: Sinyal tipi
            strategy_type: Strateji tipi
            stop_atr: Stop loss ATR çarpanı
            tp_atr: Take profit ATR çarpanı
            
        Returns:
            SynthesizedStrategy: Oluşturulan strateji
        """
        entry_conditions = [
            EntryCondition(feature=f, operator=o, value=v, required=True)
            for f, o, v in conditions
        ]
        
        exit_conditions = [
            ExitCondition(
                exit_type="stop_loss",
                value=0.0,
                atr_multiplier=stop_atr or self.default_stop_atr
            ),
            ExitCondition(
                exit_type="take_profit",
                value=0.0,
                atr_multiplier=tp_atr or self.default_tp_atr
            ),
        ]
        
        return SynthesizedStrategy(
            id=str(uuid4()),
            name=name,
            strategy_type=strategy_type,
            discovery_method=DiscoveryMethod.MANUAL,
            entry_conditions=entry_conditions,
            exit_conditions=exit_conditions,
            signal_type=signal_type,
            expected_win_rate=0.5,
            expected_profit_factor=1.5,
            expected_avg_return=0.0,
            confidence=0.5,
            sample_size=0,
            lifecycle=StrategyLifecycle.DISCOVERED,
        )
    
    def combine_strategies(
        self,
        strategies: List[SynthesizedStrategy],
        combination_type: str = "and",  # and, or, voting
        name: Optional[str] = None,
    ) -> SynthesizedStrategy:
        """
        Birden fazla stratejiyi birleştir.
        
        Args:
            strategies: Birleştirilecek stratejiler
            combination_type: Birleştirme türü
            name: Yeni strateji adı
            
        Returns:
            SynthesizedStrategy: Birleştirilmiş strateji
        """
        if not strategies:
            raise ValueError("No strategies to combine")
        
        # Tüm entry conditions'ları topla
        all_conditions = []
        for strat in strategies:
            for cond in strat.entry_conditions:
                # Duplicate check
                exists = any(
                    c.feature == cond.feature and c.operator == cond.operator
                    for c in all_conditions
                )
                if not exists:
                    # AND birleştirme için required, OR için optional
                    new_cond = EntryCondition(
                        feature=cond.feature,
                        operator=cond.operator,
                        value=cond.value,
                        weight=cond.weight,
                        required=(combination_type == "and"),
                    )
                    all_conditions.append(new_cond)
        
        # Exit conditions - en conservative olanı al
        exit_conditions = strategies[0].exit_conditions.copy()
        
        # Performans beklentisi - ortalama
        avg_wr = np.mean([s.expected_win_rate for s in strategies])
        avg_pf = np.mean([s.expected_profit_factor for s in strategies])
        
        return SynthesizedStrategy(
            id=str(uuid4()),
            name=name or f"Combined_{combination_type}_{len(strategies)}",
            strategy_type=StrategyType.HYBRID,
            discovery_method=DiscoveryMethod.BREEDING,
            entry_conditions=all_conditions,
            exit_conditions=exit_conditions,
            signal_type=strategies[0].signal_type,
            expected_win_rate=avg_wr,
            expected_profit_factor=avg_pf,
            expected_avg_return=np.mean([s.expected_avg_return for s in strategies]),
            confidence=np.mean([s.confidence for s in strategies]),
            sample_size=sum(s.sample_size for s in strategies),
            lifecycle=StrategyLifecycle.DISCOVERED,
        )
    
    def _infer_signal_type(self, pattern: DiscoveredPattern) -> SignalType:
        """Pattern'dan sinyal tipini çıkar."""
        # RSI bazlı
        for rule in pattern.rules:
            if "rsi" in rule.feature.lower():
                if rule.operator in ["<", "<="] and rule.threshold < 40:
                    return SignalType.LONG  # Oversold = Long
                elif rule.operator in [">", ">="] and rule.threshold > 60:
                    return SignalType.SHORT  # Overbought = Short
        
        # MACD bazlı
        for rule in pattern.rules:
            if "macd" in rule.feature.lower():
                if rule.operator in [">", ">="]:
                    return SignalType.LONG
                else:
                    return SignalType.SHORT
        
        # Varsayılan
        return SignalType.LONG
    
    def _infer_strategy_type(self, conditions: List[EntryCondition]) -> StrategyType:
        """Koşullardan strateji tipini çıkar."""
        features = [c.feature.lower() for c in conditions]
        
        # SMC features
        if any("ob" in f or "fvg" in f or "liquidity" in f for f in features):
            return StrategyType.SMC_BASED
        
        # OrderFlow features
        if any("delta" in f or "cvd" in f or "absorption" in f for f in features):
            return StrategyType.ORDERFLOW
        
        # Momentum features
        if any("rsi" in f or "macd" in f or "stoch" in f for f in features):
            return StrategyType.MOMENTUM
        
        # Trend features
        if any("sma" in f or "ema" in f or "adx" in f for f in features):
            return StrategyType.TREND_FOLLOWING
        
        return StrategyType.HYBRID
    
    def _create_default_exits(self, signal_type: SignalType) -> List[ExitCondition]:
        """Varsayılan çıkış koşulları."""
        return [
            ExitCondition(
                exit_type="stop_loss",
                value=0.0,
                atr_multiplier=self.default_stop_atr
            ),
            ExitCondition(
                exit_type="take_profit",
                value=0.0,
                atr_multiplier=self.default_tp_atr
            ),
            ExitCondition(
                exit_type="trailing_stop",
                value=0.0,
                atr_multiplier=2.0
            ),
            ExitCondition(
                exit_type="time_based",
                value=48.0,  # 48 saat max
                atr_multiplier=None
            ),
        ]
    
    def _generate_strategy_name(
        self,
        conditions: List[EntryCondition],
        signal_type: SignalType
    ) -> str:
        """Strateji adı oluştur."""
        parts = []
        
        for cond in conditions[:2]:  # İlk 2 koşul
            feat = cond.feature.replace("_", " ").title()
            if cond.operator in ["<", "<="]:
                parts.append(f"Low {feat}")
            elif cond.operator in [">", ">="]:
                parts.append(f"High {feat}")
        
        direction = "Long" if signal_type == SignalType.LONG else "Short"
        
        if parts:
            return f"{' + '.join(parts)} {direction}"
        
        return f"Strategy {direction} {uuid4().hex[:6]}"
    
    def _estimate_profit_factor(self, pattern: DiscoveredPattern) -> float:
        """Profit factor tahmin et."""
        if pattern.win_rate <= 0 or pattern.win_rate >= 1:
            return 1.0
        
        # Basit tahmin: win_rate / (1 - win_rate) * RR
        # Varsayılan RR = 1.5
        rr = 1.5
        pf = (pattern.win_rate * rr) / ((1 - pattern.win_rate) * 1.0)
        
        return max(0.5, min(5.0, pf))
