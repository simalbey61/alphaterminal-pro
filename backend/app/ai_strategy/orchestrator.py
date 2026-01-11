"""
AlphaTerminal Pro - AI Strategy Orchestrator
=============================================

7 katmanlı AI sistemini koordine eden ana orkestratör.

Author: AlphaTerminal Team
Version: 1.0.0
"""

from __future__ import annotations

import logging
import asyncio
from typing import Optional, List, Dict, Any, Tuple
from datetime import datetime, timedelta
from dataclasses import dataclass, field

import numpy as np
import polars as pl

from app.ai_strategy.constants import (
    TrendRegime,
    VolatilityRegime,
    StrategyLifecycle,
    SignalType,
    ApprovalThresholds,
    ValidationDefaults,
)

# Layer imports
from app.ai_strategy.data_layer import DataQualityChecker, RegimeDetector, RegimeState
from app.ai_strategy.feature_factory import IncrementalFeatureEngine, FeatureBatch, feature_store
from app.ai_strategy.pattern_discovery import DecisionTreeMiner, SHAPExplainer, DiscoveredPattern
from app.ai_strategy.strategy_generator import RuleSynthesizer, RiskCalculator, SynthesizedStrategy
from app.ai_strategy.validation_engine import (
    PurgedKFoldCV, WalkForwardAnalyzer, WalkForwardMode,
    MonteCarloSimulator, MonteCarloResult
)
from app.ai_strategy.live_execution import ApprovalChecker, PerformanceMonitor, ApprovalResult
from app.ai_strategy.evolution_engine import (
    GeneticAlgorithm, DiversityManager, RetirementManager,
    StrategyProfile, DiversityReport
)

logger = logging.getLogger(__name__)


@dataclass
class StrategyDiscoveryResult:
    """Strateji keşif sonucu."""
    discovered_patterns: List[DiscoveredPattern]
    synthesized_strategies: List[SynthesizedStrategy]
    validation_results: Dict[str, Any]
    approved_strategies: List[str]
    sandbox_strategies: List[str]
    rejected_strategies: List[str]
    discovery_time_seconds: float
    timestamp: datetime = field(default_factory=datetime.utcnow)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "discovered_patterns": len(self.discovered_patterns),
            "synthesized_strategies": len(self.synthesized_strategies),
            "approved": len(self.approved_strategies),
            "sandbox": len(self.sandbox_strategies),
            "rejected": len(self.rejected_strategies),
            "discovery_time_seconds": self.discovery_time_seconds,
            "timestamp": self.timestamp.isoformat(),
        }


@dataclass
class SignalGenerationResult:
    """Sinyal üretim sonucu."""
    symbol: str
    timeframe: str
    
    # Regime
    current_regime: RegimeState
    
    # Features
    features: FeatureBatch
    
    # Signals from strategies
    signals: List[Dict[str, Any]]
    
    # Consensus
    consensus_direction: Optional[SignalType]
    consensus_strength: float
    
    # Risk
    position_size: Optional[Dict[str, Any]]
    
    generated_at: datetime = field(default_factory=datetime.utcnow)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "symbol": self.symbol,
            "timeframe": self.timeframe,
            "regime": self.current_regime.to_dict(),
            "signal_count": len(self.signals),
            "consensus_direction": self.consensus_direction.value if self.consensus_direction else None,
            "consensus_strength": self.consensus_strength,
            "generated_at": self.generated_at.isoformat(),
        }


class AIStrategyOrchestrator:
    """
    7-Katmanlı AI Strategy System Orchestrator.
    
    Bu sınıf tüm AI katmanlarını koordine eder:
    1. Data Layer - Veri kalite ve rejim tespiti
    2. Feature Factory - 200+ feature hesaplama
    3. Pattern Discovery - Örüntü keşfi
    4. Strategy Generator - Strateji sentezi
    5. Validation Engine - Robustluk testleri
    6. Live Execution - Onay ve monitoring
    7. Evolution Engine - Evrim ve optimizasyon
    
    Example:
        ```python
        orchestrator = AIStrategyOrchestrator()
        
        # Strateji keşfi pipeline'ı
        result = await orchestrator.discover_strategies(
            historical_data=data,
            symbol="THYAO",
            timeframe="4h"
        )
        
        # Sinyal üretimi
        signals = await orchestrator.generate_signals(
            ohlcv_data=current_data,
            symbol="THYAO",
            timeframe="4h"
        )
        
        # Günlük evrim döngüsü
        await orchestrator.run_evolution_cycle()
        ```
    """
    
    def __init__(
        self,
        capital: float = 100000,
        max_risk_per_trade: float = 0.02,
        approval_thresholds: Optional[ApprovalThresholds] = None,
        validation_defaults: Optional[ValidationDefaults] = None,
    ):
        """
        Initialize AI Strategy Orchestrator.
        
        Args:
            capital: Trading sermayesi
            max_risk_per_trade: Trade başına max risk
            approval_thresholds: Onay eşikleri
            validation_defaults: Validasyon varsayılanları
        """
        self.capital = capital
        self.max_risk_per_trade = max_risk_per_trade
        self.approval_thresholds = approval_thresholds or ApprovalThresholds()
        self.validation_defaults = validation_defaults or ValidationDefaults()
        
        # Initialize layers
        self._init_layers()
        
        # Strategy storage
        self._active_strategies: Dict[str, SynthesizedStrategy] = {}
        self._strategy_performance: Dict[str, Dict] = {}
        
        logger.info("AI Strategy Orchestrator initialized")
    
    def _init_layers(self) -> None:
        """Katmanları başlat."""
        # Layer 1: Data
        self.quality_checker = DataQualityChecker()
        self.regime_detector = RegimeDetector()
        
        # Layer 2: Features
        self.feature_engine = IncrementalFeatureEngine()
        
        # Layer 3: Pattern Discovery
        self.tree_miner = DecisionTreeMiner(
            min_samples_leaf=50,
            max_depth=5,
            min_win_rate=0.55
        )
        self.shap_explainer = SHAPExplainer()
        
        # Layer 4: Strategy Generator
        self.rule_synthesizer = RuleSynthesizer()
        self.risk_calculator = RiskCalculator(
            capital=self.capital,
            max_risk_per_trade=self.max_risk_per_trade
        )
        
        # Layer 5: Validation
        self.purged_cv = PurgedKFoldCV(
            n_splits=self.validation_defaults.n_splits,
            purge_gap=self.validation_defaults.purge_gap
        )
        self.walk_forward = WalkForwardAnalyzer(
            mode=WalkForwardMode.ANCHORED,
            n_windows=12,
            train_pct=self.validation_defaults.train_pct
        )
        self.monte_carlo = MonteCarloSimulator(
            simulations=self.validation_defaults.simulations
        )
        
        # Layer 6: Live Execution
        self.approval_checker = ApprovalChecker(self.approval_thresholds)
        self.performance_monitor = PerformanceMonitor()
        
        # Layer 7: Evolution
        self.genetic_algorithm = GeneticAlgorithm(
            population_size=100,
            generations=50
        )
        self.diversity_manager = DiversityManager()
        self.retirement_manager = RetirementManager()
    
    # =========================================================================
    # STRATEGY DISCOVERY PIPELINE
    # =========================================================================
    
    async def discover_strategies(
        self,
        historical_data: pl.DataFrame,
        symbol: str,
        timeframe: str,
        min_trades: int = 100,
    ) -> StrategyDiscoveryResult:
        """
        Tam strateji keşif pipeline'ı.
        
        Args:
            historical_data: Tarihsel OHLCV verisi
            symbol: Hisse sembolü
            timeframe: Zaman dilimi
            min_trades: Minimum trade sayısı
            
        Returns:
            StrategyDiscoveryResult: Keşif sonuçları
        """
        import time
        start_time = time.time()
        
        logger.info(f"Starting strategy discovery for {symbol} {timeframe}")
        
        # Step 1: Data Quality Check
        quality_report = self.quality_checker.check(historical_data, symbol, timeframe)
        if not quality_report.passed:
            logger.warning(f"Data quality check failed: {quality_report.issues}")
            historical_data = self.quality_checker.clean(historical_data, quality_report)
        
        # Step 2: Calculate Features
        features_batch = self.feature_engine.calculate_all(
            historical_data, symbol, timeframe
        )
        
        # Step 3: Prepare Training Data
        features_array, labels, returns = self._prepare_training_data(
            historical_data, features_batch
        )
        
        if len(labels) < min_trades:
            logger.warning(f"Insufficient trades: {len(labels)} < {min_trades}")
            return self._create_empty_discovery_result(time.time() - start_time)
        
        # Step 4: Mine Patterns
        feature_names = list(features_batch.features.keys())
        patterns = self.tree_miner.mine(
            features=features_array,
            labels=labels,
            feature_names=feature_names,
            returns=returns
        )
        
        logger.info(f"Discovered {len(patterns)} significant patterns")
        
        # Step 5: Synthesize Strategies
        strategies = []
        for pattern in patterns:
            strategy = self.rule_synthesizer.synthesize_from_pattern(pattern)
            if strategy:
                strategies.append(strategy)
        
        logger.info(f"Synthesized {len(strategies)} strategies")
        
        # Step 6: Validate Strategies
        validation_results = {}
        approved, sandbox, rejected = [], [], []
        
        for strategy in strategies:
            # Quick backtest
            backtest_metrics = self._quick_backtest(
                strategy, features_array, labels, returns, feature_names
            )
            
            # Monte Carlo
            if len(returns) > 20:
                mc_result = self.monte_carlo.simulate(returns[:100])
            else:
                mc_result = None
            
            # Approval check
            approval = self.approval_checker.evaluate(
                backtest_metrics=backtest_metrics,
                monte_carlo_result=mc_result,
                robustness_score=backtest_metrics.get("robustness", 0.5)
            )
            
            validation_results[strategy.id] = {
                "backtest": backtest_metrics,
                "approval": approval.to_dict()
            }
            
            if approval.is_approved:
                approved.append(strategy.id)
                strategy.lifecycle = StrategyLifecycle.ACTIVE
                self._active_strategies[strategy.id] = strategy
            elif approval.decision.value == "sandbox":
                sandbox.append(strategy.id)
                strategy.lifecycle = StrategyLifecycle.SANDBOX
            else:
                rejected.append(strategy.id)
        
        elapsed = time.time() - start_time
        
        logger.info(
            f"Discovery complete: {len(approved)} approved, "
            f"{len(sandbox)} sandbox, {len(rejected)} rejected "
            f"in {elapsed:.2f}s"
        )
        
        return StrategyDiscoveryResult(
            discovered_patterns=patterns,
            synthesized_strategies=strategies,
            validation_results=validation_results,
            approved_strategies=approved,
            sandbox_strategies=sandbox,
            rejected_strategies=rejected,
            discovery_time_seconds=elapsed,
        )
    
    # =========================================================================
    # SIGNAL GENERATION
    # =========================================================================
    
    async def generate_signals(
        self,
        ohlcv_data: pl.DataFrame,
        symbol: str,
        timeframe: str,
        entry_price: Optional[float] = None,
    ) -> SignalGenerationResult:
        """
        Aktif stratejilerden sinyal üret.
        
        Args:
            ohlcv_data: Güncel OHLCV verisi
            symbol: Hisse sembolü
            timeframe: Zaman dilimi
            entry_price: Giriş fiyatı (position sizing için)
            
        Returns:
            SignalGenerationResult: Sinyal sonuçları
        """
        # Step 1: Regime Detection
        regime = self.regime_detector.detect(ohlcv_data)
        
        # Step 2: Calculate Features
        features = self.feature_engine.calculate_all(ohlcv_data, symbol, timeframe)
        
        # Step 3: Get Regime-Suitable Strategies
        suitable_strategies = self._get_strategies_for_regime(regime)
        
        # Step 4: Generate Signals from Each Strategy
        signals = []
        for strategy in suitable_strategies:
            signal_triggered, strength = strategy.evaluate_entry(features.features)
            
            if signal_triggered:
                signals.append({
                    "strategy_id": strategy.id,
                    "strategy_name": strategy.name,
                    "signal_type": strategy.signal_type.value,
                    "strength": strength,
                    "expected_win_rate": strategy.expected_win_rate,
                })
        
        # Step 5: Calculate Consensus
        consensus_direction, consensus_strength = self._calculate_consensus(signals)
        
        # Step 6: Position Sizing (if entry price provided)
        position_size = None
        if entry_price and consensus_direction in [SignalType.LONG, SignalType.SHORT]:
            # Get ATR for stop calculation
            atr = features.features.get("atr_14", entry_price * 0.02)
            stop_distance = atr * 1.5
            
            if consensus_direction == SignalType.LONG:
                stop_loss = entry_price - stop_distance
            else:
                stop_loss = entry_price + stop_distance
            
            size_result = self.risk_calculator.calculate_position_size(
                entry_price=entry_price,
                stop_loss=stop_loss,
                atr=atr,
                direction="long" if consensus_direction == SignalType.LONG else "short"
            )
            position_size = size_result.to_dict()
        
        return SignalGenerationResult(
            symbol=symbol,
            timeframe=timeframe,
            current_regime=regime,
            features=features,
            signals=signals,
            consensus_direction=consensus_direction,
            consensus_strength=consensus_strength,
            position_size=position_size,
        )
    
    # =========================================================================
    # EVOLUTION CYCLE
    # =========================================================================
    
    async def run_evolution_cycle(self) -> Dict[str, Any]:
        """
        Günlük evrim döngüsü.
        
        Returns:
            Dict: Evrim sonuçları
        """
        logger.info("Starting evolution cycle")
        
        results = {
            "retired": [],
            "revived": [],
            "optimized": [],
            "diversity_report": None,
        }
        
        # Step 1: Performance Check & Retirement
        for strategy_id, strategy in list(self._active_strategies.items()):
            health = self.retirement_manager.evaluate_health(
                strategy_id=strategy_id,
                strategy_name=strategy.name,
                current_lifecycle=strategy.lifecycle,
                recent_trades=self._get_recent_trades(strategy_id),
                expected_win_rate=strategy.expected_win_rate,
                expected_sharpe=1.0,
                expected_max_dd=0.15,
            )
            
            recommendation = self.retirement_manager.get_retirement_recommendation(health)
            
            if recommendation and recommendation.severity == "immediate":
                strategy.lifecycle = recommendation.recommended_action
                if recommendation.recommended_action == StrategyLifecycle.RETIRED:
                    del self._active_strategies[strategy_id]
                    results["retired"].append(strategy_id)
                    logger.info(f"Retired strategy {strategy_id}: {recommendation.reason}")
        
        # Step 2: Diversity Analysis
        report = self.diversity_manager.analyze_diversity()
        results["diversity_report"] = report.to_dict() if report else None
        
        # Step 3: Log summary
        logger.info(
            f"Evolution cycle complete: "
            f"{len(results['retired'])} retired, "
            f"{len(results['revived'])} revived"
        )
        
        return results
    
    # =========================================================================
    # HELPER METHODS
    # =========================================================================
    
    def _prepare_training_data(
        self,
        ohlcv: pl.DataFrame,
        features: FeatureBatch,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Training verisi hazırla."""
        close = ohlcv["close"].to_numpy()
        
        # Forward returns (next bar)
        forward_returns = np.zeros(len(close))
        forward_returns[:-1] = (close[1:] - close[:-1]) / close[:-1]
        
        # Labels (1 = positive return, 0 = negative)
        labels = (forward_returns > 0).astype(int)
        
        # Feature array
        feature_names = list(features.features.keys())
        n_features = len(feature_names)
        
        # Basit: son değerleri tüm bar'lara uygula (gerçekte rolling hesaplanmalı)
        # Bu placeholder - gerçek implementasyonda her bar için feature hesaplanmalı
        feature_array = np.zeros((len(close), n_features))
        for i, name in enumerate(feature_names):
            feature_array[:, i] = features.features.get(name, 0)
        
        # NaN'ları temizle
        valid_mask = ~np.isnan(feature_array).any(axis=1)
        
        return feature_array[valid_mask], labels[valid_mask], forward_returns[valid_mask]
    
    def _quick_backtest(
        self,
        strategy: SynthesizedStrategy,
        features: np.ndarray,
        labels: np.ndarray,
        returns: np.ndarray,
        feature_names: List[str],
    ) -> Dict[str, Any]:
        """Hızlı backtest."""
        # Her bar için strateji değerlendir
        signals = []
        for i in range(len(features)):
            feat_dict = {name: features[i, j] for j, name in enumerate(feature_names)}
            triggered, strength = strategy.evaluate_entry(feat_dict)
            signals.append(1 if triggered else 0)
        
        signals = np.array(signals)
        
        # Sinyal verilen bar'ların performansı
        signal_mask = signals == 1
        if signal_mask.sum() == 0:
            return {
                "win_rate": 0,
                "profit_factor": 0,
                "sharpe_ratio": 0,
                "max_drawdown": 0,
                "total_trades": 0,
            }
        
        signal_returns = returns[signal_mask]
        signal_labels = labels[signal_mask]
        
        # Metrics
        wins = signal_labels.sum()
        total = len(signal_labels)
        win_rate = wins / total if total > 0 else 0
        
        pos_returns = signal_returns[signal_returns > 0].sum()
        neg_returns = abs(signal_returns[signal_returns < 0].sum())
        profit_factor = pos_returns / neg_returns if neg_returns > 0 else 10
        
        if len(signal_returns) > 1 and np.std(signal_returns) > 0:
            sharpe = np.mean(signal_returns) / np.std(signal_returns) * np.sqrt(252)
        else:
            sharpe = 0
        
        # Max drawdown
        cumulative = np.cumsum(signal_returns)
        running_max = np.maximum.accumulate(cumulative)
        drawdown = running_max - cumulative
        max_dd = np.max(drawdown) / (np.max(running_max) + 1) if np.max(running_max) > 0 else 0
        
        return {
            "win_rate": win_rate,
            "profit_factor": profit_factor,
            "sharpe_ratio": sharpe,
            "max_drawdown": max_dd,
            "total_trades": total,
            "avg_profit_pct": np.mean(signal_returns) if len(signal_returns) > 0 else 0,
            "robustness": 0.7 if win_rate > 0.5 and profit_factor > 1.5 else 0.4,
        }
    
    def _get_strategies_for_regime(self, regime: RegimeState) -> List[SynthesizedStrategy]:
        """Regime'e uygun stratejileri al."""
        # Şimdilik tüm aktif stratejileri döndür
        # İleride regime filtering eklenebilir
        return list(self._active_strategies.values())
    
    def _calculate_consensus(
        self,
        signals: List[Dict[str, Any]]
    ) -> Tuple[Optional[SignalType], float]:
        """Sinyal konsensüsü hesapla."""
        if not signals:
            return None, 0.0
        
        long_strength = sum(
            s["strength"] * s["expected_win_rate"]
            for s in signals if s["signal_type"] == "long"
        )
        short_strength = sum(
            s["strength"] * s["expected_win_rate"]
            for s in signals if s["signal_type"] == "short"
        )
        
        total_strength = long_strength + short_strength
        if total_strength == 0:
            return None, 0.0
        
        if long_strength > short_strength * 1.5:
            return SignalType.LONG, long_strength / len(signals)
        elif short_strength > long_strength * 1.5:
            return SignalType.SHORT, short_strength / len(signals)
        else:
            return SignalType.NEUTRAL, 0.5
    
    def _get_recent_trades(self, strategy_id: str) -> List[Dict]:
        """Son trade'leri al."""
        return self.performance_monitor.get_strategy_history(strategy_id, limit=10)
    
    def _create_empty_discovery_result(self, elapsed: float) -> StrategyDiscoveryResult:
        """Boş discovery sonucu."""
        return StrategyDiscoveryResult(
            discovered_patterns=[],
            synthesized_strategies=[],
            validation_results={},
            approved_strategies=[],
            sandbox_strategies=[],
            rejected_strategies=[],
            discovery_time_seconds=elapsed,
        )
    
    # =========================================================================
    # PUBLIC API
    # =========================================================================
    
    def get_active_strategies(self) -> List[Dict[str, Any]]:
        """Aktif stratejileri listele."""
        return [s.to_dict() for s in self._active_strategies.values()]
    
    def get_strategy_count(self) -> Dict[str, int]:
        """Strateji sayıları."""
        by_lifecycle = {}
        for s in self._active_strategies.values():
            lc = s.lifecycle.value
            by_lifecycle[lc] = by_lifecycle.get(lc, 0) + 1
        return by_lifecycle
    
    async def record_trade_result(
        self,
        strategy_id: str,
        pnl: float,
        is_win: bool,
    ) -> None:
        """Trade sonucunu kaydet."""
        self.performance_monitor.record_trade(
            strategy_id=strategy_id,
            pnl=pnl,
            is_win=is_win
        )
