"""
AlphaTerminal Pro - AI Strategy System
======================================

7-Katmanlı kurumsal seviye AI strateji sistemi.

Layers:
1. Data Layer - Veri yönetimi ve kalite kontrol
2. Feature Factory - 200+ feature üretimi  
3. Pattern Discovery - Örüntü keşfi ve SHAP analizi
4. Strategy Generator - Strateji oluşturma
5. Validation Engine - Purged CV, Walk-Forward, Monte Carlo
6. Live Execution - Canlı trading ve approval
7. Evolution Engine - Genetik algoritma ve evrim

Author: AlphaTerminal Team
Version: 1.0.0
"""

# =============================================================================
# CONSTANTS & ENUMS
# =============================================================================
from app.ai_strategy.constants import (
    TrendRegime, VolatilityRegime, LiquidityRegime, MarketPhase,
    StrategyType, DiscoveryMethod, StrategyLifecycle,
    FeatureCategory, FeatureTimeframe, CalculationMode,
    ValidationStatus, ApprovalDecision, AlertSeverity, PerformanceAlert,
    ApprovalThresholds, RetirementThresholds, PositionSizingLimits,
    BacktestDefaults, ValidationDefaults, EvolutionDefaults,
    TECHNICAL_FEATURES, SMC_FEATURES, ORDERFLOW_FEATURES,
    ALPHA_FEATURES, ALL_FEATURES, STRATEGY_ZOO_CATEGORIES,
)

# LAYER 1: DATA LAYER
from app.ai_strategy.data_layer import (
    DataQualityChecker, QualityReport, RegimeDetector, RegimeState,
)

# LAYER 2: FEATURE FACTORY
from app.ai_strategy.feature_factory import (
    IncrementalFeatureEngine, FeatureBatch, FeatureStore, feature_store,
)

# LAYER 3: PATTERN DISCOVERY
from app.ai_strategy.pattern_discovery import (
    DecisionTreeMiner, TreeRule, DiscoveredPattern,
    SHAPExplainer, SHAPExplanation, FeatureImportance, FeatureInteraction,
)

# LAYER 4: STRATEGY GENERATOR
from app.ai_strategy.strategy_generator import (
    RuleSynthesizer, SynthesizedStrategy, EntryCondition, ExitCondition,
    RiskCalculator, ExecutionCosts, PositionSizeResult, RiskAssessment,
)

# LAYER 5: VALIDATION ENGINE
from app.ai_strategy.validation_engine import (
    PurgedKFoldCV, CombinatorialPurgedCV, CVFold, CVResult, PurgedCVSummary,
    WalkForwardAnalyzer, WalkForwardMode, WalkForwardWindow, WalkForwardResult,
    MonteCarloSimulator, MonteCarloResult,
)

# LAYER 6: LIVE EXECUTION
from app.ai_strategy.live_execution import (
    ApprovalChecker, ApprovalResult, CriterionResult, CriterionSeverity,
    PerformanceMonitor, MonitoringResult, PerformanceSnapshot, Alert, MonitorAction,
)

# LAYER 7: EVOLUTION ENGINE
from app.ai_strategy.evolution_engine import (
    GeneticAlgorithm, Gene, Chromosome, EvolutionResult,
    DiversityManager, StrategyProfile, DiversityReport,
    RetirementManager, RetirementRecommendation, RetirementReason, StrategyHealth,
)

__all__ = [
    # Enums
    "TrendRegime", "VolatilityRegime", "LiquidityRegime", "MarketPhase",
    "StrategyType", "DiscoveryMethod", "StrategyLifecycle",
    "FeatureCategory", "FeatureTimeframe", "CalculationMode",
    "ValidationStatus", "ApprovalDecision", "AlertSeverity", "PerformanceAlert",
    # Thresholds
    "ApprovalThresholds", "RetirementThresholds", "PositionSizingLimits",
    "BacktestDefaults", "ValidationDefaults", "EvolutionDefaults",
    # Feature Definitions
    "TECHNICAL_FEATURES", "SMC_FEATURES", "ORDERFLOW_FEATURES",
    "ALPHA_FEATURES", "ALL_FEATURES", "STRATEGY_ZOO_CATEGORIES",
    # Data Layer
    "DataQualityChecker", "QualityReport", "RegimeDetector", "RegimeState",
    # Feature Factory
    "IncrementalFeatureEngine", "FeatureBatch", "FeatureStore", "feature_store",
    # Pattern Discovery
    "DecisionTreeMiner", "TreeRule", "DiscoveredPattern",
    "SHAPExplainer", "SHAPExplanation", "FeatureImportance", "FeatureInteraction",
    # Strategy Generator
    "RuleSynthesizer", "SynthesizedStrategy", "EntryCondition", "ExitCondition",
    "RiskCalculator", "ExecutionCosts", "PositionSizeResult", "RiskAssessment",
    # Validation Engine
    "PurgedKFoldCV", "CombinatorialPurgedCV", "CVFold", "CVResult", "PurgedCVSummary",
    "WalkForwardAnalyzer", "WalkForwardMode", "WalkForwardWindow", "WalkForwardResult",
    "MonteCarloSimulator", "MonteCarloResult",
    # Live Execution
    "ApprovalChecker", "ApprovalResult", "CriterionResult", "CriterionSeverity",
    "PerformanceMonitor", "MonitoringResult", "PerformanceSnapshot", "Alert", "MonitorAction",
    # Evolution Engine
    "GeneticAlgorithm", "Gene", "Chromosome", "EvolutionResult",
    "DiversityManager", "StrategyProfile", "DiversityReport",
    "RetirementManager", "RetirementRecommendation", "RetirementReason", "StrategyHealth",
]
