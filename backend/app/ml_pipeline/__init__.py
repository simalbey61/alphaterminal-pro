"""
AlphaTerminal Pro - ML Pipeline
===============================

Enterprise-grade machine learning pipeline for trading strategies.

Features:
- Comprehensive feature engineering (100+ features)
- Multiple model types (Decision Trees, Random Forest, LSTM, etc.)
- Time-series cross-validation
- Walk-forward analysis
- Automatic strategy discovery
- Model registry and versioning

Quick Start:
    from app.ml_pipeline import TrainingPipeline, PipelineConfig
    from app.ml_pipeline.enums import ModelType, PredictionTarget
    
    config = PipelineConfig(
        model_type=ModelType.RANDOM_FOREST,
        target_type=PredictionTarget.DIRECTION,
        prediction_horizon=1
    )
    
    pipeline = TrainingPipeline(config)
    result = pipeline.run(data, symbol="THYAO")

Author: AlphaTerminal Team
Version: 1.0.0
"""

# Enums
from app.ml_pipeline.enums import (
    ModelType,
    PredictionTarget,
    FeatureCategory,
    ModelStatus,
    TrainingStatus,
    TREND_INDICATORS,
    MOMENTUM_INDICATORS,
    VOLATILITY_INDICATORS,
    VOLUME_INDICATORS,
    CANDLESTICK_PATTERNS,
    DEFAULT_HYPERPARAMS,
)

# Features
from app.ml_pipeline.features.feature_engineer import (
    FeatureDefinition,
    FeatureSet,
    FeatureEngineer,
    PriceFeatureCalculator,
    TrendFeatureCalculator,
    MomentumFeatureCalculator,
    VolatilityFeatureCalculator,
    VolumeFeatureCalculator,
    TemporalFeatureCalculator,
)

from app.ml_pipeline.features.target_generator import (
    TargetConfig,
    TargetGenerator,
)

# Models
from app.ml_pipeline.models.base_models import (
    ModelConfig,
    ModelMetadata,
    BaseModel,
    DecisionTreeModel,
    RandomForestModel,
    GradientBoostingModel,
    MLPModel,
    LSTMModel,
    ModelFactory,
)

# Evaluation
from app.ml_pipeline.evaluation.evaluator import (
    EvaluationResult,
    MetricsCalculator,
    TimeSeriesCrossValidator,
    WalkForwardValidator,
    ExpandingWindowValidator,
    ModelEvaluator,
)

# Training
from app.ml_pipeline.training.pipeline import (
    PipelineConfig,
    TrainingJob,
    TrainingPipeline,
    PipelineManager,
)

from app.ml_pipeline.training.strategy_discovery import (
    TradingRule,
    DiscoveredStrategy,
    RuleExtractor,
    StrategyDiscoverer,
)


__all__ = [
    # Enums
    "ModelType",
    "PredictionTarget",
    "FeatureCategory",
    "ModelStatus",
    "TrainingStatus",
    "TREND_INDICATORS",
    "MOMENTUM_INDICATORS",
    "VOLATILITY_INDICATORS",
    "VOLUME_INDICATORS",
    "CANDLESTICK_PATTERNS",
    "DEFAULT_HYPERPARAMS",
    
    # Features
    "FeatureDefinition",
    "FeatureSet",
    "FeatureEngineer",
    "PriceFeatureCalculator",
    "TrendFeatureCalculator",
    "MomentumFeatureCalculator",
    "VolatilityFeatureCalculator",
    "VolumeFeatureCalculator",
    "TemporalFeatureCalculator",
    "TargetConfig",
    "TargetGenerator",
    
    # Models
    "ModelConfig",
    "ModelMetadata",
    "BaseModel",
    "DecisionTreeModel",
    "RandomForestModel",
    "GradientBoostingModel",
    "MLPModel",
    "LSTMModel",
    "ModelFactory",
    
    # Evaluation
    "EvaluationResult",
    "MetricsCalculator",
    "TimeSeriesCrossValidator",
    "WalkForwardValidator",
    "ExpandingWindowValidator",
    "ModelEvaluator",
    
    # Training
    "PipelineConfig",
    "TrainingJob",
    "TrainingPipeline",
    "PipelineManager",
    
    # Strategy Discovery
    "TradingRule",
    "DiscoveredStrategy",
    "RuleExtractor",
    "StrategyDiscoverer",
]

__version__ = "1.0.0"
