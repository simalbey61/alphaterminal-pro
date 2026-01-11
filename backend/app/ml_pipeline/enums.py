"""
AlphaTerminal Pro - ML Pipeline Enums and Constants
===================================================

Enumerations and constants for ML pipeline.

Author: AlphaTerminal Team
Version: 1.0.0
"""

from enum import Enum, auto
from typing import Dict, List, Any


# =============================================================================
# MODEL TYPES
# =============================================================================

class ModelType(str, Enum):
    """Available ML model types."""
    
    # Tree-based
    DECISION_TREE = "decision_tree"
    RANDOM_FOREST = "random_forest"
    GRADIENT_BOOSTING = "gradient_boosting"
    XGBOOST = "xgboost"
    LIGHTGBM = "lightgbm"
    
    # Neural networks
    MLP = "mlp"
    LSTM = "lstm"
    GRU = "gru"
    TRANSFORMER = "transformer"
    
    # Ensemble
    VOTING_ENSEMBLE = "voting_ensemble"
    STACKING_ENSEMBLE = "stacking_ensemble"
    
    # Statistical
    LOGISTIC_REGRESSION = "logistic_regression"
    SVM = "svm"
    
    @property
    def is_neural(self) -> bool:
        """Check if model is neural network based."""
        return self in {
            ModelType.MLP, ModelType.LSTM,
            ModelType.GRU, ModelType.TRANSFORMER
        }
    
    @property
    def is_tree_based(self) -> bool:
        """Check if model is tree-based."""
        return self in {
            ModelType.DECISION_TREE, ModelType.RANDOM_FOREST,
            ModelType.GRADIENT_BOOSTING, ModelType.XGBOOST,
            ModelType.LIGHTGBM
        }
    
    @property
    def supports_feature_importance(self) -> bool:
        """Check if model supports feature importance."""
        return self.is_tree_based or self == ModelType.LOGISTIC_REGRESSION


class PredictionTarget(str, Enum):
    """Prediction target types."""
    
    # Classification
    DIRECTION = "direction"              # Up/Down
    DIRECTION_3WAY = "direction_3way"    # Up/Neutral/Down
    REGIME = "regime"                    # Market regime
    
    # Regression
    RETURN = "return"                    # Future return
    VOLATILITY = "volatility"            # Future volatility
    PRICE = "price"                      # Future price
    
    # Multi-output
    RISK_REWARD = "risk_reward"          # Risk/reward prediction


class FeatureCategory(str, Enum):
    """Feature categories."""
    
    # Price-based
    PRICE = "price"
    RETURN = "return"
    VOLATILITY = "volatility"
    
    # Technical indicators
    TREND = "trend"
    MOMENTUM = "momentum"
    VOLUME = "volume"
    OSCILLATOR = "oscillator"
    
    # Pattern
    CANDLESTICK = "candlestick"
    CHART_PATTERN = "chart_pattern"
    
    # Statistical
    STATISTICAL = "statistical"
    
    # Time-based
    TEMPORAL = "temporal"
    
    # Market
    MARKET = "market"


class ModelStatus(str, Enum):
    """Model lifecycle status."""
    
    DRAFT = "draft"
    TRAINING = "training"
    VALIDATING = "validating"
    READY = "ready"
    DEPLOYED = "deployed"
    RETIRED = "retired"
    FAILED = "failed"


class TrainingStatus(str, Enum):
    """Training job status."""
    
    QUEUED = "queued"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


# =============================================================================
# FEATURE DEFINITIONS
# =============================================================================

# Standard technical indicators
TREND_INDICATORS = {
    "sma": {"periods": [5, 10, 20, 50, 100, 200]},
    "ema": {"periods": [5, 10, 20, 50, 100]},
    "wma": {"periods": [10, 20, 50]},
    "macd": {"fast": 12, "slow": 26, "signal": 9},
    "adx": {"period": 14},
    "aroon": {"period": 25},
    "ichimoku": {},
}

MOMENTUM_INDICATORS = {
    "rsi": {"periods": [7, 14, 21]},
    "stoch": {"k_period": 14, "d_period": 3},
    "stoch_rsi": {"period": 14},
    "williams_r": {"period": 14},
    "cci": {"period": 20},
    "mfi": {"period": 14},
    "roc": {"periods": [5, 10, 20]},
    "momentum": {"periods": [10, 20]},
}

VOLATILITY_INDICATORS = {
    "atr": {"periods": [7, 14, 21]},
    "bollinger": {"period": 20, "std": 2},
    "keltner": {"period": 20, "atr_mult": 2},
    "donchian": {"period": 20},
    "std": {"periods": [10, 20, 50]},
}

VOLUME_INDICATORS = {
    "obv": {},
    "vwap": {},
    "ad": {},
    "cmf": {"period": 20},
    "force_index": {"period": 13},
    "eom": {"period": 14},
    "volume_sma": {"periods": [10, 20, 50]},
}

# Candlestick patterns
CANDLESTICK_PATTERNS = [
    "doji",
    "hammer",
    "inverted_hammer",
    "bullish_engulfing",
    "bearish_engulfing",
    "morning_star",
    "evening_star",
    "three_white_soldiers",
    "three_black_crows",
    "harami",
    "piercing_line",
    "dark_cloud_cover",
    "spinning_top",
    "marubozu",
]


# =============================================================================
# MODEL HYPERPARAMETERS
# =============================================================================

DEFAULT_HYPERPARAMS: Dict[ModelType, Dict[str, Any]] = {
    ModelType.DECISION_TREE: {
        "max_depth": [3, 5, 7, 10, None],
        "min_samples_split": [2, 5, 10, 20],
        "min_samples_leaf": [1, 2, 5, 10],
        "criterion": ["gini", "entropy"],
    },
    ModelType.RANDOM_FOREST: {
        "n_estimators": [50, 100, 200],
        "max_depth": [5, 10, 15, None],
        "min_samples_split": [2, 5, 10],
        "min_samples_leaf": [1, 2, 5],
        "max_features": ["sqrt", "log2", None],
    },
    ModelType.GRADIENT_BOOSTING: {
        "n_estimators": [50, 100, 200],
        "learning_rate": [0.01, 0.05, 0.1],
        "max_depth": [3, 5, 7],
        "min_samples_split": [2, 5],
        "subsample": [0.8, 0.9, 1.0],
    },
    ModelType.XGBOOST: {
        "n_estimators": [50, 100, 200],
        "learning_rate": [0.01, 0.05, 0.1],
        "max_depth": [3, 5, 7],
        "subsample": [0.8, 0.9, 1.0],
        "colsample_bytree": [0.8, 0.9, 1.0],
        "reg_alpha": [0, 0.1, 1],
        "reg_lambda": [0, 0.1, 1],
    },
    ModelType.LIGHTGBM: {
        "n_estimators": [50, 100, 200],
        "learning_rate": [0.01, 0.05, 0.1],
        "max_depth": [3, 5, 7, -1],
        "num_leaves": [15, 31, 63],
        "subsample": [0.8, 0.9, 1.0],
        "colsample_bytree": [0.8, 0.9, 1.0],
    },
    ModelType.LSTM: {
        "units": [32, 64, 128],
        "layers": [1, 2, 3],
        "dropout": [0.1, 0.2, 0.3],
        "learning_rate": [0.001, 0.0001],
        "batch_size": [32, 64],
        "epochs": [50, 100],
        "sequence_length": [10, 20, 30],
    },
    ModelType.MLP: {
        "hidden_layers": [(64,), (128,), (64, 32), (128, 64)],
        "activation": ["relu", "tanh"],
        "dropout": [0.1, 0.2, 0.3],
        "learning_rate": [0.001, 0.0001],
        "batch_size": [32, 64],
        "epochs": [50, 100],
    },
    ModelType.LOGISTIC_REGRESSION: {
        "C": [0.01, 0.1, 1, 10],
        "penalty": ["l1", "l2"],
        "solver": ["liblinear", "saga"],
    },
}


# =============================================================================
# EVALUATION METRICS
# =============================================================================

CLASSIFICATION_METRICS = [
    "accuracy",
    "precision",
    "recall",
    "f1_score",
    "roc_auc",
    "log_loss",
    "confusion_matrix",
    "classification_report",
]

REGRESSION_METRICS = [
    "mse",
    "rmse",
    "mae",
    "mape",
    "r2",
    "explained_variance",
]

TRADING_METRICS = [
    "directional_accuracy",
    "profit_factor",
    "sharpe_ratio",
    "sortino_ratio",
    "max_drawdown",
    "win_rate",
    "avg_win_loss_ratio",
    "calmar_ratio",
]


# =============================================================================
# CROSS-VALIDATION SETTINGS
# =============================================================================

CV_SETTINGS = {
    "time_series_split": {
        "n_splits": 5,
        "gap": 0,
        "test_size": None,
    },
    "purged_kfold": {
        "n_splits": 5,
        "embargo_pct": 0.01,
    },
    "walk_forward": {
        "train_size": 252,  # 1 year
        "test_size": 21,    # 1 month
        "step_size": 21,    # 1 month
    },
    "combinatorial_purged": {
        "n_splits": 5,
        "n_test_splits": 2,
        "embargo_pct": 0.01,
    },
}


# =============================================================================
# ALL EXPORTS
# =============================================================================

__all__ = [
    # Enums
    "ModelType",
    "PredictionTarget",
    "FeatureCategory",
    "ModelStatus",
    "TrainingStatus",
    
    # Feature definitions
    "TREND_INDICATORS",
    "MOMENTUM_INDICATORS",
    "VOLATILITY_INDICATORS",
    "VOLUME_INDICATORS",
    "CANDLESTICK_PATTERNS",
    
    # Hyperparameters
    "DEFAULT_HYPERPARAMS",
    
    # Metrics
    "CLASSIFICATION_METRICS",
    "REGRESSION_METRICS",
    "TRADING_METRICS",
    
    # CV settings
    "CV_SETTINGS",
]
