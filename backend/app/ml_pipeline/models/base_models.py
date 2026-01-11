"""
AlphaTerminal Pro - ML Model Definitions
========================================

Model wrappers and configurations for trading ML.

Author: AlphaTerminal Team
Version: 1.0.0
"""

import logging
from datetime import datetime
from typing import Optional, Dict, List, Any, Union, Tuple
from dataclasses import dataclass, field
from abc import ABC, abstractmethod
import pickle
import json
import numpy as np
import pandas as pd

from app.ml_pipeline.enums import ModelType, ModelStatus, PredictionTarget


logger = logging.getLogger(__name__)


# =============================================================================
# MODEL CONFIGURATION
# =============================================================================

@dataclass
class ModelConfig:
    """Configuration for ML model."""
    name: str
    model_type: ModelType
    target_type: PredictionTarget
    hyperparameters: Dict[str, Any] = field(default_factory=dict)
    
    # Training settings
    train_size: float = 0.8
    validation_size: float = 0.1
    test_size: float = 0.1
    random_state: int = 42
    
    # Feature settings
    feature_columns: Optional[List[str]] = None
    scale_features: bool = True
    
    # Model settings
    early_stopping: bool = True
    early_stopping_patience: int = 10
    
    # Metadata
    description: str = ""
    created_at: datetime = field(default_factory=datetime.now)


@dataclass
class ModelMetadata:
    """Metadata for trained model."""
    model_id: str
    name: str
    model_type: ModelType
    target_type: PredictionTarget
    status: ModelStatus
    
    # Training info
    training_date: datetime
    training_samples: int
    validation_samples: int
    test_samples: int
    
    # Features
    feature_names: List[str]
    feature_count: int
    
    # Performance
    metrics: Dict[str, float] = field(default_factory=dict)
    
    # Config
    hyperparameters: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "model_id": self.model_id,
            "name": self.name,
            "model_type": self.model_type.value,
            "target_type": self.target_type.value,
            "status": self.status.value,
            "training_date": self.training_date.isoformat(),
            "training_samples": self.training_samples,
            "validation_samples": self.validation_samples,
            "test_samples": self.test_samples,
            "feature_names": self.feature_names,
            "feature_count": self.feature_count,
            "metrics": self.metrics,
            "hyperparameters": self.hyperparameters,
        }


# =============================================================================
# ABSTRACT BASE MODEL
# =============================================================================

class BaseModel(ABC):
    """
    Abstract base class for ML models.
    
    Provides common interface for training, prediction, and evaluation.
    """
    
    model_type: ModelType = ModelType.DECISION_TREE
    
    def __init__(self, config: ModelConfig):
        """
        Initialize model.
        
        Args:
            config: Model configuration
        """
        self.config = config
        self.model = None
        self.scaler = None
        self.feature_names: List[str] = []
        self.is_fitted = False
        self.metadata: Optional[ModelMetadata] = None
    
    @abstractmethod
    def _create_model(self) -> Any:
        """Create the underlying model instance."""
        pass
    
    @abstractmethod
    def _fit_model(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: Optional[np.ndarray] = None,
        y_val: Optional[np.ndarray] = None
    ) -> Dict[str, Any]:
        """Fit the model and return training history/metrics."""
        pass
    
    @abstractmethod
    def _predict(self, X: np.ndarray) -> np.ndarray:
        """Make predictions."""
        pass
    
    def fit(
        self,
        X: Union[pd.DataFrame, np.ndarray],
        y: Union[pd.Series, np.ndarray],
        X_val: Optional[Union[pd.DataFrame, np.ndarray]] = None,
        y_val: Optional[Union[pd.Series, np.ndarray]] = None
    ) -> Dict[str, Any]:
        """
        Train the model.
        
        Args:
            X: Training features
            y: Training targets
            X_val: Validation features (optional)
            y_val: Validation targets (optional)
            
        Returns:
            Training history/metrics
        """
        # Store feature names
        if isinstance(X, pd.DataFrame):
            self.feature_names = list(X.columns)
            X = X.values
        
        if isinstance(y, pd.Series):
            y = y.values
        
        if X_val is not None and isinstance(X_val, pd.DataFrame):
            X_val = X_val.values
        
        if y_val is not None and isinstance(y_val, pd.Series):
            y_val = y_val.values
        
        # Scale features if configured
        if self.config.scale_features:
            from sklearn.preprocessing import StandardScaler
            self.scaler = StandardScaler()
            X = self.scaler.fit_transform(X)
            if X_val is not None:
                X_val = self.scaler.transform(X_val)
        
        # Create model
        self.model = self._create_model()
        
        # Fit model
        history = self._fit_model(X, y, X_val, y_val)
        
        self.is_fitted = True
        
        logger.info(f"Model {self.config.name} trained successfully")
        
        return history
    
    def predict(
        self,
        X: Union[pd.DataFrame, np.ndarray]
    ) -> np.ndarray:
        """
        Make predictions.
        
        Args:
            X: Features
            
        Returns:
            Predictions array
        """
        if not self.is_fitted:
            raise ValueError("Model not fitted. Call fit() first.")
        
        if isinstance(X, pd.DataFrame):
            X = X.values
        
        if self.scaler is not None:
            X = self.scaler.transform(X)
        
        return self._predict(X)
    
    def predict_proba(
        self,
        X: Union[pd.DataFrame, np.ndarray]
    ) -> np.ndarray:
        """
        Predict class probabilities.
        
        Args:
            X: Features
            
        Returns:
            Probability array
        """
        if not self.is_fitted:
            raise ValueError("Model not fitted. Call fit() first.")
        
        if isinstance(X, pd.DataFrame):
            X = X.values
        
        if self.scaler is not None:
            X = self.scaler.transform(X)
        
        if hasattr(self.model, 'predict_proba'):
            return self.model.predict_proba(X)
        else:
            # For models without predict_proba, return predictions as one-hot
            preds = self._predict(X)
            n_classes = len(np.unique(preds))
            proba = np.zeros((len(preds), n_classes))
            proba[np.arange(len(preds)), preds.astype(int)] = 1
            return proba
    
    def get_feature_importance(self) -> Optional[Dict[str, float]]:
        """
        Get feature importance scores.
        
        Returns:
            Dict mapping feature name to importance score
        """
        if not self.is_fitted or not self.model_type.supports_feature_importance:
            return None
        
        if hasattr(self.model, 'feature_importances_'):
            importances = self.model.feature_importances_
        elif hasattr(self.model, 'coef_'):
            importances = np.abs(self.model.coef_).flatten()
        else:
            return None
        
        if len(self.feature_names) != len(importances):
            return None
        
        return dict(zip(self.feature_names, importances))
    
    def save(self, path: str):
        """Save model to file."""
        data = {
            'config': self.config,
            'model': self.model,
            'scaler': self.scaler,
            'feature_names': self.feature_names,
            'is_fitted': self.is_fitted,
            'metadata': self.metadata,
        }
        
        with open(path, 'wb') as f:
            pickle.dump(data, f)
        
        logger.info(f"Model saved to {path}")
    
    @classmethod
    def load(cls, path: str) -> "BaseModel":
        """Load model from file."""
        with open(path, 'rb') as f:
            data = pickle.load(f)
        
        instance = cls(data['config'])
        instance.model = data['model']
        instance.scaler = data['scaler']
        instance.feature_names = data['feature_names']
        instance.is_fitted = data['is_fitted']
        instance.metadata = data['metadata']
        
        logger.info(f"Model loaded from {path}")
        
        return instance


# =============================================================================
# TREE-BASED MODELS
# =============================================================================

class DecisionTreeModel(BaseModel):
    """Decision Tree classifier/regressor."""
    
    model_type = ModelType.DECISION_TREE
    
    def _create_model(self) -> Any:
        from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
        
        params = self.config.hyperparameters.copy()
        params['random_state'] = self.config.random_state
        
        if self.config.target_type in [PredictionTarget.DIRECTION, PredictionTarget.DIRECTION_3WAY, PredictionTarget.REGIME]:
            return DecisionTreeClassifier(**params)
        else:
            return DecisionTreeRegressor(**params)
    
    def _fit_model(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: Optional[np.ndarray] = None,
        y_val: Optional[np.ndarray] = None
    ) -> Dict[str, Any]:
        self.model.fit(X_train, y_train)
        
        history = {
            'train_score': self.model.score(X_train, y_train)
        }
        
        if X_val is not None and y_val is not None:
            history['val_score'] = self.model.score(X_val, y_val)
        
        return history
    
    def _predict(self, X: np.ndarray) -> np.ndarray:
        return self.model.predict(X)


class RandomForestModel(BaseModel):
    """Random Forest classifier/regressor."""
    
    model_type = ModelType.RANDOM_FOREST
    
    def _create_model(self) -> Any:
        from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
        
        params = self.config.hyperparameters.copy()
        params['random_state'] = self.config.random_state
        params.setdefault('n_jobs', -1)
        
        if self.config.target_type in [PredictionTarget.DIRECTION, PredictionTarget.DIRECTION_3WAY, PredictionTarget.REGIME]:
            return RandomForestClassifier(**params)
        else:
            return RandomForestRegressor(**params)
    
    def _fit_model(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: Optional[np.ndarray] = None,
        y_val: Optional[np.ndarray] = None
    ) -> Dict[str, Any]:
        self.model.fit(X_train, y_train)
        
        history = {
            'train_score': self.model.score(X_train, y_train),
            'n_estimators': self.model.n_estimators,
        }
        
        if X_val is not None and y_val is not None:
            history['val_score'] = self.model.score(X_val, y_val)
        
        return history
    
    def _predict(self, X: np.ndarray) -> np.ndarray:
        return self.model.predict(X)


class GradientBoostingModel(BaseModel):
    """Gradient Boosting classifier/regressor."""
    
    model_type = ModelType.GRADIENT_BOOSTING
    
    def _create_model(self) -> Any:
        from sklearn.ensemble import GradientBoostingClassifier, GradientBoostingRegressor
        
        params = self.config.hyperparameters.copy()
        params['random_state'] = self.config.random_state
        
        if self.config.target_type in [PredictionTarget.DIRECTION, PredictionTarget.DIRECTION_3WAY, PredictionTarget.REGIME]:
            return GradientBoostingClassifier(**params)
        else:
            return GradientBoostingRegressor(**params)
    
    def _fit_model(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: Optional[np.ndarray] = None,
        y_val: Optional[np.ndarray] = None
    ) -> Dict[str, Any]:
        self.model.fit(X_train, y_train)
        
        history = {
            'train_score': self.model.score(X_train, y_train),
        }
        
        if X_val is not None and y_val is not None:
            history['val_score'] = self.model.score(X_val, y_val)
        
        return history
    
    def _predict(self, X: np.ndarray) -> np.ndarray:
        return self.model.predict(X)


# =============================================================================
# NEURAL NETWORK MODELS
# =============================================================================

class MLPModel(BaseModel):
    """Multi-Layer Perceptron model."""
    
    model_type = ModelType.MLP
    
    def _create_model(self) -> Any:
        from sklearn.neural_network import MLPClassifier, MLPRegressor
        
        params = self.config.hyperparameters.copy()
        params['random_state'] = self.config.random_state
        params.setdefault('hidden_layer_sizes', (64, 32))
        params.setdefault('max_iter', 500)
        
        if self.config.early_stopping:
            params['early_stopping'] = True
            params['n_iter_no_change'] = self.config.early_stopping_patience
        
        if self.config.target_type in [PredictionTarget.DIRECTION, PredictionTarget.DIRECTION_3WAY, PredictionTarget.REGIME]:
            return MLPClassifier(**params)
        else:
            return MLPRegressor(**params)
    
    def _fit_model(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: Optional[np.ndarray] = None,
        y_val: Optional[np.ndarray] = None
    ) -> Dict[str, Any]:
        self.model.fit(X_train, y_train)
        
        history = {
            'train_score': self.model.score(X_train, y_train),
            'n_iter': self.model.n_iter_,
            'loss': self.model.loss_,
        }
        
        if X_val is not None and y_val is not None:
            history['val_score'] = self.model.score(X_val, y_val)
        
        return history
    
    def _predict(self, X: np.ndarray) -> np.ndarray:
        return self.model.predict(X)


class LSTMModel(BaseModel):
    """LSTM neural network model (requires TensorFlow/Keras)."""
    
    model_type = ModelType.LSTM
    
    def __init__(self, config: ModelConfig):
        super().__init__(config)
        self.sequence_length = config.hyperparameters.get('sequence_length', 20)
        self.history = None
    
    def _create_model(self) -> Any:
        """Create LSTM model using Keras."""
        try:
            import tensorflow as tf
            from tensorflow import keras
            from tensorflow.keras import layers
        except ImportError:
            raise ImportError("TensorFlow is required for LSTM model")
        
        params = self.config.hyperparameters
        
        # Model architecture
        units = params.get('units', 64)
        n_layers = params.get('layers', 2)
        dropout = params.get('dropout', 0.2)
        
        model = keras.Sequential()
        
        # LSTM layers
        for i in range(n_layers):
            return_sequences = i < n_layers - 1
            model.add(layers.LSTM(
                units,
                return_sequences=return_sequences,
                input_shape=(self.sequence_length, self._n_features) if i == 0 else None
            ))
            model.add(layers.Dropout(dropout))
        
        # Output layer
        if self.config.target_type in [PredictionTarget.DIRECTION]:
            model.add(layers.Dense(1, activation='sigmoid'))
            model.compile(
                optimizer=keras.optimizers.Adam(learning_rate=params.get('learning_rate', 0.001)),
                loss='binary_crossentropy',
                metrics=['accuracy']
            )
        elif self.config.target_type in [PredictionTarget.DIRECTION_3WAY, PredictionTarget.REGIME]:
            n_classes = params.get('n_classes', 3)
            model.add(layers.Dense(n_classes, activation='softmax'))
            model.compile(
                optimizer=keras.optimizers.Adam(learning_rate=params.get('learning_rate', 0.001)),
                loss='sparse_categorical_crossentropy',
                metrics=['accuracy']
            )
        else:
            model.add(layers.Dense(1))
            model.compile(
                optimizer=keras.optimizers.Adam(learning_rate=params.get('learning_rate', 0.001)),
                loss='mse',
                metrics=['mae']
            )
        
        return model
    
    def _create_sequences(self, X: np.ndarray, y: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Create sequences for LSTM."""
        X_seq, y_seq = [], []
        
        for i in range(len(X) - self.sequence_length):
            X_seq.append(X[i:i + self.sequence_length])
            y_seq.append(y[i + self.sequence_length])
        
        return np.array(X_seq), np.array(y_seq)
    
    def _fit_model(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: Optional[np.ndarray] = None,
        y_val: Optional[np.ndarray] = None
    ) -> Dict[str, Any]:
        self._n_features = X_train.shape[1]
        self.model = self._create_model()
        
        # Create sequences
        X_train_seq, y_train_seq = self._create_sequences(X_train, y_train)
        
        validation_data = None
        if X_val is not None and y_val is not None:
            X_val_seq, y_val_seq = self._create_sequences(X_val, y_val)
            validation_data = (X_val_seq, y_val_seq)
        
        params = self.config.hyperparameters
        
        # Callbacks
        callbacks = []
        if self.config.early_stopping:
            try:
                from tensorflow.keras.callbacks import EarlyStopping
                callbacks.append(EarlyStopping(
                    patience=self.config.early_stopping_patience,
                    restore_best_weights=True
                ))
            except ImportError:
                pass
        
        # Train
        self.history = self.model.fit(
            X_train_seq, y_train_seq,
            validation_data=validation_data,
            epochs=params.get('epochs', 100),
            batch_size=params.get('batch_size', 32),
            callbacks=callbacks,
            verbose=0
        )
        
        return {
            'train_loss': self.history.history['loss'][-1],
            'val_loss': self.history.history.get('val_loss', [None])[-1],
            'epochs': len(self.history.history['loss']),
        }
    
    def _predict(self, X: np.ndarray) -> np.ndarray:
        # Create sequences
        X_seq = []
        for i in range(len(X) - self.sequence_length + 1):
            X_seq.append(X[i:i + self.sequence_length])
        X_seq = np.array(X_seq)
        
        preds = self.model.predict(X_seq, verbose=0)
        
        # Pad with NaN for initial sequence positions
        full_preds = np.full(len(X), np.nan)
        full_preds[self.sequence_length - 1:] = preds.flatten()
        
        return full_preds


# =============================================================================
# MODEL FACTORY
# =============================================================================

class ModelFactory:
    """Factory for creating model instances."""
    
    _models = {
        ModelType.DECISION_TREE: DecisionTreeModel,
        ModelType.RANDOM_FOREST: RandomForestModel,
        ModelType.GRADIENT_BOOSTING: GradientBoostingModel,
        ModelType.MLP: MLPModel,
        ModelType.LSTM: LSTMModel,
    }
    
    @classmethod
    def create(cls, config: ModelConfig) -> BaseModel:
        """Create model from configuration."""
        model_class = cls._models.get(config.model_type)
        
        if model_class is None:
            raise ValueError(f"Unknown model type: {config.model_type}")
        
        return model_class(config)
    
    @classmethod
    def register(cls, model_type: ModelType, model_class: type):
        """Register a new model type."""
        cls._models[model_type] = model_class
    
    @classmethod
    def available_models(cls) -> List[ModelType]:
        """Get list of available model types."""
        return list(cls._models.keys())


# =============================================================================
# ALL EXPORTS
# =============================================================================

__all__ = [
    "ModelConfig",
    "ModelMetadata",
    "BaseModel",
    "DecisionTreeModel",
    "RandomForestModel",
    "GradientBoostingModel",
    "MLPModel",
    "LSTMModel",
    "ModelFactory",
]
