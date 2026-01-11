"""
AlphaTerminal Pro - Training Pipeline
=====================================

End-to-end ML training pipeline for trading strategies.

Author: AlphaTerminal Team
Version: 1.0.0
"""

import logging
import uuid
import time
from datetime import datetime
from typing import Optional, Dict, List, Any, Tuple, Callable
from dataclasses import dataclass, field
from pathlib import Path
import json
import numpy as np
import pandas as pd

from app.ml_pipeline.enums import (
    ModelType, PredictionTarget, ModelStatus, TrainingStatus
)
from app.ml_pipeline.features.feature_engineer import FeatureEngineer
from app.ml_pipeline.features.target_generator import TargetGenerator, TargetConfig
from app.ml_pipeline.models.base_models import (
    ModelConfig, ModelMetadata, BaseModel, ModelFactory
)
from app.ml_pipeline.evaluation.evaluator import (
    ModelEvaluator, EvaluationResult, TimeSeriesCrossValidator,
    WalkForwardValidator
)


logger = logging.getLogger(__name__)


# =============================================================================
# PIPELINE CONFIGURATION
# =============================================================================

@dataclass
class PipelineConfig:
    """Configuration for training pipeline."""
    
    # Target configuration
    target_type: PredictionTarget = PredictionTarget.DIRECTION
    prediction_horizon: int = 1
    
    # Feature configuration
    feature_categories: Optional[List[str]] = None
    feature_params: Dict[str, Any] = field(default_factory=dict)
    
    # Data split
    train_ratio: float = 0.7
    val_ratio: float = 0.15
    test_ratio: float = 0.15
    
    # Model configuration
    model_type: ModelType = ModelType.RANDOM_FOREST
    model_params: Dict[str, Any] = field(default_factory=dict)
    scale_features: bool = True
    
    # Training options
    cross_validate: bool = True
    cv_splits: int = 5
    walk_forward: bool = False
    
    # Hyperparameter tuning
    tune_hyperparameters: bool = False
    tuning_iterations: int = 50
    
    # Output
    save_model: bool = True
    model_dir: str = "models"
    
    def validate(self) -> List[str]:
        """Validate configuration."""
        issues = []
        
        if self.train_ratio + self.val_ratio + self.test_ratio != 1.0:
            issues.append("Split ratios must sum to 1.0")
        
        if self.prediction_horizon < 1:
            issues.append("Prediction horizon must be >= 1")
        
        return issues


@dataclass
class TrainingJob:
    """Represents a training job."""
    job_id: str
    status: TrainingStatus
    config: PipelineConfig
    created_at: datetime
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    progress: float = 0.0
    message: str = ""
    result: Optional[Dict[str, Any]] = None
    error: Optional[str] = None


# =============================================================================
# TRAINING PIPELINE
# =============================================================================

class TrainingPipeline:
    """
    End-to-end ML training pipeline.
    
    Handles:
    - Feature engineering
    - Target generation
    - Data splitting
    - Model training
    - Cross-validation
    - Walk-forward analysis
    - Model persistence
    """
    
    def __init__(self, config: PipelineConfig):
        """
        Initialize training pipeline.
        
        Args:
            config: Pipeline configuration
        """
        self.config = config
        
        # Components
        self.feature_engineer = FeatureEngineer(params=config.feature_params)
        self.target_generator = TargetGenerator(
            TargetConfig(
                target_type=config.target_type,
                horizon=config.prediction_horizon
            )
        )
        self.evaluator = ModelEvaluator(target_type=config.target_type)
        
        # State
        self.model: Optional[BaseModel] = None
        self.feature_names: List[str] = []
        self.training_result: Optional[Dict[str, Any]] = None
    
    def run(
        self,
        data: pd.DataFrame,
        symbol: str = "UNKNOWN",
        callback: Optional[Callable[[float, str], None]] = None
    ) -> Dict[str, Any]:
        """
        Run full training pipeline.
        
        Args:
            data: OHLCV DataFrame
            symbol: Trading symbol
            callback: Progress callback (progress, message)
            
        Returns:
            Dict with training results
        """
        start_time = time.time()
        
        def update_progress(progress: float, message: str):
            if callback:
                callback(progress, message)
            logger.info(f"[{progress:.0%}] {message}")
        
        try:
            # Step 1: Feature Engineering
            update_progress(0.1, "Calculating features...")
            features_df = self.feature_engineer.calculate_all(
                data,
                categories=self.config.feature_categories
            )
            
            # Step 2: Target Generation
            update_progress(0.2, "Generating targets...")
            target, target_meta = self.target_generator.generate(
                data,
                horizon=self.config.prediction_horizon
            )
            
            # Step 3: Prepare Dataset
            update_progress(0.3, "Preparing dataset...")
            X, y, returns = self._prepare_dataset(features_df, target, data)
            
            # Step 4: Split Data
            update_progress(0.4, "Splitting data...")
            splits = self._split_data(X, y, returns)
            
            # Step 5: Train Model
            update_progress(0.5, "Training model...")
            train_result = self._train_model(
                splits['X_train'], splits['y_train'],
                splits['X_val'], splits['y_val']
            )
            
            # Step 6: Evaluate
            update_progress(0.7, "Evaluating model...")
            eval_result = self.evaluator.evaluate(
                self.model,
                splits['X_test'],
                splits['y_test'],
                splits['returns_test']
            )
            
            # Step 7: Cross-Validation (optional)
            cv_result = None
            if self.config.cross_validate:
                update_progress(0.8, "Cross-validating...")
                cv_result = self._cross_validate(X, y, returns)
            
            # Step 8: Walk-Forward Analysis (optional)
            wf_result = None
            if self.config.walk_forward:
                update_progress(0.85, "Walk-forward analysis...")
                wf_result = self._walk_forward_analysis(X, y, returns)
            
            # Step 9: Save Model
            model_path = None
            if self.config.save_model:
                update_progress(0.9, "Saving model...")
                model_path = self._save_model(symbol)
            
            # Step 10: Compile Results
            update_progress(1.0, "Complete!")
            
            execution_time = time.time() - start_time
            
            self.training_result = {
                'status': 'success',
                'symbol': symbol,
                'model_type': self.config.model_type.value,
                'target_type': self.config.target_type.value,
                'prediction_horizon': self.config.prediction_horizon,
                
                # Data info
                'total_samples': len(X),
                'train_samples': len(splits['X_train']),
                'val_samples': len(splits['X_val']),
                'test_samples': len(splits['X_test']),
                'feature_count': X.shape[1],
                'feature_names': self.feature_names,
                
                # Target info
                'target_metadata': target_meta,
                
                # Training info
                'training_history': train_result,
                
                # Evaluation
                'test_metrics': eval_result.to_dict(),
                'cv_metrics': cv_result.to_dict() if cv_result else None,
                'walk_forward': wf_result,
                
                # Feature importance
                'feature_importance': self.model.get_feature_importance(),
                
                # Model
                'model_path': model_path,
                
                # Timing
                'execution_time_seconds': execution_time,
                'timestamp': datetime.now().isoformat(),
            }
            
            return self.training_result
            
        except Exception as e:
            logger.error(f"Training failed: {e}", exc_info=True)
            return {
                'status': 'error',
                'error': str(e),
                'timestamp': datetime.now().isoformat(),
            }
    
    def _prepare_dataset(
        self,
        features_df: pd.DataFrame,
        target: pd.Series,
        data: pd.DataFrame
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Prepare dataset for training."""
        # Get feature columns (exclude OHLCV)
        exclude_cols = ['Open', 'High', 'Low', 'Close', 'Volume', 'Adj Close']
        feature_cols = [c for c in features_df.columns if c not in exclude_cols]
        
        # Create dataset
        df = features_df[feature_cols].copy()
        df['target'] = target
        
        # Calculate returns for trading metrics
        returns = data['Close'].pct_change().shift(-self.config.prediction_horizon)
        df['returns'] = returns
        
        # Drop NaN
        df = df.dropna()
        
        # Extract arrays
        X = df[feature_cols].values
        y = df['target'].values
        returns = df['returns'].values
        
        self.feature_names = feature_cols
        
        return X, y, returns
    
    def _split_data(
        self,
        X: np.ndarray,
        y: np.ndarray,
        returns: np.ndarray
    ) -> Dict[str, np.ndarray]:
        """Split data into train/val/test sets."""
        n = len(X)
        
        train_end = int(n * self.config.train_ratio)
        val_end = int(n * (self.config.train_ratio + self.config.val_ratio))
        
        return {
            'X_train': X[:train_end],
            'y_train': y[:train_end],
            'returns_train': returns[:train_end],
            
            'X_val': X[train_end:val_end],
            'y_val': y[train_end:val_end],
            'returns_val': returns[train_end:val_end],
            
            'X_test': X[val_end:],
            'y_test': y[val_end:],
            'returns_test': returns[val_end:],
        }
    
    def _train_model(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: Optional[np.ndarray] = None,
        y_val: Optional[np.ndarray] = None
    ) -> Dict[str, Any]:
        """Train the model."""
        # Create model config
        model_config = ModelConfig(
            name=f"{self.config.model_type.value}_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
            model_type=self.config.model_type,
            target_type=self.config.target_type,
            hyperparameters=self.config.model_params,
            scale_features=self.config.scale_features,
        )
        
        # Create and train model
        self.model = ModelFactory.create(model_config)
        history = self.model.fit(X_train, y_train, X_val, y_val)
        
        return history
    
    def _cross_validate(
        self,
        X: np.ndarray,
        y: np.ndarray,
        returns: np.ndarray
    ) -> EvaluationResult:
        """Perform cross-validation."""
        cv = TimeSeriesCrossValidator(
            n_splits=self.config.cv_splits,
            embargo_pct=0.01
        )
        
        model_class = ModelFactory._models[self.config.model_type]
        model_config = ModelConfig(
            name="cv_model",
            model_type=self.config.model_type,
            target_type=self.config.target_type,
            hyperparameters=self.config.model_params,
            scale_features=self.config.scale_features,
        )
        
        return self.evaluator.cross_validate(
            model_class, model_config, X, y, cv, returns
        )
    
    def _walk_forward_analysis(
        self,
        X: np.ndarray,
        y: np.ndarray,
        returns: np.ndarray
    ) -> Dict[str, Any]:
        """Perform walk-forward analysis."""
        model_class = ModelFactory._models[self.config.model_type]
        model_config = ModelConfig(
            name="wf_model",
            model_type=self.config.model_type,
            target_type=self.config.target_type,
            hyperparameters=self.config.model_params,
            scale_features=self.config.scale_features,
        )
        
        return self.evaluator.walk_forward_analysis(
            model_class, model_config, X, y,
            train_size=int(len(X) * 0.5),
            test_size=int(len(X) * 0.05),
            step_size=int(len(X) * 0.02),
            returns=returns
        )
    
    def _save_model(self, symbol: str) -> str:
        """Save trained model to disk."""
        model_dir = Path(self.config.model_dir)
        model_dir.mkdir(parents=True, exist_ok=True)
        
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        filename = f"{symbol}_{self.config.model_type.value}_{timestamp}.pkl"
        path = model_dir / filename
        
        self.model.save(str(path))
        
        # Save metadata
        meta_path = path.with_suffix('.json')
        metadata = {
            'symbol': symbol,
            'model_type': self.config.model_type.value,
            'target_type': self.config.target_type.value,
            'feature_names': self.feature_names,
            'created_at': datetime.now().isoformat(),
        }
        with open(meta_path, 'w') as f:
            json.dump(metadata, f, indent=2)
        
        return str(path)


# =============================================================================
# PIPELINE MANAGER
# =============================================================================

class PipelineManager:
    """
    Manages training pipelines and jobs.
    
    Provides job queue, status tracking, and result retrieval.
    """
    
    def __init__(self):
        self._jobs: Dict[str, TrainingJob] = {}
        self._results: Dict[str, Dict[str, Any]] = {}
    
    def create_job(self, config: PipelineConfig) -> str:
        """Create a new training job."""
        job_id = str(uuid.uuid4())
        
        job = TrainingJob(
            job_id=job_id,
            status=TrainingStatus.QUEUED,
            config=config,
            created_at=datetime.now()
        )
        
        self._jobs[job_id] = job
        
        return job_id
    
    def run_job(
        self,
        job_id: str,
        data: pd.DataFrame,
        symbol: str
    ) -> Dict[str, Any]:
        """Run a training job."""
        if job_id not in self._jobs:
            raise ValueError(f"Job not found: {job_id}")
        
        job = self._jobs[job_id]
        job.status = TrainingStatus.RUNNING
        job.started_at = datetime.now()
        
        def progress_callback(progress: float, message: str):
            job.progress = progress
            job.message = message
        
        try:
            pipeline = TrainingPipeline(job.config)
            result = pipeline.run(data, symbol, callback=progress_callback)
            
            job.status = TrainingStatus.COMPLETED
            job.completed_at = datetime.now()
            job.result = result
            
            self._results[job_id] = result
            
            return result
            
        except Exception as e:
            job.status = TrainingStatus.FAILED
            job.completed_at = datetime.now()
            job.error = str(e)
            
            return {'status': 'error', 'error': str(e)}
    
    def get_job_status(self, job_id: str) -> Dict[str, Any]:
        """Get job status."""
        if job_id not in self._jobs:
            return {'error': 'Job not found'}
        
        job = self._jobs[job_id]
        
        return {
            'job_id': job.job_id,
            'status': job.status.value,
            'progress': job.progress,
            'message': job.message,
            'created_at': job.created_at.isoformat(),
            'started_at': job.started_at.isoformat() if job.started_at else None,
            'completed_at': job.completed_at.isoformat() if job.completed_at else None,
            'error': job.error,
        }
    
    def get_result(self, job_id: str) -> Optional[Dict[str, Any]]:
        """Get job result."""
        return self._results.get(job_id)
    
    def list_jobs(self) -> List[Dict[str, Any]]:
        """List all jobs."""
        return [self.get_job_status(jid) for jid in self._jobs]


# =============================================================================
# ALL EXPORTS
# =============================================================================

__all__ = [
    "PipelineConfig",
    "TrainingJob",
    "TrainingPipeline",
    "PipelineManager",
]
