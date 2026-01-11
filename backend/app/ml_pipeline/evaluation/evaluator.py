"""
AlphaTerminal Pro - Model Evaluation
====================================

Comprehensive model evaluation and cross-validation.

Author: AlphaTerminal Team
Version: 1.0.0
"""

import logging
import time
from datetime import datetime
from typing import Optional, Dict, List, Any, Tuple, Callable
from dataclasses import dataclass, field
import numpy as np
import pandas as pd

from app.ml_pipeline.enums import PredictionTarget


logger = logging.getLogger(__name__)


# =============================================================================
# EVALUATION RESULT
# =============================================================================

@dataclass
class EvaluationResult:
    """Results from model evaluation."""
    
    # Basic metrics
    accuracy: Optional[float] = None
    precision: Optional[float] = None
    recall: Optional[float] = None
    f1_score: Optional[float] = None
    
    # Classification specific
    roc_auc: Optional[float] = None
    log_loss: Optional[float] = None
    confusion_matrix: Optional[np.ndarray] = None
    
    # Regression specific
    mse: Optional[float] = None
    rmse: Optional[float] = None
    mae: Optional[float] = None
    r2: Optional[float] = None
    
    # Trading specific
    directional_accuracy: Optional[float] = None
    profit_factor: Optional[float] = None
    sharpe_ratio: Optional[float] = None
    max_drawdown: Optional[float] = None
    win_rate: Optional[float] = None
    
    # Cross-validation
    cv_scores: Optional[List[float]] = None
    cv_mean: Optional[float] = None
    cv_std: Optional[float] = None
    
    # Metadata
    n_samples: int = 0
    evaluation_time: float = 0.0
    
    def to_dict(self) -> Dict[str, Any]:
        result = {}
        for k, v in self.__dict__.items():
            if v is not None:
                if isinstance(v, np.ndarray):
                    result[k] = v.tolist()
                else:
                    result[k] = v
        return result


# =============================================================================
# METRICS CALCULATOR
# =============================================================================

class MetricsCalculator:
    """Calculate various evaluation metrics."""
    
    @staticmethod
    def classification_metrics(
        y_true: np.ndarray,
        y_pred: np.ndarray,
        y_proba: Optional[np.ndarray] = None
    ) -> Dict[str, Any]:
        """Calculate classification metrics."""
        from sklearn.metrics import (
            accuracy_score, precision_score, recall_score, f1_score,
            roc_auc_score, log_loss, confusion_matrix
        )
        
        metrics = {
            'accuracy': accuracy_score(y_true, y_pred),
            'precision': precision_score(y_true, y_pred, average='weighted', zero_division=0),
            'recall': recall_score(y_true, y_pred, average='weighted', zero_division=0),
            'f1_score': f1_score(y_true, y_pred, average='weighted', zero_division=0),
        }
        
        if y_proba is not None:
            try:
                if y_proba.ndim == 1 or y_proba.shape[1] == 2:
                    proba = y_proba[:, 1] if y_proba.ndim > 1 else y_proba
                    metrics['roc_auc'] = roc_auc_score(y_true, proba)
                else:
                    metrics['roc_auc'] = roc_auc_score(
                        y_true, y_proba, multi_class='ovr', average='weighted'
                    )
                metrics['log_loss'] = log_loss(y_true, y_proba)
            except Exception as e:
                logger.warning(f"Could not calculate AUC/log_loss: {e}")
        
        metrics['confusion_matrix'] = confusion_matrix(y_true, y_pred)
        
        return metrics
    
    @staticmethod
    def regression_metrics(
        y_true: np.ndarray,
        y_pred: np.ndarray
    ) -> Dict[str, float]:
        """Calculate regression metrics."""
        from sklearn.metrics import (
            mean_squared_error, mean_absolute_error, r2_score
        )
        
        mse = mean_squared_error(y_true, y_pred)
        
        return {
            'mse': mse,
            'rmse': np.sqrt(mse),
            'mae': mean_absolute_error(y_true, y_pred),
            'r2': r2_score(y_true, y_pred),
        }
    
    @staticmethod
    def trading_metrics(
        y_true: np.ndarray,
        y_pred: np.ndarray,
        returns: Optional[np.ndarray] = None
    ) -> Dict[str, float]:
        """Calculate trading-specific metrics."""
        metrics = {}
        
        if returns is not None:
            true_direction = np.sign(returns)
            
            # Handle different prediction formats
            if set(np.unique(y_pred)).issubset({0, 1}):
                pred_direction = np.where(y_pred == 1, 1, -1)
            elif set(np.unique(y_pred)).issubset({-1, 0, 1}):
                pred_direction = y_pred
            else:
                pred_direction = np.sign(y_pred)
            
            # Directional accuracy
            valid_mask = true_direction != 0
            if valid_mask.sum() > 0:
                directional_correct = (true_direction[valid_mask] == pred_direction[valid_mask]).sum()
                metrics['directional_accuracy'] = directional_correct / valid_mask.sum()
            
            # Strategy returns
            strategy_returns = returns * pred_direction
            
            # Sharpe ratio (annualized)
            if len(strategy_returns) > 0 and strategy_returns.std() > 0:
                metrics['sharpe_ratio'] = (
                    strategy_returns.mean() / strategy_returns.std() * np.sqrt(252)
                )
            
            # Profit factor
            gains = strategy_returns[strategy_returns > 0].sum()
            losses = abs(strategy_returns[strategy_returns < 0].sum())
            metrics['profit_factor'] = gains / (losses + 1e-10)
            
            # Max drawdown
            cumulative = (1 + strategy_returns).cumprod()
            rolling_max = np.maximum.accumulate(cumulative)
            drawdowns = (cumulative - rolling_max) / (rolling_max + 1e-10)
            metrics['max_drawdown'] = abs(drawdowns.min())
            
            # Win rate
            wins = (strategy_returns > 0).sum()
            total = (strategy_returns != 0).sum()
            metrics['win_rate'] = wins / (total + 1e-10)
            
            # Average win/loss
            avg_win = strategy_returns[strategy_returns > 0].mean() if (strategy_returns > 0).any() else 0
            avg_loss = abs(strategy_returns[strategy_returns < 0].mean()) if (strategy_returns < 0).any() else 0
            metrics['avg_win_loss_ratio'] = avg_win / (avg_loss + 1e-10)
        
        return metrics


# =============================================================================
# CROSS-VALIDATION
# =============================================================================

class TimeSeriesCrossValidator:
    """
    Cross-validation for time series data.
    
    Implements various CV strategies that respect temporal order.
    """
    
    def __init__(
        self,
        n_splits: int = 5,
        gap: int = 0,
        test_size: Optional[int] = None,
        embargo_pct: float = 0.0
    ):
        """
        Initialize cross-validator.
        
        Args:
            n_splits: Number of splits
            gap: Gap between train and test
            test_size: Size of test set (None = auto)
            embargo_pct: Embargo percentage for purging
        """
        self.n_splits = n_splits
        self.gap = gap
        self.test_size = test_size
        self.embargo_pct = embargo_pct
    
    def split(
        self,
        X: np.ndarray,
        y: Optional[np.ndarray] = None
    ) -> List[Tuple[np.ndarray, np.ndarray]]:
        """
        Generate train/test indices for each split.
        
        Args:
            X: Features array
            y: Target array (optional)
            
        Returns:
            List of (train_indices, test_indices) tuples
        """
        n_samples = len(X)
        
        if self.test_size is None:
            test_size = n_samples // (self.n_splits + 1)
        else:
            test_size = self.test_size
        
        indices = np.arange(n_samples)
        splits = []
        
        for i in range(self.n_splits):
            test_start = (i + 1) * test_size + self.gap
            test_end = test_start + test_size
            
            if test_end > n_samples:
                break
            
            embargo = int(test_size * self.embargo_pct)
            train_end = test_start - self.gap - embargo
            
            train_indices = indices[:train_end]
            test_indices = indices[test_start:test_end]
            
            splits.append((train_indices, test_indices))
        
        return splits
    
    def get_n_splits(self) -> int:
        return self.n_splits


class WalkForwardValidator:
    """
    Walk-forward validation for trading strategies.
    
    Simulates real trading conditions by training on past data
    and testing on future data in a rolling manner.
    """
    
    def __init__(
        self,
        train_size: int = 252,
        test_size: int = 21,
        step_size: int = 21
    ):
        """
        Initialize walk-forward validator.
        
        Args:
            train_size: Training window size
            test_size: Test window size
            step_size: Step size between windows
        """
        self.train_size = train_size
        self.test_size = test_size
        self.step_size = step_size
    
    def split(
        self,
        X: np.ndarray,
        y: Optional[np.ndarray] = None
    ) -> List[Tuple[np.ndarray, np.ndarray]]:
        """Generate walk-forward splits."""
        n_samples = len(X)
        indices = np.arange(n_samples)
        splits = []
        
        start = 0
        while start + self.train_size + self.test_size <= n_samples:
            train_start = start
            train_end = start + self.train_size
            test_start = train_end
            test_end = test_start + self.test_size
            
            train_indices = indices[train_start:train_end]
            test_indices = indices[test_start:test_end]
            
            splits.append((train_indices, test_indices))
            
            start += self.step_size
        
        return splits
    
    def get_n_splits(self, n_samples: int) -> int:
        """Calculate number of splits for given sample size."""
        return (n_samples - self.train_size - self.test_size) // self.step_size + 1


class ExpandingWindowValidator:
    """
    Expanding window cross-validation.
    
    Training set grows with each split while test set moves forward.
    """
    
    def __init__(
        self,
        initial_train_size: int = 252,
        test_size: int = 21,
        step_size: int = 21
    ):
        """
        Initialize expanding window validator.
        
        Args:
            initial_train_size: Initial training window size
            test_size: Test window size
            step_size: Step size between windows
        """
        self.initial_train_size = initial_train_size
        self.test_size = test_size
        self.step_size = step_size
    
    def split(
        self,
        X: np.ndarray,
        y: Optional[np.ndarray] = None
    ) -> List[Tuple[np.ndarray, np.ndarray]]:
        """Generate expanding window splits."""
        n_samples = len(X)
        indices = np.arange(n_samples)
        splits = []
        
        train_end = self.initial_train_size
        
        while train_end + self.test_size <= n_samples:
            train_indices = indices[:train_end]
            test_indices = indices[train_end:train_end + self.test_size]
            
            splits.append((train_indices, test_indices))
            
            train_end += self.step_size
        
        return splits


# =============================================================================
# MODEL EVALUATOR
# =============================================================================

class ModelEvaluator:
    """
    Comprehensive model evaluation.
    
    Performs various types of evaluation including:
    - Single train/test split
    - Cross-validation
    - Walk-forward analysis
    - Out-of-sample testing
    """
    
    def __init__(
        self,
        target_type: PredictionTarget = PredictionTarget.DIRECTION
    ):
        """
        Initialize evaluator.
        
        Args:
            target_type: Type of prediction target
        """
        self.target_type = target_type
        self.metrics_calculator = MetricsCalculator()
    
    def evaluate(
        self,
        model,
        X_test: np.ndarray,
        y_test: np.ndarray,
        returns: Optional[np.ndarray] = None
    ) -> EvaluationResult:
        """
        Evaluate model on test set.
        
        Args:
            model: Trained model
            X_test: Test features
            y_test: Test targets
            returns: Actual returns for trading metrics
            
        Returns:
            EvaluationResult with all metrics
        """
        start_time = time.time()
        
        # Get predictions
        y_pred = model.predict(X_test)
        
        # Get probabilities if available
        y_proba = None
        if hasattr(model, 'predict_proba'):
            try:
                y_proba = model.predict_proba(X_test)
            except:
                pass
        
        # Remove NaN values
        mask = ~(np.isnan(y_pred) | np.isnan(y_test))
        y_pred = y_pred[mask]
        y_test = y_test[mask]
        if y_proba is not None:
            y_proba = y_proba[mask]
        if returns is not None:
            returns = returns[mask]
        
        result = EvaluationResult(n_samples=len(y_test))
        
        # Classification metrics
        if self.target_type in [PredictionTarget.DIRECTION, 
                                 PredictionTarget.DIRECTION_3WAY,
                                 PredictionTarget.REGIME]:
            metrics = self.metrics_calculator.classification_metrics(
                y_test, y_pred, y_proba
            )
            result.accuracy = metrics.get('accuracy')
            result.precision = metrics.get('precision')
            result.recall = metrics.get('recall')
            result.f1_score = metrics.get('f1_score')
            result.roc_auc = metrics.get('roc_auc')
            result.log_loss = metrics.get('log_loss')
            result.confusion_matrix = metrics.get('confusion_matrix')
        
        # Regression metrics
        if self.target_type in [PredictionTarget.RETURN,
                                 PredictionTarget.VOLATILITY,
                                 PredictionTarget.PRICE]:
            metrics = self.metrics_calculator.regression_metrics(y_test, y_pred)
            result.mse = metrics.get('mse')
            result.rmse = metrics.get('rmse')
            result.mae = metrics.get('mae')
            result.r2 = metrics.get('r2')
        
        # Trading metrics
        if returns is not None:
            metrics = self.metrics_calculator.trading_metrics(y_test, y_pred, returns)
            result.directional_accuracy = metrics.get('directional_accuracy')
            result.profit_factor = metrics.get('profit_factor')
            result.sharpe_ratio = metrics.get('sharpe_ratio')
            result.max_drawdown = metrics.get('max_drawdown')
            result.win_rate = metrics.get('win_rate')
        
        result.evaluation_time = time.time() - start_time
        
        return result
    
    def cross_validate(
        self,
        model_class,
        model_config,
        X: np.ndarray,
        y: np.ndarray,
        cv=None,
        returns: Optional[np.ndarray] = None,
        scoring: str = 'accuracy'
    ) -> EvaluationResult:
        """
        Perform cross-validation.
        
        Args:
            model_class: Model class to instantiate
            model_config: Model configuration
            X: Features
            y: Targets
            cv: Cross-validator (default: TimeSeriesCrossValidator)
            returns: Returns for trading metrics
            scoring: Primary scoring metric
            
        Returns:
            EvaluationResult with CV scores
        """
        if cv is None:
            cv = TimeSeriesCrossValidator(n_splits=5)
        
        scores = []
        all_results = []
        
        for train_idx, test_idx in cv.split(X, y):
            X_train, X_test = X[train_idx], X[test_idx]
            y_train, y_test = y[train_idx], y[test_idx]
            
            ret_test = returns[test_idx] if returns is not None else None
            
            # Create and train model
            model = model_class(model_config)
            model.fit(X_train, y_train)
            
            # Evaluate
            result = self.evaluate(model, X_test, y_test, ret_test)
            all_results.append(result)
            
            # Get primary score
            score = getattr(result, scoring, result.accuracy)
            if score is not None:
                scores.append(score)
        
        # Aggregate results
        final_result = EvaluationResult(
            cv_scores=scores,
            cv_mean=np.mean(scores) if scores else None,
            cv_std=np.std(scores) if scores else None,
        )
        
        # Average other metrics
        for attr in ['accuracy', 'precision', 'recall', 'f1_score', 'roc_auc',
                     'sharpe_ratio', 'profit_factor', 'max_drawdown', 'win_rate']:
            values = [getattr(r, attr) for r in all_results if getattr(r, attr) is not None]
            if values:
                setattr(final_result, attr, np.mean(values))
        
        return final_result
    
    def walk_forward_analysis(
        self,
        model_class,
        model_config,
        X: np.ndarray,
        y: np.ndarray,
        train_size: int = 252,
        test_size: int = 21,
        step_size: int = 21,
        returns: Optional[np.ndarray] = None
    ) -> Dict[str, Any]:
        """
        Perform walk-forward analysis.
        
        Args:
            model_class: Model class
            model_config: Model configuration
            X: Features
            y: Targets
            train_size: Training window
            test_size: Test window
            step_size: Step size
            returns: Returns for trading metrics
            
        Returns:
            Dict with detailed walk-forward results
        """
        validator = WalkForwardValidator(
            train_size=train_size,
            test_size=test_size,
            step_size=step_size
        )
        
        all_predictions = []
        all_actuals = []
        all_returns = []
        window_results = []
        
        splits = validator.split(X, y)
        
        for i, (train_idx, test_idx) in enumerate(splits):
            X_train, X_test = X[train_idx], X[test_idx]
            y_train, y_test = y[train_idx], y[test_idx]
            
            # Train model
            model = model_class(model_config)
            model.fit(X_train, y_train)
            
            # Predict
            y_pred = model.predict(X_test)
            
            all_predictions.extend(y_pred)
            all_actuals.extend(y_test)
            
            if returns is not None:
                ret_test = returns[test_idx]
                all_returns.extend(ret_test)
                
                # Window result
                result = self.evaluate(model, X_test, y_test, ret_test)
                window_results.append({
                    'window': i,
                    'train_start': train_idx[0],
                    'train_end': train_idx[-1],
                    'test_start': test_idx[0],
                    'test_end': test_idx[-1],
                    **result.to_dict()
                })
        
        # Overall metrics
        all_predictions = np.array(all_predictions)
        all_actuals = np.array(all_actuals)
        all_returns = np.array(all_returns) if returns is not None else None
        
        # Create dummy model for overall evaluation
        class DummyModel:
            def __init__(self, predictions):
                self._predictions = predictions
            def predict(self, X):
                return self._predictions[:len(X)]
        
        dummy = DummyModel(all_predictions)
        overall_result = self.evaluate(dummy, np.zeros((len(all_actuals), 1)), 
                                       all_actuals, all_returns)
        
        return {
            'overall': overall_result.to_dict(),
            'windows': window_results,
            'n_windows': len(splits),
            'predictions': all_predictions,
            'actuals': all_actuals,
        }


# =============================================================================
# ALL EXPORTS
# =============================================================================

__all__ = [
    "EvaluationResult",
    "MetricsCalculator",
    "TimeSeriesCrossValidator",
    "WalkForwardValidator",
    "ExpandingWindowValidator",
    "ModelEvaluator",
]
