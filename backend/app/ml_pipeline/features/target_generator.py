"""
AlphaTerminal Pro - Target Generator
====================================

Generate prediction targets for ML models.

Author: AlphaTerminal Team
Version: 1.0.0
"""

import logging
from typing import Optional, Dict, List, Any, Union, Tuple
from dataclasses import dataclass
import numpy as np
import pandas as pd

from app.ml_pipeline.enums import PredictionTarget


logger = logging.getLogger(__name__)


# =============================================================================
# TARGET CONFIGURATION
# =============================================================================

@dataclass
class TargetConfig:
    """Configuration for target generation."""
    target_type: PredictionTarget
    horizon: int = 1  # Prediction horizon in bars
    threshold: float = 0.0  # Threshold for classification
    use_log_returns: bool = True
    
    # For multi-class
    num_classes: int = 2
    class_thresholds: Optional[List[float]] = None
    
    # For volatility targets
    vol_window: int = 20


# =============================================================================
# TARGET GENERATOR
# =============================================================================

class TargetGenerator:
    """
    Generate prediction targets from OHLCV data.
    
    Supports various target types:
    - Binary classification (direction)
    - Multi-class classification (regime)
    - Regression (returns, volatility)
    """
    
    def __init__(self, config: Optional[TargetConfig] = None):
        """
        Initialize target generator.
        
        Args:
            config: Target configuration
        """
        self.config = config or TargetConfig(target_type=PredictionTarget.DIRECTION)
    
    def generate(
        self,
        df: pd.DataFrame,
        target_type: Optional[PredictionTarget] = None,
        horizon: Optional[int] = None,
        **kwargs
    ) -> Tuple[pd.Series, Dict[str, Any]]:
        """
        Generate target variable.
        
        Args:
            df: OHLCV DataFrame
            target_type: Override target type
            horizon: Override prediction horizon
            **kwargs: Additional parameters
            
        Returns:
            Tuple of (target series, metadata dict)
        """
        target_type = target_type or self.config.target_type
        horizon = horizon or self.config.horizon
        
        generators = {
            PredictionTarget.DIRECTION: self._generate_direction,
            PredictionTarget.DIRECTION_3WAY: self._generate_direction_3way,
            PredictionTarget.RETURN: self._generate_return,
            PredictionTarget.VOLATILITY: self._generate_volatility,
            PredictionTarget.REGIME: self._generate_regime,
            PredictionTarget.PRICE: self._generate_price,
        }
        
        if target_type not in generators:
            raise ValueError(f"Unsupported target type: {target_type}")
        
        return generators[target_type](df, horizon, **kwargs)
    
    def _generate_direction(
        self,
        df: pd.DataFrame,
        horizon: int,
        threshold: Optional[float] = None,
        **kwargs
    ) -> Tuple[pd.Series, Dict[str, Any]]:
        """
        Generate binary direction target.
        
        1 = price goes up
        0 = price goes down
        """
        threshold = threshold if threshold is not None else self.config.threshold
        
        if self.config.use_log_returns:
            future_return = np.log(df['Close'].shift(-horizon) / df['Close'])
        else:
            future_return = df['Close'].pct_change(horizon).shift(-horizon)
        
        # Binary classification
        target = (future_return > threshold).astype(int)
        
        # Metadata
        meta = {
            "target_type": PredictionTarget.DIRECTION.value,
            "horizon": horizon,
            "threshold": threshold,
            "class_distribution": target.value_counts(normalize=True).to_dict(),
            "positive_rate": target.mean(),
        }
        
        return target, meta
    
    def _generate_direction_3way(
        self,
        df: pd.DataFrame,
        horizon: int,
        thresholds: Optional[List[float]] = None,
        **kwargs
    ) -> Tuple[pd.Series, Dict[str, Any]]:
        """
        Generate 3-way direction target.
        
        2 = strong up
        1 = neutral / sideways
        0 = strong down
        """
        thresholds = thresholds or self.config.class_thresholds or [-0.01, 0.01]
        
        if self.config.use_log_returns:
            future_return = np.log(df['Close'].shift(-horizon) / df['Close'])
        else:
            future_return = df['Close'].pct_change(horizon).shift(-horizon)
        
        # 3-way classification
        target = pd.Series(1, index=df.index)  # Default neutral
        target[future_return <= thresholds[0]] = 0  # Down
        target[future_return >= thresholds[1]] = 2  # Up
        
        meta = {
            "target_type": PredictionTarget.DIRECTION_3WAY.value,
            "horizon": horizon,
            "thresholds": thresholds,
            "class_distribution": target.value_counts(normalize=True).to_dict(),
        }
        
        return target, meta
    
    def _generate_return(
        self,
        df: pd.DataFrame,
        horizon: int,
        **kwargs
    ) -> Tuple[pd.Series, Dict[str, Any]]:
        """
        Generate return regression target.
        """
        if self.config.use_log_returns:
            target = np.log(df['Close'].shift(-horizon) / df['Close'])
        else:
            target = df['Close'].pct_change(horizon).shift(-horizon)
        
        meta = {
            "target_type": PredictionTarget.RETURN.value,
            "horizon": horizon,
            "mean": target.mean(),
            "std": target.std(),
            "min": target.min(),
            "max": target.max(),
        }
        
        return target, meta
    
    def _generate_volatility(
        self,
        df: pd.DataFrame,
        horizon: int,
        window: Optional[int] = None,
        **kwargs
    ) -> Tuple[pd.Series, Dict[str, Any]]:
        """
        Generate volatility regression target.
        
        Predicts future realized volatility.
        """
        window = window or self.config.vol_window
        
        # Calculate future realized volatility
        log_returns = np.log(df['Close'] / df['Close'].shift(1))
        
        # Future volatility (shifted back)
        future_vol = log_returns.rolling(window).std().shift(-horizon) * np.sqrt(252)
        
        meta = {
            "target_type": PredictionTarget.VOLATILITY.value,
            "horizon": horizon,
            "window": window,
            "mean_vol": future_vol.mean(),
            "std_vol": future_vol.std(),
        }
        
        return future_vol, meta
    
    def _generate_regime(
        self,
        df: pd.DataFrame,
        horizon: int,
        n_regimes: int = 3,
        **kwargs
    ) -> Tuple[pd.Series, Dict[str, Any]]:
        """
        Generate market regime target.
        
        Based on returns and volatility:
        0 = Bear (low returns, high vol)
        1 = Neutral (mixed)
        2 = Bull (high returns, low vol)
        """
        returns = df['Close'].pct_change(20)
        volatility = returns.rolling(20).std()
        
        # Normalize
        returns_z = (returns - returns.mean()) / returns.std()
        vol_z = (volatility - volatility.mean()) / volatility.std()
        
        # Simple regime classification
        score = returns_z - 0.5 * vol_z
        
        # Quantile-based classification
        target = pd.qcut(score, n_regimes, labels=False, duplicates='drop')
        target = target.shift(-horizon)  # Future regime
        
        meta = {
            "target_type": PredictionTarget.REGIME.value,
            "horizon": horizon,
            "n_regimes": n_regimes,
            "class_distribution": target.value_counts(normalize=True).to_dict() if not target.isna().all() else {},
        }
        
        return target, meta
    
    def _generate_price(
        self,
        df: pd.DataFrame,
        horizon: int,
        **kwargs
    ) -> Tuple[pd.Series, Dict[str, Any]]:
        """
        Generate price prediction target.
        """
        target = df['Close'].shift(-horizon)
        
        meta = {
            "target_type": PredictionTarget.PRICE.value,
            "horizon": horizon,
            "mean_price": target.mean(),
            "std_price": target.std(),
        }
        
        return target, meta
    
    # =========================================================================
    # UTILITY METHODS
    # =========================================================================
    
    def create_labels(
        self,
        df: pd.DataFrame,
        method: str = "triple_barrier",
        **kwargs
    ) -> Tuple[pd.Series, pd.Series]:
        """
        Create labels using triple barrier method.
        
        Returns:
            Tuple of (labels, returns)
        """
        if method == "triple_barrier":
            return self._triple_barrier_labels(df, **kwargs)
        else:
            raise ValueError(f"Unknown labeling method: {method}")
    
    def _triple_barrier_labels(
        self,
        df: pd.DataFrame,
        profit_taking: float = 0.02,
        stop_loss: float = 0.02,
        max_holding: int = 10,
        **kwargs
    ) -> Tuple[pd.Series, pd.Series]:
        """
        Triple barrier labeling method.
        
        Creates labels based on which barrier is hit first:
        - Upper barrier (profit taking)
        - Lower barrier (stop loss)
        - Vertical barrier (time limit)
        
        Args:
            df: OHLCV DataFrame
            profit_taking: Profit taking threshold
            stop_loss: Stop loss threshold
            max_holding: Maximum holding period
            
        Returns:
            Tuple of (labels, returns)
        """
        close = df['Close']
        n = len(df)
        
        labels = pd.Series(index=df.index, dtype=float)
        returns = pd.Series(index=df.index, dtype=float)
        
        for i in range(n - max_holding):
            entry_price = close.iloc[i]
            upper = entry_price * (1 + profit_taking)
            lower = entry_price * (1 - stop_loss)
            
            # Look forward
            future_prices = close.iloc[i+1:i+max_holding+1]
            
            # Check barriers
            upper_hit = (future_prices >= upper).idxmax() if (future_prices >= upper).any() else None
            lower_hit = (future_prices <= lower).idxmax() if (future_prices <= lower).any() else None
            vertical_hit = future_prices.index[-1] if len(future_prices) > 0 else None
            
            # Determine which barrier was hit first
            if upper_hit and lower_hit:
                if upper_hit <= lower_hit:
                    labels.iloc[i] = 1  # Win
                    returns.iloc[i] = profit_taking
                else:
                    labels.iloc[i] = -1  # Loss
                    returns.iloc[i] = -stop_loss
            elif upper_hit:
                labels.iloc[i] = 1
                returns.iloc[i] = profit_taking
            elif lower_hit:
                labels.iloc[i] = -1
                returns.iloc[i] = -stop_loss
            elif vertical_hit:
                # Time barrier hit
                exit_price = close.loc[vertical_hit]
                ret = (exit_price - entry_price) / entry_price
                labels.iloc[i] = 1 if ret > 0 else (-1 if ret < 0 else 0)
                returns.iloc[i] = ret
        
        return labels, returns
    
    def balance_classes(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        method: str = "undersample"
    ) -> Tuple[pd.DataFrame, pd.Series]:
        """
        Balance class distribution.
        
        Args:
            X: Feature DataFrame
            y: Target Series
            method: 'undersample' or 'oversample'
            
        Returns:
            Balanced X and y
        """
        # Remove NaN
        mask = ~y.isna()
        X = X[mask]
        y = y[mask]
        
        if method == "undersample":
            # Undersample majority class
            class_counts = y.value_counts()
            min_count = class_counts.min()
            
            indices = []
            for cls in y.unique():
                cls_indices = y[y == cls].index
                sampled = np.random.choice(cls_indices, size=min_count, replace=False)
                indices.extend(sampled)
            
            return X.loc[indices], y.loc[indices]
        
        elif method == "oversample":
            # Oversample minority class
            class_counts = y.value_counts()
            max_count = class_counts.max()
            
            dfs = []
            for cls in y.unique():
                cls_X = X[y == cls]
                cls_y = y[y == cls]
                
                if len(cls_y) < max_count:
                    # Sample with replacement
                    indices = np.random.choice(cls_y.index, size=max_count, replace=True)
                    dfs.append((X.loc[indices], y.loc[indices]))
                else:
                    dfs.append((cls_X, cls_y))
            
            X_balanced = pd.concat([d[0] for d in dfs])
            y_balanced = pd.concat([d[1] for d in dfs])
            
            return X_balanced, y_balanced
        
        else:
            raise ValueError(f"Unknown balancing method: {method}")


# =============================================================================
# ALL EXPORTS
# =============================================================================

__all__ = [
    "TargetConfig",
    "TargetGenerator",
]
