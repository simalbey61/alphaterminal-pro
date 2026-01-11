"""
AlphaTerminal Pro - Strategy Discovery
======================================

Automatic trading strategy discovery using ML.

Author: AlphaTerminal Team
Version: 1.0.0
"""

import logging
from datetime import datetime
from typing import Optional, Dict, List, Any, Tuple
from dataclasses import dataclass, field
import numpy as np
import pandas as pd

from app.ml_pipeline.enums import ModelType, PredictionTarget
from app.ml_pipeline.features.feature_engineer import FeatureEngineer
from app.ml_pipeline.models.base_models import ModelFactory, ModelConfig


logger = logging.getLogger(__name__)


# =============================================================================
# TRADING RULE
# =============================================================================

@dataclass
class TradingRule:
    """Single trading rule extracted from decision tree."""
    
    conditions: List[Dict[str, Any]]  # List of conditions
    prediction: int  # 0 = sell/short, 1 = buy/long
    confidence: float  # Confidence score
    support: int  # Number of samples supporting this rule
    
    # Performance metrics
    win_rate: Optional[float] = None
    avg_return: Optional[float] = None
    profit_factor: Optional[float] = None
    
    def to_string(self, feature_names: List[str] = None) -> str:
        """Convert rule to human-readable string."""
        parts = []
        
        for cond in self.conditions:
            feature = cond['feature']
            if feature_names and isinstance(feature, int):
                feature = feature_names[feature]
            
            threshold = cond['threshold']
            operator = '<=' if cond['direction'] == 'left' else '>'
            
            parts.append(f"{feature} {operator} {threshold:.4f}")
        
        direction = "BUY" if self.prediction == 1 else "SELL"
        
        return f"IF {' AND '.join(parts)} THEN {direction} (confidence: {self.confidence:.1%})"
    
    def evaluate(self, row: pd.Series) -> bool:
        """Evaluate if rule applies to a data row."""
        for cond in self.conditions:
            feature = cond['feature']
            threshold = cond['threshold']
            
            value = row[feature] if isinstance(feature, str) else row.iloc[feature]
            
            if cond['direction'] == 'left':
                if value > threshold:
                    return False
            else:
                if value <= threshold:
                    return False
        
        return True


@dataclass
class DiscoveredStrategy:
    """A discovered trading strategy."""
    
    strategy_id: str
    name: str
    rules: List[TradingRule]
    
    # Performance
    accuracy: float
    win_rate: float
    profit_factor: float
    sharpe_ratio: float
    max_drawdown: float
    
    # Metadata
    symbol: str
    interval: str
    training_period: str
    sample_size: int
    
    # Configuration
    feature_names: List[str] = field(default_factory=list)
    model_params: Dict[str, Any] = field(default_factory=dict)
    
    created_at: datetime = field(default_factory=datetime.now)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'strategy_id': self.strategy_id,
            'name': self.name,
            'rules': [
                {
                    'conditions': r.conditions,
                    'prediction': r.prediction,
                    'confidence': r.confidence,
                    'support': r.support,
                    'rule_string': r.to_string(self.feature_names)
                }
                for r in self.rules
            ],
            'accuracy': self.accuracy,
            'win_rate': self.win_rate,
            'profit_factor': self.profit_factor,
            'sharpe_ratio': self.sharpe_ratio,
            'max_drawdown': self.max_drawdown,
            'symbol': self.symbol,
            'interval': self.interval,
            'training_period': self.training_period,
            'sample_size': self.sample_size,
            'created_at': self.created_at.isoformat(),
        }


# =============================================================================
# RULE EXTRACTOR
# =============================================================================

class RuleExtractor:
    """
    Extract interpretable rules from tree-based models.
    
    Converts decision tree paths into trading rules.
    """
    
    def __init__(self, max_depth: int = 5, min_samples: int = 20):
        """
        Initialize rule extractor.
        
        Args:
            max_depth: Maximum rule depth
            min_samples: Minimum samples for rule
        """
        self.max_depth = max_depth
        self.min_samples = min_samples
    
    def extract_from_tree(
        self,
        tree_model,
        feature_names: List[str],
        X: np.ndarray,
        y: np.ndarray
    ) -> List[TradingRule]:
        """
        Extract rules from decision tree.
        
        Args:
            tree_model: Fitted decision tree
            feature_names: Feature names
            X: Training features
            y: Training targets
            
        Returns:
            List of TradingRule objects
        """
        rules = []
        
        # Get tree structure
        tree = tree_model.tree_
        
        # Extract paths
        paths = self._get_all_paths(tree)
        
        for path in paths:
            if len(path['conditions']) > self.max_depth:
                continue
            
            if path['samples'] < self.min_samples:
                continue
            
            # Calculate performance
            mask = self._apply_conditions(X, path['conditions'])
            if mask.sum() < self.min_samples:
                continue
            
            # Get prediction and confidence
            leaf_values = y[mask]
            prediction = int(np.round(leaf_values.mean()))
            confidence = (leaf_values == prediction).mean()
            
            rule = TradingRule(
                conditions=path['conditions'],
                prediction=prediction,
                confidence=confidence,
                support=int(mask.sum())
            )
            
            rules.append(rule)
        
        # Sort by confidence
        rules.sort(key=lambda r: r.confidence, reverse=True)
        
        return rules
    
    def _get_all_paths(self, tree) -> List[Dict[str, Any]]:
        """Get all paths from root to leaves."""
        paths = []
        
        def traverse(node_id: int, conditions: List[Dict], depth: int):
            # Check if leaf
            if tree.children_left[node_id] == tree.children_right[node_id]:
                paths.append({
                    'conditions': conditions.copy(),
                    'samples': tree.n_node_samples[node_id],
                    'value': tree.value[node_id]
                })
                return
            
            if depth >= self.max_depth:
                return
            
            feature = tree.feature[node_id]
            threshold = tree.threshold[node_id]
            
            # Left child (<=)
            left_cond = conditions + [{
                'feature': feature,
                'threshold': threshold,
                'direction': 'left'
            }]
            traverse(tree.children_left[node_id], left_cond, depth + 1)
            
            # Right child (>)
            right_cond = conditions + [{
                'feature': feature,
                'threshold': threshold,
                'direction': 'right'
            }]
            traverse(tree.children_right[node_id], right_cond, depth + 1)
        
        traverse(0, [], 0)
        
        return paths
    
    def _apply_conditions(
        self,
        X: np.ndarray,
        conditions: List[Dict]
    ) -> np.ndarray:
        """Apply conditions to get matching rows."""
        mask = np.ones(len(X), dtype=bool)
        
        for cond in conditions:
            feature = cond['feature']
            threshold = cond['threshold']
            
            if cond['direction'] == 'left':
                mask &= (X[:, feature] <= threshold)
            else:
                mask &= (X[:, feature] > threshold)
        
        return mask
    
    def extract_from_forest(
        self,
        forest_model,
        feature_names: List[str],
        X: np.ndarray,
        y: np.ndarray,
        top_n: int = 20
    ) -> List[TradingRule]:
        """
        Extract rules from random forest.
        
        Aggregates rules from multiple trees.
        """
        all_rules = []
        
        for tree in forest_model.estimators_:
            rules = self.extract_from_tree(tree, feature_names, X, y)
            all_rules.extend(rules[:5])  # Top 5 from each tree
        
        # Deduplicate and sort
        unique_rules = self._deduplicate_rules(all_rules)
        unique_rules.sort(key=lambda r: r.confidence * r.support, reverse=True)
        
        return unique_rules[:top_n]
    
    def _deduplicate_rules(
        self,
        rules: List[TradingRule]
    ) -> List[TradingRule]:
        """Remove duplicate rules."""
        seen = set()
        unique = []
        
        for rule in rules:
            # Create hash from conditions
            cond_str = str(sorted([
                (c['feature'], c['threshold'], c['direction'])
                for c in rule.conditions
            ]))
            
            if cond_str not in seen:
                seen.add(cond_str)
                unique.append(rule)
        
        return unique


# =============================================================================
# STRATEGY DISCOVERER
# =============================================================================

class StrategyDiscoverer:
    """
    Discover profitable trading strategies using ML.
    
    Process:
    1. Engineer features
    2. Train interpretable models
    3. Extract trading rules
    4. Backtest and validate rules
    5. Package into strategy
    """
    
    def __init__(
        self,
        max_depth: int = 5,
        min_samples: int = 30,
        min_confidence: float = 0.55,
        min_profit_factor: float = 1.2
    ):
        """
        Initialize strategy discoverer.
        
        Args:
            max_depth: Maximum rule depth
            min_samples: Minimum samples per rule
            min_confidence: Minimum rule confidence
            min_profit_factor: Minimum profit factor
        """
        self.max_depth = max_depth
        self.min_samples = min_samples
        self.min_confidence = min_confidence
        self.min_profit_factor = min_profit_factor
        
        self.feature_engineer = FeatureEngineer()
        self.rule_extractor = RuleExtractor(max_depth, min_samples)
    
    def discover(
        self,
        data: pd.DataFrame,
        symbol: str,
        interval: str = "1d",
        n_strategies: int = 5
    ) -> List[DiscoveredStrategy]:
        """
        Discover trading strategies from data.
        
        Args:
            data: OHLCV DataFrame
            symbol: Trading symbol
            interval: Data interval
            n_strategies: Number of strategies to return
            
        Returns:
            List of DiscoveredStrategy objects
        """
        logger.info(f"Starting strategy discovery for {symbol}")
        
        # Step 1: Feature Engineering
        features_df = self.feature_engineer.calculate_all(data)
        
        # Step 2: Create target
        returns = data['Close'].pct_change().shift(-1)
        target = (returns > 0).astype(int)
        
        # Step 3: Prepare data
        exclude_cols = ['Open', 'High', 'Low', 'Close', 'Volume', 'Adj Close']
        feature_cols = [c for c in features_df.columns if c not in exclude_cols]
        
        df = features_df[feature_cols].copy()
        df['target'] = target
        df['returns'] = returns
        df = df.dropna()
        
        X = df[feature_cols].values
        y = df['target'].values
        returns_arr = df['returns'].values
        
        # Step 4: Train models
        strategies = []
        
        # Try different model configurations
        configs = [
            {'max_depth': 3, 'min_samples_leaf': 30},
            {'max_depth': 4, 'min_samples_leaf': 20},
            {'max_depth': 5, 'min_samples_leaf': 15},
        ]
        
        for i, params in enumerate(configs):
            try:
                strategy = self._discover_single(
                    X, y, returns_arr, feature_cols,
                    params, symbol, interval, len(df), i
                )
                
                if strategy and self._validate_strategy(strategy):
                    strategies.append(strategy)
                    
            except Exception as e:
                logger.warning(f"Strategy discovery failed for config {i}: {e}")
        
        # Sort by Sharpe ratio
        strategies.sort(key=lambda s: s.sharpe_ratio, reverse=True)
        
        logger.info(f"Discovered {len(strategies)} valid strategies")
        
        return strategies[:n_strategies]
    
    def _discover_single(
        self,
        X: np.ndarray,
        y: np.ndarray,
        returns: np.ndarray,
        feature_names: List[str],
        model_params: Dict[str, Any],
        symbol: str,
        interval: str,
        sample_size: int,
        index: int
    ) -> Optional[DiscoveredStrategy]:
        """Discover single strategy with given params."""
        from sklearn.tree import DecisionTreeClassifier
        from sklearn.model_selection import train_test_split
        
        # Split data
        X_train, X_test, y_train, y_test, ret_train, ret_test = train_test_split(
            X, y, returns, test_size=0.3, shuffle=False
        )
        
        # Train decision tree
        model = DecisionTreeClassifier(
            random_state=42,
            **model_params
        )
        model.fit(X_train, y_train)
        
        # Extract rules
        rules = self.rule_extractor.extract_from_tree(
            model, feature_names, X_train, y_train
        )
        
        # Filter by confidence
        rules = [r for r in rules if r.confidence >= self.min_confidence]
        
        if not rules:
            return None
        
        # Evaluate on test set
        y_pred = model.predict(X_test)
        accuracy = (y_pred == y_test).mean()
        
        # Calculate trading metrics
        positions = np.where(y_pred == 1, 1, -1)
        strategy_returns = ret_test * positions
        
        wins = (strategy_returns > 0).sum()
        total = (strategy_returns != 0).sum()
        win_rate = wins / (total + 1e-10)
        
        gains = strategy_returns[strategy_returns > 0].sum()
        losses = abs(strategy_returns[strategy_returns < 0].sum())
        profit_factor = gains / (losses + 1e-10)
        
        # Sharpe ratio
        if strategy_returns.std() > 0:
            sharpe = strategy_returns.mean() / strategy_returns.std() * np.sqrt(252)
        else:
            sharpe = 0
        
        # Max drawdown
        cumulative = (1 + strategy_returns).cumprod()
        rolling_max = np.maximum.accumulate(cumulative)
        drawdowns = (cumulative - rolling_max) / (rolling_max + 1e-10)
        max_dd = abs(drawdowns.min())
        
        # Create strategy
        import uuid
        strategy_id = str(uuid.uuid4())[:8]
        
        strategy = DiscoveredStrategy(
            strategy_id=strategy_id,
            name=f"AutoStrategy_{symbol}_{strategy_id}",
            rules=rules[:10],  # Top 10 rules
            accuracy=accuracy,
            win_rate=win_rate,
            profit_factor=profit_factor,
            sharpe_ratio=sharpe,
            max_drawdown=max_dd,
            symbol=symbol,
            interval=interval,
            training_period=f"{sample_size} bars",
            sample_size=sample_size,
            feature_names=feature_names,
            model_params=model_params
        )
        
        return strategy
    
    def _validate_strategy(self, strategy: DiscoveredStrategy) -> bool:
        """Validate strategy meets minimum criteria."""
        if strategy.accuracy < 0.5:
            return False
        
        if strategy.profit_factor < self.min_profit_factor:
            return False
        
        if strategy.sharpe_ratio < 0:
            return False
        
        if len(strategy.rules) < 1:
            return False
        
        return True


# =============================================================================
# ALL EXPORTS
# =============================================================================

__all__ = [
    "TradingRule",
    "DiscoveredStrategy",
    "RuleExtractor",
    "StrategyDiscoverer",
]
