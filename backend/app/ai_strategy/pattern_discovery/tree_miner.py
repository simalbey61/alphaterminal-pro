"""
AlphaTerminal Pro - Decision Tree Miner
=======================================

Kazanan trade örüntülerini keşfetmek için Decision Tree tabanlı mining.

Author: AlphaTerminal Team
Version: 1.0.0
"""

from __future__ import annotations

import logging
from typing import Optional, List, Dict, Any, Tuple
from datetime import datetime
from dataclasses import dataclass, field

import numpy as np
from scipy import stats

logger = logging.getLogger(__name__)


@dataclass
class TreeRule:
    """Decision tree'den çıkarılan kural."""
    feature: str
    operator: str
    threshold: float
    
    def to_dict(self) -> Dict[str, Any]:
        return {"feature": self.feature, "operator": self.operator, "threshold": self.threshold}
    
    def evaluate(self, features: Dict[str, float]) -> bool:
        if self.feature not in features:
            return False
        value = features[self.feature]
        if self.operator == "<":
            return value < self.threshold
        elif self.operator == ">":
            return value > self.threshold
        elif self.operator == "<=":
            return value <= self.threshold
        elif self.operator == ">=":
            return value >= self.threshold
        return False
    
    def __str__(self) -> str:
        return f"{self.feature} {self.operator} {self.threshold:.4f}"


@dataclass
class DiscoveredPattern:
    """Keşfedilen örüntü."""
    rules: List[TreeRule]
    confidence: float
    support: int
    win_rate: float
    avg_return: float
    sample_size: int
    chi_square_stat: Optional[float] = None
    chi_square_pvalue: Optional[float] = None
    is_significant: bool = False
    depth: int = 0
    discovered_at: datetime = field(default_factory=datetime.utcnow)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "rules": [r.to_dict() for r in self.rules],
            "confidence": self.confidence,
            "support": self.support,
            "win_rate": self.win_rate,
            "avg_return": self.avg_return,
            "sample_size": self.sample_size,
            "chi_square_stat": self.chi_square_stat,
            "chi_square_pvalue": self.chi_square_pvalue,
            "is_significant": self.is_significant,
            "depth": self.depth,
        }
    
    def evaluate(self, features: Dict[str, float]) -> bool:
        return all(rule.evaluate(features) for rule in self.rules)
    
    def __str__(self) -> str:
        rules_str = " AND ".join(str(r) for r in self.rules)
        return f"Pattern [{rules_str}] (WR: {self.win_rate:.1%}, Support: {self.support})"


class DecisionTreeMiner:
    """
    Kazanan trade örüntülerini keşfeden Decision Tree miner.
    
    Example:
        ```python
        miner = DecisionTreeMiner()
        patterns = miner.mine(trades_df, features_df)
        ```
    """
    
    def __init__(
        self,
        min_samples_leaf: int = 50,
        max_depth: int = 5,
        min_info_gain: float = 0.05,
        significance_level: float = 0.05,
        min_win_rate: float = 0.55,
    ):
        self.min_samples_leaf = min_samples_leaf
        self.max_depth = max_depth
        self.min_info_gain = min_info_gain
        self.significance_level = significance_level
        self.min_win_rate = min_win_rate
    
    def mine(
        self,
        features: np.ndarray,
        labels: np.ndarray,
        feature_names: List[str],
        returns: Optional[np.ndarray] = None,
    ) -> List[DiscoveredPattern]:
        """Örüntüleri keşfet."""
        if len(features) < self.min_samples_leaf * 2:
            logger.warning(f"Insufficient samples: {len(features)}")
            return []
        
        patterns = []
        self._recursive_mine(
            features=features,
            labels=labels,
            feature_names=feature_names,
            returns=returns,
            current_rules=[],
            depth=0,
            patterns=patterns,
        )
        
        significant_patterns = [
            p for p in patterns
            if p.is_significant and p.win_rate >= self.min_win_rate
        ]
        significant_patterns.sort(key=lambda p: p.win_rate, reverse=True)
        
        logger.info(f"Mining complete: {len(patterns)} patterns, {len(significant_patterns)} significant")
        return significant_patterns
    
    def _recursive_mine(
        self,
        features: np.ndarray,
        labels: np.ndarray,
        feature_names: List[str],
        returns: Optional[np.ndarray],
        current_rules: List[TreeRule],
        depth: int,
        patterns: List[DiscoveredPattern],
    ) -> None:
        """Recursive örüntü keşfi."""
        n_samples = len(labels)
        
        if depth >= self.max_depth or n_samples < self.min_samples_leaf * 2:
            return
        
        if current_rules:
            pattern = self._create_pattern(labels, returns, current_rules, depth)
            if pattern:
                patterns.append(pattern)
        
        best_split = self._find_best_split(features, labels, feature_names)
        if best_split is None:
            return
        
        feature_idx, threshold, info_gain = best_split
        feature_name = feature_names[feature_idx]
        
        if info_gain < self.min_info_gain:
            return
        
        left_mask = features[:, feature_idx] <= threshold
        right_mask = ~left_mask
        
        if left_mask.sum() >= self.min_samples_leaf:
            left_rule = TreeRule(feature=feature_name, operator="<=", threshold=threshold)
            self._recursive_mine(
                features[left_mask], labels[left_mask], feature_names,
                returns[left_mask] if returns is not None else None,
                current_rules + [left_rule], depth + 1, patterns,
            )
        
        if right_mask.sum() >= self.min_samples_leaf:
            right_rule = TreeRule(feature=feature_name, operator=">", threshold=threshold)
            self._recursive_mine(
                features[right_mask], labels[right_mask], feature_names,
                returns[right_mask] if returns is not None else None,
                current_rules + [right_rule], depth + 1, patterns,
            )
    
    def _find_best_split(
        self,
        features: np.ndarray,
        labels: np.ndarray,
        feature_names: List[str],
    ) -> Optional[Tuple[int, float, float]]:
        """En iyi split noktasını bul."""
        best_gain = -1
        best_split = None
        base_entropy = self._entropy(labels)
        
        for feat_idx in range(features.shape[1]):
            feat_values = features[:, feat_idx]
            unique_values = np.unique(feat_values)
            
            if len(unique_values) < 2:
                continue
            
            thresholds = (unique_values[:-1] + unique_values[1:]) / 2
            
            for thresh in thresholds:
                left_mask = feat_values <= thresh
                right_mask = ~left_mask
                
                if left_mask.sum() < self.min_samples_leaf or right_mask.sum() < self.min_samples_leaf:
                    continue
                
                n_left, n_right = left_mask.sum(), right_mask.sum()
                n_total = n_left + n_right
                
                weighted_entropy = (
                    (n_left / n_total) * self._entropy(labels[left_mask]) +
                    (n_right / n_total) * self._entropy(labels[right_mask])
                )
                
                info_gain = base_entropy - weighted_entropy
                
                if info_gain > best_gain:
                    best_gain = info_gain
                    best_split = (feat_idx, thresh, info_gain)
        
        return best_split
    
    def _entropy(self, labels: np.ndarray) -> float:
        """Binary entropy hesapla."""
        if len(labels) == 0:
            return 0.0
        p = np.mean(labels)
        if p == 0 or p == 1:
            return 0.0
        return -p * np.log2(p) - (1 - p) * np.log2(1 - p)
    
    def _create_pattern(
        self,
        labels: np.ndarray,
        returns: Optional[np.ndarray],
        rules: List[TreeRule],
        depth: int,
    ) -> Optional[DiscoveredPattern]:
        """Pattern oluştur ve istatistiksel test yap."""
        n_samples = len(labels)
        if n_samples < self.min_samples_leaf:
            return None
        
        wins = np.sum(labels)
        losses = n_samples - wins
        win_rate = wins / n_samples
        
        avg_return = np.mean(returns) if returns is not None else 0.0
        
        # Chi-square test
        expected = n_samples * 0.5
        chi_stat = ((wins - expected) ** 2 + (losses - expected) ** 2) / expected
        chi_pvalue = 1 - stats.chi2.cdf(chi_stat, df=1)
        
        is_significant = chi_pvalue < self.significance_level
        
        confidence = win_rate * (1 - chi_pvalue)
        
        return DiscoveredPattern(
            rules=rules.copy(),
            confidence=confidence,
            support=n_samples,
            win_rate=win_rate,
            avg_return=avg_return,
            sample_size=n_samples,
            chi_square_stat=chi_stat,
            chi_square_pvalue=chi_pvalue,
            is_significant=is_significant,
            depth=depth,
        )
    
    def evaluate_pattern(
        self,
        pattern: DiscoveredPattern,
        features: np.ndarray,
        labels: np.ndarray,
        feature_names: List[str],
    ) -> Dict[str, Any]:
        """Pattern'ı yeni veri üzerinde değerlendir."""
        feature_dict_list = [
            {name: features[i, j] for j, name in enumerate(feature_names)}
            for i in range(len(features))
        ]
        
        matches = np.array([pattern.evaluate(fd) for fd in feature_dict_list])
        
        if matches.sum() == 0:
            return {"matches": 0, "win_rate": 0.0, "stability": 0.0}
        
        matched_labels = labels[matches]
        new_win_rate = np.mean(matched_labels)
        
        stability = 1 - abs(new_win_rate - pattern.win_rate) / max(pattern.win_rate, 0.01)
        
        return {
            "matches": int(matches.sum()),
            "win_rate": new_win_rate,
            "original_win_rate": pattern.win_rate,
            "stability": stability,
        }
