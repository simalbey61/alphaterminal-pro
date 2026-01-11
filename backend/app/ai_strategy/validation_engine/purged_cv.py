"""
AlphaTerminal Pro - Purged K-Fold Cross Validation
==================================================

Marcos Lopez de Prado'nun Purged & Embargoed CV implementasyonu.
Data leakage'ı önleyen finansal cross-validation.

Author: AlphaTerminal Team
Version: 1.0.0
"""

from __future__ import annotations

import logging
from typing import Optional, List, Dict, Any, Tuple, Iterator, Generator
from datetime import datetime, timedelta
from dataclasses import dataclass, field

import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class CVFold:
    """Cross-validation fold bilgisi."""
    fold_index: int
    train_indices: np.ndarray
    test_indices: np.ndarray
    purged_indices: np.ndarray
    embargo_indices: np.ndarray
    train_start: Optional[datetime] = None
    train_end: Optional[datetime] = None
    test_start: Optional[datetime] = None
    test_end: Optional[datetime] = None


@dataclass
class CVResult:
    """Cross-validation sonucu."""
    fold_index: int
    train_score: float
    test_score: float
    train_size: int
    test_size: int
    metrics: Dict[str, float] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "fold_index": self.fold_index,
            "train_score": self.train_score,
            "test_score": self.test_score,
            "train_size": self.train_size,
            "test_size": self.test_size,
            "metrics": self.metrics,
        }


@dataclass
class PurgedCVSummary:
    """Purged CV özeti."""
    n_splits: int
    purge_gap: int
    embargo_pct: float
    fold_results: List[CVResult]
    mean_train_score: float
    mean_test_score: float
    std_test_score: float
    overfitting_score: float  # train-test gap
    consistency_score: float  # % positive test folds
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "n_splits": self.n_splits,
            "purge_gap": self.purge_gap,
            "embargo_pct": self.embargo_pct,
            "fold_results": [r.to_dict() for r in self.fold_results],
            "mean_train_score": self.mean_train_score,
            "mean_test_score": self.mean_test_score,
            "std_test_score": self.std_test_score,
            "overfitting_score": self.overfitting_score,
            "consistency_score": self.consistency_score,
        }
    
    @property
    def is_robust(self) -> bool:
        """Strateji robust mu?"""
        return (
            self.overfitting_score < 0.3 and  # Max 30% overfit
            self.consistency_score >= 0.6 and  # Min 60% positive folds
            self.std_test_score < self.mean_test_score  # Std < mean
        )


class PurgedKFoldCV:
    """
    Purged K-Fold Cross-Validation.
    
    Lopez de Prado'nun "Advances in Financial Machine Learning" kitabından:
    - Purging: Test seti etrafındaki örnekleri train'den çıkar
    - Embargo: Test sonrası ek güvenlik marjı
    
    Bu sayede feature'ların geçmişten test setine bilgi sızdırması önlenir.
    
    Example:
        ```python
        cv = PurgedKFoldCV(n_splits=5, purge_gap=20)
        
        for train_idx, test_idx in cv.split(X, timestamps):
            model.fit(X[train_idx], y[train_idx])
            score = model.score(X[test_idx], y[test_idx])
        
        # Veya tam analiz
        summary = cv.cross_validate(model, X, y, timestamps)
        print(f"Consistency: {summary.consistency_score:.1%}")
        ```
    """
    
    def __init__(
        self,
        n_splits: int = 5,
        purge_gap: int = 20,
        embargo_pct: float = 0.01,
    ):
        """
        Initialize Purged K-Fold CV.
        
        Args:
            n_splits: Fold sayısı
            purge_gap: Purge edilecek sample sayısı (her iki yönde)
            embargo_pct: Test sonrası embargo yüzdesi
        """
        self.n_splits = n_splits
        self.purge_gap = purge_gap
        self.embargo_pct = embargo_pct
    
    def split(
        self,
        X: np.ndarray,
        timestamps: Optional[np.ndarray] = None,
        groups: Optional[np.ndarray] = None,
    ) -> Generator[Tuple[np.ndarray, np.ndarray], None, None]:
        """
        Cross-validation split generator.
        
        Args:
            X: Feature matrix
            timestamps: Zaman damgaları (opsiyonel)
            groups: Grup etiketleri (opsiyonel)
            
        Yields:
            Tuple[train_indices, test_indices]
        """
        n_samples = len(X)
        indices = np.arange(n_samples)
        
        # Fold boyutu
        fold_size = n_samples // self.n_splits
        
        for fold_idx in range(self.n_splits):
            # Test indices
            test_start = fold_idx * fold_size
            test_end = (fold_idx + 1) * fold_size if fold_idx < self.n_splits - 1 else n_samples
            test_indices = indices[test_start:test_end]
            
            # Purge indices (test etrafında)
            purge_start = max(0, test_start - self.purge_gap)
            purge_end = min(n_samples, test_end + self.purge_gap)
            
            # Embargo indices (test sonrası)
            embargo_size = int(n_samples * self.embargo_pct)
            embargo_end = min(n_samples, test_end + embargo_size)
            
            # Train indices (purge ve embargo dışında kalanlar)
            excluded = set(range(purge_start, embargo_end))
            train_indices = np.array([i for i in indices if i not in excluded])
            
            yield train_indices, test_indices
    
    def get_folds(
        self,
        X: np.ndarray,
        timestamps: Optional[np.ndarray] = None,
    ) -> List[CVFold]:
        """
        Detaylı fold bilgilerini al.
        
        Args:
            X: Feature matrix
            timestamps: Zaman damgaları
            
        Returns:
            List[CVFold]: Fold detayları
        """
        n_samples = len(X)
        indices = np.arange(n_samples)
        folds = []
        
        fold_size = n_samples // self.n_splits
        
        for fold_idx in range(self.n_splits):
            test_start = fold_idx * fold_size
            test_end = (fold_idx + 1) * fold_size if fold_idx < self.n_splits - 1 else n_samples
            test_indices = indices[test_start:test_end]
            
            purge_start = max(0, test_start - self.purge_gap)
            purge_end = min(n_samples, test_end + self.purge_gap)
            purge_indices = np.array([i for i in range(purge_start, purge_end) if i not in test_indices])
            
            embargo_size = int(n_samples * self.embargo_pct)
            embargo_start = test_end
            embargo_end = min(n_samples, test_end + embargo_size)
            embargo_indices = indices[embargo_start:embargo_end]
            
            excluded = set(range(purge_start, embargo_end))
            train_indices = np.array([i for i in indices if i not in excluded])
            
            fold = CVFold(
                fold_index=fold_idx,
                train_indices=train_indices,
                test_indices=test_indices,
                purged_indices=purge_indices,
                embargo_indices=embargo_indices,
            )
            
            if timestamps is not None:
                fold.train_start = timestamps[train_indices[0]] if len(train_indices) > 0 else None
                fold.train_end = timestamps[train_indices[-1]] if len(train_indices) > 0 else None
                fold.test_start = timestamps[test_indices[0]] if len(test_indices) > 0 else None
                fold.test_end = timestamps[test_indices[-1]] if len(test_indices) > 0 else None
            
            folds.append(fold)
        
        return folds
    
    def cross_validate(
        self,
        model: Any,
        X: np.ndarray,
        y: np.ndarray,
        timestamps: Optional[np.ndarray] = None,
        scoring: str = "accuracy",
        return_train_score: bool = True,
    ) -> PurgedCVSummary:
        """
        Tam cross-validation çalıştır.
        
        Args:
            model: Scikit-learn uyumlu model
            X: Feature matrix
            y: Labels
            timestamps: Zaman damgaları
            scoring: Skor metriği
            return_train_score: Train skoru hesapla
            
        Returns:
            PurgedCVSummary: CV sonuç özeti
        """
        fold_results = []
        
        for fold_idx, (train_idx, test_idx) in enumerate(self.split(X, timestamps)):
            # Clone model (her fold için temiz model)
            from sklearn.base import clone
            fold_model = clone(model)
            
            # Train
            fold_model.fit(X[train_idx], y[train_idx])
            
            # Scores
            test_score = self._calculate_score(fold_model, X[test_idx], y[test_idx], scoring)
            train_score = self._calculate_score(fold_model, X[train_idx], y[train_idx], scoring) if return_train_score else 0.0
            
            # Additional metrics
            metrics = self._calculate_metrics(fold_model, X[test_idx], y[test_idx])
            
            fold_results.append(CVResult(
                fold_index=fold_idx,
                train_score=train_score,
                test_score=test_score,
                train_size=len(train_idx),
                test_size=len(test_idx),
                metrics=metrics,
            ))
        
        # Summary statistics
        train_scores = [r.train_score for r in fold_results]
        test_scores = [r.test_score for r in fold_results]
        
        mean_train = np.mean(train_scores)
        mean_test = np.mean(test_scores)
        std_test = np.std(test_scores)
        
        # Overfitting score (train-test gap)
        overfitting = (mean_train - mean_test) / max(mean_train, 0.01)
        
        # Consistency (% positive test scores)
        positive_threshold = 0.5 if scoring == "accuracy" else 0.0
        positive_folds = sum(1 for s in test_scores if s > positive_threshold)
        consistency = positive_folds / len(test_scores)
        
        return PurgedCVSummary(
            n_splits=self.n_splits,
            purge_gap=self.purge_gap,
            embargo_pct=self.embargo_pct,
            fold_results=fold_results,
            mean_train_score=mean_train,
            mean_test_score=mean_test,
            std_test_score=std_test,
            overfitting_score=overfitting,
            consistency_score=consistency,
        )
    
    def _calculate_score(
        self,
        model: Any,
        X: np.ndarray,
        y: np.ndarray,
        scoring: str,
    ) -> float:
        """Skor hesapla."""
        if scoring == "accuracy":
            return model.score(X, y)
        elif scoring == "roc_auc":
            from sklearn.metrics import roc_auc_score
            try:
                proba = model.predict_proba(X)[:, 1]
                return roc_auc_score(y, proba)
            except:
                return model.score(X, y)
        elif scoring == "f1":
            from sklearn.metrics import f1_score
            pred = model.predict(X)
            return f1_score(y, pred)
        else:
            return model.score(X, y)
    
    def _calculate_metrics(
        self,
        model: Any,
        X: np.ndarray,
        y: np.ndarray,
    ) -> Dict[str, float]:
        """Ek metrikler hesapla."""
        metrics = {}
        
        try:
            from sklearn.metrics import precision_score, recall_score, f1_score
            
            pred = model.predict(X)
            metrics["precision"] = precision_score(y, pred, zero_division=0)
            metrics["recall"] = recall_score(y, pred, zero_division=0)
            metrics["f1"] = f1_score(y, pred, zero_division=0)
        except Exception as e:
            logger.warning(f"Metric calculation error: {e}")
        
        return metrics


class CombinatorialPurgedCV:
    """
    Combinatorial Purged Cross-Validation (CPCV).
    
    Daha kapsamlı test için fold kombinasyonları kullanır.
    """
    
    def __init__(
        self,
        n_splits: int = 5,
        n_test_splits: int = 2,
        purge_gap: int = 20,
        embargo_pct: float = 0.01,
    ):
        """
        Initialize CPCV.
        
        Args:
            n_splits: Toplam split sayısı
            n_test_splits: Test için kullanılacak split sayısı
            purge_gap: Purge gap
            embargo_pct: Embargo yüzdesi
        """
        self.n_splits = n_splits
        self.n_test_splits = n_test_splits
        self.purge_gap = purge_gap
        self.embargo_pct = embargo_pct
    
    def split(
        self,
        X: np.ndarray,
        timestamps: Optional[np.ndarray] = None,
    ) -> Generator[Tuple[np.ndarray, np.ndarray], None, None]:
        """Combinatorial split generator."""
        from itertools import combinations
        
        n_samples = len(X)
        indices = np.arange(n_samples)
        fold_size = n_samples // self.n_splits
        
        # Fold boundaries
        fold_bounds = [(i * fold_size, (i + 1) * fold_size if i < self.n_splits - 1 else n_samples)
                       for i in range(self.n_splits)]
        
        # Test fold kombinasyonları
        for test_folds in combinations(range(self.n_splits), self.n_test_splits):
            test_indices = []
            for fold_idx in test_folds:
                start, end = fold_bounds[fold_idx]
                test_indices.extend(range(start, end))
            test_indices = np.array(test_indices)
            
            # Purge & embargo
            test_min, test_max = test_indices.min(), test_indices.max()
            purge_start = max(0, test_min - self.purge_gap)
            embargo_end = min(n_samples, test_max + int(n_samples * self.embargo_pct) + self.purge_gap)
            
            excluded = set(range(purge_start, embargo_end))
            train_indices = np.array([i for i in indices if i not in excluded])
            
            yield train_indices, test_indices
