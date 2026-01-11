"""
AlphaTerminal Pro - Walk Forward Analyzer
=========================================

Walk-Forward Analysis ile out-of-sample strateji doğrulama.

Author: AlphaTerminal Team
Version: 1.0.0
"""

from __future__ import annotations

import logging
from typing import Optional, List, Dict, Any, Tuple
from datetime import datetime
from dataclasses import dataclass, field
from enum import Enum

import numpy as np

logger = logging.getLogger(__name__)


class WalkForwardMode(str, Enum):
    """Walk-forward modu."""
    ANCHORED = "anchored"     # Training window büyür
    ROLLING = "rolling"       # Sabit training window kayar
    EXPANDING = "expanding"   # Minimum ile başla, sonra anchored


@dataclass
class WalkForwardWindow:
    """Walk-forward pencere bilgisi."""
    window_index: int
    train_start_idx: int
    train_end_idx: int
    test_start_idx: int
    test_end_idx: int
    train_size: int
    test_size: int
    
    # Performans metrikleri
    in_sample_metrics: Dict[str, float] = field(default_factory=dict)
    out_of_sample_metrics: Dict[str, float] = field(default_factory=dict)
    
    # Timestamps
    train_start_date: Optional[datetime] = None
    train_end_date: Optional[datetime] = None
    test_start_date: Optional[datetime] = None
    test_end_date: Optional[datetime] = None
    
    @property
    def passed(self) -> bool:
        """Pencere başarılı mı?"""
        oos_pf = self.out_of_sample_metrics.get("profit_factor", 0)
        oos_wr = self.out_of_sample_metrics.get("win_rate", 0)
        return oos_pf > 1.0 and oos_wr > 0.45
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "window_index": self.window_index,
            "train_size": self.train_size,
            "test_size": self.test_size,
            "in_sample_metrics": self.in_sample_metrics,
            "out_of_sample_metrics": self.out_of_sample_metrics,
            "passed": self.passed,
            "train_start_date": self.train_start_date.isoformat() if self.train_start_date else None,
            "test_end_date": self.test_end_date.isoformat() if self.test_end_date else None,
        }


@dataclass
class WalkForwardResult:
    """Walk-forward analiz sonucu."""
    mode: WalkForwardMode
    total_windows: int
    windows_passed: int
    consistency_score: float
    
    # Ortalama metrikler
    avg_in_sample: Dict[str, float]
    avg_out_of_sample: Dict[str, float]
    
    # Degradasyon (IS vs OOS farkı)
    performance_degradation: float
    
    # Pencere detayları
    windows: List[WalkForwardWindow]
    
    # Robustluk değerlendirmesi
    is_robust: bool
    recommendation: str
    
    calculated_at: datetime = field(default_factory=datetime.utcnow)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "mode": self.mode.value,
            "total_windows": self.total_windows,
            "windows_passed": self.windows_passed,
            "consistency_score": self.consistency_score,
            "avg_in_sample": self.avg_in_sample,
            "avg_out_of_sample": self.avg_out_of_sample,
            "performance_degradation": self.performance_degradation,
            "windows": [w.to_dict() for w in self.windows],
            "is_robust": self.is_robust,
            "recommendation": self.recommendation,
            "calculated_at": self.calculated_at.isoformat(),
        }


class WalkForwardAnalyzer:
    """
    Walk-Forward Analysis.
    
    Strateji performansının zaman içinde tutarlılığını test eder.
    Her pencerede:
    1. In-sample (train) periyodunda optimize et
    2. Out-of-sample (test) periyodunda değerlendir
    3. Pencereyi kaydır ve tekrarla
    
    Example:
        ```python
        analyzer = WalkForwardAnalyzer(
            mode=WalkForwardMode.ANCHORED,
            n_windows=12,
            train_pct=0.7
        )
        
        result = analyzer.analyze(strategy, data)
        
        print(f"Consistency: {result.consistency_score:.1%}")
        print(f"Is Robust: {result.is_robust}")
        ```
    """
    
    def __init__(
        self,
        mode: WalkForwardMode = WalkForwardMode.ANCHORED,
        n_windows: int = 12,
        train_pct: float = 0.7,
        min_train_size: int = 100,
        min_test_size: int = 30,
    ):
        """
        Initialize Walk-Forward Analyzer.
        
        Args:
            mode: Walk-forward modu
            n_windows: Pencere sayısı
            train_pct: Train periyodu oranı
            min_train_size: Minimum train boyutu
            min_test_size: Minimum test boyutu
        """
        self.mode = mode
        self.n_windows = n_windows
        self.train_pct = train_pct
        self.min_train_size = min_train_size
        self.min_test_size = min_test_size
    
    def generate_windows(
        self,
        n_samples: int,
        timestamps: Optional[np.ndarray] = None,
    ) -> List[WalkForwardWindow]:
        """
        Walk-forward pencerelerini oluştur.
        
        Args:
            n_samples: Toplam örnek sayısı
            timestamps: Zaman damgaları
            
        Returns:
            List[WalkForwardWindow]: Pencere listesi
        """
        windows = []
        
        if self.mode == WalkForwardMode.ROLLING:
            windows = self._generate_rolling_windows(n_samples, timestamps)
        elif self.mode == WalkForwardMode.ANCHORED:
            windows = self._generate_anchored_windows(n_samples, timestamps)
        elif self.mode == WalkForwardMode.EXPANDING:
            windows = self._generate_expanding_windows(n_samples, timestamps)
        
        return windows
    
    def _generate_rolling_windows(
        self,
        n_samples: int,
        timestamps: Optional[np.ndarray],
    ) -> List[WalkForwardWindow]:
        """Rolling (sabit boyut) pencereler."""
        windows = []
        
        # Her pencere için sabit train ve test boyutu
        total_window = n_samples // self.n_windows
        train_size = int(total_window * self.train_pct)
        test_size = total_window - train_size
        
        for i in range(self.n_windows):
            train_start = i * test_size
            train_end = train_start + train_size
            test_start = train_end
            test_end = test_start + test_size
            
            if test_end > n_samples:
                break
            
            window = WalkForwardWindow(
                window_index=i,
                train_start_idx=train_start,
                train_end_idx=train_end,
                test_start_idx=test_start,
                test_end_idx=test_end,
                train_size=train_end - train_start,
                test_size=test_end - test_start,
            )
            
            if timestamps is not None:
                window.train_start_date = timestamps[train_start]
                window.train_end_date = timestamps[train_end - 1]
                window.test_start_date = timestamps[test_start]
                window.test_end_date = timestamps[test_end - 1]
            
            windows.append(window)
        
        return windows
    
    def _generate_anchored_windows(
        self,
        n_samples: int,
        timestamps: Optional[np.ndarray],
    ) -> List[WalkForwardWindow]:
        """Anchored (büyüyen train) pencereler."""
        windows = []
        
        # Test boyutu sabit, train büyür
        test_size = n_samples // (self.n_windows + 1)
        
        for i in range(self.n_windows):
            train_start = 0  # Her zaman baştan
            train_end = (i + 1) * test_size
            test_start = train_end
            test_end = min(test_start + test_size, n_samples)
            
            if train_end - train_start < self.min_train_size:
                continue
            if test_end - test_start < self.min_test_size:
                continue
            
            window = WalkForwardWindow(
                window_index=i,
                train_start_idx=train_start,
                train_end_idx=train_end,
                test_start_idx=test_start,
                test_end_idx=test_end,
                train_size=train_end - train_start,
                test_size=test_end - test_start,
            )
            
            if timestamps is not None:
                window.train_start_date = timestamps[train_start]
                window.train_end_date = timestamps[train_end - 1]
                window.test_start_date = timestamps[test_start]
                window.test_end_date = timestamps[min(test_end - 1, len(timestamps) - 1)]
            
            windows.append(window)
        
        return windows
    
    def _generate_expanding_windows(
        self,
        n_samples: int,
        timestamps: Optional[np.ndarray],
    ) -> List[WalkForwardWindow]:
        """Expanding (minimum ile başla) pencereler."""
        windows = []
        
        test_size = max(self.min_test_size, n_samples // (self.n_windows + 1))
        initial_train = self.min_train_size
        
        for i in range(self.n_windows):
            train_start = 0
            train_end = initial_train + i * test_size
            test_start = train_end
            test_end = min(test_start + test_size, n_samples)
            
            if test_end > n_samples:
                break
            
            window = WalkForwardWindow(
                window_index=i,
                train_start_idx=train_start,
                train_end_idx=train_end,
                test_start_idx=test_start,
                test_end_idx=test_end,
                train_size=train_end - train_start,
                test_size=test_end - test_start,
            )
            
            if timestamps is not None:
                window.train_start_date = timestamps[train_start]
                window.train_end_date = timestamps[train_end - 1]
                window.test_start_date = timestamps[test_start]
                window.test_end_date = timestamps[min(test_end - 1, len(timestamps) - 1)]
            
            windows.append(window)
        
        return windows
    
    def analyze(
        self,
        features: np.ndarray,
        labels: np.ndarray,
        returns: np.ndarray,
        model_factory: callable,
        timestamps: Optional[np.ndarray] = None,
    ) -> WalkForwardResult:
        """
        Walk-forward analiz çalıştır.
        
        Args:
            features: Feature matrix
            labels: Labels (win/loss)
            returns: Trade returns
            model_factory: Model oluşturucu fonksiyon
            timestamps: Zaman damgaları
            
        Returns:
            WalkForwardResult: Analiz sonucu
        """
        windows = self.generate_windows(len(features), timestamps)
        
        if len(windows) == 0:
            logger.warning("No valid windows generated")
            return self._create_empty_result()
        
        for window in windows:
            # Train indices
            train_idx = slice(window.train_start_idx, window.train_end_idx)
            test_idx = slice(window.test_start_idx, window.test_end_idx)
            
            # Train
            model = model_factory()
            try:
                model.fit(features[train_idx], labels[train_idx])
                
                # In-sample metrics
                train_pred = model.predict(features[train_idx])
                window.in_sample_metrics = self._calculate_metrics(
                    labels[train_idx], train_pred, returns[train_idx]
                )
                
                # Out-of-sample metrics
                test_pred = model.predict(features[test_idx])
                window.out_of_sample_metrics = self._calculate_metrics(
                    labels[test_idx], test_pred, returns[test_idx]
                )
                
            except Exception as e:
                logger.error(f"Window {window.window_index} error: {e}")
                window.in_sample_metrics = {}
                window.out_of_sample_metrics = {}
        
        return self._compile_results(windows)
    
    def _calculate_metrics(
        self,
        labels: np.ndarray,
        predictions: np.ndarray,
        returns: np.ndarray,
    ) -> Dict[str, float]:
        """Performans metriklerini hesapla."""
        # Win rate
        correct = (labels == predictions).sum()
        win_rate = correct / len(labels) if len(labels) > 0 else 0
        
        # Profit factor (simplistic)
        wins = returns[returns > 0].sum()
        losses = abs(returns[returns < 0].sum())
        profit_factor = wins / losses if losses > 0 else 10.0
        
        # Total return
        total_return = returns.sum()
        
        # Sharpe (simplified)
        if len(returns) > 1 and returns.std() > 0:
            sharpe = returns.mean() / returns.std() * np.sqrt(252)
        else:
            sharpe = 0.0
        
        return {
            "win_rate": win_rate,
            "profit_factor": profit_factor,
            "total_return": total_return,
            "sharpe": sharpe,
            "n_trades": len(labels),
        }
    
    def _compile_results(self, windows: List[WalkForwardWindow]) -> WalkForwardResult:
        """Sonuçları derle."""
        # Passed windows
        passed = sum(1 for w in windows if w.passed)
        consistency = passed / len(windows) if windows else 0
        
        # Average metrics
        avg_is = {}
        avg_oos = {}
        
        for metric in ["win_rate", "profit_factor", "sharpe"]:
            is_values = [w.in_sample_metrics.get(metric, 0) for w in windows if w.in_sample_metrics]
            oos_values = [w.out_of_sample_metrics.get(metric, 0) for w in windows if w.out_of_sample_metrics]
            
            avg_is[metric] = np.mean(is_values) if is_values else 0
            avg_oos[metric] = np.mean(oos_values) if oos_values else 0
        
        # Performance degradation
        is_pf = avg_is.get("profit_factor", 1)
        oos_pf = avg_oos.get("profit_factor", 1)
        degradation = (is_pf - oos_pf) / max(is_pf, 0.01)
        
        # Robustness check
        is_robust = (
            consistency >= 0.6 and
            degradation < 0.4 and
            avg_oos.get("profit_factor", 0) > 1.0
        )
        
        # Recommendation
        if is_robust:
            recommendation = "Strategy shows acceptable robustness for live trading"
        elif consistency >= 0.5:
            recommendation = "Strategy shows moderate robustness, consider sandbox testing"
        else:
            recommendation = "Strategy shows poor out-of-sample performance, needs improvement"
        
        return WalkForwardResult(
            mode=self.mode,
            total_windows=len(windows),
            windows_passed=passed,
            consistency_score=consistency,
            avg_in_sample=avg_is,
            avg_out_of_sample=avg_oos,
            performance_degradation=degradation,
            windows=windows,
            is_robust=is_robust,
            recommendation=recommendation,
        )
    
    def _create_empty_result(self) -> WalkForwardResult:
        """Boş sonuç oluştur."""
        return WalkForwardResult(
            mode=self.mode,
            total_windows=0,
            windows_passed=0,
            consistency_score=0.0,
            avg_in_sample={},
            avg_out_of_sample={},
            performance_degradation=1.0,
            windows=[],
            is_robust=False,
            recommendation="Insufficient data for walk-forward analysis",
        )
