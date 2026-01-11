"""
AlphaTerminal Pro - Diversity Manager (Strategy Zoo)
===================================================

Strateji havuzu çeşitlilik yönetimi ve regime-based organizasyon.

Author: AlphaTerminal Team
Version: 1.0.0
"""

from __future__ import annotations

import logging
from typing import Optional, List, Dict, Any, Set
from datetime import datetime
from dataclasses import dataclass, field
from collections import defaultdict

import numpy as np

from app.ai_strategy.constants import (
    TrendRegime,
    VolatilityRegime,
    EvolutionDefaults,
    STRATEGY_ZOO_CATEGORIES,
)

logger = logging.getLogger(__name__)


@dataclass
class StrategyProfile:
    """Strateji profili."""
    strategy_id: str
    name: str
    
    # Regime suitability
    target_trend_regimes: List[TrendRegime]
    target_volatility_regimes: List[VolatilityRegime]
    
    # Performance metrics
    win_rate: float
    sharpe_ratio: float
    profit_factor: float
    max_drawdown: float
    
    # Feature usage
    features_used: List[str]
    
    # Correlation data
    return_series: Optional[np.ndarray] = None
    
    # Zoo category
    zoo_category: Optional[str] = None
    
    # Status
    is_active: bool = True
    created_at: datetime = field(default_factory=datetime.utcnow)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "strategy_id": self.strategy_id,
            "name": self.name,
            "target_trend_regimes": [r.value for r in self.target_trend_regimes],
            "target_volatility_regimes": [r.value for r in self.target_volatility_regimes],
            "win_rate": self.win_rate,
            "sharpe_ratio": self.sharpe_ratio,
            "profit_factor": self.profit_factor,
            "max_drawdown": self.max_drawdown,
            "features_used": self.features_used,
            "zoo_category": self.zoo_category,
            "is_active": self.is_active,
        }


@dataclass
class DiversityReport:
    """Çeşitlilik raporu."""
    total_strategies: int
    active_strategies: int
    
    # Correlation metrics
    avg_pairwise_correlation: float
    max_correlation: float
    min_correlation: float
    highly_correlated_pairs: List[Tuple[str, str, float]]
    
    # Coverage
    regime_coverage: Dict[str, int]
    feature_coverage: Dict[str, int]
    zoo_distribution: Dict[str, int]
    
    # Recommendations
    underrepresented_regimes: List[str]
    redundant_strategies: List[str]
    diversification_score: float
    
    generated_at: datetime = field(default_factory=datetime.utcnow)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "total_strategies": self.total_strategies,
            "active_strategies": self.active_strategies,
            "avg_pairwise_correlation": self.avg_pairwise_correlation,
            "max_correlation": self.max_correlation,
            "highly_correlated_pairs": self.highly_correlated_pairs,
            "regime_coverage": self.regime_coverage,
            "zoo_distribution": self.zoo_distribution,
            "underrepresented_regimes": self.underrepresented_regimes,
            "redundant_strategies": self.redundant_strategies,
            "diversification_score": self.diversification_score,
        }


class DiversityManager:
    """
    Strategy Zoo ve çeşitlilik yönetimi.
    
    Özellikler:
    - Regime bazlı strateji kategorilendirme
    - Korelasyon analizi
    - Otomatik çeşitlilik optimizasyonu
    - Redundant strateji tespiti
    
    Strategy Zoo Categories:
    - Bull Market Specialists
    - Bear Market Specialists
    - Sideways/Range Traders
    - High Volatility Players
    - Low Volatility Players
    - All-Weather Strategies
    
    Example:
        ```python
        manager = DiversityManager()
        
        # Strateji ekle
        manager.add_strategy(strategy_profile)
        
        # Çeşitlilik raporu
        report = manager.analyze_diversity()
        
        # Regime için en iyi strateji
        best = manager.get_strategies_for_regime(
            TrendRegime.BULL,
            VolatilityRegime.NORMAL
        )
        ```
    """
    
    def __init__(
        self,
        max_correlation: float = 0.70,
        min_strategies_per_regime: int = 3,
        defaults: Optional[EvolutionDefaults] = None,
    ):
        """
        Initialize Diversity Manager.
        
        Args:
            max_correlation: Maximum kabul edilebilir korelasyon
            min_strategies_per_regime: Regime başına minimum strateji
            defaults: Varsayılan değerler
        """
        self.max_correlation = max_correlation
        self.min_strategies_per_regime = min_strategies_per_regime
        self.defaults = defaults or EvolutionDefaults()
        
        # Strategy storage
        self._strategies: Dict[str, StrategyProfile] = {}
        
        # Correlation cache
        self._correlation_matrix: Optional[np.ndarray] = None
        self._correlation_ids: List[str] = []
    
    def add_strategy(self, profile: StrategyProfile) -> bool:
        """
        Strateji ekle.
        
        Args:
            profile: Strateji profili
            
        Returns:
            bool: Eklendi mi? (False = çok benzer strateji var)
        """
        # Redundancy check
        if self._is_redundant(profile):
            logger.warning(f"Strategy {profile.name} is too similar to existing strategies")
            return False
        
        # Assign zoo category
        profile.zoo_category = self._assign_zoo_category(profile)
        
        self._strategies[profile.strategy_id] = profile
        
        # Invalidate correlation cache
        self._correlation_matrix = None
        
        logger.info(f"Added strategy {profile.name} to {profile.zoo_category}")
        return True
    
    def remove_strategy(self, strategy_id: str) -> bool:
        """Strateji kaldır."""
        if strategy_id in self._strategies:
            del self._strategies[strategy_id]
            self._correlation_matrix = None
            return True
        return False
    
    def get_strategy(self, strategy_id: str) -> Optional[StrategyProfile]:
        """Strateji al."""
        return self._strategies.get(strategy_id)
    
    def get_all_strategies(self, active_only: bool = True) -> List[StrategyProfile]:
        """Tüm stratejileri al."""
        strategies = list(self._strategies.values())
        if active_only:
            strategies = [s for s in strategies if s.is_active]
        return strategies
    
    def get_strategies_for_regime(
        self,
        trend_regime: TrendRegime,
        volatility_regime: VolatilityRegime,
        top_n: int = 5,
    ) -> List[StrategyProfile]:
        """
        Belirli regime için en iyi stratejileri al.
        
        Args:
            trend_regime: Trend rejimi
            volatility_regime: Volatilite rejimi
            top_n: Kaç strateji döndürülsün
            
        Returns:
            List[StrategyProfile]: Uygun stratejiler (Sharpe'a göre sıralı)
        """
        suitable = []
        
        for strategy in self._strategies.values():
            if not strategy.is_active:
                continue
            
            # Regime match
            trend_match = (
                trend_regime in strategy.target_trend_regimes or
                not strategy.target_trend_regimes  # All-weather
            )
            vol_match = (
                volatility_regime in strategy.target_volatility_regimes or
                not strategy.target_volatility_regimes  # All-weather
            )
            
            if trend_match and vol_match:
                suitable.append(strategy)
        
        # Sort by Sharpe
        suitable.sort(key=lambda s: s.sharpe_ratio, reverse=True)
        
        return suitable[:top_n]
    
    def get_strategies_by_category(self, category: str) -> List[StrategyProfile]:
        """Kategori bazlı stratejiler."""
        return [
            s for s in self._strategies.values()
            if s.zoo_category == category and s.is_active
        ]
    
    def analyze_diversity(self) -> DiversityReport:
        """
        Çeşitlilik analizi yap.
        
        Returns:
            DiversityReport: Çeşitlilik raporu
        """
        strategies = list(self._strategies.values())
        active_strategies = [s for s in strategies if s.is_active]
        
        if len(active_strategies) < 2:
            return self._create_empty_report(len(strategies), len(active_strategies))
        
        # Correlation analysis
        corr_matrix, corr_ids = self._calculate_correlations()
        
        if corr_matrix is not None and len(corr_matrix) > 1:
            # Upper triangle (excluding diagonal)
            upper_tri = np.triu(corr_matrix, k=1)
            correlations = upper_tri[upper_tri != 0]
            
            avg_corr = float(np.mean(np.abs(correlations))) if len(correlations) > 0 else 0
            max_corr = float(np.max(np.abs(correlations))) if len(correlations) > 0 else 0
            min_corr = float(np.min(np.abs(correlations))) if len(correlations) > 0 else 0
            
            # Highly correlated pairs
            highly_correlated = []
            for i in range(len(corr_matrix)):
                for j in range(i + 1, len(corr_matrix)):
                    if abs(corr_matrix[i, j]) > self.max_correlation:
                        highly_correlated.append((
                            corr_ids[i],
                            corr_ids[j],
                            float(corr_matrix[i, j])
                        ))
        else:
            avg_corr, max_corr, min_corr = 0, 0, 0
            highly_correlated = []
        
        # Regime coverage
        regime_coverage = defaultdict(int)
        for s in active_strategies:
            for regime in s.target_trend_regimes:
                regime_coverage[regime.value] += 1
            for regime in s.target_volatility_regimes:
                regime_coverage[regime.value] += 1
        
        # Zoo distribution
        zoo_distribution = defaultdict(int)
        for s in active_strategies:
            if s.zoo_category:
                zoo_distribution[s.zoo_category] += 1
        
        # Feature coverage
        feature_coverage = defaultdict(int)
        for s in active_strategies:
            for feat in s.features_used:
                feature_coverage[feat] += 1
        
        # Underrepresented regimes
        underrepresented = []
        all_trend_regimes = [r.value for r in TrendRegime]
        all_vol_regimes = [r.value for r in VolatilityRegime]
        
        for regime in all_trend_regimes + all_vol_regimes:
            if regime_coverage.get(regime, 0) < self.min_strategies_per_regime:
                underrepresented.append(regime)
        
        # Redundant strategies
        redundant = [pair[0] for pair in highly_correlated]
        redundant = list(set(redundant))
        
        # Diversification score (0-100)
        div_score = self._calculate_diversification_score(
            avg_corr, len(underrepresented), zoo_distribution
        )
        
        return DiversityReport(
            total_strategies=len(strategies),
            active_strategies=len(active_strategies),
            avg_pairwise_correlation=avg_corr,
            max_correlation=max_corr,
            min_correlation=min_corr,
            highly_correlated_pairs=highly_correlated,
            regime_coverage=dict(regime_coverage),
            feature_coverage=dict(feature_coverage),
            zoo_distribution=dict(zoo_distribution),
            underrepresented_regimes=underrepresented,
            redundant_strategies=redundant,
            diversification_score=div_score,
        )
    
    def optimize_allocation(
        self,
        current_regime: TrendRegime,
        current_volatility: VolatilityRegime,
        target_count: int = 5,
    ) -> Dict[str, float]:
        """
        Mevcut regime için optimal strateji ağırlıkları.
        
        Args:
            current_regime: Mevcut trend rejimi
            current_volatility: Mevcut volatilite rejimi
            target_count: Hedef strateji sayısı
            
        Returns:
            Dict[strategy_id, weight]: Strateji ağırlıkları
        """
        # Get suitable strategies
        suitable = self.get_strategies_for_regime(
            current_regime, current_volatility, target_count * 2
        )
        
        if not suitable:
            return {}
        
        # Select diverse subset
        selected = self._select_diverse_subset(suitable, target_count)
        
        # Equal weight (could be optimized with mean-variance)
        weight = 1.0 / len(selected)
        
        return {s.strategy_id: weight for s in selected}
    
    def _is_redundant(self, new_profile: StrategyProfile) -> bool:
        """Yeni strateji redundant mı?"""
        if new_profile.return_series is None:
            # Korelasyon hesaplanamaz, feature overlap'e bak
            for existing in self._strategies.values():
                if not existing.is_active:
                    continue
                
                # Feature overlap
                overlap = set(new_profile.features_used) & set(existing.features_used)
                overlap_ratio = len(overlap) / max(len(new_profile.features_used), 1)
                
                if overlap_ratio > 0.8:  # 80% feature overlap
                    return True
            
            return False
        
        # Korelasyon bazlı kontrol
        for existing in self._strategies.values():
            if not existing.is_active or existing.return_series is None:
                continue
            
            # Align series
            min_len = min(len(new_profile.return_series), len(existing.return_series))
            if min_len < 10:
                continue
            
            corr = np.corrcoef(
                new_profile.return_series[-min_len:],
                existing.return_series[-min_len:]
            )[0, 1]
            
            if not np.isnan(corr) and abs(corr) > self.max_correlation:
                return True
        
        return False
    
    def _assign_zoo_category(self, profile: StrategyProfile) -> str:
        """Zoo kategorisi ata."""
        # Trend regime bazlı
        bullish_regimes = {TrendRegime.STRONG_BULL, TrendRegime.BULL, TrendRegime.WEAK_BULL}
        bearish_regimes = {TrendRegime.STRONG_BEAR, TrendRegime.BEAR, TrendRegime.WEAK_BEAR}
        
        target_trends = set(profile.target_trend_regimes)
        target_vols = set(profile.target_volatility_regimes)
        
        if target_trends <= bullish_regimes:
            return "bull_specialist"
        elif target_trends <= bearish_regimes:
            return "bear_specialist"
        elif TrendRegime.SIDEWAYS in target_trends and len(target_trends) == 1:
            return "sideways_trader"
        elif target_vols <= {VolatilityRegime.HIGH, VolatilityRegime.EXTREME}:
            return "high_vol_player"
        elif target_vols <= {VolatilityRegime.LOW, VolatilityRegime.VERY_LOW}:
            return "low_vol_player"
        else:
            return "all_weather"
    
    def _calculate_correlations(self) -> Tuple[Optional[np.ndarray], List[str]]:
        """Korelasyon matrisi hesapla."""
        if self._correlation_matrix is not None:
            return self._correlation_matrix, self._correlation_ids
        
        strategies_with_returns = [
            s for s in self._strategies.values()
            if s.is_active and s.return_series is not None and len(s.return_series) > 10
        ]
        
        if len(strategies_with_returns) < 2:
            return None, []
        
        # Align returns
        min_len = min(len(s.return_series) for s in strategies_with_returns)
        returns_matrix = np.array([
            s.return_series[-min_len:] for s in strategies_with_returns
        ])
        
        corr_matrix = np.corrcoef(returns_matrix)
        corr_ids = [s.strategy_id for s in strategies_with_returns]
        
        self._correlation_matrix = corr_matrix
        self._correlation_ids = corr_ids
        
        return corr_matrix, corr_ids
    
    def _select_diverse_subset(
        self,
        strategies: List[StrategyProfile],
        target_count: int,
    ) -> List[StrategyProfile]:
        """Çeşitli bir alt küme seç."""
        if len(strategies) <= target_count:
            return strategies
        
        selected = [strategies[0]]  # Start with best Sharpe
        
        while len(selected) < target_count:
            best_candidate = None
            best_diversity = -1
            
            for candidate in strategies:
                if candidate in selected:
                    continue
                
                # Calculate diversity from selected
                min_feature_overlap = 1.0
                for s in selected:
                    overlap = len(set(candidate.features_used) & set(s.features_used))
                    overlap_ratio = overlap / max(len(candidate.features_used), 1)
                    min_feature_overlap = min(min_feature_overlap, 1 - overlap_ratio)
                
                if min_feature_overlap > best_diversity:
                    best_diversity = min_feature_overlap
                    best_candidate = candidate
            
            if best_candidate:
                selected.append(best_candidate)
            else:
                break
        
        return selected
    
    def _calculate_diversification_score(
        self,
        avg_corr: float,
        underrep_count: int,
        zoo_dist: Dict[str, int],
    ) -> float:
        """Çeşitlilik skoru hesapla (0-100)."""
        score = 100.0
        
        # Correlation penalty
        score -= avg_corr * 40
        
        # Underrepresented penalty
        score -= underrep_count * 5
        
        # Zoo distribution bonus
        if len(zoo_dist) >= 4:
            score += 10
        elif len(zoo_dist) >= 2:
            score += 5
        
        return max(0, min(100, score))
    
    def _create_empty_report(self, total: int, active: int) -> DiversityReport:
        """Boş rapor oluştur."""
        return DiversityReport(
            total_strategies=total,
            active_strategies=active,
            avg_pairwise_correlation=0,
            max_correlation=0,
            min_correlation=0,
            highly_correlated_pairs=[],
            regime_coverage={},
            feature_coverage={},
            zoo_distribution={},
            underrepresented_regimes=list(TrendRegime.__members__.keys()),
            redundant_strategies=[],
            diversification_score=0,
        )


# Type hint fix
from typing import Tuple
