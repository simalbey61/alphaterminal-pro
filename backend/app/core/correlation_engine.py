"""
AlphaTerminal Pro - Correlation Engine v4.2
============================================

Kurumsal Seviye Korelasyon Analiz Motoru

Özellikler:
- Pairwise Correlation Matrix
- Rolling Correlation
- Sector Correlation Analysis
- Beta Clustering
- Diversification Score
- Correlation Regime Detection
- Risk Contribution Analysis

Author: AlphaTerminal Team
Version: 4.2.0
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, field
from enum import Enum
from scipy import stats
from scipy.cluster.hierarchy import linkage, fcluster
from scipy.spatial.distance import squareform

from app.core.config import logger, CORRELATION_CONFIG, get_sector


# ═══════════════════════════════════════════════════════════════════════════════
# ENUMS
# ═══════════════════════════════════════════════════════════════════════════════

class CorrelationRegime(Enum):
    """Korelasyon rejimi"""
    CRISIS = "CRISIS"           # Tüm korelasyonlar yükseliyor
    NORMAL = "NORMAL"           # Normal dağılım
    ROTATION = "ROTATION"       # Sektör rotasyonu
    DIVERGENCE = "DIVERGENCE"   # Düşük korelasyon


class ClusterType(Enum):
    """Kümeleme tipi"""
    HIGH_BETA = "HIGH_BETA"
    LOW_BETA = "LOW_BETA"
    DEFENSIVE = "DEFENSIVE"
    GROWTH = "GROWTH"
    VALUE = "VALUE"


# ═══════════════════════════════════════════════════════════════════════════════
# DATA CLASSES
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass
class PairCorrelation:
    """İki hisse arasındaki korelasyon"""
    symbol_1: str
    symbol_2: str
    correlation: float
    rolling_corr_20d: float
    rolling_corr_60d: float
    beta_ratio: float
    cointegrated: bool
    hedge_ratio: float


@dataclass
class SectorCorrelation:
    """Sektör korelasyon analizi"""
    sector: str
    intra_correlation: float  # Sektör içi ortalama korelasyon
    inter_correlations: Dict[str, float]  # Diğer sektörlerle korelasyon
    beta_to_index: float
    top_correlated_stocks: List[Tuple[str, float]]
    diversification_benefit: float


@dataclass
class StockCluster:
    """Hisse kümesi"""
    cluster_id: int
    cluster_type: ClusterType
    symbols: List[str]
    centroid_correlation: float
    avg_beta: float
    avg_volatility: float
    sector_composition: Dict[str, int]


@dataclass 
class RiskContribution:
    """Risk katkısı analizi"""
    symbol: str
    marginal_risk: float
    component_risk: float
    percent_contribution: float
    diversification_ratio: float


@dataclass
class PortfolioCorrelation:
    """Portföy korelasyon analizi"""
    correlation_matrix: pd.DataFrame
    avg_correlation: float
    max_correlation: Tuple[str, str, float]
    min_correlation: Tuple[str, str, float]
    effective_n: float  # Efektif bağımsız varlık sayısı
    diversification_score: float  # 0-100
    regime: CorrelationRegime
    risk_contributions: List[RiskContribution]
    clusters: List[StockCluster]
    warnings: List[str]


@dataclass
class CorrelationChange:
    """Korelasyon değişim analizi"""
    symbol_1: str
    symbol_2: str
    current_corr: float
    prev_corr: float
    change: float
    change_percent: float
    is_significant: bool
    trend: str  # "INCREASING", "DECREASING", "STABLE"


# ═══════════════════════════════════════════════════════════════════════════════
# CORRELATION ENGINE CLASS
# ═══════════════════════════════════════════════════════════════════════════════

class CorrelationEngine:
    """
    Kurumsal Seviye Korelasyon Analiz Motoru v4.2
    
    Özellikler:
    - Pairwise correlation hesaplama
    - Rolling correlation tracking
    - Sector correlation analysis
    - Hierarchical clustering
    - Risk contribution analysis
    - Correlation regime detection
    - Diversification scoring
    """
    
    def __init__(self, config=None):
        self.config = config or CORRELATION_CONFIG
        self._cache: Dict[str, any] = {}
    
    # ═══════════════════════════════════════════════════════════════════════════
    # CORRELATION MATRIX
    # ═══════════════════════════════════════════════════════════════════════════
    
    def calculate_correlation_matrix(
        self,
        returns_dict: Dict[str, pd.Series],
        method: str = "pearson"
    ) -> pd.DataFrame:
        """
        Korelasyon matrisi hesapla
        
        Args:
            returns_dict: {symbol: returns_series}
            method: "pearson", "spearman", "kendall"
            
        Returns:
            Correlation matrix DataFrame
        """
        # DataFrame oluştur
        df = pd.DataFrame(returns_dict)
        df = df.dropna()
        
        if len(df) < self.config.min_periods:
            logger.warning(f"Yetersiz veri: {len(df)} < {self.config.min_periods}")
            return pd.DataFrame()
        
        # Korelasyon hesapla
        if method == "pearson":
            corr_matrix = df.corr(method='pearson')
        elif method == "spearman":
            corr_matrix = df.corr(method='spearman')
        elif method == "kendall":
            corr_matrix = df.corr(method='kendall')
        else:
            corr_matrix = df.corr()
        
        return corr_matrix
    
    def calculate_rolling_correlation(
        self,
        returns_1: pd.Series,
        returns_2: pd.Series,
        windows: List[int] = None
    ) -> Dict[int, pd.Series]:
        """
        Rolling korelasyon hesapla
        
        Args:
            returns_1: İlk hisse getirileri
            returns_2: İkinci hisse getirileri
            windows: Rolling window boyutları
            
        Returns:
            {window: rolling_correlation_series}
        """
        if windows is None:
            windows = self.config.rolling_windows
        
        aligned = pd.DataFrame({
            'r1': returns_1,
            'r2': returns_2
        }).dropna()
        
        result = {}
        for w in windows:
            if len(aligned) >= w:
                rolling_corr = aligned['r1'].rolling(w).corr(aligned['r2'])
                result[w] = rolling_corr
        
        return result
    
    # ═══════════════════════════════════════════════════════════════════════════
    # PAIR ANALYSIS
    # ═══════════════════════════════════════════════════════════════════════════
    
    def analyze_pair(
        self,
        symbol_1: str,
        symbol_2: str,
        returns_1: pd.Series,
        returns_2: pd.Series,
        prices_1: pd.Series = None,
        prices_2: pd.Series = None
    ) -> PairCorrelation:
        """
        İki hisse arasındaki korelasyon analizi
        
        Args:
            symbol_1, symbol_2: Hisse kodları
            returns_1, returns_2: Getiri serileri
            prices_1, prices_2: Fiyat serileri (cointegration için)
            
        Returns:
            PairCorrelation dataclass
        """
        aligned = pd.DataFrame({
            'r1': returns_1,
            'r2': returns_2
        }).dropna()
        
        if len(aligned) < 30:
            return PairCorrelation(
                symbol_1=symbol_1, symbol_2=symbol_2,
                correlation=0, rolling_corr_20d=0, rolling_corr_60d=0,
                beta_ratio=1, cointegrated=False, hedge_ratio=1
            )
        
        # Static correlation
        correlation = aligned['r1'].corr(aligned['r2'])
        
        # Rolling correlations
        rolling_20 = aligned['r1'].rolling(20).corr(aligned['r2']).iloc[-1] if len(aligned) >= 20 else correlation
        rolling_60 = aligned['r1'].rolling(60).corr(aligned['r2']).iloc[-1] if len(aligned) >= 60 else correlation
        
        # Beta ratio (volatility adjusted)
        vol_1 = aligned['r1'].std()
        vol_2 = aligned['r2'].std()
        beta_ratio = (vol_1 / vol_2) if vol_2 != 0 else 1
        
        # Cointegration test (basitleştirilmiş)
        cointegrated = False
        hedge_ratio = 1.0
        
        if prices_1 is not None and prices_2 is not None:
            try:
                # OLS regression for hedge ratio
                aligned_prices = pd.DataFrame({
                    'p1': prices_1,
                    'p2': prices_2
                }).dropna()
                
                if len(aligned_prices) >= 60:
                    slope, intercept, r_value, _, _ = stats.linregress(
                        aligned_prices['p2'], aligned_prices['p1']
                    )
                    hedge_ratio = slope
                    
                    # Spread stationarity check (ADF-like)
                    spread = aligned_prices['p1'] - hedge_ratio * aligned_prices['p2']
                    spread_mean = spread.mean()
                    spread_std = spread.std()
                    
                    # Mean reversion check
                    crossings = ((spread.shift(1) - spread_mean) * (spread - spread_mean) < 0).sum()
                    expected_crossings = len(spread) * 0.3
                    
                    cointegrated = crossings > expected_crossings and r_value ** 2 > 0.7
            except Exception as e:
                logger.debug(f"Cointegration test failed: {e}")
        
        return PairCorrelation(
            symbol_1=symbol_1,
            symbol_2=symbol_2,
            correlation=round(correlation, 4),
            rolling_corr_20d=round(rolling_20, 4) if not pd.isna(rolling_20) else 0,
            rolling_corr_60d=round(rolling_60, 4) if not pd.isna(rolling_60) else 0,
            beta_ratio=round(beta_ratio, 4),
            cointegrated=cointegrated,
            hedge_ratio=round(hedge_ratio, 4)
        )
    
    # ═══════════════════════════════════════════════════════════════════════════
    # SECTOR ANALYSIS
    # ═══════════════════════════════════════════════════════════════════════════
    
    def analyze_sector_correlations(
        self,
        returns_dict: Dict[str, pd.Series],
        index_returns: pd.Series = None
    ) -> Dict[str, SectorCorrelation]:
        """
        Sektör korelasyon analizi
        
        Args:
            returns_dict: {symbol: returns}
            index_returns: Endeks getirileri
            
        Returns:
            {sector: SectorCorrelation}
        """
        # Sembolleri sektörlere ayır
        sector_symbols: Dict[str, List[str]] = {}
        
        for symbol in returns_dict.keys():
            sector = get_sector(symbol)
            if sector:
                if sector not in sector_symbols:
                    sector_symbols[sector] = []
                sector_symbols[sector].append(symbol)
        
        results = {}
        
        for sector, symbols in sector_symbols.items():
            if len(symbols) < 2:
                continue
            
            # Sektör içi korelasyon
            sector_returns = {s: returns_dict[s] for s in symbols if s in returns_dict}
            
            if len(sector_returns) < 2:
                continue
            
            corr_matrix = self.calculate_correlation_matrix(sector_returns)
            
            if corr_matrix.empty:
                continue
            
            # Ortalama intra-sector korelasyon (diagonal hariç)
            mask = np.triu(np.ones_like(corr_matrix, dtype=bool), k=1)
            intra_corr = corr_matrix.where(mask).stack().mean()
            
            # Diğer sektörlerle korelasyon
            inter_corrs = {}
            for other_sector, other_symbols in sector_symbols.items():
                if other_sector == sector:
                    continue
                
                # Her sektörden birer temsilci
                if symbols and other_symbols:
                    sym1 = symbols[0]
                    sym2 = other_symbols[0]
                    if sym1 in returns_dict and sym2 in returns_dict:
                        corr = returns_dict[sym1].corr(returns_dict[sym2])
                        if not pd.isna(corr):
                            inter_corrs[other_sector] = round(corr, 4)
            
            # Index beta
            beta = 1.0
            if index_returns is not None:
                sector_avg = pd.DataFrame(sector_returns).mean(axis=1)
                aligned = pd.DataFrame({
                    'sector': sector_avg,
                    'index': index_returns
                }).dropna()
                
                if len(aligned) >= 30:
                    cov = aligned['sector'].cov(aligned['index'])
                    var = aligned['index'].var()
                    beta = cov / var if var != 0 else 1.0
            
            # Top correlated stocks
            top_stocks = []
            for sym in symbols:
                if sym in returns_dict and index_returns is not None:
                    corr = returns_dict[sym].corr(index_returns)
                    if not pd.isna(corr):
                        top_stocks.append((sym, round(corr, 4)))
            
            top_stocks.sort(key=lambda x: abs(x[1]), reverse=True)
            
            # Diversification benefit
            n = len(symbols)
            if n > 1 and not pd.isna(intra_corr):
                div_benefit = (1 - intra_corr) * 100
            else:
                div_benefit = 0
            
            results[sector] = SectorCorrelation(
                sector=sector,
                intra_correlation=round(intra_corr, 4) if not pd.isna(intra_corr) else 0,
                inter_correlations=inter_corrs,
                beta_to_index=round(beta, 4),
                top_correlated_stocks=top_stocks[:5],
                diversification_benefit=round(div_benefit, 2)
            )
        
        return results
    
    # ═══════════════════════════════════════════════════════════════════════════
    # CLUSTERING
    # ═══════════════════════════════════════════════════════════════════════════
    
    def cluster_stocks(
        self,
        returns_dict: Dict[str, pd.Series],
        n_clusters: int = None
    ) -> List[StockCluster]:
        """
        Hisseleri korelasyona göre kümele
        
        Args:
            returns_dict: {symbol: returns}
            n_clusters: Küme sayısı (None = otomatik)
            
        Returns:
            StockCluster listesi
        """
        if len(returns_dict) < 3:
            return []
        
        # Korelasyon matrisi
        corr_matrix = self.calculate_correlation_matrix(returns_dict)
        
        if corr_matrix.empty:
            return []
        
        # Distance matrix (1 - correlation)
        dist_matrix = 1 - corr_matrix.abs()
        np.fill_diagonal(dist_matrix.values, 0)
        
        # Hierarchical clustering
        try:
            condensed_dist = squareform(dist_matrix.values)
            linkage_matrix = linkage(condensed_dist, method='ward')
            
            # Optimal cluster sayısı
            if n_clusters is None:
                n_clusters = min(5, len(returns_dict) // 3)
                n_clusters = max(2, n_clusters)
            
            cluster_labels = fcluster(linkage_matrix, n_clusters, criterion='maxclust')
        except Exception as e:
            logger.error(f"Clustering error: {e}")
            return []
        
        symbols = list(returns_dict.keys())
        clusters = []
        
        for cluster_id in range(1, n_clusters + 1):
            cluster_symbols = [symbols[i] for i, label in enumerate(cluster_labels) if label == cluster_id]
            
            if not cluster_symbols:
                continue
            
            # Cluster statistics
            cluster_returns = {s: returns_dict[s] for s in cluster_symbols}
            cluster_corr = self.calculate_correlation_matrix(cluster_returns)
            
            if not cluster_corr.empty:
                mask = np.triu(np.ones_like(cluster_corr, dtype=bool), k=1)
                centroid_corr = cluster_corr.where(mask).stack().mean()
            else:
                centroid_corr = 0
            
            # Average volatility
            avg_vol = np.mean([returns_dict[s].std() * np.sqrt(252) for s in cluster_symbols])
            
            # Sector composition
            sector_comp = {}
            for s in cluster_symbols:
                sector = get_sector(s)
                if sector:
                    sector_comp[sector] = sector_comp.get(sector, 0) + 1
            
            # Cluster type (basit heuristik)
            if avg_vol > 0.4:
                cluster_type = ClusterType.HIGH_BETA
            elif avg_vol < 0.2:
                cluster_type = ClusterType.DEFENSIVE
            else:
                cluster_type = ClusterType.GROWTH
            
            clusters.append(StockCluster(
                cluster_id=cluster_id,
                cluster_type=cluster_type,
                symbols=cluster_symbols,
                centroid_correlation=round(centroid_corr, 4) if not pd.isna(centroid_corr) else 0,
                avg_beta=1.0,  # Placeholder
                avg_volatility=round(avg_vol, 4),
                sector_composition=sector_comp
            ))
        
        return clusters
    
    # ═══════════════════════════════════════════════════════════════════════════
    # RISK CONTRIBUTION
    # ═══════════════════════════════════════════════════════════════════════════
    
    def calculate_risk_contributions(
        self,
        returns_dict: Dict[str, pd.Series],
        weights: Dict[str, float] = None
    ) -> List[RiskContribution]:
        """
        Risk katkısı analizi
        
        Args:
            returns_dict: {symbol: returns}
            weights: {symbol: weight} (None = eşit ağırlık)
            
        Returns:
            RiskContribution listesi
        """
        symbols = list(returns_dict.keys())
        n = len(symbols)
        
        if n < 2:
            return []
        
        # Weights
        if weights is None:
            weights = {s: 1/n for s in symbols}
        
        w = np.array([weights.get(s, 1/n) for s in symbols])
        w = w / w.sum()  # Normalize
        
        # Covariance matrix
        df = pd.DataFrame(returns_dict)
        cov_matrix = df.cov() * 252  # Annualized
        
        if cov_matrix.empty:
            return []
        
        cov = cov_matrix.values
        
        # Portfolio variance
        port_var = w @ cov @ w
        port_std = np.sqrt(port_var)
        
        results = []
        
        for i, symbol in enumerate(symbols):
            # Marginal risk contribution
            marginal = (cov @ w)[i] / port_std if port_std > 0 else 0
            
            # Component risk contribution
            component = w[i] * marginal
            
            # Percent contribution
            pct_contrib = component / port_std * 100 if port_std > 0 else 0
            
            # Diversification ratio (individual vs contribution)
            individual_vol = np.sqrt(cov[i, i])
            div_ratio = individual_vol / (marginal + 1e-10)
            
            results.append(RiskContribution(
                symbol=symbol,
                marginal_risk=round(marginal * 100, 4),
                component_risk=round(component * 100, 4),
                percent_contribution=round(pct_contrib, 2),
                diversification_ratio=round(div_ratio, 4)
            ))
        
        # Sort by contribution
        results.sort(key=lambda x: x.percent_contribution, reverse=True)
        
        return results
    
    # ═══════════════════════════════════════════════════════════════════════════
    # REGIME DETECTION
    # ═══════════════════════════════════════════════════════════════════════════
    
    def detect_correlation_regime(
        self,
        returns_dict: Dict[str, pd.Series],
        lookback: int = 60
    ) -> Tuple[CorrelationRegime, float]:
        """
        Korelasyon rejimi tespiti
        
        Args:
            returns_dict: {symbol: returns}
            lookback: Geriye bakış periyodu
            
        Returns:
            (regime, confidence)
        """
        if len(returns_dict) < 3:
            return CorrelationRegime.NORMAL, 0.5
        
        # Son N günlük korelasyon
        recent_returns = {
            s: r.iloc[-lookback:] if len(r) >= lookback else r
            for s, r in returns_dict.items()
        }
        
        corr_matrix = self.calculate_correlation_matrix(recent_returns)
        
        if corr_matrix.empty:
            return CorrelationRegime.NORMAL, 0.5
        
        # Ortalama korelasyon
        mask = np.triu(np.ones_like(corr_matrix, dtype=bool), k=1)
        avg_corr = corr_matrix.where(mask).stack().mean()
        
        # Korelasyon std
        corr_std = corr_matrix.where(mask).stack().std()
        
        # Tarihsel karşılaştırma (basitleştirilmiş)
        historical_avg = 0.3  # Varsayılan normal seviye
        
        if avg_corr > self.config.crisis_threshold:
            regime = CorrelationRegime.CRISIS
            confidence = min((avg_corr - historical_avg) / 0.3, 1.0)
        elif avg_corr < self.config.divergence_threshold:
            regime = CorrelationRegime.DIVERGENCE
            confidence = min((historical_avg - avg_corr) / 0.2, 1.0)
        elif corr_std > 0.3:
            regime = CorrelationRegime.ROTATION
            confidence = min(corr_std / 0.5, 1.0)
        else:
            regime = CorrelationRegime.NORMAL
            confidence = 1 - abs(avg_corr - historical_avg) / 0.3
        
        return regime, round(max(0, min(confidence, 1)), 2)
    
    # ═══════════════════════════════════════════════════════════════════════════
    # DIVERSIFICATION SCORE
    # ═══════════════════════════════════════════════════════════════════════════
    
    def calculate_diversification_score(
        self,
        returns_dict: Dict[str, pd.Series],
        weights: Dict[str, float] = None
    ) -> Tuple[float, float]:
        """
        Diversifikasyon skoru hesapla
        
        Args:
            returns_dict: {symbol: returns}
            weights: Ağırlıklar
            
        Returns:
            (diversification_score, effective_n)
        """
        n = len(returns_dict)
        
        if n < 2:
            return 0.0, 1.0
        
        corr_matrix = self.calculate_correlation_matrix(returns_dict)
        
        if corr_matrix.empty:
            return 0.0, 1.0
        
        # Ortalama korelasyon
        mask = np.triu(np.ones_like(corr_matrix, dtype=bool), k=1)
        avg_corr = corr_matrix.where(mask).stack().mean()
        
        if pd.isna(avg_corr):
            avg_corr = 0.5
        
        # Effective N (bağımsız varlık sayısı tahmini)
        # N_eff = N / (1 + (N-1) * avg_corr)
        effective_n = n / (1 + (n - 1) * avg_corr) if avg_corr >= 0 else n
        
        # Diversification score (0-100)
        # Perfect div = avg_corr = 0, score = 100
        # No div = avg_corr = 1, score = 0
        div_score = (1 - avg_corr) * 100
        
        # Bonus for more stocks
        n_bonus = min(n / 10, 1) * 10  # Max 10 puan bonus
        
        final_score = min(div_score + n_bonus, 100)
        
        return round(final_score, 2), round(effective_n, 2)
    
    # ═══════════════════════════════════════════════════════════════════════════
    # CORRELATION CHANGES
    # ═══════════════════════════════════════════════════════════════════════════
    
    def detect_correlation_changes(
        self,
        returns_dict: Dict[str, pd.Series],
        short_window: int = 20,
        long_window: int = 60
    ) -> List[CorrelationChange]:
        """
        Önemli korelasyon değişimlerini tespit et
        
        Args:
            returns_dict: {symbol: returns}
            short_window: Kısa dönem penceresi
            long_window: Uzun dönem penceresi
            
        Returns:
            CorrelationChange listesi
        """
        symbols = list(returns_dict.keys())
        changes = []
        
        for i in range(len(symbols)):
            for j in range(i + 1, len(symbols)):
                sym1, sym2 = symbols[i], symbols[j]
                r1, r2 = returns_dict[sym1], returns_dict[sym2]
                
                aligned = pd.DataFrame({'r1': r1, 'r2': r2}).dropna()
                
                if len(aligned) < long_window:
                    continue
                
                # Short-term correlation
                short_corr = aligned['r1'].iloc[-short_window:].corr(aligned['r2'].iloc[-short_window:])
                
                # Long-term correlation
                long_corr = aligned['r1'].iloc[-long_window:].corr(aligned['r2'].iloc[-long_window:])
                
                if pd.isna(short_corr) or pd.isna(long_corr):
                    continue
                
                change = short_corr - long_corr
                change_pct = (change / abs(long_corr)) * 100 if long_corr != 0 else 0
                
                # Significance
                is_significant = abs(change) > self.config.significant_change_threshold
                
                # Trend
                if change > 0.1:
                    trend = "INCREASING"
                elif change < -0.1:
                    trend = "DECREASING"
                else:
                    trend = "STABLE"
                
                if is_significant:
                    changes.append(CorrelationChange(
                        symbol_1=sym1,
                        symbol_2=sym2,
                        current_corr=round(short_corr, 4),
                        prev_corr=round(long_corr, 4),
                        change=round(change, 4),
                        change_percent=round(change_pct, 2),
                        is_significant=is_significant,
                        trend=trend
                    ))
        
        # En büyük değişimlere göre sırala
        changes.sort(key=lambda x: abs(x.change), reverse=True)
        
        return changes
    
    # ═══════════════════════════════════════════════════════════════════════════
    # MAIN ANALYSIS
    # ═══════════════════════════════════════════════════════════════════════════
    
    def analyze_portfolio(
        self,
        returns_dict: Dict[str, pd.Series],
        weights: Dict[str, float] = None,
        index_returns: pd.Series = None
    ) -> PortfolioCorrelation:
        """
        Kapsamlı portföy korelasyon analizi
        
        Args:
            returns_dict: {symbol: returns}
            weights: Portföy ağırlıkları
            index_returns: Endeks getirileri
            
        Returns:
            PortfolioCorrelation dataclass
        """
        warnings = []
        
        if len(returns_dict) < 2:
            return self._empty_analysis()
        
        try:
            # Correlation matrix
            corr_matrix = self.calculate_correlation_matrix(returns_dict)
            
            if corr_matrix.empty:
                return self._empty_analysis()
            
            # Stats
            mask = np.triu(np.ones_like(corr_matrix, dtype=bool), k=1)
            upper_triangle = corr_matrix.where(mask).stack()
            
            avg_corr = upper_triangle.mean()
            
            # Max/Min correlation
            max_idx = upper_triangle.idxmax()
            min_idx = upper_triangle.idxmin()
            
            max_corr = (max_idx[0], max_idx[1], upper_triangle[max_idx])
            min_corr = (min_idx[0], min_idx[1], upper_triangle[min_idx])
            
            # Diversification
            div_score, effective_n = self.calculate_diversification_score(returns_dict, weights)
            
            # Regime
            regime, regime_conf = self.detect_correlation_regime(returns_dict)
            
            # Risk contributions
            risk_contribs = self.calculate_risk_contributions(returns_dict, weights)
            
            # Clustering
            clusters = self.cluster_stocks(returns_dict)
            
            # Warnings
            if avg_corr > 0.7:
                warnings.append(f"⚠️ Yüksek ortalama korelasyon: {avg_corr:.2f}")
            
            if max_corr[2] > 0.9:
                warnings.append(f"⚠️ Çok yüksek korelasyon: {max_corr[0]}-{max_corr[1]} ({max_corr[2]:.2f})")
            
            if effective_n < len(returns_dict) / 2:
                warnings.append(f"⚠️ Düşük efektif çeşitlendirme: {effective_n:.1f}/{len(returns_dict)}")
            
            if regime == CorrelationRegime.CRISIS:
                warnings.append("⚠️ Kriz rejimi tespit edildi - korelasyonlar yükseliyor")
            
            return PortfolioCorrelation(
                correlation_matrix=corr_matrix,
                avg_correlation=round(avg_corr, 4),
                max_correlation=(max_corr[0], max_corr[1], round(max_corr[2], 4)),
                min_correlation=(min_corr[0], min_corr[1], round(min_corr[2], 4)),
                effective_n=effective_n,
                diversification_score=div_score,
                regime=regime,
                risk_contributions=risk_contribs,
                clusters=clusters,
                warnings=warnings
            )
        
        except Exception as e:
            logger.error(f"❌ Correlation Analysis Error: {e}")
            import traceback
            traceback.print_exc()
            return self._empty_analysis()
    
    def _empty_analysis(self) -> PortfolioCorrelation:
        """Boş analiz sonucu"""
        return PortfolioCorrelation(
            correlation_matrix=pd.DataFrame(),
            avg_correlation=0,
            max_correlation=("", "", 0),
            min_correlation=("", "", 0),
            effective_n=0,
            diversification_score=0,
            regime=CorrelationRegime.NORMAL,
            risk_contributions=[],
            clusters=[],
            warnings=["Yetersiz veri"]
        )


# ═══════════════════════════════════════════════════════════════════════════════
# TEST
# ═══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    print("Correlation Engine v4.2 - Test")
    print("=" * 60)
    
    # Test data
    np.random.seed(42)
    dates = pd.date_range(start='2024-01-01', periods=252, freq='D')
    
    # Simüle edilmiş getiriler (korelasyonlu)
    market = np.random.randn(252) * 0.02
    
    returns_dict = {
        'THYAO': pd.Series(market * 1.2 + np.random.randn(252) * 0.01, index=dates),
        'TCELL': pd.Series(market * 0.8 + np.random.randn(252) * 0.015, index=dates),
        'GARAN': pd.Series(market * 1.5 + np.random.randn(252) * 0.02, index=dates),
        'AKBNK': pd.Series(market * 1.4 + np.random.randn(252) * 0.018, index=dates),
        'EREGL': pd.Series(market * 0.6 + np.random.randn(252) * 0.025, index=dates),
    }
    
    engine = CorrelationEngine()
    analysis = engine.analyze_portfolio(returns_dict)
    
    print(f"\n📊 KORELASYON ANALİZİ")
    print("=" * 60)
    print(f"📈 Ortalama Korelasyon: {analysis.avg_correlation:.4f}")
    print(f"📊 Max Korelasyon: {analysis.max_correlation[0]}-{analysis.max_correlation[1]} ({analysis.max_correlation[2]:.4f})")
    print(f"📉 Min Korelasyon: {analysis.min_correlation[0]}-{analysis.min_correlation[1]} ({analysis.min_correlation[2]:.4f})")
    print(f"🎯 Diversifikasyon Skoru: {analysis.diversification_score}/100")
    print(f"📊 Efektif N: {analysis.effective_n:.2f}/{len(returns_dict)}")
    print(f"🌡️ Rejim: {analysis.regime.value}")
    print(f"\n📊 Risk Katkıları:")
    for rc in analysis.risk_contributions[:3]:
        print(f"  {rc.symbol}: {rc.percent_contribution:.1f}%")
    print(f"\n⚠️ Uyarılar: {len(analysis.warnings)}")
    for w in analysis.warnings:
        print(f"  {w}")
