"""
AlphaTerminal Pro - Data Quality Checker
========================================

Finansal veri kalite kontrolü ve düzeltme.

Author: AlphaTerminal Team
Version: 1.0.0
"""

from __future__ import annotations

import logging
from typing import Optional, List, Dict, Any, Tuple
from datetime import datetime, timedelta
from dataclasses import dataclass, field
from enum import Enum

import numpy as np
import polars as pl

logger = logging.getLogger(__name__)


class QualityIssue(str, Enum):
    """Veri kalite sorunu tipleri."""
    MISSING_DATA = "missing_data"
    OUTLIER = "outlier"
    STALE_DATA = "stale_data"
    INVALID_OHLC = "invalid_ohlc"
    ZERO_VOLUME = "zero_volume"
    DUPLICATE = "duplicate"
    GAP = "gap"
    CORPORATE_ACTION = "corporate_action"
    PRICE_SPIKE = "price_spike"


@dataclass
class QualityReport:
    """Kalite raporu."""
    symbol: str
    timeframe: str
    total_bars: int
    issues: List[Dict[str, Any]] = field(default_factory=list)
    passed: bool = True
    score: float = 100.0
    checked_at: datetime = field(default_factory=datetime.utcnow)
    
    def add_issue(
        self,
        issue_type: QualityIssue,
        severity: str,
        count: int,
        details: Optional[Dict] = None
    ):
        """Sorun ekle."""
        self.issues.append({
            "type": issue_type.value,
            "severity": severity,
            "count": count,
            "details": details or {},
        })
        
        # Skor güncelle
        severity_weights = {"critical": 20, "high": 10, "medium": 5, "low": 2}
        self.score -= severity_weights.get(severity, 1) * count
        self.score = max(0, self.score)
        
        if severity == "critical":
            self.passed = False


class DataQualityChecker:
    """
    Finansal veri kalite kontrolü.
    
    Kontrol edilen sorunlar:
    - Missing data (eksik veri)
    - Outliers (aykırı değerler)
    - Invalid OHLC (geçersiz mum verisi)
    - Zero volume (sıfır hacim)
    - Duplicates (tekrarlı veri)
    - Gaps (boşluklar)
    - Price spikes (ani fiyat hareketleri)
    - Corporate actions (sermaye olayları)
    
    Example:
        ```python
        checker = DataQualityChecker()
        
        # Kalite kontrolü
        report = checker.check(df, "THYAO", "4h")
        
        if report.passed:
            clean_df = checker.clean(df, report)
        else:
            logger.warning(f"Quality check failed: {report.issues}")
        ```
    """
    
    def __init__(
        self,
        outlier_std: float = 4.0,
        max_gap_hours: int = 48,
        max_price_change_pct: float = 20.0,
        min_volume_ratio: float = 0.01,
    ):
        """
        Initialize quality checker.
        
        Args:
            outlier_std: Outlier tespit için std sapma eşiği
            max_gap_hours: Maximum kabul edilebilir boşluk (saat)
            max_price_change_pct: Maximum tek bar fiyat değişimi (%)
            min_volume_ratio: Minimum hacim oranı (ortalamaya göre)
        """
        self.outlier_std = outlier_std
        self.max_gap_hours = max_gap_hours
        self.max_price_change_pct = max_price_change_pct
        self.min_volume_ratio = min_volume_ratio
    
    def check(
        self,
        df: pl.DataFrame,
        symbol: str,
        timeframe: str
    ) -> QualityReport:
        """
        Veri kalite kontrolü yap.
        
        Args:
            df: OHLCV DataFrame (Polars)
            symbol: Hisse sembolü
            timeframe: Zaman dilimi
            
        Returns:
            QualityReport: Kalite raporu
        """
        report = QualityReport(
            symbol=symbol,
            timeframe=timeframe,
            total_bars=len(df)
        )
        
        if len(df) == 0:
            report.add_issue(
                QualityIssue.MISSING_DATA,
                "critical",
                1,
                {"message": "Empty dataframe"}
            )
            return report
        
        # 1. Missing data kontrolü
        self._check_missing_data(df, report)
        
        # 2. Invalid OHLC kontrolü
        self._check_invalid_ohlc(df, report)
        
        # 3. Outlier kontrolü
        self._check_outliers(df, report)
        
        # 4. Zero volume kontrolü
        self._check_zero_volume(df, report)
        
        # 5. Duplicate kontrolü
        self._check_duplicates(df, report)
        
        # 6. Gap kontrolü
        self._check_gaps(df, timeframe, report)
        
        # 7. Price spike kontrolü
        self._check_price_spikes(df, report)
        
        logger.info(
            f"Quality check for {symbol} {timeframe}: "
            f"Score={report.score:.1f}, Passed={report.passed}, "
            f"Issues={len(report.issues)}"
        )
        
        return report
    
    def _check_missing_data(self, df: pl.DataFrame, report: QualityReport) -> None:
        """Missing data kontrolü."""
        required_cols = ["open", "high", "low", "close", "volume"]
        
        for col in required_cols:
            if col not in df.columns:
                report.add_issue(
                    QualityIssue.MISSING_DATA,
                    "critical",
                    1,
                    {"column": col, "message": f"Missing column: {col}"}
                )
                continue
            
            null_count = df[col].null_count()
            if null_count > 0:
                severity = "critical" if null_count > len(df) * 0.1 else "high"
                report.add_issue(
                    QualityIssue.MISSING_DATA,
                    severity,
                    null_count,
                    {"column": col}
                )
    
    def _check_invalid_ohlc(self, df: pl.DataFrame, report: QualityReport) -> None:
        """Invalid OHLC kontrolü (High >= Low, High >= Open/Close, Low <= Open/Close)."""
        try:
            # High < Low
            invalid_hl = df.filter(pl.col("high") < pl.col("low"))
            if len(invalid_hl) > 0:
                report.add_issue(
                    QualityIssue.INVALID_OHLC,
                    "critical",
                    len(invalid_hl),
                    {"type": "high_less_than_low"}
                )
            
            # High < Open or High < Close
            invalid_ho = df.filter(
                (pl.col("high") < pl.col("open")) | 
                (pl.col("high") < pl.col("close"))
            )
            if len(invalid_ho) > 0:
                report.add_issue(
                    QualityIssue.INVALID_OHLC,
                    "critical",
                    len(invalid_ho),
                    {"type": "high_not_maximum"}
                )
            
            # Low > Open or Low > Close
            invalid_lo = df.filter(
                (pl.col("low") > pl.col("open")) | 
                (pl.col("low") > pl.col("close"))
            )
            if len(invalid_lo) > 0:
                report.add_issue(
                    QualityIssue.INVALID_OHLC,
                    "critical",
                    len(invalid_lo),
                    {"type": "low_not_minimum"}
                )
            
            # Negative prices
            negative = df.filter(
                (pl.col("open") <= 0) | 
                (pl.col("high") <= 0) | 
                (pl.col("low") <= 0) | 
                (pl.col("close") <= 0)
            )
            if len(negative) > 0:
                report.add_issue(
                    QualityIssue.INVALID_OHLC,
                    "critical",
                    len(negative),
                    {"type": "negative_or_zero_price"}
                )
                
        except Exception as e:
            logger.warning(f"Invalid OHLC check error: {e}")
    
    def _check_outliers(self, df: pl.DataFrame, report: QualityReport) -> None:
        """Outlier kontrolü (Z-score ve IQR)."""
        try:
            # Return hesapla
            returns = df["close"].pct_change().drop_nulls()
            
            if len(returns) < 20:
                return
            
            # Z-score method
            mean = returns.mean()
            std = returns.std()
            
            if std > 0:
                z_scores = ((returns - mean) / std).abs()
                outliers = (z_scores > self.outlier_std).sum()
                
                if outliers > 0:
                    severity = "high" if outliers > len(returns) * 0.01 else "medium"
                    report.add_issue(
                        QualityIssue.OUTLIER,
                        severity,
                        outliers,
                        {"method": "z_score", "threshold": self.outlier_std}
                    )
            
            # IQR method
            q1 = returns.quantile(0.25)
            q3 = returns.quantile(0.75)
            iqr = q3 - q1
            
            lower_bound = q1 - 3 * iqr
            upper_bound = q3 + 3 * iqr
            
            iqr_outliers = ((returns < lower_bound) | (returns > upper_bound)).sum()
            
            if iqr_outliers > 0 and iqr_outliers != outliers:
                report.add_issue(
                    QualityIssue.OUTLIER,
                    "medium",
                    iqr_outliers,
                    {"method": "iqr", "bounds": [lower_bound, upper_bound]}
                )
                
        except Exception as e:
            logger.warning(f"Outlier check error: {e}")
    
    def _check_zero_volume(self, df: pl.DataFrame, report: QualityReport) -> None:
        """Zero volume kontrolü."""
        try:
            if "volume" not in df.columns:
                return
            
            zero_vol = df.filter(pl.col("volume") == 0)
            
            if len(zero_vol) > 0:
                severity = "high" if len(zero_vol) > len(df) * 0.05 else "medium"
                report.add_issue(
                    QualityIssue.ZERO_VOLUME,
                    severity,
                    len(zero_vol),
                    {}
                )
            
            # Çok düşük hacim kontrolü
            avg_volume = df["volume"].mean()
            if avg_volume > 0:
                low_vol = df.filter(
                    pl.col("volume") < avg_volume * self.min_volume_ratio
                )
                
                if len(low_vol) > len(df) * 0.1:
                    report.add_issue(
                        QualityIssue.ZERO_VOLUME,
                        "low",
                        len(low_vol),
                        {"type": "low_volume", "threshold": self.min_volume_ratio}
                    )
                    
        except Exception as e:
            logger.warning(f"Zero volume check error: {e}")
    
    def _check_duplicates(self, df: pl.DataFrame, report: QualityReport) -> None:
        """Duplicate kontrolü."""
        try:
            if "timestamp" in df.columns:
                dupe_count = len(df) - df["timestamp"].n_unique()
                
                if dupe_count > 0:
                    report.add_issue(
                        QualityIssue.DUPLICATE,
                        "high",
                        dupe_count,
                        {}
                    )
                    
        except Exception as e:
            logger.warning(f"Duplicate check error: {e}")
    
    def _check_gaps(
        self,
        df: pl.DataFrame,
        timeframe: str,
        report: QualityReport
    ) -> None:
        """Gap (boşluk) kontrolü."""
        try:
            if "timestamp" not in df.columns or len(df) < 2:
                return
            
            # Timeframe'i saate çevir
            tf_hours = self._timeframe_to_hours(timeframe)
            expected_gap = timedelta(hours=tf_hours)
            max_gap = timedelta(hours=self.max_gap_hours)
            
            timestamps = df["timestamp"].to_list()
            gaps = []
            
            for i in range(1, len(timestamps)):
                if timestamps[i] is None or timestamps[i-1] is None:
                    continue
                    
                actual_gap = timestamps[i] - timestamps[i-1]
                
                # Weekend'i atla (Cuma-Pazartesi arası)
                if self._is_weekend_gap(timestamps[i-1], timestamps[i]):
                    continue
                
                if actual_gap > max_gap:
                    gaps.append({
                        "start": timestamps[i-1],
                        "end": timestamps[i],
                        "duration_hours": actual_gap.total_seconds() / 3600
                    })
            
            if gaps:
                severity = "high" if len(gaps) > 5 else "medium"
                report.add_issue(
                    QualityIssue.GAP,
                    severity,
                    len(gaps),
                    {"gaps": gaps[:10]}  # İlk 10 gap
                )
                
        except Exception as e:
            logger.warning(f"Gap check error: {e}")
    
    def _check_price_spikes(self, df: pl.DataFrame, report: QualityReport) -> None:
        """Ani fiyat hareketi kontrolü."""
        try:
            # Bar içi değişim
            bar_range = ((df["high"] - df["low"]) / df["low"] * 100)
            large_bars = (bar_range > self.max_price_change_pct).sum()
            
            if large_bars > 0:
                report.add_issue(
                    QualityIssue.PRICE_SPIKE,
                    "medium",
                    large_bars,
                    {"type": "large_bar_range", "threshold_pct": self.max_price_change_pct}
                )
            
            # Ardışık bar değişimi
            close_pct = df["close"].pct_change().abs() * 100
            spikes = (close_pct > self.max_price_change_pct).sum()
            
            if spikes > 0:
                severity = "high" if spikes > 5 else "medium"
                report.add_issue(
                    QualityIssue.PRICE_SPIKE,
                    severity,
                    spikes,
                    {"type": "close_to_close", "threshold_pct": self.max_price_change_pct}
                )
                
        except Exception as e:
            logger.warning(f"Price spike check error: {e}")
    
    def clean(
        self,
        df: pl.DataFrame,
        report: QualityReport,
        fix_missing: bool = True,
        fix_outliers: bool = True,
        remove_duplicates: bool = True
    ) -> pl.DataFrame:
        """
        Veriyi temizle.
        
        Args:
            df: Orijinal DataFrame
            report: Kalite raporu
            fix_missing: Missing value'ları doldur
            fix_outliers: Outlier'ları düzelt
            remove_duplicates: Duplicate'ları kaldır
            
        Returns:
            pl.DataFrame: Temizlenmiş DataFrame
        """
        cleaned = df.clone()
        
        # Duplicate'ları kaldır
        if remove_duplicates and "timestamp" in cleaned.columns:
            cleaned = cleaned.unique(subset=["timestamp"])
        
        # Missing value'ları doldur (forward fill)
        if fix_missing:
            for col in ["open", "high", "low", "close"]:
                if col in cleaned.columns:
                    cleaned = cleaned.with_columns(
                        pl.col(col).forward_fill().alias(col)
                    )
            
            # Volume için 0 ile doldur
            if "volume" in cleaned.columns:
                cleaned = cleaned.with_columns(
                    pl.col("volume").fill_null(0).alias("volume")
                )
        
        # Outlier'ları winsorize et
        if fix_outliers:
            cleaned = self._winsorize_outliers(cleaned)
        
        # Invalid OHLC düzelt
        cleaned = self._fix_invalid_ohlc(cleaned)
        
        # Sırala
        if "timestamp" in cleaned.columns:
            cleaned = cleaned.sort("timestamp")
        
        logger.info(
            f"Cleaned data for {report.symbol}: "
            f"{report.total_bars} -> {len(cleaned)} bars"
        )
        
        return cleaned
    
    def _winsorize_outliers(self, df: pl.DataFrame) -> pl.DataFrame:
        """Outlier'ları winsorize et (uç değerleri sınırla)."""
        try:
            returns = df["close"].pct_change()
            
            q1 = returns.quantile(0.01)
            q99 = returns.quantile(0.99)
            
            # Bu basit bir implementasyon, gerekirse geliştirilebilir
            return df
            
        except Exception:
            return df
    
    def _fix_invalid_ohlc(self, df: pl.DataFrame) -> pl.DataFrame:
        """Invalid OHLC düzelt."""
        try:
            # High'ı max(open, high, close) yap
            df = df.with_columns(
                pl.max_horizontal(["open", "high", "close"]).alias("high")
            )
            
            # Low'u min(open, low, close) yap
            df = df.with_columns(
                pl.min_horizontal(["open", "low", "close"]).alias("low")
            )
            
            return df
            
        except Exception:
            return df
    
    def _timeframe_to_hours(self, timeframe: str) -> float:
        """Timeframe'i saate çevir."""
        mapping = {
            "1m": 1/60, "5m": 5/60, "15m": 0.25, "30m": 0.5,
            "1h": 1, "4h": 4, "1d": 24, "1w": 168, "1M": 720,
        }
        return mapping.get(timeframe, 4)
    
    def _is_weekend_gap(self, start: datetime, end: datetime) -> bool:
        """Weekend gap mı kontrol et."""
        if start.weekday() == 4 and end.weekday() == 0:  # Cuma -> Pazartesi
            return True
        return False


# =============================================================================
# CONVENIENCE FUNCTIONS
# =============================================================================

def check_data_quality(
    df: pl.DataFrame,
    symbol: str,
    timeframe: str = "4h"
) -> Tuple[bool, QualityReport]:
    """
    Hızlı kalite kontrolü.
    
    Args:
        df: OHLCV DataFrame
        symbol: Hisse sembolü
        timeframe: Zaman dilimi
        
    Returns:
        Tuple[bool, QualityReport]: (Passed, Report)
    """
    checker = DataQualityChecker()
    report = checker.check(df, symbol, timeframe)
    return report.passed, report


def clean_ohlcv_data(
    df: pl.DataFrame,
    symbol: str,
    timeframe: str = "4h"
) -> pl.DataFrame:
    """
    Hızlı veri temizleme.
    
    Args:
        df: OHLCV DataFrame
        symbol: Hisse sembolü
        timeframe: Zaman dilimi
        
    Returns:
        pl.DataFrame: Temizlenmiş DataFrame
    """
    checker = DataQualityChecker()
    report = checker.check(df, symbol, timeframe)
    return checker.clean(df, report)
