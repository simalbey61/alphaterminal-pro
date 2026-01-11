"""
AlphaTerminal Pro - AI Learning Models
======================================

AI strateji sistemi için öğrenme modelleri:
- Winner History: Kazananların geçmişi
- Discovered Pattern: Keşfedilen patternler
- Evolution Log: Strateji evrim geçmişi
- Market Regime: Piyasa rejimi

Author: AlphaTerminal Team
Version: 1.0.0
"""

from __future__ import annotations

import uuid
from datetime import datetime, date
from decimal import Decimal
from typing import Optional, List, Dict, Any, TYPE_CHECKING

from sqlalchemy import String, Numeric, ForeignKey, Index, Text, Integer, Boolean, Date
from sqlalchemy.dialects.postgresql import UUID, JSONB
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.db.models.base import BaseModel

if TYPE_CHECKING:
    from app.db.models.stock import StockModel
    from app.db.models.strategy import AIStrategyModel


class WinnerHistoryModel(BaseModel):
    """
    Kazanan hisse geçmişi modeli.
    
    Belirli periyotlarda yüksek getiri sağlayan hisselerin
    pre-move feature'larını saklar. AI sistemi bu verileri
    pattern keşfi için kullanır.
    
    Attributes:
        symbol: Hisse sembolü
        stock_id: Hisse foreign key
        date: Tarih
        period: Periyot (daily, weekly, monthly)
        gain_percent: Kazanç yüzdesi
        
        pre_move_features: Hareket öncesi feature'lar (JSON)
        market_regime: O anki piyasa rejimi
        sector_strength: Sektör gücü
        xu100_trend: XU100 trendi
        
        volume_spike: Hacim spike var mıydı
        news_catalyst: Haber katalizörü var mıydı
    """
    
    __tablename__ = "winner_histories"
    
    # İlişkiler
    symbol: Mapped[str] = mapped_column(
        String(20),
        nullable=False,
        index=True,
        comment="Hisse sembolü"
    )
    stock_id: Mapped[Optional[uuid.UUID]] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("stocks.id", ondelete="SET NULL"),
        nullable=True,
        index=True,
        comment="Hisse foreign key"
    )
    
    # Tarih ve periyot
    date: Mapped[date] = mapped_column(
        Date,
        nullable=False,
        index=True,
        comment="Tarih"
    )
    period: Mapped[str] = mapped_column(
        String(20),
        nullable=False,
        index=True,
        comment="Periyot (daily, weekly, monthly)"
    )
    
    # Performans
    gain_percent: Mapped[Decimal] = mapped_column(
        Numeric(8, 4),
        nullable=False,
        index=True,
        comment="Kazanç yüzdesi"
    )
    open_price: Mapped[Decimal] = mapped_column(
        Numeric(12, 4),
        nullable=False,
        comment="Açılış fiyatı"
    )
    close_price: Mapped[Decimal] = mapped_column(
        Numeric(12, 4),
        nullable=False,
        comment="Kapanış fiyatı"
    )
    high_price: Mapped[Decimal] = mapped_column(
        Numeric(12, 4),
        nullable=False,
        comment="En yüksek fiyat"
    )
    low_price: Mapped[Decimal] = mapped_column(
        Numeric(12, 4),
        nullable=False,
        comment="En düşük fiyat"
    )
    volume: Mapped[int] = mapped_column(
        Integer,
        nullable=False,
        comment="Hacim"
    )
    
    # Pre-move features (hareket öncesi 1-5-10 gün)
    pre_move_features: Mapped[Dict[str, Any]] = mapped_column(
        JSONB,
        nullable=False,
        comment="Hareket öncesi feature'lar (200+ feature)"
    )
    
    # Piyasa bağlamı
    market_regime: Mapped[Optional[str]] = mapped_column(
        String(30),
        nullable=True,
        index=True,
        comment="Piyasa rejimi"
    )
    sector_strength: Mapped[Optional[Decimal]] = mapped_column(
        Numeric(5, 4),
        nullable=True,
        comment="Sektör gücü (0-1)"
    )
    xu100_trend: Mapped[Optional[str]] = mapped_column(
        String(20),
        nullable=True,
        comment="XU100 trendi"
    )
    xu100_change: Mapped[Optional[Decimal]] = mapped_column(
        Numeric(8, 4),
        nullable=True,
        comment="XU100 değişimi (%)"
    )
    
    # Katalizörler
    volume_spike: Mapped[bool] = mapped_column(
        Boolean,
        default=False,
        nullable=False,
        comment="Hacim spike var mıydı"
    )
    volume_ratio: Mapped[Optional[Decimal]] = mapped_column(
        Numeric(6, 2),
        nullable=True,
        comment="Hacim oranı (ortalamaya göre)"
    )
    news_catalyst: Mapped[bool] = mapped_column(
        Boolean,
        default=False,
        nullable=False,
        comment="Haber katalizörü var mıydı"
    )
    
    # SMC verileri (hareket öncesi)
    smc_structure: Mapped[Optional[str]] = mapped_column(
        String(20),
        nullable=True,
        comment="SMC yapısı"
    )
    had_order_block: Mapped[bool] = mapped_column(
        Boolean,
        default=False,
        nullable=False,
        comment="Order block var mıydı"
    )
    had_fvg: Mapped[bool] = mapped_column(
        Boolean,
        default=False,
        nullable=False,
        comment="FVG var mıydı"
    )
    had_liquidity_sweep: Mapped[bool] = mapped_column(
        Boolean,
        default=False,
        nullable=False,
        comment="Liquidity sweep var mıydı"
    )
    
    # Relationship
    stock: Mapped[Optional["StockModel"]] = relationship(
        "StockModel",
        back_populates="winner_histories",
        lazy="selectin"
    )
    
    # Indexes
    __table_args__ = (
        Index("ix_winners_date_period", "date", "period"),
        Index("ix_winners_symbol_date", "symbol", "date"),
        Index("ix_winners_gain_period", "gain_percent", "period"),
        Index("ix_winners_market_regime", "market_regime", "date"),
    )
    
    def __repr__(self) -> str:
        """String representation."""
        return f"<WinnerHistory({self.symbol} {self.date} +{self.gain_percent:.2f}%)>"


class DiscoveredPatternModel(BaseModel):
    """
    Keşfedilen pattern modeli.
    
    Pattern Discovery Engine tarafından keşfedilen
    patternleri saklar.
    
    Attributes:
        pattern_type: Pattern türü
        conditions: Pattern koşulları (JSON)
        confidence: Güven seviyesi
        support: Destek oranı
        sample_size: Örnek sayısı
        
        avg_return: Ortalama getiri
        win_rate: Kazanma oranı
        
        is_active: Aktif mi
        converted_to_strategy_id: Strateji dönüştürüldü mü
    """
    
    __tablename__ = "discovered_patterns"
    
    # Pattern türü ve koşulları
    pattern_type: Mapped[str] = mapped_column(
        String(50),
        nullable=False,
        index=True,
        comment="Pattern türü (decision_tree, association, clustering, sequence)"
    )
    name: Mapped[str] = mapped_column(
        String(200),
        nullable=False,
        comment="Pattern adı"
    )
    conditions: Mapped[List[Dict[str, Any]]] = mapped_column(
        JSONB,
        nullable=False,
        comment="Pattern koşulları"
    )
    
    # İstatistikler
    confidence: Mapped[Decimal] = mapped_column(
        Numeric(5, 4),
        nullable=False,
        index=True,
        comment="Güven seviyesi (0-1)"
    )
    support: Mapped[Decimal] = mapped_column(
        Numeric(5, 4),
        nullable=False,
        comment="Destek oranı (0-1)"
    )
    lift: Mapped[Optional[Decimal]] = mapped_column(
        Numeric(6, 3),
        nullable=True,
        comment="Lift değeri (association rules)"
    )
    sample_size: Mapped[int] = mapped_column(
        Integer,
        nullable=False,
        comment="Örnek sayısı"
    )
    
    # Performans
    avg_return: Mapped[Decimal] = mapped_column(
        Numeric(8, 4),
        nullable=False,
        comment="Ortalama getiri (%)"
    )
    win_rate: Mapped[Decimal] = mapped_column(
        Numeric(5, 4),
        nullable=False,
        comment="Kazanma oranı (0-1)"
    )
    profit_factor: Mapped[Optional[Decimal]] = mapped_column(
        Numeric(6, 3),
        nullable=True,
        comment="Profit factor"
    )
    
    # Durum
    is_active: Mapped[bool] = mapped_column(
        Boolean,
        default=True,
        nullable=False,
        index=True,
        comment="Aktif mi"
    )
    is_validated: Mapped[bool] = mapped_column(
        Boolean,
        default=False,
        nullable=False,
        comment="Doğrulandı mı"
    )
    
    # Strateji dönüşümü
    converted_to_strategy_id: Mapped[Optional[uuid.UUID]] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("ai_strategies.id", ondelete="SET NULL"),
        nullable=True,
        comment="Dönüştürülen strateji ID"
    )
    converted_at: Mapped[Optional[datetime]] = mapped_column(
        nullable=True,
        comment="Dönüşüm zamanı"
    )
    
    # Keşif detayları
    discovery_details: Mapped[Optional[Dict[str, Any]]] = mapped_column(
        JSONB,
        nullable=True,
        comment="Keşif detayları"
    )
    feature_importance: Mapped[Optional[Dict[str, float]]] = mapped_column(
        JSONB,
        nullable=True,
        comment="Feature önem dereceleri"
    )
    
    # Indexes
    __table_args__ = (
        Index("ix_patterns_type_active", "pattern_type", "is_active"),
        Index("ix_patterns_confidence", "confidence", "is_active"),
        Index("ix_patterns_win_rate", "win_rate", "is_active"),
    )
    
    def __repr__(self) -> str:
        """String representation."""
        return f"<DiscoveredPattern({self.name} - {self.confidence:.2%})>"


class EvolutionLogModel(BaseModel):
    """
    Strateji evrim logu modeli.
    
    AI stratejilerinin evrim geçmişini saklar.
    
    Attributes:
        strategy_id: Strateji foreign key
        change_type: Değişiklik türü
        old_value: Eski değer
        new_value: Yeni değer
        reason: Değişiklik sebebi
        impact_score: Etki skoru
    """
    
    __tablename__ = "evolution_logs"
    
    # İlişkiler
    strategy_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("ai_strategies.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
        comment="Strateji foreign key"
    )
    
    # Değişiklik bilgileri
    change_type: Mapped[str] = mapped_column(
        String(50),
        nullable=False,
        index=True,
        comment="Değişiklik türü"
    )
    old_value: Mapped[Optional[Dict[str, Any]]] = mapped_column(
        JSONB,
        nullable=True,
        comment="Eski değer"
    )
    new_value: Mapped[Optional[Dict[str, Any]]] = mapped_column(
        JSONB,
        nullable=True,
        comment="Yeni değer"
    )
    
    # Açıklama
    reason: Mapped[str] = mapped_column(
        Text,
        nullable=False,
        comment="Değişiklik sebebi"
    )
    
    # Etki
    impact_score: Mapped[Optional[Decimal]] = mapped_column(
        Numeric(5, 4),
        nullable=True,
        comment="Etki skoru"
    )
    
    # Tetikleyici
    trigger_signal_id: Mapped[Optional[uuid.UUID]] = mapped_column(
        UUID(as_uuid=True),
        nullable=True,
        comment="Tetikleyen sinyal ID"
    )
    
    # İlişki
    strategy: Mapped["AIStrategyModel"] = relationship(
        "AIStrategyModel",
        back_populates="evolution_logs",
        lazy="selectin"
    )
    
    # Indexes
    __table_args__ = (
        Index("ix_evolution_strategy_type", "strategy_id", "change_type"),
        Index("ix_evolution_created", "created_at", "strategy_id"),
    )
    
    def __repr__(self) -> str:
        """String representation."""
        return f"<EvolutionLog({self.change_type} @ {self.created_at})>"


class MarketRegimeModel(BaseModel):
    """
    Piyasa rejimi modeli.
    
    Günlük piyasa rejimi bilgilerini saklar.
    
    Attributes:
        date: Tarih
        trend_regime: Trend rejimi
        volatility_regime: Volatilite rejimi
        xu100_trend: XU100 trendi
        breadth: Piyasa genişliği
        sector_rotation: Sektör rotasyonu
    """
    
    __tablename__ = "market_regimes"
    
    # Override id - date primary key olacak
    date: Mapped[date] = mapped_column(
        Date,
        primary_key=True,
        comment="Tarih"
    )
    
    # Rejim bilgileri
    trend_regime: Mapped[str] = mapped_column(
        String(30),
        nullable=False,
        index=True,
        comment="Trend rejimi (bull, bear, sideways)"
    )
    volatility_regime: Mapped[str] = mapped_column(
        String(30),
        nullable=False,
        index=True,
        comment="Volatilite rejimi (low, normal, high)"
    )
    
    # XU100 verileri
    xu100_close: Mapped[Decimal] = mapped_column(
        Numeric(12, 2),
        nullable=False,
        comment="XU100 kapanış"
    )
    xu100_change_pct: Mapped[Decimal] = mapped_column(
        Numeric(8, 4),
        nullable=False,
        comment="XU100 değişim (%)"
    )
    xu100_trend: Mapped[str] = mapped_column(
        String(20),
        nullable=False,
        comment="XU100 trendi"
    )
    
    # Piyasa genişliği
    breadth: Mapped[Decimal] = mapped_column(
        Numeric(5, 4),
        nullable=False,
        comment="MA üstü hisse oranı"
    )
    advancing: Mapped[int] = mapped_column(
        Integer,
        nullable=False,
        comment="Yükselen hisse sayısı"
    )
    declining: Mapped[int] = mapped_column(
        Integer,
        nullable=False,
        comment="Düşen hisse sayısı"
    )
    unchanged: Mapped[int] = mapped_column(
        Integer,
        default=0,
        nullable=False,
        comment="Değişmeyen hisse sayısı"
    )
    
    # Volatilite
    avg_atr_pct: Mapped[Decimal] = mapped_column(
        Numeric(6, 4),
        nullable=False,
        comment="Ortalama ATR yüzdesi"
    )
    vix_equivalent: Mapped[Optional[Decimal]] = mapped_column(
        Numeric(6, 2),
        nullable=True,
        comment="VIX eşdeğeri"
    )
    
    # Sektör
    leading_sector: Mapped[Optional[str]] = mapped_column(
        String(30),
        nullable=True,
        comment="Lider sektör"
    )
    lagging_sector: Mapped[Optional[str]] = mapped_column(
        String(30),
        nullable=True,
        comment="Gerileyen sektör"
    )
    sector_dispersion: Mapped[Optional[Decimal]] = mapped_column(
        Numeric(6, 4),
        nullable=True,
        comment="Sektör dağılımı"
    )
    
    # Hacim
    total_volume: Mapped[int] = mapped_column(
        Integer,
        nullable=False,
        comment="Toplam hacim"
    )
    volume_ratio: Mapped[Decimal] = mapped_column(
        Numeric(6, 2),
        nullable=False,
        comment="Hacim oranı (20 günlük ortalamaya göre)"
    )
    
    def __repr__(self) -> str:
        """String representation."""
        return f"<MarketRegime({self.date} - {self.trend_regime}/{self.volatility_regime})>"
    
    @property
    def is_bullish(self) -> bool:
        """Boğa piyasası mı."""
        return self.trend_regime == "bull"
    
    @property
    def is_bearish(self) -> bool:
        """Ayı piyasası mı."""
        return self.trend_regime == "bear"
    
    @property
    def is_high_volatility(self) -> bool:
        """Yüksek volatilite mi."""
        return self.volatility_regime == "high"


class StrategyPerformanceModel(BaseModel):
    """
    Strateji günlük performans modeli.
    
    AI stratejilerinin günlük performansını saklar.
    """
    
    __tablename__ = "strategy_performances"
    
    # İlişkiler
    strategy_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("ai_strategies.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
        comment="Strateji foreign key"
    )
    date: Mapped[date] = mapped_column(
        Date,
        nullable=False,
        index=True,
        comment="Tarih"
    )
    
    # Sinyal istatistikleri
    total_signals: Mapped[int] = mapped_column(
        Integer,
        default=0,
        nullable=False,
        comment="Toplam sinyal"
    )
    wins: Mapped[int] = mapped_column(
        Integer,
        default=0,
        nullable=False,
        comment="Kazanan"
    )
    losses: Mapped[int] = mapped_column(
        Integer,
        default=0,
        nullable=False,
        comment="Kaybeden"
    )
    
    # Performans
    win_rate: Mapped[Optional[Decimal]] = mapped_column(
        Numeric(5, 4),
        nullable=True,
        comment="Kazanma oranı"
    )
    profit_factor: Mapped[Optional[Decimal]] = mapped_column(
        Numeric(6, 3),
        nullable=True,
        comment="Profit factor"
    )
    total_return: Mapped[Decimal] = mapped_column(
        Numeric(10, 4),
        default=Decimal("0"),
        nullable=False,
        comment="Toplam getiri (%)"
    )
    
    # Risk
    max_drawdown: Mapped[Decimal] = mapped_column(
        Numeric(6, 4),
        default=Decimal("0"),
        nullable=False,
        comment="Maksimum drawdown"
    )
    sharpe_ratio: Mapped[Optional[Decimal]] = mapped_column(
        Numeric(6, 3),
        nullable=True,
        comment="Sharpe oranı"
    )
    
    # Relationship
    strategy: Mapped["AIStrategyModel"] = relationship(
        "AIStrategyModel",
        back_populates="performances",
        lazy="selectin"
    )
    
    # Indexes
    __table_args__ = (
        Index("ix_perf_strategy_date", "strategy_id", "date"),
    )
    
    def __repr__(self) -> str:
        """String representation."""
        return f"<StrategyPerformance({self.strategy_id} @ {self.date})>"
