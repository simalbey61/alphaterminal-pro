"""
AlphaTerminal Pro - AI Strategy Model
=====================================

AI strateji veritabanı modeli.

Author: AlphaTerminal Team
Version: 1.0.0
"""

from __future__ import annotations

import uuid
from datetime import datetime
from decimal import Decimal
from typing import Optional, List, Dict, Any, TYPE_CHECKING

from sqlalchemy import String, Numeric, Index, Text, Integer
from sqlalchemy.dialects.postgresql import UUID, JSONB, ARRAY
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.db.models.base import BaseModel
from app.config import StrategyStatus

if TYPE_CHECKING:
    from app.db.models.signal import SignalModel
    from app.db.models.strategy_performance import StrategyPerformanceModel
    from app.db.models.evolution_log import EvolutionLogModel


class AIStrategyModel(BaseModel):
    """
    AI strateji modeli.
    
    Pattern Discovery Engine tarafından keşfedilen ve
    Strategy Generator tarafından oluşturulan AI stratejilerini saklar.
    
    Attributes:
        name: Strateji adı
        version: Versiyon numarası
        description: Strateji açıklaması
        
        conditions: Strateji koşulları (JSON array)
        confidence: Başlangıç güven seviyesi
        sample_size: Keşif örnek sayısı
        
        status: Strateji durumu
        
        stop_loss_atr: Stop loss ATR çarpanı
        take_profit_r: Take profit R çarpanı
        position_size_pct: Pozisyon boyutu yüzdesi
        
        discovery_method: Keşif yöntemi
        parent_strategy_ids: Genetik crossover parent'ları
        
        performance_score: Güncel performans skoru
    
    Relationships:
        signals: Bu strateji tarafından üretilen sinyaller
        performances: Performans kayıtları
        evolution_logs: Evrim geçmişi
    """
    
    __tablename__ = "ai_strategies"
    
    # Temel bilgiler
    name: Mapped[str] = mapped_column(
        String(100),
        nullable=False,
        index=True,
        comment="Strateji adı"
    )
    version: Mapped[str] = mapped_column(
        String(20),
        default="1.0.0",
        nullable=False,
        comment="Versiyon numarası"
    )
    description: Mapped[Optional[str]] = mapped_column(
        Text,
        nullable=True,
        comment="Strateji açıklaması"
    )
    
    # Strateji kuralları
    conditions: Mapped[List[Dict[str, Any]]] = mapped_column(
        JSONB,
        nullable=False,
        comment="Strateji koşulları [{indicator, operator, value}, ...]"
    )
    
    # Keşif bilgileri
    confidence: Mapped[Decimal] = mapped_column(
        Numeric(5, 4),
        nullable=False,
        comment="Başlangıç güven seviyesi (0-1)"
    )
    sample_size: Mapped[int] = mapped_column(
        Integer,
        nullable=False,
        comment="Keşif örnek sayısı"
    )
    discovery_method: Mapped[str] = mapped_column(
        String(50),
        nullable=False,
        comment="Keşif yöntemi (decision_tree, association, clustering, genetic)"
    )
    
    # Durum
    status: Mapped[str] = mapped_column(
        String(30),
        default=StrategyStatus.PENDING_VALIDATION.value,
        nullable=False,
        index=True,
        comment="Strateji durumu"
    )
    
    # Risk parametreleri
    stop_loss_atr: Mapped[Decimal] = mapped_column(
        Numeric(4, 2),
        default=Decimal("1.5"),
        nullable=False,
        comment="Stop loss ATR çarpanı"
    )
    take_profit_r: Mapped[Decimal] = mapped_column(
        Numeric(4, 2),
        default=Decimal("2.0"),
        nullable=False,
        comment="Take profit R çarpanı"
    )
    position_size_pct: Mapped[Decimal] = mapped_column(
        Numeric(5, 4),
        default=Decimal("0.02"),
        nullable=False,
        comment="Pozisyon boyutu yüzdesi"
    )
    
    # Genetik algoritma bilgileri
    parent_strategy_ids: Mapped[Optional[List[uuid.UUID]]] = mapped_column(
        ARRAY(UUID(as_uuid=True)),
        nullable=True,
        comment="Genetik crossover parent ID'leri"
    )
    generation: Mapped[int] = mapped_column(
        Integer,
        default=1,
        nullable=False,
        comment="Genetik jenerasyon numarası"
    )
    
    # Performans skoru
    performance_score: Mapped[Decimal] = mapped_column(
        Numeric(6, 4),
        default=Decimal("0"),
        nullable=False,
        index=True,
        comment="Güncel performans skoru"
    )
    
    # Backtest sonuçları
    backtest_win_rate: Mapped[Optional[Decimal]] = mapped_column(
        Numeric(5, 4),
        nullable=True,
        comment="Backtest kazanma oranı"
    )
    backtest_profit_factor: Mapped[Optional[Decimal]] = mapped_column(
        Numeric(6, 3),
        nullable=True,
        comment="Backtest profit factor"
    )
    backtest_sharpe: Mapped[Optional[Decimal]] = mapped_column(
        Numeric(6, 3),
        nullable=True,
        comment="Backtest Sharpe oranı"
    )
    backtest_max_drawdown: Mapped[Optional[Decimal]] = mapped_column(
        Numeric(5, 4),
        nullable=True,
        comment="Backtest maksimum drawdown"
    )
    backtest_total_trades: Mapped[Optional[int]] = mapped_column(
        Integer,
        nullable=True,
        comment="Backtest toplam trade sayısı"
    )
    backtest_period_days: Mapped[Optional[int]] = mapped_column(
        Integer,
        nullable=True,
        comment="Backtest süresi (gün)"
    )
    
    # Walk-forward validation sonuçları
    walkforward_consistency: Mapped[Optional[Decimal]] = mapped_column(
        Numeric(5, 4),
        nullable=True,
        comment="Walk-forward tutarlılık skoru"
    )
    walkforward_windows_passed: Mapped[Optional[int]] = mapped_column(
        Integer,
        nullable=True,
        comment="Geçilen walk-forward pencere sayısı"
    )
    
    # Canlı performans
    live_win_rate: Mapped[Optional[Decimal]] = mapped_column(
        Numeric(5, 4),
        nullable=True,
        comment="Canlı kazanma oranı"
    )
    live_profit_factor: Mapped[Optional[Decimal]] = mapped_column(
        Numeric(6, 3),
        nullable=True,
        comment="Canlı profit factor"
    )
    live_sharpe: Mapped[Optional[Decimal]] = mapped_column(
        Numeric(6, 3),
        nullable=True,
        comment="Canlı Sharpe oranı"
    )
    live_total_trades: Mapped[int] = mapped_column(
        Integer,
        default=0,
        nullable=False,
        comment="Canlı toplam trade sayısı"
    )
    live_total_pnl: Mapped[Decimal] = mapped_column(
        Numeric(14, 4),
        default=Decimal("0"),
        nullable=False,
        comment="Canlı toplam P&L"
    )
    live_wins: Mapped[int] = mapped_column(
        Integer,
        default=0,
        nullable=False,
        comment="Canlı kazanç sayısı"
    )
    live_losses: Mapped[int] = mapped_column(
        Integer,
        default=0,
        nullable=False,
        comment="Canlı kayıp sayısı"
    )
    
    # Tarihler
    approved_at: Mapped[Optional[datetime]] = mapped_column(
        nullable=True,
        comment="Onay zamanı"
    )
    paused_at: Mapped[Optional[datetime]] = mapped_column(
        nullable=True,
        comment="Duraklatma zamanı"
    )
    retired_at: Mapped[Optional[datetime]] = mapped_column(
        nullable=True,
        comment="Emekli edilme zamanı"
    )
    last_signal_at: Mapped[Optional[datetime]] = mapped_column(
        nullable=True,
        comment="Son sinyal zamanı"
    )
    last_evolution_at: Mapped[Optional[datetime]] = mapped_column(
        nullable=True,
        comment="Son evrim zamanı"
    )
    
    # Ek meta veriler
    metadata: Mapped[Optional[Dict[str, Any]]] = mapped_column(
        JSONB,
        nullable=True,
        comment="Ek meta veriler"
    )
    
    # Relationships
    signals: Mapped[List["SignalModel"]] = relationship(
        "SignalModel",
        back_populates="strategy",
        lazy="selectin"
    )
    
    performances: Mapped[List["StrategyPerformanceModel"]] = relationship(
        "StrategyPerformanceModel",
        back_populates="strategy",
        cascade="all, delete-orphan",
        lazy="selectin"
    )
    
    evolution_logs: Mapped[List["EvolutionLogModel"]] = relationship(
        "EvolutionLogModel",
        back_populates="strategy",
        cascade="all, delete-orphan",
        lazy="selectin"
    )
    
    # Indexes
    __table_args__ = (
        Index("ix_strategies_status_performance", "status", "performance_score"),
        Index("ix_strategies_discovery_method", "discovery_method", "status"),
        Index("ix_strategies_generation", "generation", "status"),
        Index("ix_strategies_approved_at", "approved_at", "status"),
    )
    
    def __repr__(self) -> str:
        """String representation."""
        return f"<AIStrategy({self.name} v{self.version} [{self.status}])>"
    
    @property
    def is_active(self) -> bool:
        """Strateji aktif mi."""
        return self.status == StrategyStatus.ACTIVE.value
    
    @property
    def is_pending(self) -> bool:
        """Strateji onay bekliyor mu."""
        return self.status == StrategyStatus.PENDING_VALIDATION.value
    
    @property
    def is_retired(self) -> bool:
        """Strateji emekli mi."""
        return self.status == StrategyStatus.RETIRED.value
    
    @property
    def actual_win_rate(self) -> Optional[Decimal]:
        """Güncel kazanma oranı."""
        if self.live_total_trades > 0:
            return self.live_win_rate
        return self.backtest_win_rate
    
    @property
    def conditions_summary(self) -> str:
        """Koşulların özeti."""
        if not self.conditions:
            return "No conditions"
        
        summaries = []
        for cond in self.conditions[:3]:  # İlk 3 koşul
            indicator = cond.get("indicator", "?")
            operator = cond.get("operator", "?")
            value = cond.get("value", "?")
            summaries.append(f"{indicator} {operator} {value}")
        
        result = " AND ".join(summaries)
        if len(self.conditions) > 3:
            result += f" (+{len(self.conditions) - 3} more)"
        
        return result
    
    def approve(self) -> None:
        """Stratejiyi onayla."""
        self.status = StrategyStatus.ACTIVE.value
        self.approved_at = datetime.utcnow()
    
    def pause(self) -> None:
        """Stratejiyi duraklat."""
        self.status = StrategyStatus.PAUSED.value
        self.paused_at = datetime.utcnow()
    
    def activate(self) -> None:
        """Stratejiyi aktifleştir."""
        self.status = StrategyStatus.ACTIVE.value
        self.paused_at = None
    
    def retire(self) -> None:
        """Stratejiyi emekli et."""
        self.status = StrategyStatus.RETIRED.value
        self.retired_at = datetime.utcnow()
    
    def record_trade_result(self, is_win: bool, pnl: Decimal) -> None:
        """
        Trade sonucunu kaydet.
        
        Args:
            is_win: Kazanç mı
            pnl: P&L miktarı
        """
        self.live_total_trades += 1
        self.live_total_pnl += pnl
        
        if is_win:
            self.live_wins += 1
        else:
            self.live_losses += 1
        
        # Win rate güncelle
        if self.live_total_trades > 0:
            self.live_win_rate = Decimal(str(self.live_wins / self.live_total_trades))
        
        # Profit factor güncelle (basitleştirilmiş)
        if self.live_losses > 0 and self.live_wins > 0:
            avg_win = self.live_total_pnl / self.live_wins if self.live_total_pnl > 0 else Decimal("0")
            # Bu basitleştirilmiş bir hesaplama
            self.live_profit_factor = Decimal(str(
                (self.live_wins * float(avg_win)) / 
                (self.live_losses * abs(float(self.live_total_pnl) / self.live_total_trades))
            )) if self.live_total_trades > 0 else None
    
    def update_performance_score(self) -> None:
        """Performans skorunu güncelle."""
        score = Decimal("0")
        
        # Win rate katkısı (0.3)
        if self.live_win_rate:
            score += self.live_win_rate * Decimal("0.3")
        elif self.backtest_win_rate:
            score += self.backtest_win_rate * Decimal("0.2")
        
        # Profit factor katkısı (0.3)
        if self.live_profit_factor and self.live_profit_factor > 0:
            pf_score = min(self.live_profit_factor / Decimal("3"), Decimal("1"))
            score += pf_score * Decimal("0.3")
        elif self.backtest_profit_factor and self.backtest_profit_factor > 0:
            pf_score = min(self.backtest_profit_factor / Decimal("3"), Decimal("1"))
            score += pf_score * Decimal("0.2")
        
        # Sharpe katkısı (0.2)
        if self.live_sharpe and self.live_sharpe > 0:
            sharpe_score = min(self.live_sharpe / Decimal("2"), Decimal("1"))
            score += sharpe_score * Decimal("0.2")
        elif self.backtest_sharpe and self.backtest_sharpe > 0:
            sharpe_score = min(self.backtest_sharpe / Decimal("2"), Decimal("1"))
            score += sharpe_score * Decimal("0.15")
        
        # Trade sayısı katkısı (0.2)
        if self.live_total_trades > 0:
            trade_score = min(Decimal(str(self.live_total_trades / 50)), Decimal("1"))
            score += trade_score * Decimal("0.2")
        
        self.performance_score = score
    
    def increment_version(self) -> None:
        """Versiyon numarasını artır."""
        parts = self.version.split(".")
        if len(parts) == 3:
            parts[2] = str(int(parts[2]) + 1)
            self.version = ".".join(parts)
        else:
            self.version = "1.0.1"
        
        self.last_evolution_at = datetime.utcnow()
