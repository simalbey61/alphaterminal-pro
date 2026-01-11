"""
AlphaTerminal Pro - Signal Model
================================

Trading sinyalleri veritabanı modeli.

Author: AlphaTerminal Team
Version: 1.0.0
"""

from __future__ import annotations

import uuid
from datetime import datetime
from decimal import Decimal
from typing import Optional, List, Dict, Any, TYPE_CHECKING

from sqlalchemy import String, Numeric, ForeignKey, Index, Text, Enum as SQLEnum
from sqlalchemy.dialects.postgresql import UUID, JSONB
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.db.models.base import BaseModel
from app.config import SignalType, SignalTier, SignalStatus

if TYPE_CHECKING:
    from app.db.models.stock import StockModel
    from app.db.models.strategy import AIStrategyModel
    from app.db.models.position import PositionModel


class SignalModel(BaseModel):
    """
    Trading sinyali modeli.
    
    SMC, OrderFlow, Alpha ve AI stratejilerinden üretilen
    trading sinyallerini saklar.
    
    Attributes:
        symbol: Hisse sembolü
        stock_id: Hisse foreign key
        strategy_id: AI strateji foreign key (opsiyonel)
        
        signal_type: Sinyal türü (LONG/SHORT)
        tier: Sinyal kalitesi (TIER1/TIER2/TIER3)
        status: Sinyal durumu
        
        entry_price: Giriş fiyatı
        stop_loss: Stop loss fiyatı
        take_profit_1/2/3: Take profit seviyeleri
        
        total_score: Toplam skor
        smc_score: SMC skoru
        orderflow_score: Order flow skoru
        alpha_score: Alpha skoru
        ml_score: ML skoru
        mtf_score: Multi-timeframe skoru
        
        confidence: Güven seviyesi
        risk_reward: Risk/Reward oranı
        setup_type: Setup türü
        
        reasoning: Sinyal gerekçeleri (JSON)
        metadata: Ek meta veriler (JSON)
        
        result_pnl: Sonuç P&L
        result_pnl_pct: Sonuç P&L yüzdesi
        closed_at: Kapanış zamanı
        closed_reason: Kapanış sebebi
    
    Relationships:
        stock: İlgili hisse
        strategy: Üretilen AI stratejisi
        positions: Bu sinyalden açılan pozisyonlar
    """
    
    __tablename__ = "signals"
    
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
    strategy_id: Mapped[Optional[uuid.UUID]] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("ai_strategies.id", ondelete="SET NULL"),
        nullable=True,
        index=True,
        comment="AI strateji foreign key"
    )
    
    # Sinyal türü ve durumu
    signal_type: Mapped[str] = mapped_column(
        String(10),
        nullable=False,
        index=True,
        comment="Sinyal türü (LONG/SHORT)"
    )
    tier: Mapped[str] = mapped_column(
        String(10),
        nullable=False,
        index=True,
        comment="Sinyal kalitesi (TIER1/TIER2/TIER3)"
    )
    status: Mapped[str] = mapped_column(
        String(20),
        default=SignalStatus.ACTIVE.value,
        nullable=False,
        index=True,
        comment="Sinyal durumu"
    )
    
    # Fiyat seviyeleri
    entry_price: Mapped[Decimal] = mapped_column(
        Numeric(12, 4),
        nullable=False,
        comment="Giriş fiyatı"
    )
    stop_loss: Mapped[Decimal] = mapped_column(
        Numeric(12, 4),
        nullable=False,
        comment="Stop loss fiyatı"
    )
    take_profit_1: Mapped[Decimal] = mapped_column(
        Numeric(12, 4),
        nullable=False,
        comment="Take profit 1"
    )
    take_profit_2: Mapped[Optional[Decimal]] = mapped_column(
        Numeric(12, 4),
        nullable=True,
        comment="Take profit 2"
    )
    take_profit_3: Mapped[Optional[Decimal]] = mapped_column(
        Numeric(12, 4),
        nullable=True,
        comment="Take profit 3"
    )
    
    # Skorlar
    total_score: Mapped[Decimal] = mapped_column(
        Numeric(5, 2),
        nullable=False,
        index=True,
        comment="Toplam skor (0-100)"
    )
    smc_score: Mapped[Optional[Decimal]] = mapped_column(
        Numeric(5, 2),
        nullable=True,
        comment="SMC skoru (0-100)"
    )
    orderflow_score: Mapped[Optional[Decimal]] = mapped_column(
        Numeric(5, 2),
        nullable=True,
        comment="Order flow skoru (0-100)"
    )
    alpha_score: Mapped[Optional[Decimal]] = mapped_column(
        Numeric(5, 2),
        nullable=True,
        comment="Alpha skoru (0-100)"
    )
    ml_score: Mapped[Optional[Decimal]] = mapped_column(
        Numeric(5, 2),
        nullable=True,
        comment="ML skoru (0-100)"
    )
    mtf_score: Mapped[Optional[Decimal]] = mapped_column(
        Numeric(5, 2),
        nullable=True,
        comment="MTF skoru (0-100)"
    )
    
    # Meta bilgiler
    confidence: Mapped[Decimal] = mapped_column(
        Numeric(5, 4),
        nullable=False,
        comment="Güven seviyesi (0-1)"
    )
    risk_reward: Mapped[Decimal] = mapped_column(
        Numeric(5, 2),
        nullable=False,
        comment="Risk/Reward oranı"
    )
    setup_type: Mapped[str] = mapped_column(
        String(50),
        nullable=False,
        comment="Setup türü (ORDER_BLOCK, LIQUIDITY_SWEEP, vb.)"
    )
    timeframe: Mapped[str] = mapped_column(
        String(10),
        default="4h",
        nullable=False,
        comment="Zaman dilimi"
    )
    
    # JSON alanlar
    reasoning: Mapped[Optional[List[str]]] = mapped_column(
        JSONB,
        nullable=True,
        comment="Sinyal gerekçeleri"
    )
    smc_data: Mapped[Optional[Dict[str, Any]]] = mapped_column(
        JSONB,
        nullable=True,
        comment="SMC analiz verileri"
    )
    orderflow_data: Mapped[Optional[Dict[str, Any]]] = mapped_column(
        JSONB,
        nullable=True,
        comment="Order flow verileri"
    )
    metadata: Mapped[Optional[Dict[str, Any]]] = mapped_column(
        JSONB,
        nullable=True,
        comment="Ek meta veriler"
    )
    
    # Sonuç bilgileri
    result_pnl: Mapped[Optional[Decimal]] = mapped_column(
        Numeric(12, 4),
        nullable=True,
        comment="Sonuç P&L (TRY)"
    )
    result_pnl_pct: Mapped[Optional[Decimal]] = mapped_column(
        Numeric(8, 4),
        nullable=True,
        comment="Sonuç P&L (%)"
    )
    exit_price: Mapped[Optional[Decimal]] = mapped_column(
        Numeric(12, 4),
        nullable=True,
        comment="Çıkış fiyatı"
    )
    closed_at: Mapped[Optional[datetime]] = mapped_column(
        nullable=True,
        comment="Kapanış zamanı"
    )
    closed_reason: Mapped[Optional[str]] = mapped_column(
        String(50),
        nullable=True,
        comment="Kapanış sebebi"
    )
    
    # Geçerlilik
    valid_until: Mapped[Optional[datetime]] = mapped_column(
        nullable=True,
        comment="Geçerlilik süresi"
    )
    
    # Relationships
    stock: Mapped[Optional["StockModel"]] = relationship(
        "StockModel",
        back_populates="signals",
        lazy="selectin"
    )
    
    strategy: Mapped[Optional["AIStrategyModel"]] = relationship(
        "AIStrategyModel",
        back_populates="signals",
        lazy="selectin"
    )
    
    positions: Mapped[List["PositionModel"]] = relationship(
        "PositionModel",
        back_populates="signal",
        cascade="all, delete-orphan",
        lazy="selectin"
    )
    
    # Indexes
    __table_args__ = (
        Index("ix_signals_symbol_status", "symbol", "status"),
        Index("ix_signals_tier_status", "tier", "status"),
        Index("ix_signals_created_status", "created_at", "status"),
        Index("ix_signals_strategy_status", "strategy_id", "status"),
        Index("ix_signals_total_score", "total_score", "status"),
    )
    
    def __repr__(self) -> str:
        """String representation."""
        return f"<Signal({self.symbol} {self.signal_type} {self.tier})>"
    
    @property
    def is_active(self) -> bool:
        """Sinyal aktif mi."""
        return self.status == SignalStatus.ACTIVE.value
    
    @property
    def is_closed(self) -> bool:
        """Sinyal kapalı mı."""
        return self.status in [
            SignalStatus.STOPPED.value,
            SignalStatus.CLOSED.value,
            SignalStatus.EXPIRED.value,
            SignalStatus.TP1_HIT.value,
            SignalStatus.TP2_HIT.value,
            SignalStatus.TP3_HIT.value,
        ]
    
    @property
    def is_profitable(self) -> Optional[bool]:
        """Karlı mı."""
        if self.result_pnl is None:
            return None
        return self.result_pnl > 0
    
    @property
    def stop_distance_pct(self) -> Decimal:
        """Stop loss mesafesi yüzdesi."""
        if self.entry_price == 0:
            return Decimal("0")
        return abs(self.entry_price - self.stop_loss) / self.entry_price * 100
    
    @property
    def tp1_distance_pct(self) -> Decimal:
        """TP1 mesafesi yüzdesi."""
        if self.entry_price == 0:
            return Decimal("0")
        return abs(self.take_profit_1 - self.entry_price) / self.entry_price * 100
    
    def close(
        self,
        exit_price: Decimal,
        reason: str,
        pnl: Optional[Decimal] = None
    ) -> None:
        """
        Sinyali kapat.
        
        Args:
            exit_price: Çıkış fiyatı
            reason: Kapanış sebebi
            pnl: P&L (opsiyonel, hesaplanır)
        """
        self.exit_price = exit_price
        self.closed_at = datetime.utcnow()
        self.closed_reason = reason
        
        # P&L hesapla
        if pnl is not None:
            self.result_pnl = pnl
        else:
            if self.signal_type == SignalType.LONG.value:
                self.result_pnl = exit_price - self.entry_price
            else:
                self.result_pnl = self.entry_price - exit_price
        
        # P&L yüzdesi
        if self.entry_price > 0:
            self.result_pnl_pct = (self.result_pnl / self.entry_price) * 100
        
        # Durumu güncelle
        if reason == "stopped":
            self.status = SignalStatus.STOPPED.value
        elif reason == "tp1":
            self.status = SignalStatus.TP1_HIT.value
        elif reason == "tp2":
            self.status = SignalStatus.TP2_HIT.value
        elif reason == "tp3":
            self.status = SignalStatus.TP3_HIT.value
        elif reason == "expired":
            self.status = SignalStatus.EXPIRED.value
        else:
            self.status = SignalStatus.CLOSED.value
    
    def to_telegram_message(self) -> str:
        """Telegram mesaj formatı."""
        emoji = "🟢" if self.signal_type == SignalType.LONG.value else "🔴"
        tier_emoji = {"TIER1": "🥇", "TIER2": "🥈", "TIER3": "🥉"}.get(self.tier, "")
        
        message = f"""
{emoji} **{self.symbol}** - {self.signal_type} {tier_emoji}
━━━━━━━━━━━━━━━━━━━━━

📊 **Setup:** `{self.setup_type}`
💯 **Skor:** `{self.total_score:.0f}/100`

💰 **Giriş:** `{self.entry_price:.2f}`
🛑 **Stop:** `{self.stop_loss:.2f}` ({self.stop_distance_pct:.1f}%)
🎯 **TP1:** `{self.take_profit_1:.2f}` ({self.tp1_distance_pct:.1f}%)

📈 **R:R:** `1:{self.risk_reward:.1f}`
🔥 **Güven:** `{float(self.confidence)*100:.0f}%`

⏰ {self.created_at.strftime('%H:%M:%S')}
"""
        return message.strip()
