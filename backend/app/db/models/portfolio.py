"""
AlphaTerminal Pro - Portfolio & Position Models
===============================================

Portfolio ve pozisyon veritabanı modelleri.

Author: AlphaTerminal Team
Version: 1.0.0
"""

from __future__ import annotations

import uuid
from datetime import datetime
from decimal import Decimal
from typing import Optional, List, Dict, Any, TYPE_CHECKING

from sqlalchemy import String, Numeric, ForeignKey, Index, Text, Integer, Boolean
from sqlalchemy.dialects.postgresql import UUID, JSONB
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.db.models.base import BaseModel
from app.config import SignalType

if TYPE_CHECKING:
    from app.db.models.user import UserModel
    from app.db.models.signal import SignalModel


class PortfolioModel(BaseModel):
    """
    Portfolio modeli.
    
    Kullanıcının sanal veya gerçek portfolyolarını tutar.
    
    Attributes:
        user_id: Kullanıcı foreign key
        name: Portfolio adı
        description: Portfolio açıklaması
        capital: Başlangıç sermayesi
        current_value: Güncel değer
        is_default: Varsayılan portfolio mu
        is_paper: Kağıt trade mi (sanal)
        
        total_pnl: Toplam P&L
        total_pnl_pct: Toplam P&L yüzdesi
        total_trades: Toplam trade sayısı
        winning_trades: Kazanan trade sayısı
        
        max_drawdown: Maksimum drawdown
        sharpe_ratio: Sharpe oranı
        
    Relationships:
        user: Portfolio sahibi
        positions: Portfolyodaki pozisyonlar
    """
    
    __tablename__ = "portfolios"
    
    # İlişkiler
    user_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("users.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
        comment="Kullanıcı foreign key"
    )
    
    # Temel bilgiler
    name: Mapped[str] = mapped_column(
        String(100),
        nullable=False,
        comment="Portfolio adı"
    )
    description: Mapped[Optional[str]] = mapped_column(
        Text,
        nullable=True,
        comment="Portfolio açıklaması"
    )
    
    # Sermaye
    capital: Mapped[Decimal] = mapped_column(
        Numeric(14, 2),
        nullable=False,
        comment="Başlangıç sermayesi (TRY)"
    )
    current_value: Mapped[Decimal] = mapped_column(
        Numeric(14, 2),
        nullable=False,
        comment="Güncel değer (TRY)"
    )
    available_capital: Mapped[Decimal] = mapped_column(
        Numeric(14, 2),
        nullable=False,
        comment="Kullanılabilir sermaye (TRY)"
    )
    
    # Ayarlar
    is_default: Mapped[bool] = mapped_column(
        Boolean,
        default=False,
        nullable=False,
        index=True,
        comment="Varsayılan portfolio mu"
    )
    is_paper: Mapped[bool] = mapped_column(
        Boolean,
        default=True,
        nullable=False,
        comment="Kağıt trade mi (sanal)"
    )
    
    # Risk ayarları
    max_positions: Mapped[int] = mapped_column(
        Integer,
        default=5,
        nullable=False,
        comment="Maksimum pozisyon sayısı"
    )
    max_risk_per_trade: Mapped[Decimal] = mapped_column(
        Numeric(5, 4),
        default=Decimal("0.02"),
        nullable=False,
        comment="Trade başına maksimum risk"
    )
    max_portfolio_risk: Mapped[Decimal] = mapped_column(
        Numeric(5, 4),
        default=Decimal("0.06"),
        nullable=False,
        comment="Maksimum portfolio riski"
    )
    
    # Performans
    total_pnl: Mapped[Decimal] = mapped_column(
        Numeric(14, 4),
        default=Decimal("0"),
        nullable=False,
        comment="Toplam P&L (TRY)"
    )
    total_pnl_pct: Mapped[Decimal] = mapped_column(
        Numeric(8, 4),
        default=Decimal("0"),
        nullable=False,
        comment="Toplam P&L (%)"
    )
    
    # Trade istatistikleri
    total_trades: Mapped[int] = mapped_column(
        Integer,
        default=0,
        nullable=False,
        comment="Toplam trade sayısı"
    )
    winning_trades: Mapped[int] = mapped_column(
        Integer,
        default=0,
        nullable=False,
        comment="Kazanan trade sayısı"
    )
    losing_trades: Mapped[int] = mapped_column(
        Integer,
        default=0,
        nullable=False,
        comment="Kaybeden trade sayısı"
    )
    
    # Risk metrikleri
    max_drawdown: Mapped[Decimal] = mapped_column(
        Numeric(8, 4),
        default=Decimal("0"),
        nullable=False,
        comment="Maksimum drawdown (%)"
    )
    sharpe_ratio: Mapped[Optional[Decimal]] = mapped_column(
        Numeric(6, 3),
        nullable=True,
        comment="Sharpe oranı"
    )
    profit_factor: Mapped[Optional[Decimal]] = mapped_column(
        Numeric(6, 3),
        nullable=True,
        comment="Profit factor"
    )
    
    # Zirve değer (drawdown hesabı için)
    peak_value: Mapped[Decimal] = mapped_column(
        Numeric(14, 2),
        nullable=False,
        comment="Zirve değer (TRY)"
    )
    
    # Relationships
    user: Mapped["UserModel"] = relationship(
        "UserModel",
        back_populates="portfolios",
        lazy="selectin"
    )
    
    positions: Mapped[List["PositionModel"]] = relationship(
        "PositionModel",
        back_populates="portfolio",
        cascade="all, delete-orphan",
        lazy="selectin"
    )
    
    # Indexes
    __table_args__ = (
        Index("ix_portfolios_user_default", "user_id", "is_default"),
    )
    
    def __repr__(self) -> str:
        """String representation."""
        return f"<Portfolio({self.name} - {self.current_value:.2f} TRY)>"
    
    @property
    def win_rate(self) -> Optional[Decimal]:
        """Kazanma oranı."""
        if self.total_trades == 0:
            return None
        return Decimal(str(self.winning_trades / self.total_trades))
    
    @property
    def current_drawdown(self) -> Decimal:
        """Güncel drawdown."""
        if self.peak_value == 0:
            return Decimal("0")
        return (self.peak_value - self.current_value) / self.peak_value
    
    @property
    def open_positions_count(self) -> int:
        """Açık pozisyon sayısı."""
        return len([p for p in self.positions if p.is_open])
    
    @property
    def can_open_position(self) -> bool:
        """Yeni pozisyon açılabilir mi."""
        return self.open_positions_count < self.max_positions
    
    def update_value(self, new_value: Decimal) -> None:
        """Portfolio değerini güncelle."""
        self.current_value = new_value
        
        # Zirve kontrolü
        if new_value > self.peak_value:
            self.peak_value = new_value
        
        # P&L güncelle
        self.total_pnl = new_value - self.capital
        if self.capital > 0:
            self.total_pnl_pct = (self.total_pnl / self.capital) * 100
        
        # Max drawdown güncelle
        current_dd = self.current_drawdown
        if current_dd > self.max_drawdown:
            self.max_drawdown = current_dd
    
    def record_trade(self, is_win: bool, pnl: Decimal) -> None:
        """Trade sonucunu kaydet."""
        self.total_trades += 1
        
        if is_win:
            self.winning_trades += 1
        else:
            self.losing_trades += 1


class PositionModel(BaseModel):
    """
    Pozisyon modeli.
    
    Portfolyodaki açık ve kapalı pozisyonları tutar.
    
    Attributes:
        portfolio_id: Portfolio foreign key
        signal_id: Sinyal foreign key
        symbol: Hisse sembolü
        
        direction: Pozisyon yönü (LONG/SHORT)
        quantity: Miktar
        entry_price: Giriş fiyatı
        current_price: Güncel fiyat
        
        stop_loss: Stop loss fiyatı
        take_profit: Take profit fiyatı
        
        unrealized_pnl: Gerçekleşmemiş P&L
        realized_pnl: Gerçekleşmiş P&L
        
        status: Pozisyon durumu (open/closed)
        exit_price: Çıkış fiyatı
        exit_reason: Çıkış sebebi
        closed_at: Kapanış zamanı
        
    Relationships:
        portfolio: Ait olduğu portfolio
        signal: İlgili sinyal
    """
    
    __tablename__ = "positions"
    
    # İlişkiler
    portfolio_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("portfolios.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
        comment="Portfolio foreign key"
    )
    signal_id: Mapped[Optional[uuid.UUID]] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("signals.id", ondelete="SET NULL"),
        nullable=True,
        index=True,
        comment="Sinyal foreign key"
    )
    
    # Temel bilgiler
    symbol: Mapped[str] = mapped_column(
        String(20),
        nullable=False,
        index=True,
        comment="Hisse sembolü"
    )
    direction: Mapped[str] = mapped_column(
        String(10),
        nullable=False,
        comment="Pozisyon yönü (LONG/SHORT)"
    )
    
    # Pozisyon detayları
    quantity: Mapped[int] = mapped_column(
        Integer,
        nullable=False,
        comment="Miktar (lot)"
    )
    entry_price: Mapped[Decimal] = mapped_column(
        Numeric(12, 4),
        nullable=False,
        comment="Giriş fiyatı"
    )
    current_price: Mapped[Optional[Decimal]] = mapped_column(
        Numeric(12, 4),
        nullable=True,
        comment="Güncel fiyat"
    )
    
    # Risk seviyeleri
    stop_loss: Mapped[Decimal] = mapped_column(
        Numeric(12, 4),
        nullable=False,
        comment="Stop loss fiyatı"
    )
    take_profit: Mapped[Decimal] = mapped_column(
        Numeric(12, 4),
        nullable=False,
        comment="Take profit fiyatı"
    )
    trailing_stop: Mapped[Optional[Decimal]] = mapped_column(
        Numeric(12, 4),
        nullable=True,
        comment="Trailing stop fiyatı"
    )
    
    # P&L
    unrealized_pnl: Mapped[Decimal] = mapped_column(
        Numeric(14, 4),
        default=Decimal("0"),
        nullable=False,
        comment="Gerçekleşmemiş P&L (TRY)"
    )
    unrealized_pnl_pct: Mapped[Decimal] = mapped_column(
        Numeric(8, 4),
        default=Decimal("0"),
        nullable=False,
        comment="Gerçekleşmemiş P&L (%)"
    )
    realized_pnl: Mapped[Optional[Decimal]] = mapped_column(
        Numeric(14, 4),
        nullable=True,
        comment="Gerçekleşmiş P&L (TRY)"
    )
    realized_pnl_pct: Mapped[Optional[Decimal]] = mapped_column(
        Numeric(8, 4),
        nullable=True,
        comment="Gerçekleşmiş P&L (%)"
    )
    
    # Pozisyon değeri
    position_value: Mapped[Decimal] = mapped_column(
        Numeric(14, 2),
        nullable=False,
        comment="Pozisyon değeri (TRY)"
    )
    risk_amount: Mapped[Decimal] = mapped_column(
        Numeric(14, 2),
        nullable=False,
        comment="Risk miktarı (TRY)"
    )
    
    # Durum
    status: Mapped[str] = mapped_column(
        String(20),
        default="open",
        nullable=False,
        index=True,
        comment="Pozisyon durumu (open/closed)"
    )
    
    # Kapanış bilgileri
    exit_price: Mapped[Optional[Decimal]] = mapped_column(
        Numeric(12, 4),
        nullable=True,
        comment="Çıkış fiyatı"
    )
    exit_reason: Mapped[Optional[str]] = mapped_column(
        String(50),
        nullable=True,
        comment="Çıkış sebebi"
    )
    closed_at: Mapped[Optional[datetime]] = mapped_column(
        nullable=True,
        comment="Kapanış zamanı"
    )
    
    # Notlar
    notes: Mapped[Optional[str]] = mapped_column(
        Text,
        nullable=True,
        comment="Notlar"
    )
    
    # Relationships
    portfolio: Mapped["PortfolioModel"] = relationship(
        "PortfolioModel",
        back_populates="positions",
        lazy="selectin"
    )
    
    signal: Mapped[Optional["SignalModel"]] = relationship(
        "SignalModel",
        back_populates="positions",
        lazy="selectin"
    )
    
    # Indexes
    __table_args__ = (
        Index("ix_positions_portfolio_status", "portfolio_id", "status"),
        Index("ix_positions_symbol_status", "symbol", "status"),
    )
    
    def __repr__(self) -> str:
        """String representation."""
        return f"<Position({self.symbol} {self.direction} {self.quantity} @ {self.entry_price})>"
    
    @property
    def is_open(self) -> bool:
        """Pozisyon açık mı."""
        return self.status == "open"
    
    @property
    def is_profitable(self) -> bool:
        """Karlı mı."""
        return self.unrealized_pnl > 0
    
    @property
    def holding_duration(self) -> Optional[int]:
        """Tutma süresi (dakika)."""
        if self.closed_at:
            return int((self.closed_at - self.created_at).total_seconds() / 60)
        return int((datetime.utcnow() - self.created_at).total_seconds() / 60)
    
    def update_price(self, price: Decimal) -> None:
        """
        Güncel fiyatı güncelle ve P&L hesapla.
        
        Args:
            price: Güncel fiyat
        """
        self.current_price = price
        
        # P&L hesapla
        if self.direction == SignalType.LONG.value:
            self.unrealized_pnl = (price - self.entry_price) * self.quantity
        else:
            self.unrealized_pnl = (self.entry_price - price) * self.quantity
        
        # P&L yüzdesi
        if self.entry_price > 0:
            self.unrealized_pnl_pct = (self.unrealized_pnl / (self.entry_price * self.quantity)) * 100
    
    def close(self, exit_price: Decimal, reason: str) -> None:
        """
        Pozisyonu kapat.
        
        Args:
            exit_price: Çıkış fiyatı
            reason: Kapanış sebebi
        """
        self.exit_price = exit_price
        self.exit_reason = reason
        self.closed_at = datetime.utcnow()
        self.status = "closed"
        
        # P&L hesapla
        if self.direction == SignalType.LONG.value:
            self.realized_pnl = (exit_price - self.entry_price) * self.quantity
        else:
            self.realized_pnl = (self.entry_price - exit_price) * self.quantity
        
        # P&L yüzdesi
        if self.entry_price > 0:
            self.realized_pnl_pct = (self.realized_pnl / (self.entry_price * self.quantity)) * 100
        
        self.unrealized_pnl = Decimal("0")
        self.unrealized_pnl_pct = Decimal("0")
    
    def should_stop(self, current_price: Decimal) -> bool:
        """Stop seviyesine ulaşıldı mı."""
        if self.direction == SignalType.LONG.value:
            return current_price <= self.stop_loss
        else:
            return current_price >= self.stop_loss
    
    def should_take_profit(self, current_price: Decimal) -> bool:
        """Take profit seviyesine ulaşıldı mı."""
        if self.direction == SignalType.LONG.value:
            return current_price >= self.take_profit
        else:
            return current_price <= self.take_profit
