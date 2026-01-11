"""
AlphaTerminal Pro - Stock Model
===============================

Hisse senedi veritabanı modeli.

Author: AlphaTerminal Team
Version: 1.0.0
"""

from __future__ import annotations

from datetime import datetime
from decimal import Decimal
from typing import Optional, List, TYPE_CHECKING

from sqlalchemy import String, Numeric, Index, Text
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.db.models.base import FullModel

if TYPE_CHECKING:
    from app.db.models.signal import SignalModel
    from app.db.models.winner_history import WinnerHistoryModel


class StockModel(FullModel):
    """
    Hisse senedi modeli.
    
    BIST'te işlem gören hisse senetlerinin temel bilgilerini tutar.
    
    Attributes:
        symbol: Hisse sembolü (örn: "THYAO")
        yahoo_symbol: Yahoo Finance sembolü (örn: "THYAO.IS")
        name: Şirket adı
        sector: Sektör kodu
        sub_sector: Alt sektör
        market_cap: Piyasa değeri
        free_float: Halka açıklık oranı
        lot_size: Lot büyüklüğü
        description: Şirket açıklaması
        
        last_price: Son fiyat
        last_volume: Son hacim
        last_updated_at: Son güncelleme zamanı
        
        day_change: Günlük değişim
        day_change_pct: Günlük değişim yüzdesi
        week_change_pct: Haftalık değişim yüzdesi
        month_change_pct: Aylık değişim yüzdesi
        year_change_pct: Yıllık değişim yüzdesi
    
    Relationships:
        signals: Bu hisseye ait sinyaller
        winner_histories: Kazanan geçmişi kayıtları
    """
    
    __tablename__ = "stocks"
    
    # Temel bilgiler
    symbol: Mapped[str] = mapped_column(
        String(20),
        unique=True,
        nullable=False,
        index=True,
        comment="Hisse sembolü (örn: THYAO)"
    )
    yahoo_symbol: Mapped[str] = mapped_column(
        String(25),
        unique=True,
        nullable=False,
        index=True,
        comment="Yahoo Finance sembolü (örn: THYAO.IS)"
    )
    name: Mapped[Optional[str]] = mapped_column(
        String(200),
        nullable=True,
        comment="Şirket adı"
    )
    
    # Sektör bilgileri
    sector: Mapped[Optional[str]] = mapped_column(
        String(50),
        nullable=True,
        index=True,
        comment="Sektör kodu"
    )
    sub_sector: Mapped[Optional[str]] = mapped_column(
        String(100),
        nullable=True,
        comment="Alt sektör"
    )
    
    # Şirket bilgileri
    market_cap: Mapped[Optional[Decimal]] = mapped_column(
        Numeric(20, 2),
        nullable=True,
        comment="Piyasa değeri (TRY)"
    )
    free_float: Mapped[Optional[Decimal]] = mapped_column(
        Numeric(5, 2),
        nullable=True,
        comment="Halka açıklık oranı (%)"
    )
    lot_size: Mapped[int] = mapped_column(
        default=1,
        nullable=False,
        comment="Lot büyüklüğü"
    )
    description: Mapped[Optional[str]] = mapped_column(
        Text,
        nullable=True,
        comment="Şirket açıklaması"
    )
    
    # Güncel fiyat bilgileri
    last_price: Mapped[Optional[Decimal]] = mapped_column(
        Numeric(12, 4),
        nullable=True,
        comment="Son fiyat"
    )
    last_volume: Mapped[Optional[int]] = mapped_column(
        nullable=True,
        comment="Son hacim"
    )
    last_updated_at: Mapped[Optional[datetime]] = mapped_column(
        nullable=True,
        comment="Son güncelleme zamanı"
    )
    
    # Değişim bilgileri
    day_change: Mapped[Optional[Decimal]] = mapped_column(
        Numeric(12, 4),
        nullable=True,
        comment="Günlük değişim (TRY)"
    )
    day_change_pct: Mapped[Optional[Decimal]] = mapped_column(
        Numeric(8, 4),
        nullable=True,
        index=True,
        comment="Günlük değişim (%)"
    )
    week_change_pct: Mapped[Optional[Decimal]] = mapped_column(
        Numeric(8, 4),
        nullable=True,
        comment="Haftalık değişim (%)"
    )
    month_change_pct: Mapped[Optional[Decimal]] = mapped_column(
        Numeric(8, 4),
        nullable=True,
        comment="Aylık değişim (%)"
    )
    year_change_pct: Mapped[Optional[Decimal]] = mapped_column(
        Numeric(8, 4),
        nullable=True,
        comment="Yıllık değişim (%)"
    )
    
    # Teknik veriler (güncellenebilir)
    atr: Mapped[Optional[Decimal]] = mapped_column(
        Numeric(12, 4),
        nullable=True,
        comment="Average True Range"
    )
    rsi: Mapped[Optional[Decimal]] = mapped_column(
        Numeric(6, 2),
        nullable=True,
        comment="RSI (14)"
    )
    
    # Relationships
    signals: Mapped[List["SignalModel"]] = relationship(
        "SignalModel",
        back_populates="stock",
        cascade="all, delete-orphan",
        lazy="selectin"
    )
    
    winner_histories: Mapped[List["WinnerHistoryModel"]] = relationship(
        "WinnerHistoryModel",
        back_populates="stock",
        cascade="all, delete-orphan",
        lazy="selectin"
    )
    
    # Indexes
    __table_args__ = (
        Index("ix_stocks_sector_active", "sector", "is_active"),
        Index("ix_stocks_day_change_pct", "day_change_pct", "is_active"),
        Index("ix_stocks_market_cap", "market_cap", "is_active"),
    )
    
    def __repr__(self) -> str:
        """String representation."""
        return f"<Stock({self.symbol})>"
    
    @property
    def is_bist30(self) -> bool:
        """BIST30 hissesi mi kontrol et."""
        bist30 = [
            "THYAO", "GARAN", "AKBNK", "YKBNK", "ISCTR", "EREGL", "BIMAS",
            "ASELS", "KCHOL", "TUPRS", "SISE", "SAHOL", "FROTO", "TOASO",
            "TCELL", "PGSUS", "ARCLK", "TAVHL", "PETKM", "SASA", "EKGYO",
            "HEKTS", "GUBRF", "KONTR", "ENKAI", "TKFEN", "TTKOM", "KRDMD",
            "SOKM", "MGROS"
        ]
        return self.symbol in bist30
    
    @property
    def is_bist100(self) -> bool:
        """BIST100 hissesi mi kontrol et (basitleştirilmiş)."""
        # Market cap'e göre basit kontrol
        if self.market_cap:
            return self.market_cap > Decimal("1000000000")  # 1 Milyar TL üstü
        return self.is_bist30
    
    @property
    def price_display(self) -> str:
        """Fiyat gösterimi."""
        if self.last_price:
            return f"{self.last_price:.2f} TRY"
        return "N/A"
    
    @property
    def change_display(self) -> str:
        """Değişim gösterimi."""
        if self.day_change_pct:
            sign = "+" if self.day_change_pct > 0 else ""
            return f"{sign}{self.day_change_pct:.2f}%"
        return "N/A"
    
    def update_price(
        self,
        price: Decimal,
        volume: int,
        day_change: Optional[Decimal] = None,
        day_change_pct: Optional[Decimal] = None
    ) -> None:
        """Fiyat bilgilerini güncelle."""
        self.last_price = price
        self.last_volume = volume
        self.last_updated_at = datetime.utcnow()
        
        if day_change is not None:
            self.day_change = day_change
        if day_change_pct is not None:
            self.day_change_pct = day_change_pct
