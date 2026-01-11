"""
AlphaTerminal Pro - User Related Models
=======================================

Kullanıcı ile ilişkili modeller:
- Watchlist: İzleme listesi
- Notification: Bildirimler

Author: AlphaTerminal Team
Version: 1.0.0
"""

from __future__ import annotations

import uuid
from datetime import datetime
from typing import Optional, Dict, Any, TYPE_CHECKING

from sqlalchemy import String, Boolean, ForeignKey, Index, Text, UniqueConstraint
from sqlalchemy.dialects.postgresql import UUID, JSONB
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.db.models.base import BaseModel

if TYPE_CHECKING:
    from app.db.models.user import UserModel


class WatchlistModel(BaseModel):
    """
    İzleme listesi modeli.
    
    Kullanıcının takip ettiği hisseleri tutar.
    
    Attributes:
        user_id: Kullanıcı foreign key
        symbol: Hisse sembolü
        notes: Notlar
        alert_price_above: Üst fiyat alarmı
        alert_price_below: Alt fiyat alarmı
        alert_enabled: Alarm aktif mi
    """
    
    __tablename__ = "watchlists"
    
    # İlişkiler
    user_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("users.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
        comment="Kullanıcı foreign key"
    )
    
    # Hisse bilgisi
    symbol: Mapped[str] = mapped_column(
        String(20),
        nullable=False,
        index=True,
        comment="Hisse sembolü"
    )
    
    # Notlar
    notes: Mapped[Optional[str]] = mapped_column(
        Text,
        nullable=True,
        comment="Kullanıcı notları"
    )
    
    # Fiyat alarmları
    alert_price_above: Mapped[Optional[float]] = mapped_column(
        nullable=True,
        comment="Üst fiyat alarmı"
    )
    alert_price_below: Mapped[Optional[float]] = mapped_column(
        nullable=True,
        comment="Alt fiyat alarmı"
    )
    alert_enabled: Mapped[bool] = mapped_column(
        Boolean,
        default=True,
        nullable=False,
        comment="Alarm aktif mi"
    )
    
    # Kategori/Etiket
    category: Mapped[Optional[str]] = mapped_column(
        String(50),
        nullable=True,
        comment="Kategori"
    )
    tags: Mapped[Optional[list]] = mapped_column(
        JSONB,
        nullable=True,
        comment="Etiketler"
    )
    
    # Sıralama
    sort_order: Mapped[int] = mapped_column(
        default=0,
        nullable=False,
        comment="Sıralama"
    )
    
    # Relationship
    user: Mapped["UserModel"] = relationship(
        "UserModel",
        back_populates="watchlists",
        lazy="selectin"
    )
    
    # Constraints & Indexes
    __table_args__ = (
        UniqueConstraint("user_id", "symbol", name="uq_watchlist_user_symbol"),
        Index("ix_watchlist_user_symbol", "user_id", "symbol"),
    )
    
    def __repr__(self) -> str:
        """String representation."""
        return f"<Watchlist({self.symbol})>"


class NotificationModel(BaseModel):
    """
    Bildirim modeli.
    
    Kullanıcı bildirimlerini tutar.
    
    Attributes:
        user_id: Kullanıcı foreign key
        type: Bildirim türü (signal, alert, strategy, system)
        title: Başlık
        message: Mesaj
        data: Ek veri (JSON)
        is_read: Okundu mu
        read_at: Okunma zamanı
    """
    
    __tablename__ = "notifications"
    
    # İlişkiler
    user_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("users.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
        comment="Kullanıcı foreign key"
    )
    
    # Bildirim içeriği
    type: Mapped[str] = mapped_column(
        String(30),
        nullable=False,
        index=True,
        comment="Bildirim türü"
    )
    title: Mapped[str] = mapped_column(
        String(200),
        nullable=False,
        comment="Başlık"
    )
    message: Mapped[str] = mapped_column(
        Text,
        nullable=False,
        comment="Mesaj"
    )
    
    # Ek veri
    data: Mapped[Optional[Dict[str, Any]]] = mapped_column(
        JSONB,
        nullable=True,
        comment="Ek veri"
    )
    
    # Link
    action_url: Mapped[Optional[str]] = mapped_column(
        String(500),
        nullable=True,
        comment="Aksiyon URL"
    )
    
    # Durum
    is_read: Mapped[bool] = mapped_column(
        Boolean,
        default=False,
        nullable=False,
        index=True,
        comment="Okundu mu"
    )
    read_at: Mapped[Optional[datetime]] = mapped_column(
        nullable=True,
        comment="Okunma zamanı"
    )
    
    # Öncelik
    priority: Mapped[str] = mapped_column(
        String(20),
        default="normal",
        nullable=False,
        comment="Öncelik (low, normal, high, urgent)"
    )
    
    # Süre
    expires_at: Mapped[Optional[datetime]] = mapped_column(
        nullable=True,
        comment="Son geçerlilik zamanı"
    )
    
    # Relationship
    user: Mapped["UserModel"] = relationship(
        "UserModel",
        back_populates="notifications",
        lazy="selectin"
    )
    
    # Indexes
    __table_args__ = (
        Index("ix_notifications_user_read", "user_id", "is_read"),
        Index("ix_notifications_user_type", "user_id", "type"),
        Index("ix_notifications_created", "created_at", "user_id"),
    )
    
    def __repr__(self) -> str:
        """String representation."""
        return f"<Notification({self.type}: {self.title[:30]})>"
    
    def mark_as_read(self) -> None:
        """Bildirimi okundu olarak işaretle."""
        self.is_read = True
        self.read_at = datetime.utcnow()
    
    @property
    def is_expired(self) -> bool:
        """Süresi dolmuş mu."""
        if self.expires_at is None:
            return False
        return datetime.utcnow() > self.expires_at
