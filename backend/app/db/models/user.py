"""
AlphaTerminal Pro - User Model
==============================

Kullanıcı veritabanı modeli.

Author: AlphaTerminal Team
Version: 1.0.0
"""

from __future__ import annotations

import uuid
from datetime import datetime
from typing import Optional, List, TYPE_CHECKING

from sqlalchemy import String, Boolean, BigInteger, Text, Index
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.db.models.base import FullModel

if TYPE_CHECKING:
    from app.db.models.portfolio import PortfolioModel
    from app.db.models.watchlist import WatchlistModel
    from app.db.models.notification import NotificationModel


class UserModel(FullModel):
    """
    Kullanıcı modeli.
    
    Hem web hem de Telegram kullanıcılarını destekler.
    
    Attributes:
        telegram_id: Telegram kullanıcı ID'si (opsiyonel)
        email: E-posta adresi (opsiyonel)
        username: Kullanıcı adı (opsiyonel)
        password_hash: Hashlenmiş şifre (web kullanıcıları için)
        full_name: Tam ad
        is_admin: Admin yetkisi var mı
        is_premium: Premium kullanıcı mı
        last_login_at: Son giriş zamanı
        preferences: Kullanıcı tercihleri (JSON)
    
    Relationships:
        portfolios: Kullanıcının portföyleri
        watchlists: İzleme listeleri
        notifications: Bildirimler
    """
    
    __tablename__ = "users"
    
    # Telegram entegrasyonu
    telegram_id: Mapped[Optional[int]] = mapped_column(
        BigInteger,
        unique=True,
        nullable=True,
        index=True,
        comment="Telegram kullanıcı ID"
    )
    telegram_username: Mapped[Optional[str]] = mapped_column(
        String(100),
        nullable=True,
        comment="Telegram kullanıcı adı"
    )
    
    # Web authentication
    email: Mapped[Optional[str]] = mapped_column(
        String(255),
        unique=True,
        nullable=True,
        index=True,
        comment="E-posta adresi"
    )
    username: Mapped[Optional[str]] = mapped_column(
        String(50),
        unique=True,
        nullable=True,
        index=True,
        comment="Kullanıcı adı"
    )
    password_hash: Mapped[Optional[str]] = mapped_column(
        String(255),
        nullable=True,
        comment="Hashlenmiş şifre"
    )
    
    # Profil bilgileri
    full_name: Mapped[Optional[str]] = mapped_column(
        String(100),
        nullable=True,
        comment="Tam ad"
    )
    avatar_url: Mapped[Optional[str]] = mapped_column(
        String(500),
        nullable=True,
        comment="Avatar URL"
    )
    
    # Yetki ve durum
    is_admin: Mapped[bool] = mapped_column(
        Boolean,
        default=False,
        nullable=False,
        index=True,
        comment="Admin yetkisi"
    )
    is_premium: Mapped[bool] = mapped_column(
        Boolean,
        default=False,
        nullable=False,
        index=True,
        comment="Premium kullanıcı"
    )
    is_verified: Mapped[bool] = mapped_column(
        Boolean,
        default=False,
        nullable=False,
        comment="E-posta doğrulandı mı"
    )
    
    # Oturum bilgileri
    last_login_at: Mapped[Optional[datetime]] = mapped_column(
        nullable=True,
        comment="Son giriş zamanı"
    )
    last_activity_at: Mapped[Optional[datetime]] = mapped_column(
        nullable=True,
        comment="Son aktivite zamanı"
    )
    
    # Tercihler (JSON)
    preferences: Mapped[Optional[dict]] = mapped_column(
        default=dict,
        nullable=True,
        comment="Kullanıcı tercihleri"
    )
    
    # Relationships
    portfolios: Mapped[List["PortfolioModel"]] = relationship(
        "PortfolioModel",
        back_populates="user",
        cascade="all, delete-orphan",
        lazy="selectin"
    )
    
    watchlists: Mapped[List["WatchlistModel"]] = relationship(
        "WatchlistModel",
        back_populates="user",
        cascade="all, delete-orphan",
        lazy="selectin"
    )
    
    notifications: Mapped[List["NotificationModel"]] = relationship(
        "NotificationModel",
        back_populates="user",
        cascade="all, delete-orphan",
        lazy="selectin"
    )
    
    # Indexes
    __table_args__ = (
        Index("ix_users_telegram_id_active", "telegram_id", "is_active"),
        Index("ix_users_email_active", "email", "is_active"),
        Index("ix_users_premium_active", "is_premium", "is_active"),
    )
    
    def __repr__(self) -> str:
        """String representation."""
        identifier = self.email or self.telegram_username or str(self.id)[:8]
        return f"<User({identifier})>"
    
    @property
    def display_name(self) -> str:
        """Görüntülenecek isim."""
        if self.full_name:
            return self.full_name
        if self.username:
            return self.username
        if self.telegram_username:
            return f"@{self.telegram_username}"
        if self.email:
            return self.email.split("@")[0]
        return f"User-{str(self.id)[:8]}"
    
    @property
    def is_telegram_user(self) -> bool:
        """Telegram kullanıcısı mı."""
        return self.telegram_id is not None
    
    @property
    def is_web_user(self) -> bool:
        """Web kullanıcısı mı."""
        return self.email is not None
    
    def update_last_login(self) -> None:
        """Son giriş zamanını güncelle."""
        self.last_login_at = datetime.utcnow()
        self.last_activity_at = datetime.utcnow()
    
    def update_last_activity(self) -> None:
        """Son aktivite zamanını güncelle."""
        self.last_activity_at = datetime.utcnow()
    
    def get_preference(self, key: str, default=None):
        """Tercih değeri al."""
        if self.preferences is None:
            return default
        return self.preferences.get(key, default)
    
    def set_preference(self, key: str, value) -> None:
        """Tercih değeri ayarla."""
        if self.preferences is None:
            self.preferences = {}
        self.preferences[key] = value
