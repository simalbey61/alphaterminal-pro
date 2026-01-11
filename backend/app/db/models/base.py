"""
AlphaTerminal Pro - Database Base Model
=======================================

SQLAlchemy base sınıfı ve tüm modellerde kullanılan ortak alanlar.

Author: AlphaTerminal Team
Version: 1.0.0
"""

from __future__ import annotations

import uuid
from datetime import datetime
from typing import Any, Dict, Optional

from sqlalchemy import Column, DateTime, String, Boolean, text
from sqlalchemy.dialects.postgresql import UUID
from sqlalchemy.ext.declarative import declared_attr
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column


class Base(DeclarativeBase):
    """
    Tüm SQLAlchemy modelleri için base sınıf.
    
    Özellikler:
        - Otomatik tablo adı oluşturma
        - Ortak alanlar (id, created_at, updated_at)
        - repr ve dict metodları
    """
    
    # Type annotation mapping
    type_annotation_map = {
        uuid.UUID: UUID(as_uuid=True),
        datetime: DateTime(timezone=True),
    }
    
    @declared_attr.directive
    def __tablename__(cls) -> str:
        """
        Sınıf adından otomatik tablo adı oluştur.
        
        Örnek: UserModel -> users
        """
        # CamelCase'i snake_case'e çevir
        name = cls.__name__
        if name.endswith("Model"):
            name = name[:-5]  # "Model" suffix'ini kaldır
        
        # CamelCase -> snake_case
        result = []
        for i, char in enumerate(name):
            if char.isupper() and i > 0:
                result.append("_")
            result.append(char.lower())
        
        # Çoğul yap (basit kural)
        table_name = "".join(result)
        if table_name.endswith("y"):
            return table_name[:-1] + "ies"
        elif table_name.endswith("s"):
            return table_name + "es"
        else:
            return table_name + "s"
    
    def to_dict(self) -> Dict[str, Any]:
        """
        Model instance'ını dictionary'e çevir.
        
        Returns:
            Dict[str, Any]: Model verileri
        """
        result = {}
        for column in self.__table__.columns:
            value = getattr(self, column.name)
            
            # UUID'yi string'e çevir
            if isinstance(value, uuid.UUID):
                value = str(value)
            # Datetime'ı ISO formatına çevir
            elif isinstance(value, datetime):
                value = value.isoformat()
            
            result[column.name] = value
        
        return result
    
    def __repr__(self) -> str:
        """String representation."""
        class_name = self.__class__.__name__
        
        # Primary key'i bul
        pk_columns = [col.name for col in self.__table__.primary_key.columns]
        pk_values = [f"{col}={getattr(self, col)!r}" for col in pk_columns]
        
        return f"<{class_name}({', '.join(pk_values)})>"


class TimestampMixin:
    """
    Timestamp alanları için mixin.
    
    created_at ve updated_at alanlarını ekler.
    """
    
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        default=datetime.utcnow,
        server_default=text("CURRENT_TIMESTAMP"),
        nullable=False,
        comment="Oluşturulma zamanı"
    )
    
    updated_at: Mapped[Optional[datetime]] = mapped_column(
        DateTime(timezone=True),
        default=None,
        onupdate=datetime.utcnow,
        nullable=True,
        comment="Güncellenme zamanı"
    )


class UUIDMixin:
    """
    UUID primary key için mixin.
    """
    
    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        primary_key=True,
        default=uuid.uuid4,
        comment="Benzersiz tanımlayıcı"
    )


class SoftDeleteMixin:
    """
    Soft delete için mixin.
    
    Kayıtları silmek yerine is_deleted flag'ini set eder.
    """
    
    is_deleted: Mapped[bool] = mapped_column(
        Boolean,
        default=False,
        nullable=False,
        index=True,
        comment="Silinmiş mi?"
    )
    
    deleted_at: Mapped[Optional[datetime]] = mapped_column(
        DateTime(timezone=True),
        default=None,
        nullable=True,
        comment="Silinme zamanı"
    )
    
    def soft_delete(self) -> None:
        """Kaydı soft delete olarak işaretle."""
        self.is_deleted = True
        self.deleted_at = datetime.utcnow()
    
    def restore(self) -> None:
        """Silinmiş kaydı geri yükle."""
        self.is_deleted = False
        self.deleted_at = None


class ActiveMixin:
    """
    Aktif/Pasif durumu için mixin.
    """
    
    is_active: Mapped[bool] = mapped_column(
        Boolean,
        default=True,
        nullable=False,
        index=True,
        comment="Aktif mi?"
    )
    
    def activate(self) -> None:
        """Kaydı aktifleştir."""
        self.is_active = True
    
    def deactivate(self) -> None:
        """Kaydı pasifleştir."""
        self.is_active = False


class BaseModel(Base, UUIDMixin, TimestampMixin):
    """
    Standart base model.
    
    UUID primary key, created_at ve updated_at alanlarını içerir.
    """
    
    __abstract__ = True


class SoftDeleteModel(BaseModel, SoftDeleteMixin):
    """
    Soft delete destekli base model.
    """
    
    __abstract__ = True


class FullModel(BaseModel, SoftDeleteMixin, ActiveMixin):
    """
    Tüm mixin'leri içeren tam base model.
    """
    
    __abstract__ = True
