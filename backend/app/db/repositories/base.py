"""
AlphaTerminal Pro - Base Repository
===================================

Generic CRUD işlemleri için base repository sınıfı.
Repository pattern implementasyonu.

Author: AlphaTerminal Team
Version: 1.0.0
"""

from __future__ import annotations

import uuid
import logging
from typing import (
    Any,
    Dict,
    Generic,
    List,
    Optional,
    Sequence,
    Type,
    TypeVar,
    Union,
)
from datetime import datetime

from sqlalchemy import select, update, delete, func, and_, or_
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import selectinload, joinedload
from sqlalchemy.sql import Select

from app.db.models.base import Base

logger = logging.getLogger(__name__)

# Generic type for model
ModelType = TypeVar("ModelType", bound=Base)


class BaseRepository(Generic[ModelType]):
    """
    Generic base repository.
    
    Tüm repository'ler için temel CRUD işlemlerini sağlar.
    
    Type Parameters:
        ModelType: SQLAlchemy model tipi
        
    Attributes:
        model: Model sınıfı
        session: Async database session
        
    Example:
        ```python
        class UserRepository(BaseRepository[UserModel]):
            def __init__(self, session: AsyncSession):
                super().__init__(UserModel, session)
            
            async def find_by_email(self, email: str) -> Optional[UserModel]:
                return await self.find_one(email=email)
        ```
    """
    
    def __init__(self, model: Type[ModelType], session: AsyncSession):
        """
        Repository'yi initialize et.
        
        Args:
            model: SQLAlchemy model sınıfı
            session: Async database session
        """
        self.model = model
        self.session = session
    
    # =========================================================================
    # CREATE
    # =========================================================================
    
    async def create(self, **kwargs: Any) -> ModelType:
        """
        Yeni kayıt oluştur.
        
        Args:
            **kwargs: Model alanları
            
        Returns:
            ModelType: Oluşturulan model instance
            
        Example:
            ```python
            user = await repo.create(
                email="test@example.com",
                username="testuser"
            )
            ```
        """
        instance = self.model(**kwargs)
        self.session.add(instance)
        await self.session.flush()
        await self.session.refresh(instance)
        logger.debug(f"Created {self.model.__name__}: {instance}")
        return instance
    
    async def create_many(self, items: List[Dict[str, Any]]) -> List[ModelType]:
        """
        Birden fazla kayıt oluştur.
        
        Args:
            items: Model alanlarının listesi
            
        Returns:
            List[ModelType]: Oluşturulan model instance'ları
        """
        instances = [self.model(**item) for item in items]
        self.session.add_all(instances)
        await self.session.flush()
        
        for instance in instances:
            await self.session.refresh(instance)
        
        logger.debug(f"Created {len(instances)} {self.model.__name__} records")
        return instances
    
    # =========================================================================
    # READ
    # =========================================================================
    
    async def get(self, id: uuid.UUID) -> Optional[ModelType]:
        """
        ID ile kayıt getir.
        
        Args:
            id: Kayıt UUID'si
            
        Returns:
            Optional[ModelType]: Model instance veya None
        """
        return await self.session.get(self.model, id)
    
    async def get_or_raise(self, id: uuid.UUID) -> ModelType:
        """
        ID ile kayıt getir veya hata fırlat.
        
        Args:
            id: Kayıt UUID'si
            
        Returns:
            ModelType: Model instance
            
        Raises:
            ValueError: Kayıt bulunamazsa
        """
        instance = await self.get(id)
        if instance is None:
            raise ValueError(f"{self.model.__name__} with id {id} not found")
        return instance
    
    async def find_one(self, **filters: Any) -> Optional[ModelType]:
        """
        Filtrelere göre tek kayıt bul.
        
        Args:
            **filters: Filtre koşulları
            
        Returns:
            Optional[ModelType]: Model instance veya None
            
        Example:
            ```python
            user = await repo.find_one(email="test@example.com")
            ```
        """
        query = select(self.model).filter_by(**filters)
        result = await self.session.execute(query)
        return result.scalar_one_or_none()
    
    async def find_many(
        self,
        *,
        filters: Optional[Dict[str, Any]] = None,
        order_by: Optional[str] = None,
        order_desc: bool = False,
        limit: Optional[int] = None,
        offset: Optional[int] = None,
    ) -> Sequence[ModelType]:
        """
        Filtrelere göre birden fazla kayıt bul.
        
        Args:
            filters: Filtre koşulları
            order_by: Sıralama alanı
            order_desc: Azalan sıralama
            limit: Maksimum kayıt sayısı
            offset: Başlangıç offset'i
            
        Returns:
            Sequence[ModelType]: Model instance'ları
        """
        query = select(self.model)
        
        # Filtreler
        if filters:
            query = query.filter_by(**filters)
        
        # Sıralama
        if order_by:
            order_column = getattr(self.model, order_by)
            query = query.order_by(
                order_column.desc() if order_desc else order_column
            )
        
        # Pagination
        if offset is not None:
            query = query.offset(offset)
        if limit is not None:
            query = query.limit(limit)
        
        result = await self.session.execute(query)
        return result.scalars().all()
    
    async def find_all(self) -> Sequence[ModelType]:
        """
        Tüm kayıtları getir.
        
        Returns:
            Sequence[ModelType]: Tüm model instance'ları
        """
        query = select(self.model)
        result = await self.session.execute(query)
        return result.scalars().all()
    
    async def exists(self, **filters: Any) -> bool:
        """
        Kayıt var mı kontrol et.
        
        Args:
            **filters: Filtre koşulları
            
        Returns:
            bool: Kayıt var mı
        """
        query = select(func.count()).select_from(self.model).filter_by(**filters)
        result = await self.session.execute(query)
        return result.scalar_one() > 0
    
    async def count(self, **filters: Any) -> int:
        """
        Kayıt sayısını al.
        
        Args:
            **filters: Filtre koşulları
            
        Returns:
            int: Kayıt sayısı
        """
        query = select(func.count()).select_from(self.model)
        if filters:
            query = query.filter_by(**filters)
        result = await self.session.execute(query)
        return result.scalar_one()
    
    # =========================================================================
    # UPDATE
    # =========================================================================
    
    async def update(
        self,
        id: uuid.UUID,
        **kwargs: Any
    ) -> Optional[ModelType]:
        """
        Kaydı güncelle.
        
        Args:
            id: Kayıt UUID'si
            **kwargs: Güncellenecek alanlar
            
        Returns:
            Optional[ModelType]: Güncellenen model instance veya None
        """
        instance = await self.get(id)
        if instance is None:
            return None
        
        for key, value in kwargs.items():
            if hasattr(instance, key):
                setattr(instance, key, value)
        
        # updated_at varsa güncelle
        if hasattr(instance, "updated_at"):
            instance.updated_at = datetime.utcnow()
        
        await self.session.flush()
        await self.session.refresh(instance)
        
        logger.debug(f"Updated {self.model.__name__}: {instance}")
        return instance
    
    async def update_many(
        self,
        filters: Dict[str, Any],
        **kwargs: Any
    ) -> int:
        """
        Birden fazla kaydı güncelle.
        
        Args:
            filters: Filtre koşulları
            **kwargs: Güncellenecek alanlar
            
        Returns:
            int: Güncellenen kayıt sayısı
        """
        # updated_at varsa ekle
        if hasattr(self.model, "updated_at"):
            kwargs["updated_at"] = datetime.utcnow()
        
        query = (
            update(self.model)
            .filter_by(**filters)
            .values(**kwargs)
        )
        result = await self.session.execute(query)
        logger.debug(f"Updated {result.rowcount} {self.model.__name__} records")
        return result.rowcount
    
    # =========================================================================
    # DELETE
    # =========================================================================
    
    async def delete(self, id: uuid.UUID) -> bool:
        """
        Kaydı sil.
        
        Args:
            id: Kayıt UUID'si
            
        Returns:
            bool: Başarılı mı
        """
        instance = await self.get(id)
        if instance is None:
            return False
        
        await self.session.delete(instance)
        logger.debug(f"Deleted {self.model.__name__} with id {id}")
        return True
    
    async def delete_many(self, **filters: Any) -> int:
        """
        Birden fazla kaydı sil.
        
        Args:
            **filters: Filtre koşulları
            
        Returns:
            int: Silinen kayıt sayısı
        """
        query = delete(self.model).filter_by(**filters)
        result = await self.session.execute(query)
        logger.debug(f"Deleted {result.rowcount} {self.model.__name__} records")
        return result.rowcount
    
    async def soft_delete(self, id: uuid.UUID) -> bool:
        """
        Kaydı soft delete yap.
        
        Args:
            id: Kayıt UUID'si
            
        Returns:
            bool: Başarılı mı
        """
        instance = await self.get(id)
        if instance is None:
            return False
        
        if hasattr(instance, "soft_delete"):
            instance.soft_delete()
            await self.session.flush()
            logger.debug(f"Soft deleted {self.model.__name__} with id {id}")
            return True
        
        # soft_delete yoksa normal delete
        return await self.delete(id)
    
    # =========================================================================
    # ADVANCED QUERIES
    # =========================================================================
    
    async def execute_query(self, query: Select) -> Sequence[ModelType]:
        """
        Custom query çalıştır.
        
        Args:
            query: SQLAlchemy Select query
            
        Returns:
            Sequence[ModelType]: Sonuçlar
        """
        result = await self.session.execute(query)
        return result.scalars().all()
    
    async def paginate(
        self,
        page: int = 1,
        per_page: int = 20,
        filters: Optional[Dict[str, Any]] = None,
        order_by: Optional[str] = None,
        order_desc: bool = False,
    ) -> Dict[str, Any]:
        """
        Sayfalanmış sonuçlar getir.
        
        Args:
            page: Sayfa numarası (1'den başlar)
            per_page: Sayfa başına kayıt
            filters: Filtre koşulları
            order_by: Sıralama alanı
            order_desc: Azalan sıralama
            
        Returns:
            dict: Sayfalanmış sonuçlar
            
        Example:
            ```python
            result = await repo.paginate(
                page=1,
                per_page=20,
                filters={"status": "active"},
                order_by="created_at",
                order_desc=True
            )
            # {
            #     "items": [...],
            #     "total": 100,
            #     "page": 1,
            #     "per_page": 20,
            #     "pages": 5,
            #     "has_next": True,
            #     "has_prev": False
            # }
            ```
        """
        # Toplam sayı
        total = await self.count(**(filters or {}))
        
        # Offset hesapla
        offset = (page - 1) * per_page
        
        # Kayıtları getir
        items = await self.find_many(
            filters=filters,
            order_by=order_by,
            order_desc=order_desc,
            limit=per_page,
            offset=offset,
        )
        
        # Sayfa sayısı
        pages = (total + per_page - 1) // per_page if per_page > 0 else 0
        
        return {
            "items": items,
            "total": total,
            "page": page,
            "per_page": per_page,
            "pages": pages,
            "has_next": page < pages,
            "has_prev": page > 1,
        }
    
    async def find_in(
        self,
        field: str,
        values: List[Any]
    ) -> Sequence[ModelType]:
        """
        IN query ile kayıtları bul.
        
        Args:
            field: Alan adı
            values: Değerler listesi
            
        Returns:
            Sequence[ModelType]: Bulunan kayıtlar
        """
        if not values:
            return []
        
        column = getattr(self.model, field)
        query = select(self.model).where(column.in_(values))
        result = await self.session.execute(query)
        return result.scalars().all()
    
    async def find_between(
        self,
        field: str,
        start: Any,
        end: Any
    ) -> Sequence[ModelType]:
        """
        BETWEEN query ile kayıtları bul.
        
        Args:
            field: Alan adı
            start: Başlangıç değeri
            end: Bitiş değeri
            
        Returns:
            Sequence[ModelType]: Bulunan kayıtlar
        """
        column = getattr(self.model, field)
        query = select(self.model).where(column.between(start, end))
        result = await self.session.execute(query)
        return result.scalars().all()
    
    async def search(
        self,
        field: str,
        term: str,
        case_sensitive: bool = False
    ) -> Sequence[ModelType]:
        """
        LIKE query ile arama yap.
        
        Args:
            field: Alan adı
            term: Arama terimi
            case_sensitive: Büyük/küçük harf duyarlı mı
            
        Returns:
            Sequence[ModelType]: Bulunan kayıtlar
        """
        column = getattr(self.model, field)
        
        if case_sensitive:
            query = select(self.model).where(column.like(f"%{term}%"))
        else:
            query = select(self.model).where(column.ilike(f"%{term}%"))
        
        result = await self.session.execute(query)
        return result.scalars().all()
    
    # =========================================================================
    # RELATIONSHIP LOADING
    # =========================================================================
    
    async def get_with_relations(
        self,
        id: uuid.UUID,
        *relations: str
    ) -> Optional[ModelType]:
        """
        İlişkilerle birlikte kayıt getir.
        
        Args:
            id: Kayıt UUID'si
            *relations: Yüklenecek ilişki adları
            
        Returns:
            Optional[ModelType]: Model instance veya None
            
        Example:
            ```python
            user = await repo.get_with_relations(
                user_id,
                "portfolios",
                "watchlists"
            )
            ```
        """
        query = select(self.model).where(self.model.id == id)
        
        for relation in relations:
            query = query.options(selectinload(getattr(self.model, relation)))
        
        result = await self.session.execute(query)
        return result.scalar_one_or_none()
    
    async def find_with_relations(
        self,
        filters: Optional[Dict[str, Any]] = None,
        relations: Optional[List[str]] = None,
        limit: Optional[int] = None,
    ) -> Sequence[ModelType]:
        """
        İlişkilerle birlikte kayıtları bul.
        
        Args:
            filters: Filtre koşulları
            relations: Yüklenecek ilişki adları
            limit: Maksimum kayıt sayısı
            
        Returns:
            Sequence[ModelType]: Model instance'ları
        """
        query = select(self.model)
        
        if filters:
            query = query.filter_by(**filters)
        
        if relations:
            for relation in relations:
                query = query.options(selectinload(getattr(self.model, relation)))
        
        if limit:
            query = query.limit(limit)
        
        result = await self.session.execute(query)
        return result.scalars().all()
    
    # =========================================================================
    # AGGREGATIONS
    # =========================================================================
    
    async def sum(self, field: str, **filters: Any) -> Optional[float]:
        """
        Alan toplamını al.
        
        Args:
            field: Alan adı
            **filters: Filtre koşulları
            
        Returns:
            Optional[float]: Toplam değer
        """
        column = getattr(self.model, field)
        query = select(func.sum(column)).select_from(self.model)
        
        if filters:
            query = query.filter_by(**filters)
        
        result = await self.session.execute(query)
        return result.scalar_one()
    
    async def avg(self, field: str, **filters: Any) -> Optional[float]:
        """
        Alan ortalamasını al.
        
        Args:
            field: Alan adı
            **filters: Filtre koşulları
            
        Returns:
            Optional[float]: Ortalama değer
        """
        column = getattr(self.model, field)
        query = select(func.avg(column)).select_from(self.model)
        
        if filters:
            query = query.filter_by(**filters)
        
        result = await self.session.execute(query)
        return result.scalar_one()
    
    async def max(self, field: str, **filters: Any) -> Optional[Any]:
        """
        Alan maksimum değerini al.
        
        Args:
            field: Alan adı
            **filters: Filtre koşulları
            
        Returns:
            Optional[Any]: Maksimum değer
        """
        column = getattr(self.model, field)
        query = select(func.max(column)).select_from(self.model)
        
        if filters:
            query = query.filter_by(**filters)
        
        result = await self.session.execute(query)
        return result.scalar_one()
    
    async def min(self, field: str, **filters: Any) -> Optional[Any]:
        """
        Alan minimum değerini al.
        
        Args:
            field: Alan adı
            **filters: Filtre koşulları
            
        Returns:
            Optional[Any]: Minimum değer
        """
        column = getattr(self.model, field)
        query = select(func.min(column)).select_from(self.model)
        
        if filters:
            query = query.filter_by(**filters)
        
        result = await self.session.execute(query)
        return result.scalar_one()
