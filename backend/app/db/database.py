"""
AlphaTerminal Pro - Database Connection & Session
=================================================

PostgreSQL veritabanı bağlantısı ve session yönetimi.
Async SQLAlchemy kullanır.

Author: AlphaTerminal Team
Version: 1.0.0
"""

from __future__ import annotations

import logging
from contextlib import asynccontextmanager
from typing import AsyncGenerator, Optional

from sqlalchemy import event, text
from sqlalchemy.ext.asyncio import (
    AsyncSession,
    AsyncEngine,
    async_sessionmaker,
    create_async_engine,
)
from sqlalchemy.orm import Session
from sqlalchemy.pool import Pool

from app.config import settings
from app.db.models import Base

logger = logging.getLogger(__name__)


class DatabaseManager:
    """
    Veritabanı bağlantı yöneticisi.
    
    Singleton pattern ile tek bir instance üzerinden
    tüm veritabanı işlemlerini yönetir.
    
    Attributes:
        engine: Async SQLAlchemy engine
        session_factory: Session oluşturucu
        
    Example:
        ```python
        db = DatabaseManager()
        await db.initialize()
        
        async with db.session() as session:
            result = await session.execute(select(User))
            users = result.scalars().all()
        
        await db.close()
        ```
    """
    
    _instance: Optional["DatabaseManager"] = None
    _engine: Optional[AsyncEngine] = None
    _session_factory: Optional[async_sessionmaker[AsyncSession]] = None
    
    def __new__(cls) -> "DatabaseManager":
        """Singleton pattern."""
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance
    
    @property
    def engine(self) -> AsyncEngine:
        """Engine'e erişim."""
        if self._engine is None:
            raise RuntimeError("Database not initialized. Call initialize() first.")
        return self._engine
    
    @property
    def session_factory(self) -> async_sessionmaker[AsyncSession]:
        """Session factory'ye erişim."""
        if self._session_factory is None:
            raise RuntimeError("Database not initialized. Call initialize() first.")
        return self._session_factory
    
    async def initialize(self) -> None:
        """
        Veritabanı bağlantısını başlat.
        
        Engine ve session factory oluşturur,
        connection pool'u yapılandırır.
        """
        if self._engine is not None:
            logger.warning("Database already initialized")
            return
        
        logger.info(f"Initializing database connection to {settings.database.host}:{settings.database.port}")
        
        # Engine oluştur
        self._engine = create_async_engine(
            settings.database.url,
            echo=settings.database.echo,
            pool_size=settings.database.pool_size,
            max_overflow=settings.database.max_overflow,
            pool_timeout=settings.database.pool_timeout,
            pool_recycle=settings.database.pool_recycle,
            pool_pre_ping=True,  # Connection health check
        )
        
        # Session factory oluştur
        self._session_factory = async_sessionmaker(
            bind=self._engine,
            class_=AsyncSession,
            expire_on_commit=False,
            autocommit=False,
            autoflush=False,
        )
        
        # Pool event listener'ları ekle
        self._setup_pool_events()
        
        logger.info("Database connection initialized successfully")
    
    def _setup_pool_events(self) -> None:
        """Connection pool event listener'larını ayarla."""
        
        @event.listens_for(self._engine.sync_engine, "connect")
        def on_connect(dbapi_connection, connection_record):
            """Yeni bağlantı oluşturulduğunda."""
            logger.debug("New database connection established")
        
        @event.listens_for(self._engine.sync_engine, "checkout")
        def on_checkout(dbapi_connection, connection_record, connection_proxy):
            """Pool'dan bağlantı alındığında."""
            logger.debug("Connection checked out from pool")
        
        @event.listens_for(self._engine.sync_engine, "checkin")
        def on_checkin(dbapi_connection, connection_record):
            """Pool'a bağlantı geri verildiğinde."""
            logger.debug("Connection returned to pool")
    
    async def close(self) -> None:
        """
        Veritabanı bağlantısını kapat.
        
        Tüm açık connection'ları kapatır ve
        pool'u temizler.
        """
        if self._engine is not None:
            logger.info("Closing database connection")
            await self._engine.dispose()
            self._engine = None
            self._session_factory = None
            logger.info("Database connection closed")
    
    @asynccontextmanager
    async def session(self) -> AsyncGenerator[AsyncSession, None]:
        """
        Session context manager.
        
        Yields:
            AsyncSession: Veritabanı session'ı
            
        Example:
            ```python
            async with db.session() as session:
                user = await session.get(User, user_id)
            ```
        """
        session = self.session_factory()
        try:
            yield session
            await session.commit()
        except Exception as e:
            await session.rollback()
            logger.error(f"Database session error: {e}")
            raise
        finally:
            await session.close()
    
    @asynccontextmanager
    async def transaction(self) -> AsyncGenerator[AsyncSession, None]:
        """
        Transaction context manager.
        
        Nested transaction desteği ile explicit
        transaction yönetimi sağlar.
        
        Yields:
            AsyncSession: Transaction içindeki session
            
        Example:
            ```python
            async with db.transaction() as session:
                # Bu blok başarılı olursa commit edilir
                # Hata olursa rollback edilir
                session.add(user)
                session.add(profile)
            ```
        """
        session = self.session_factory()
        try:
            async with session.begin():
                yield session
        except Exception as e:
            logger.error(f"Transaction error: {e}")
            raise
        finally:
            await session.close()
    
    async def create_tables(self) -> None:
        """
        Tüm tabloları oluştur.
        
        Development ve testing için kullanılır.
        Production'da Alembic migration kullanılmalı.
        """
        logger.info("Creating database tables")
        async with self._engine.begin() as conn:
            await conn.run_sync(Base.metadata.create_all)
        logger.info("Database tables created")
    
    async def drop_tables(self) -> None:
        """
        Tüm tabloları sil.
        
        DİKKAT: Bu işlem geri alınamaz!
        Sadece testing için kullanılmalı.
        """
        logger.warning("Dropping all database tables")
        async with self._engine.begin() as conn:
            await conn.run_sync(Base.metadata.drop_all)
        logger.warning("All database tables dropped")
    
    async def health_check(self) -> bool:
        """
        Veritabanı bağlantı kontrolü.
        
        Returns:
            bool: Bağlantı sağlıklı mı
        """
        try:
            async with self.session() as session:
                await session.execute(text("SELECT 1"))
            return True
        except Exception as e:
            logger.error(f"Database health check failed: {e}")
            return False
    
    async def get_pool_status(self) -> dict:
        """
        Connection pool durumunu döndür.
        
        Returns:
            dict: Pool istatistikleri
        """
        pool = self._engine.pool
        return {
            "pool_size": pool.size(),
            "checked_out": pool.checkedout(),
            "overflow": pool.overflow(),
            "checked_in": pool.checkedin(),
        }


# Global database manager instance
db = DatabaseManager()


async def get_session() -> AsyncGenerator[AsyncSession, None]:
    """
    FastAPI dependency için session generator.
    
    Yields:
        AsyncSession: Veritabanı session'ı
        
    Example:
        ```python
        @router.get("/users")
        async def get_users(session: AsyncSession = Depends(get_session)):
            result = await session.execute(select(User))
            return result.scalars().all()
        ```
    """
    async with db.session() as session:
        yield session


async def init_db() -> None:
    """
    Veritabanını başlat.
    
    Application startup'ta çağrılır.
    """
    await db.initialize()


async def close_db() -> None:
    """
    Veritabanını kapat.
    
    Application shutdown'da çağrılır.
    """
    await db.close()
