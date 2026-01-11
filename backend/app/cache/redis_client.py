"""
AlphaTerminal Pro - Redis Cache Client
======================================

Redis cache bağlantısı ve yönetimi.

Author: AlphaTerminal Team
Version: 1.0.0
"""

from __future__ import annotations

import json
import logging
import pickle
from typing import Any, Optional, Union, List, TypeVar, Type
from datetime import timedelta

import redis.asyncio as redis
from redis.asyncio import Redis

from app.config import settings

logger = logging.getLogger(__name__)

T = TypeVar("T")


class RedisClient:
    """
    Redis cache client.
    
    Async Redis bağlantısı ve cache işlemleri sağlar.
    
    Example:
        ```python
        cache = RedisClient()
        await cache.initialize()
        
        # String cache
        await cache.set("key", "value", ttl=300)
        value = await cache.get("key")
        
        # JSON cache
        await cache.set_json("user:123", {"name": "Test"})
        user = await cache.get_json("user:123")
        
        # Pickle cache (complex objects)
        await cache.set_pickle("df:THYAO", dataframe)
        df = await cache.get_pickle("df:THYAO")
        
        await cache.close()
        ```
    """
    
    _instance: Optional["RedisClient"] = None
    _client: Optional[Redis] = None
    
    def __new__(cls) -> "RedisClient":
        """Singleton pattern."""
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance
    
    @property
    def client(self) -> Redis:
        """Redis client'a erişim."""
        if self._client is None:
            raise RuntimeError("Redis not initialized. Call initialize() first.")
        return self._client
    
    async def initialize(self) -> None:
        """
        Redis bağlantısını başlat.
        """
        if self._client is not None:
            logger.warning("Redis already initialized")
            return
        
        logger.info(f"Initializing Redis connection to {settings.redis.host}:{settings.redis.port}")
        
        self._client = redis.Redis(
            host=settings.redis.host,
            port=settings.redis.port,
            db=settings.redis.db,
            password=settings.redis.password,
            decode_responses=False,  # Binary için False
            socket_timeout=5,
            socket_connect_timeout=5,
        )
        
        # Bağlantı testi
        await self._client.ping()
        logger.info("Redis connection initialized successfully")
    
    async def close(self) -> None:
        """
        Redis bağlantısını kapat.
        """
        if self._client is not None:
            logger.info("Closing Redis connection")
            await self._client.close()
            self._client = None
            logger.info("Redis connection closed")
    
    async def health_check(self) -> bool:
        """
        Redis bağlantı kontrolü.
        
        Returns:
            bool: Bağlantı sağlıklı mı
        """
        try:
            await self._client.ping()
            return True
        except Exception as e:
            logger.error(f"Redis health check failed: {e}")
            return False
    
    # =========================================================================
    # BASIC OPERATIONS
    # =========================================================================
    
    async def get(self, key: str) -> Optional[str]:
        """
        String değer al.
        
        Args:
            key: Cache anahtarı
            
        Returns:
            Optional[str]: Cache değeri veya None
        """
        try:
            value = await self.client.get(key)
            if value is not None:
                return value.decode("utf-8")
            return None
        except Exception as e:
            logger.error(f"Redis GET error for key {key}: {e}")
            return None
    
    async def set(
        self,
        key: str,
        value: str,
        ttl: Optional[int] = None
    ) -> bool:
        """
        String değer kaydet.
        
        Args:
            key: Cache anahtarı
            value: Değer
            ttl: TTL (saniye)
            
        Returns:
            bool: Başarılı mı
        """
        try:
            if ttl:
                await self.client.setex(key, ttl, value)
            else:
                await self.client.set(key, value)
            return True
        except Exception as e:
            logger.error(f"Redis SET error for key {key}: {e}")
            return False
    
    async def delete(self, key: str) -> bool:
        """
        Anahtarı sil.
        
        Args:
            key: Cache anahtarı
            
        Returns:
            bool: Silindi mi
        """
        try:
            result = await self.client.delete(key)
            return result > 0
        except Exception as e:
            logger.error(f"Redis DELETE error for key {key}: {e}")
            return False
    
    async def exists(self, key: str) -> bool:
        """
        Anahtar var mı kontrol et.
        
        Args:
            key: Cache anahtarı
            
        Returns:
            bool: Var mı
        """
        try:
            return await self.client.exists(key) > 0
        except Exception as e:
            logger.error(f"Redis EXISTS error for key {key}: {e}")
            return False
    
    async def expire(self, key: str, ttl: int) -> bool:
        """
        TTL ayarla.
        
        Args:
            key: Cache anahtarı
            ttl: TTL (saniye)
            
        Returns:
            bool: Başarılı mı
        """
        try:
            return await self.client.expire(key, ttl)
        except Exception as e:
            logger.error(f"Redis EXPIRE error for key {key}: {e}")
            return False
    
    async def ttl(self, key: str) -> int:
        """
        Kalan TTL al.
        
        Args:
            key: Cache anahtarı
            
        Returns:
            int: Kalan saniye (-1: TTL yok, -2: anahtar yok)
        """
        try:
            return await self.client.ttl(key)
        except Exception as e:
            logger.error(f"Redis TTL error for key {key}: {e}")
            return -2
    
    # =========================================================================
    # JSON OPERATIONS
    # =========================================================================
    
    async def get_json(self, key: str) -> Optional[Any]:
        """
        JSON değer al.
        
        Args:
            key: Cache anahtarı
            
        Returns:
            Optional[Any]: Parse edilmiş JSON veya None
        """
        try:
            value = await self.client.get(key)
            if value is not None:
                return json.loads(value.decode("utf-8"))
            return None
        except Exception as e:
            logger.error(f"Redis GET JSON error for key {key}: {e}")
            return None
    
    async def set_json(
        self,
        key: str,
        value: Any,
        ttl: Optional[int] = None
    ) -> bool:
        """
        JSON değer kaydet.
        
        Args:
            key: Cache anahtarı
            value: JSON serializable değer
            ttl: TTL (saniye)
            
        Returns:
            bool: Başarılı mı
        """
        try:
            json_str = json.dumps(value, default=str)
            if ttl:
                await self.client.setex(key, ttl, json_str)
            else:
                await self.client.set(key, json_str)
            return True
        except Exception as e:
            logger.error(f"Redis SET JSON error for key {key}: {e}")
            return False
    
    # =========================================================================
    # PICKLE OPERATIONS (Complex Objects)
    # =========================================================================
    
    async def get_pickle(self, key: str) -> Optional[Any]:
        """
        Pickle değer al (DataFrame, numpy array, vb.).
        
        Args:
            key: Cache anahtarı
            
        Returns:
            Optional[Any]: Deserialize edilmiş obje veya None
        """
        try:
            value = await self.client.get(key)
            if value is not None:
                return pickle.loads(value)
            return None
        except Exception as e:
            logger.error(f"Redis GET PICKLE error for key {key}: {e}")
            return None
    
    async def set_pickle(
        self,
        key: str,
        value: Any,
        ttl: Optional[int] = None
    ) -> bool:
        """
        Pickle değer kaydet.
        
        Args:
            key: Cache anahtarı
            value: Herhangi bir Python objesi
            ttl: TTL (saniye)
            
        Returns:
            bool: Başarılı mı
        """
        try:
            pickled = pickle.dumps(value)
            if ttl:
                await self.client.setex(key, ttl, pickled)
            else:
                await self.client.set(key, pickled)
            return True
        except Exception as e:
            logger.error(f"Redis SET PICKLE error for key {key}: {e}")
            return False
    
    # =========================================================================
    # HASH OPERATIONS
    # =========================================================================
    
    async def hget(self, name: str, key: str) -> Optional[str]:
        """Hash field al."""
        try:
            value = await self.client.hget(name, key)
            if value is not None:
                return value.decode("utf-8")
            return None
        except Exception as e:
            logger.error(f"Redis HGET error: {e}")
            return None
    
    async def hset(self, name: str, key: str, value: str) -> bool:
        """Hash field ayarla."""
        try:
            await self.client.hset(name, key, value)
            return True
        except Exception as e:
            logger.error(f"Redis HSET error: {e}")
            return False
    
    async def hgetall(self, name: str) -> dict:
        """Tüm hash field'larını al."""
        try:
            result = await self.client.hgetall(name)
            return {k.decode("utf-8"): v.decode("utf-8") for k, v in result.items()}
        except Exception as e:
            logger.error(f"Redis HGETALL error: {e}")
            return {}
    
    async def hdel(self, name: str, *keys: str) -> int:
        """Hash field'larını sil."""
        try:
            return await self.client.hdel(name, *keys)
        except Exception as e:
            logger.error(f"Redis HDEL error: {e}")
            return 0
    
    # =========================================================================
    # LIST OPERATIONS
    # =========================================================================
    
    async def lpush(self, key: str, *values: str) -> int:
        """Liste başına ekle."""
        try:
            return await self.client.lpush(key, *values)
        except Exception as e:
            logger.error(f"Redis LPUSH error: {e}")
            return 0
    
    async def rpush(self, key: str, *values: str) -> int:
        """Liste sonuna ekle."""
        try:
            return await self.client.rpush(key, *values)
        except Exception as e:
            logger.error(f"Redis RPUSH error: {e}")
            return 0
    
    async def lrange(self, key: str, start: int, end: int) -> List[str]:
        """Liste aralığını al."""
        try:
            result = await self.client.lrange(key, start, end)
            return [v.decode("utf-8") for v in result]
        except Exception as e:
            logger.error(f"Redis LRANGE error: {e}")
            return []
    
    async def llen(self, key: str) -> int:
        """Liste uzunluğu."""
        try:
            return await self.client.llen(key)
        except Exception as e:
            logger.error(f"Redis LLEN error: {e}")
            return 0
    
    async def ltrim(self, key: str, start: int, end: int) -> bool:
        """Listeyi kırp."""
        try:
            await self.client.ltrim(key, start, end)
            return True
        except Exception as e:
            logger.error(f"Redis LTRIM error: {e}")
            return False
    
    # =========================================================================
    # SET OPERATIONS
    # =========================================================================
    
    async def sadd(self, key: str, *members: str) -> int:
        """Set'e üye ekle."""
        try:
            return await self.client.sadd(key, *members)
        except Exception as e:
            logger.error(f"Redis SADD error: {e}")
            return 0
    
    async def srem(self, key: str, *members: str) -> int:
        """Set'ten üye sil."""
        try:
            return await self.client.srem(key, *members)
        except Exception as e:
            logger.error(f"Redis SREM error: {e}")
            return 0
    
    async def smembers(self, key: str) -> set:
        """Set üyelerini al."""
        try:
            result = await self.client.smembers(key)
            return {v.decode("utf-8") for v in result}
        except Exception as e:
            logger.error(f"Redis SMEMBERS error: {e}")
            return set()
    
    async def sismember(self, key: str, member: str) -> bool:
        """Üye set'te mi kontrol et."""
        try:
            return await self.client.sismember(key, member)
        except Exception as e:
            logger.error(f"Redis SISMEMBER error: {e}")
            return False
    
    # =========================================================================
    # SORTED SET OPERATIONS
    # =========================================================================
    
    async def zadd(self, key: str, mapping: dict) -> int:
        """Sorted set'e ekle."""
        try:
            return await self.client.zadd(key, mapping)
        except Exception as e:
            logger.error(f"Redis ZADD error: {e}")
            return 0
    
    async def zrange(
        self,
        key: str,
        start: int,
        end: int,
        withscores: bool = False
    ) -> Union[List[str], List[tuple]]:
        """Sorted set aralığını al."""
        try:
            result = await self.client.zrange(key, start, end, withscores=withscores)
            if withscores:
                return [(v[0].decode("utf-8"), v[1]) for v in result]
            return [v.decode("utf-8") for v in result]
        except Exception as e:
            logger.error(f"Redis ZRANGE error: {e}")
            return []
    
    async def zrevrange(
        self,
        key: str,
        start: int,
        end: int,
        withscores: bool = False
    ) -> Union[List[str], List[tuple]]:
        """Sorted set'i ters sırada al."""
        try:
            result = await self.client.zrevrange(key, start, end, withscores=withscores)
            if withscores:
                return [(v[0].decode("utf-8"), v[1]) for v in result]
            return [v.decode("utf-8") for v in result]
        except Exception as e:
            logger.error(f"Redis ZREVRANGE error: {e}")
            return []
    
    # =========================================================================
    # PATTERN OPERATIONS
    # =========================================================================
    
    async def keys(self, pattern: str) -> List[str]:
        """
        Pattern ile anahtarları bul.
        
        DİKKAT: Production'da dikkatli kullanın!
        """
        try:
            result = await self.client.keys(pattern)
            return [k.decode("utf-8") for k in result]
        except Exception as e:
            logger.error(f"Redis KEYS error: {e}")
            return []
    
    async def delete_pattern(self, pattern: str) -> int:
        """
        Pattern ile anahtarları sil.
        
        Args:
            pattern: Glob pattern (örn: "cache:*")
            
        Returns:
            int: Silinen anahtar sayısı
        """
        try:
            keys = await self.keys(pattern)
            if keys:
                return await self.client.delete(*keys)
            return 0
        except Exception as e:
            logger.error(f"Redis DELETE PATTERN error: {e}")
            return 0
    
    # =========================================================================
    # PUB/SUB
    # =========================================================================
    
    async def publish(self, channel: str, message: str) -> int:
        """
        Kanala mesaj yayınla.
        
        Args:
            channel: Kanal adı
            message: Mesaj
            
        Returns:
            int: Mesajı alan subscriber sayısı
        """
        try:
            return await self.client.publish(channel, message)
        except Exception as e:
            logger.error(f"Redis PUBLISH error: {e}")
            return 0
    
    async def subscribe(self, *channels: str):
        """
        Kanallara abone ol.
        
        Returns:
            PubSub: Pub/Sub objesi
        """
        pubsub = self.client.pubsub()
        await pubsub.subscribe(*channels)
        return pubsub
    
    # =========================================================================
    # UTILITY
    # =========================================================================
    
    async def flush_db(self) -> bool:
        """
        Veritabanını temizle.
        
        DİKKAT: Tüm veriler silinir!
        """
        try:
            await self.client.flushdb()
            logger.warning("Redis database flushed")
            return True
        except Exception as e:
            logger.error(f"Redis FLUSHDB error: {e}")
            return False
    
    async def info(self) -> dict:
        """Redis server bilgilerini al."""
        try:
            return await self.client.info()
        except Exception as e:
            logger.error(f"Redis INFO error: {e}")
            return {}
    
    async def dbsize(self) -> int:
        """Veritabanındaki anahtar sayısı."""
        try:
            return await self.client.dbsize()
        except Exception as e:
            logger.error(f"Redis DBSIZE error: {e}")
            return 0


# Global Redis client instance
cache = RedisClient()


async def init_cache() -> None:
    """Cache'i başlat."""
    await cache.initialize()


async def close_cache() -> None:
    """Cache'i kapat."""
    await cache.close()
