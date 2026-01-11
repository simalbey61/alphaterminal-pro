"""
AlphaTerminal Pro - Data Cache System
=====================================

Multi-tier caching system for market data.
Supports memory and disk caching with TTL.

Author: AlphaTerminal Team
Version: 1.0.0
"""

import logging
import hashlib
import pickle
import time
import threading
import os
from datetime import datetime, timedelta
from typing import Optional, Dict, List, Any, Callable
from pathlib import Path
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
import json

import pandas as pd

from app.data_providers.enums import DataInterval, DataSource
from app.data_providers.models import MarketData, CacheEntry
from app.data_providers.exceptions import (
    CacheError, CacheMissError, CacheWriteError,
    CacheSerializationError
)


logger = logging.getLogger(__name__)


# =============================================================================
# CACHE KEY BUILDER
# =============================================================================

class CacheKeyBuilder:
    """Build consistent cache keys."""
    
    VERSION = "v1"
    
    @classmethod
    def build(
        cls,
        symbol: str,
        interval: DataInterval,
        source: Optional[DataSource] = None,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None,
        **extra
    ) -> str:
        """
        Build cache key from parameters.
        
        Format: {version}:{symbol}:{interval}:{source}:{date_hash}:{extra_hash}
        """
        parts = [
            cls.VERSION,
            symbol.upper(),
            interval.value,
            source.value if source else "any"
        ]
        
        # Add date range hash
        if start_date or end_date:
            date_str = f"{start_date or 'none'}:{end_date or 'none'}"
            date_hash = hashlib.md5(date_str.encode()).hexdigest()[:8]
            parts.append(date_hash)
        else:
            parts.append("latest")
        
        # Add extra params hash if any
        if extra:
            extra_str = json.dumps(extra, sort_keys=True)
            extra_hash = hashlib.md5(extra_str.encode()).hexdigest()[:8]
            parts.append(extra_hash)
        
        return ":".join(parts)
    
    @classmethod
    def parse(cls, key: str) -> Dict[str, str]:
        """Parse cache key into components."""
        parts = key.split(":")
        
        result = {
            "version": parts[0] if len(parts) > 0 else None,
            "symbol": parts[1] if len(parts) > 1 else None,
            "interval": parts[2] if len(parts) > 2 else None,
            "source": parts[3] if len(parts) > 3 else None,
        }
        
        return result


# =============================================================================
# ABSTRACT CACHE BACKEND
# =============================================================================

class CacheBackend(ABC):
    """Abstract base class for cache backends."""
    
    @abstractmethod
    def get(self, key: str) -> Optional[CacheEntry]:
        """Get entry from cache."""
        pass
    
    @abstractmethod
    def set(self, key: str, entry: CacheEntry) -> bool:
        """Set entry in cache."""
        pass
    
    @abstractmethod
    def delete(self, key: str) -> bool:
        """Delete entry from cache."""
        pass
    
    @abstractmethod
    def exists(self, key: str) -> bool:
        """Check if key exists."""
        pass
    
    @abstractmethod
    def clear(self) -> int:
        """Clear all entries. Returns count of deleted entries."""
        pass
    
    @abstractmethod
    def keys(self, pattern: Optional[str] = None) -> List[str]:
        """Get all keys, optionally matching pattern."""
        pass
    
    @abstractmethod
    def size(self) -> int:
        """Get number of entries in cache."""
        pass


# =============================================================================
# MEMORY CACHE BACKEND
# =============================================================================

class MemoryCacheBackend(CacheBackend):
    """
    In-memory cache backend using dict.
    
    Fast but limited by available memory.
    Data lost on restart.
    """
    
    def __init__(self, max_entries: int = 1000):
        """
        Initialize memory cache.
        
        Args:
            max_entries: Maximum entries to store (LRU eviction)
        """
        self.max_entries = max_entries
        self._cache: Dict[str, CacheEntry] = {}
        self._access_order: List[str] = []
        self._lock = threading.RLock()
    
    def get(self, key: str) -> Optional[CacheEntry]:
        with self._lock:
            entry = self._cache.get(key)
            
            if entry is None:
                return None
            
            # Check expiration
            if entry.is_expired:
                self._remove(key)
                return None
            
            # Update access order (LRU)
            self._touch(key)
            entry.touch()
            
            return entry
    
    def set(self, key: str, entry: CacheEntry) -> bool:
        with self._lock:
            # Evict if at capacity
            while len(self._cache) >= self.max_entries:
                self._evict_lru()
            
            self._cache[key] = entry
            self._touch(key)
            
            return True
    
    def delete(self, key: str) -> bool:
        with self._lock:
            return self._remove(key)
    
    def exists(self, key: str) -> bool:
        with self._lock:
            if key not in self._cache:
                return False
            
            # Check expiration
            if self._cache[key].is_expired:
                self._remove(key)
                return False
            
            return True
    
    def clear(self) -> int:
        with self._lock:
            count = len(self._cache)
            self._cache.clear()
            self._access_order.clear()
            return count
    
    def keys(self, pattern: Optional[str] = None) -> List[str]:
        with self._lock:
            all_keys = list(self._cache.keys())
            
            if pattern is None:
                return all_keys
            
            # Simple pattern matching (prefix)
            return [k for k in all_keys if k.startswith(pattern)]
    
    def size(self) -> int:
        return len(self._cache)
    
    def _touch(self, key: str):
        """Update access order for LRU."""
        if key in self._access_order:
            self._access_order.remove(key)
        self._access_order.append(key)
    
    def _remove(self, key: str) -> bool:
        """Remove key from cache."""
        if key in self._cache:
            del self._cache[key]
            if key in self._access_order:
                self._access_order.remove(key)
            return True
        return False
    
    def _evict_lru(self):
        """Evict least recently used entry."""
        if self._access_order:
            lru_key = self._access_order.pop(0)
            if lru_key in self._cache:
                del self._cache[lru_key]
                logger.debug(f"Evicted LRU cache entry: {lru_key}")


# =============================================================================
# DISK CACHE BACKEND
# =============================================================================

class DiskCacheBackend(CacheBackend):
    """
    Disk-based cache backend using pickle files.
    
    Persistent but slower than memory.
    Suitable for larger datasets.
    """
    
    def __init__(
        self,
        cache_dir: str = "cache/data",
        max_size_mb: int = 500
    ):
        """
        Initialize disk cache.
        
        Args:
            cache_dir: Directory for cache files
            max_size_mb: Maximum cache size in MB
        """
        self.cache_dir = Path(cache_dir)
        self.max_size_bytes = max_size_mb * 1024 * 1024
        
        self._lock = threading.RLock()
        self._metadata: Dict[str, Dict] = {}
        
        # Create directory
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        
        # Load metadata
        self._load_metadata()
    
    def get(self, key: str) -> Optional[CacheEntry]:
        with self._lock:
            if key not in self._metadata:
                return None
            
            meta = self._metadata[key]
            
            # Check expiration
            if datetime.fromisoformat(meta["expires_at"]) <= datetime.now():
                self.delete(key)
                return None
            
            # Load from disk
            try:
                file_path = self._key_to_path(key)
                
                if not file_path.exists():
                    del self._metadata[key]
                    return None
                
                with open(file_path, "rb") as f:
                    entry = pickle.load(f)
                
                entry.touch()
                self._metadata[key]["hits"] = entry.hits
                self._save_metadata()
                
                return entry
                
            except Exception as e:
                logger.error(f"Failed to load cache entry {key}: {e}")
                self.delete(key)
                return None
    
    def set(self, key: str, entry: CacheEntry) -> bool:
        with self._lock:
            try:
                # Check size limits
                self._enforce_size_limit()
                
                file_path = self._key_to_path(key)
                file_path.parent.mkdir(parents=True, exist_ok=True)
                
                with open(file_path, "wb") as f:
                    pickle.dump(entry, f)
                
                # Update metadata
                self._metadata[key] = {
                    "created_at": entry.created_at.isoformat(),
                    "expires_at": entry.expires_at.isoformat(),
                    "hits": entry.hits,
                    "size_bytes": file_path.stat().st_size
                }
                self._save_metadata()
                
                return True
                
            except Exception as e:
                logger.error(f"Failed to write cache entry {key}: {e}")
                raise CacheWriteError(
                    f"Failed to write cache: {e}",
                    details={"key": key}
                )
    
    def delete(self, key: str) -> bool:
        with self._lock:
            file_path = self._key_to_path(key)
            
            deleted = False
            if file_path.exists():
                file_path.unlink()
                deleted = True
            
            if key in self._metadata:
                del self._metadata[key]
                self._save_metadata()
                deleted = True
            
            return deleted
    
    def exists(self, key: str) -> bool:
        with self._lock:
            if key not in self._metadata:
                return False
            
            # Check expiration
            meta = self._metadata[key]
            if datetime.fromisoformat(meta["expires_at"]) <= datetime.now():
                self.delete(key)
                return False
            
            return self._key_to_path(key).exists()
    
    def clear(self) -> int:
        with self._lock:
            count = 0
            
            for key in list(self._metadata.keys()):
                if self.delete(key):
                    count += 1
            
            return count
    
    def keys(self, pattern: Optional[str] = None) -> List[str]:
        with self._lock:
            all_keys = list(self._metadata.keys())
            
            if pattern is None:
                return all_keys
            
            return [k for k in all_keys if k.startswith(pattern)]
    
    def size(self) -> int:
        return len(self._metadata)
    
    def total_size_bytes(self) -> int:
        """Get total cache size in bytes."""
        return sum(m.get("size_bytes", 0) for m in self._metadata.values())
    
    def _key_to_path(self, key: str) -> Path:
        """Convert key to file path."""
        # Use hash for filename to avoid path issues
        key_hash = hashlib.md5(key.encode()).hexdigest()
        return self.cache_dir / f"{key_hash}.pkl"
    
    def _load_metadata(self):
        """Load metadata from disk."""
        meta_path = self.cache_dir / "_metadata.json"
        
        if meta_path.exists():
            try:
                with open(meta_path, "r") as f:
                    self._metadata = json.load(f)
                logger.debug(f"Loaded {len(self._metadata)} cache entries metadata")
            except Exception as e:
                logger.warning(f"Failed to load cache metadata: {e}")
                self._metadata = {}
    
    def _save_metadata(self):
        """Save metadata to disk."""
        meta_path = self.cache_dir / "_metadata.json"
        
        try:
            with open(meta_path, "w") as f:
                json.dump(self._metadata, f)
        except Exception as e:
            logger.warning(f"Failed to save cache metadata: {e}")
    
    def _enforce_size_limit(self):
        """Enforce maximum cache size by removing oldest entries."""
        while self.total_size_bytes() > self.max_size_bytes:
            # Find oldest entry
            oldest_key = min(
                self._metadata.keys(),
                key=lambda k: self._metadata[k].get("created_at", "")
            )
            self.delete(oldest_key)
            logger.debug(f"Evicted cache entry for size limit: {oldest_key}")


# =============================================================================
# TIERED CACHE
# =============================================================================

class TieredCache:
    """
    Multi-tier cache combining memory and disk.
    
    L1: Fast memory cache for hot data
    L2: Disk cache for larger/cold data
    
    Features:
    - Automatic tier promotion/demotion
    - Configurable TTL per tier
    - Cache warming
    - Statistics tracking
    """
    
    def __init__(
        self,
        memory_max_entries: int = 500,
        disk_cache_dir: str = "cache/data",
        disk_max_size_mb: int = 500,
        default_memory_ttl: int = 300,      # 5 minutes
        default_disk_ttl: int = 3600,       # 1 hour
    ):
        """
        Initialize tiered cache.
        
        Args:
            memory_max_entries: Max entries in memory
            disk_cache_dir: Disk cache directory
            disk_max_size_mb: Max disk cache size
            default_memory_ttl: Default memory TTL (seconds)
            default_disk_ttl: Default disk TTL (seconds)
        """
        self.l1 = MemoryCacheBackend(max_entries=memory_max_entries)
        self.l2 = DiskCacheBackend(
            cache_dir=disk_cache_dir,
            max_size_mb=disk_max_size_mb
        )
        
        self.default_memory_ttl = default_memory_ttl
        self.default_disk_ttl = default_disk_ttl
        
        # Statistics
        self._stats = {
            "l1_hits": 0,
            "l2_hits": 0,
            "misses": 0,
            "writes": 0,
        }
    
    def get(
        self,
        key: str,
        promote_to_l1: bool = True
    ) -> Optional[MarketData]:
        """
        Get data from cache.
        
        Checks L1 first, then L2.
        Optionally promotes L2 hits to L1.
        
        Args:
            key: Cache key
            promote_to_l1: Promote L2 hits to L1
            
        Returns:
            MarketData or None if not found
        """
        # Try L1 (memory)
        entry = self.l1.get(key)
        if entry is not None:
            self._stats["l1_hits"] += 1
            return entry.data
        
        # Try L2 (disk)
        entry = self.l2.get(key)
        if entry is not None:
            self._stats["l2_hits"] += 1
            
            # Promote to L1
            if promote_to_l1:
                l1_entry = CacheEntry(
                    key=key,
                    data=entry.data,
                    expires_at=datetime.now() + timedelta(seconds=self.default_memory_ttl)
                )
                self.l1.set(key, l1_entry)
            
            return entry.data
        
        self._stats["misses"] += 1
        return None
    
    def set(
        self,
        key: str,
        data: MarketData,
        memory_ttl: Optional[int] = None,
        disk_ttl: Optional[int] = None,
        memory_only: bool = False,
        disk_only: bool = False
    ) -> bool:
        """
        Set data in cache.
        
        By default writes to both tiers.
        
        Args:
            key: Cache key
            data: MarketData to cache
            memory_ttl: Memory TTL override
            disk_ttl: Disk TTL override
            memory_only: Only cache in memory
            disk_only: Only cache on disk
            
        Returns:
            True if successful
        """
        self._stats["writes"] += 1
        
        memory_ttl = memory_ttl or self.default_memory_ttl
        disk_ttl = disk_ttl or self.default_disk_ttl
        
        success = True
        
        # Write to L1
        if not disk_only:
            l1_entry = CacheEntry(
                key=key,
                data=data,
                expires_at=datetime.now() + timedelta(seconds=memory_ttl)
            )
            success = self.l1.set(key, l1_entry) and success
        
        # Write to L2
        if not memory_only:
            l2_entry = CacheEntry(
                key=key,
                data=data,
                expires_at=datetime.now() + timedelta(seconds=disk_ttl)
            )
            try:
                success = self.l2.set(key, l2_entry) and success
            except CacheWriteError:
                # Disk write failed, but memory succeeded
                pass
        
        return success
    
    def delete(self, key: str) -> bool:
        """Delete from all tiers."""
        l1_deleted = self.l1.delete(key)
        l2_deleted = self.l2.delete(key)
        return l1_deleted or l2_deleted
    
    def exists(self, key: str) -> bool:
        """Check if key exists in any tier."""
        return self.l1.exists(key) or self.l2.exists(key)
    
    def clear(self) -> int:
        """Clear all tiers."""
        l1_count = self.l1.clear()
        l2_count = self.l2.clear()
        return l1_count + l2_count
    
    def invalidate_symbol(self, symbol: str) -> int:
        """Invalidate all cache entries for a symbol."""
        pattern = f"v1:{symbol.upper()}:"
        
        count = 0
        for key in self.l1.keys(pattern):
            if self.l1.delete(key):
                count += 1
        
        for key in self.l2.keys(pattern):
            if self.l2.delete(key):
                count += 1
        
        return count
    
    def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics."""
        total_requests = (
            self._stats["l1_hits"] + 
            self._stats["l2_hits"] + 
            self._stats["misses"]
        )
        
        return {
            "l1_hits": self._stats["l1_hits"],
            "l2_hits": self._stats["l2_hits"],
            "total_hits": self._stats["l1_hits"] + self._stats["l2_hits"],
            "misses": self._stats["misses"],
            "writes": self._stats["writes"],
            "hit_rate": (self._stats["l1_hits"] + self._stats["l2_hits"]) / max(1, total_requests),
            "l1_size": self.l1.size(),
            "l2_size": self.l2.size(),
            "l2_size_bytes": self.l2.total_size_bytes(),
        }


# =============================================================================
# CACHE MANAGER
# =============================================================================

class DataCacheManager:
    """
    High-level cache manager for market data.
    
    Provides simple interface for caching MarketData with
    automatic key generation and TTL management.
    """
    
    _instance: Optional["DataCacheManager"] = None
    
    def __init__(
        self,
        cache_dir: str = "cache/data",
        memory_max_entries: int = 500,
        disk_max_size_mb: int = 500,
        intraday_ttl: int = 300,        # 5 min
        daily_ttl: int = 3600,          # 1 hour
        symbol_info_ttl: int = 86400,   # 24 hours
    ):
        """Initialize cache manager."""
        self.cache = TieredCache(
            memory_max_entries=memory_max_entries,
            disk_cache_dir=cache_dir,
            disk_max_size_mb=disk_max_size_mb
        )
        
        self.intraday_ttl = intraday_ttl
        self.daily_ttl = daily_ttl
        self.symbol_info_ttl = symbol_info_ttl
    
    @classmethod
    def get_instance(cls) -> "DataCacheManager":
        """Get singleton instance."""
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance
    
    def get_market_data(
        self,
        symbol: str,
        interval: DataInterval,
        source: Optional[DataSource] = None,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None
    ) -> Optional[MarketData]:
        """
        Get cached market data.
        
        Args:
            symbol: Trading symbol
            interval: Data interval
            source: Data source (optional)
            start_date: Start date (optional)
            end_date: End date (optional)
            
        Returns:
            Cached MarketData or None
        """
        key = CacheKeyBuilder.build(
            symbol=symbol,
            interval=interval,
            source=source,
            start_date=start_date,
            end_date=end_date
        )
        
        return self.cache.get(key)
    
    def set_market_data(
        self,
        data: MarketData,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None,
        ttl_override: Optional[int] = None
    ) -> bool:
        """
        Cache market data.
        
        Args:
            data: MarketData to cache
            start_date: Start date for key
            end_date: End date for key
            ttl_override: Override default TTL
            
        Returns:
            True if cached successfully
        """
        key = CacheKeyBuilder.build(
            symbol=data.symbol,
            interval=data.interval,
            source=data.source,
            start_date=start_date,
            end_date=end_date
        )
        
        # Determine TTL
        if ttl_override:
            ttl = ttl_override
        elif data.interval.is_intraday:
            ttl = self.intraday_ttl
        else:
            ttl = self.daily_ttl
        
        return self.cache.set(key, data, memory_ttl=min(ttl, 600), disk_ttl=ttl)
    
    def invalidate(
        self,
        symbol: Optional[str] = None,
        interval: Optional[DataInterval] = None
    ) -> int:
        """
        Invalidate cache entries.
        
        Args:
            symbol: Symbol to invalidate (all if None)
            interval: Interval to invalidate (all if None)
            
        Returns:
            Number of entries invalidated
        """
        if symbol:
            return self.cache.invalidate_symbol(symbol)
        else:
            return self.cache.clear()
    
    def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics."""
        return self.cache.get_stats()


# =============================================================================
# ALL EXPORTS
# =============================================================================

__all__ = [
    "CacheKeyBuilder",
    "CacheBackend",
    "MemoryCacheBackend",
    "DiskCacheBackend",
    "TieredCache",
    "DataCacheManager",
]
