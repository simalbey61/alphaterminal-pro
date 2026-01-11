"""
AlphaTerminal Pro - Feature Store
=================================

Redis tabanlı feature depolama ve yönetimi.

Author: AlphaTerminal Team
Version: 1.0.0
"""

from __future__ import annotations

import logging
import json
import pickle
from typing import Optional, List, Dict, Any, Set
from datetime import datetime, timedelta
from dataclasses import dataclass
import asyncio

from app.cache import cache, CacheKeys, CacheTTL

logger = logging.getLogger(__name__)


@dataclass
class StoredFeature:
    """Depolanan feature."""
    name: str
    value: float
    timestamp: datetime
    symbol: str
    timeframe: str
    ttl_seconds: int = 3600
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "value": self.value,
            "timestamp": self.timestamp.isoformat(),
            "symbol": self.symbol,
            "timeframe": self.timeframe,
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "StoredFeature":
        return cls(
            name=data["name"],
            value=data["value"],
            timestamp=datetime.fromisoformat(data["timestamp"]),
            symbol=data["symbol"],
            timeframe=data["timeframe"],
        )


class FeatureStore:
    """
    Redis tabanlı feature store.
    
    Özellikler:
    - Feature caching ve persistence
    - Batch operations
    - TTL management
    - Versioning
    - History tracking
    
    Key Patterns:
    - Single feature: features:{symbol}:{timeframe}:{feature_name}
    - Batch: features:{symbol}:{timeframe}:batch
    - History: features:{symbol}:{timeframe}:{feature_name}:history
    
    Example:
        ```python
        store = FeatureStore()
        
        # Tekil feature kaydet
        await store.set_feature("THYAO", "4h", "rsi_14", 65.5)
        
        # Batch kaydet
        await store.set_batch("THYAO", "4h", {"rsi_14": 65.5, "macd": 0.5})
        
        # Oku
        value = await store.get_feature("THYAO", "4h", "rsi_14")
        batch = await store.get_batch("THYAO", "4h")
        ```
    """
    
    # Key prefixes
    PREFIX = "features"
    BATCH_SUFFIX = "batch"
    HISTORY_SUFFIX = "history"
    META_SUFFIX = "meta"
    
    # Default TTLs
    DEFAULT_TTL = 3600  # 1 saat
    BATCH_TTL = 1800    # 30 dakika
    HISTORY_TTL = 86400  # 24 saat
    
    def __init__(
        self,
        default_ttl: int = DEFAULT_TTL,
        history_enabled: bool = True,
        max_history_size: int = 100,
    ):
        """
        Initialize feature store.
        
        Args:
            default_ttl: Varsayılan TTL (saniye)
            history_enabled: Geçmiş kaydı aktif mi?
            max_history_size: Maximum geçmiş kayıt sayısı
        """
        self.default_ttl = default_ttl
        self.history_enabled = history_enabled
        self.max_history_size = max_history_size
    
    # =========================================================================
    # SINGLE FEATURE OPERATIONS
    # =========================================================================
    
    async def set_feature(
        self,
        symbol: str,
        timeframe: str,
        feature_name: str,
        value: float,
        timestamp: Optional[datetime] = None,
        ttl: Optional[int] = None,
    ) -> bool:
        """
        Tekil feature kaydet.
        
        Args:
            symbol: Hisse sembolü
            timeframe: Zaman dilimi
            feature_name: Feature adı
            value: Feature değeri
            timestamp: Zaman damgası
            ttl: TTL (saniye)
            
        Returns:
            bool: Başarılı mı?
        """
        try:
            key = self._make_key(symbol, timeframe, feature_name)
            
            data = {
                "value": value,
                "timestamp": (timestamp or datetime.utcnow()).isoformat(),
                "symbol": symbol,
                "timeframe": timeframe,
                "feature": feature_name,
            }
            
            await cache.set_json(key, data, ttl=ttl or self.default_ttl)
            
            # History kaydet
            if self.history_enabled:
                await self._add_to_history(symbol, timeframe, feature_name, value, timestamp)
            
            return True
            
        except Exception as e:
            logger.error(f"Error setting feature {feature_name}: {e}")
            return False
    
    async def get_feature(
        self,
        symbol: str,
        timeframe: str,
        feature_name: str,
    ) -> Optional[float]:
        """
        Tekil feature oku.
        
        Args:
            symbol: Hisse sembolü
            timeframe: Zaman dilimi
            feature_name: Feature adı
            
        Returns:
            Optional[float]: Feature değeri veya None
        """
        try:
            key = self._make_key(symbol, timeframe, feature_name)
            data = await cache.get_json(key)
            
            if data:
                return data.get("value")
            
            return None
            
        except Exception as e:
            logger.error(f"Error getting feature {feature_name}: {e}")
            return None
    
    async def delete_feature(
        self,
        symbol: str,
        timeframe: str,
        feature_name: str,
    ) -> bool:
        """Feature sil."""
        try:
            key = self._make_key(symbol, timeframe, feature_name)
            await cache.delete(key)
            return True
        except Exception as e:
            logger.error(f"Error deleting feature {feature_name}: {e}")
            return False
    
    # =========================================================================
    # BATCH OPERATIONS
    # =========================================================================
    
    async def set_batch(
        self,
        symbol: str,
        timeframe: str,
        features: Dict[str, float],
        timestamp: Optional[datetime] = None,
        ttl: Optional[int] = None,
    ) -> bool:
        """
        Batch feature kaydet.
        
        Args:
            symbol: Hisse sembolü
            timeframe: Zaman dilimi
            features: Feature dict {name: value}
            timestamp: Zaman damgası
            ttl: TTL (saniye)
            
        Returns:
            bool: Başarılı mı?
        """
        try:
            key = self._make_batch_key(symbol, timeframe)
            ts = timestamp or datetime.utcnow()
            
            data = {
                "features": features,
                "timestamp": ts.isoformat(),
                "symbol": symbol,
                "timeframe": timeframe,
                "count": len(features),
            }
            
            await cache.set_json(key, data, ttl=ttl or self.BATCH_TTL)
            
            # Ayrıca tekil key'lere de kaydet (parallel)
            tasks = []
            for name, value in features.items():
                tasks.append(
                    self.set_feature(symbol, timeframe, name, value, ts, ttl)
                )
            
            await asyncio.gather(*tasks, return_exceptions=True)
            
            logger.debug(f"Batch stored: {symbol}:{timeframe} - {len(features)} features")
            return True
            
        except Exception as e:
            logger.error(f"Error setting batch for {symbol}: {e}")
            return False
    
    async def get_batch(
        self,
        symbol: str,
        timeframe: str,
    ) -> Optional[Dict[str, float]]:
        """
        Batch feature oku.
        
        Args:
            symbol: Hisse sembolü
            timeframe: Zaman dilimi
            
        Returns:
            Optional[Dict[str, float]]: Features veya None
        """
        try:
            key = self._make_batch_key(symbol, timeframe)
            data = await cache.get_json(key)
            
            if data:
                return data.get("features")
            
            return None
            
        except Exception as e:
            logger.error(f"Error getting batch for {symbol}: {e}")
            return None
    
    async def get_batch_with_meta(
        self,
        symbol: str,
        timeframe: str,
    ) -> Optional[Dict[str, Any]]:
        """Batch feature ve metadata oku."""
        try:
            key = self._make_batch_key(symbol, timeframe)
            return await cache.get_json(key)
        except Exception as e:
            logger.error(f"Error getting batch with meta for {symbol}: {e}")
            return None
    
    # =========================================================================
    # MULTI-SYMBOL OPERATIONS
    # =========================================================================
    
    async def get_features_for_symbols(
        self,
        symbols: List[str],
        timeframe: str,
        feature_names: List[str],
    ) -> Dict[str, Dict[str, float]]:
        """
        Birden fazla sembol için feature'ları oku.
        
        Args:
            symbols: Sembol listesi
            timeframe: Zaman dilimi
            feature_names: Feature adları
            
        Returns:
            Dict[symbol, Dict[feature, value]]
        """
        result = {}
        
        tasks = []
        for symbol in symbols:
            tasks.append(self.get_batch(symbol, timeframe))
        
        batches = await asyncio.gather(*tasks, return_exceptions=True)
        
        for i, batch in enumerate(batches):
            symbol = symbols[i]
            if isinstance(batch, dict):
                result[symbol] = {
                    name: batch.get(name)
                    for name in feature_names
                    if name in batch
                }
            else:
                result[symbol] = {}
        
        return result
    
    async def set_features_for_symbols(
        self,
        data: Dict[str, Dict[str, float]],
        timeframe: str,
        timestamp: Optional[datetime] = None,
    ) -> int:
        """
        Birden fazla sembol için feature'ları kaydet.
        
        Args:
            data: {symbol: {feature: value}}
            timeframe: Zaman dilimi
            timestamp: Zaman damgası
            
        Returns:
            int: Başarılı kayıt sayısı
        """
        ts = timestamp or datetime.utcnow()
        
        tasks = []
        for symbol, features in data.items():
            tasks.append(self.set_batch(symbol, timeframe, features, ts))
        
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        return sum(1 for r in results if r is True)
    
    # =========================================================================
    # HISTORY OPERATIONS
    # =========================================================================
    
    async def _add_to_history(
        self,
        symbol: str,
        timeframe: str,
        feature_name: str,
        value: float,
        timestamp: Optional[datetime] = None,
    ) -> None:
        """Feature geçmişine ekle."""
        try:
            key = self._make_history_key(symbol, timeframe, feature_name)
            ts = timestamp or datetime.utcnow()
            
            entry = {
                "value": value,
                "timestamp": ts.isoformat(),
            }
            
            # List'e ekle (LPUSH)
            await cache.lpush(key, json.dumps(entry))
            
            # Boyutu sınırla (LTRIM)
            await cache.ltrim(key, 0, self.max_history_size - 1)
            
            # TTL ayarla
            await cache.expire(key, self.HISTORY_TTL)
            
        except Exception as e:
            logger.warning(f"Error adding to history: {e}")
    
    async def get_feature_history(
        self,
        symbol: str,
        timeframe: str,
        feature_name: str,
        limit: int = 100,
    ) -> List[Dict[str, Any]]:
        """
        Feature geçmişini oku.
        
        Args:
            symbol: Hisse sembolü
            timeframe: Zaman dilimi
            feature_name: Feature adı
            limit: Maksimum kayıt sayısı
            
        Returns:
            List[Dict]: Geçmiş kayıtları (yeniden eskiye)
        """
        try:
            key = self._make_history_key(symbol, timeframe, feature_name)
            entries = await cache.lrange(key, 0, limit - 1)
            
            return [json.loads(entry) for entry in entries]
            
        except Exception as e:
            logger.error(f"Error getting history for {feature_name}: {e}")
            return []
    
    # =========================================================================
    # UTILITY METHODS
    # =========================================================================
    
    async def exists(
        self,
        symbol: str,
        timeframe: str,
        feature_name: Optional[str] = None,
    ) -> bool:
        """Feature veya batch var mı?"""
        if feature_name:
            key = self._make_key(symbol, timeframe, feature_name)
        else:
            key = self._make_batch_key(symbol, timeframe)
        
        return await cache.exists(key)
    
    async def get_age(
        self,
        symbol: str,
        timeframe: str,
        feature_name: Optional[str] = None,
    ) -> Optional[int]:
        """Feature yaşını saniye olarak döndür."""
        if feature_name:
            key = self._make_key(symbol, timeframe, feature_name)
        else:
            key = self._make_batch_key(symbol, timeframe)
        
        data = await cache.get_json(key)
        if not data:
            return None
        
        ts = datetime.fromisoformat(data["timestamp"])
        return int((datetime.utcnow() - ts).total_seconds())
    
    async def is_stale(
        self,
        symbol: str,
        timeframe: str,
        max_age_seconds: int = 300,
        feature_name: Optional[str] = None,
    ) -> bool:
        """Feature stale mı (eski)?"""
        age = await self.get_age(symbol, timeframe, feature_name)
        if age is None:
            return True
        return age > max_age_seconds
    
    async def delete_all(
        self,
        symbol: str,
        timeframe: str,
    ) -> int:
        """Sembol için tüm feature'ları sil."""
        pattern = f"{self.PREFIX}:{symbol}:{timeframe}:*"
        return await cache.delete_pattern(pattern)
    
    async def list_features(
        self,
        symbol: str,
        timeframe: str,
    ) -> List[str]:
        """Mevcut feature listesi."""
        pattern = f"{self.PREFIX}:{symbol}:{timeframe}:*"
        keys = await cache.keys(pattern)
        
        # Key'lerden feature adlarını çıkar
        features = []
        batch_key = self._make_batch_key(symbol, timeframe)
        
        for key in keys:
            if key == batch_key or key.endswith(":history"):
                continue
            
            parts = key.split(":")
            if len(parts) >= 4:
                features.append(parts[-1])
        
        return features
    
    async def get_stats(self) -> Dict[str, Any]:
        """Store istatistikleri."""
        try:
            all_keys = await cache.keys(f"{self.PREFIX}:*")
            
            symbols = set()
            timeframes = set()
            feature_count = 0
            batch_count = 0
            
            for key in all_keys:
                parts = key.split(":")
                if len(parts) >= 3:
                    symbols.add(parts[1])
                    timeframes.add(parts[2])
                
                if key.endswith(":batch"):
                    batch_count += 1
                elif not key.endswith(":history"):
                    feature_count += 1
            
            return {
                "total_keys": len(all_keys),
                "symbols": len(symbols),
                "timeframes": len(timeframes),
                "feature_entries": feature_count,
                "batch_entries": batch_count,
            }
            
        except Exception as e:
            logger.error(f"Error getting stats: {e}")
            return {}
    
    # =========================================================================
    # KEY BUILDERS
    # =========================================================================
    
    def _make_key(self, symbol: str, timeframe: str, feature_name: str) -> str:
        """Tekil feature key."""
        return f"{self.PREFIX}:{symbol}:{timeframe}:{feature_name}"
    
    def _make_batch_key(self, symbol: str, timeframe: str) -> str:
        """Batch key."""
        return f"{self.PREFIX}:{symbol}:{timeframe}:{self.BATCH_SUFFIX}"
    
    def _make_history_key(self, symbol: str, timeframe: str, feature_name: str) -> str:
        """History key."""
        return f"{self.PREFIX}:{symbol}:{timeframe}:{feature_name}:{self.HISTORY_SUFFIX}"


# =============================================================================
# SINGLETON INSTANCE
# =============================================================================

feature_store = FeatureStore()


# =============================================================================
# CONVENIENCE FUNCTIONS
# =============================================================================

async def store_features(
    symbol: str,
    timeframe: str,
    features: Dict[str, float],
    timestamp: Optional[datetime] = None,
) -> bool:
    """Hızlı feature kaydetme."""
    return await feature_store.set_batch(symbol, timeframe, features, timestamp)


async def load_features(
    symbol: str,
    timeframe: str,
) -> Optional[Dict[str, float]]:
    """Hızlı feature okuma."""
    return await feature_store.get_batch(symbol, timeframe)
