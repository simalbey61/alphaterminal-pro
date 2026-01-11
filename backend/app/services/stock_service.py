"""
AlphaTerminal Pro - Stock Service
=================================

Hisse senedi business logic servisi.

Author: AlphaTerminal Team
Version: 1.0.0
"""

from __future__ import annotations

import logging
from typing import Optional, List, Dict, Any
from decimal import Decimal
from datetime import datetime, timedelta

from sqlalchemy.ext.asyncio import AsyncSession

from app.config import settings, SECTORS, SECTOR_META
from app.db.models import StockModel
from app.db.repositories import StockRepository
from app.cache import cache, CacheKeys, CacheTTL

logger = logging.getLogger(__name__)


class StockService:
    """
    Hisse senedi servisi.
    
    Hisse ile ilgili tüm business logic bu serviste toplanır.
    Repository pattern ile veritabanı erişiminden soyutlanmıştır.
    
    Example:
        ```python
        service = StockService(session)
        
        # Hisse bilgisi al
        stock = await service.get_stock("THYAO")
        
        # Sektör analizi
        analysis = await service.analyze_sector("BANK")
        
        # Fiyat güncelleme
        await service.update_prices(price_data)
        ```
    """
    
    def __init__(self, session: AsyncSession):
        """
        Initialize stock service.
        
        Args:
            session: Database session
        """
        self.session = session
        self.repo = StockRepository(session)
    
    # =========================================================================
    # READ OPERATIONS
    # =========================================================================
    
    async def get_stock(self, symbol: str) -> Optional[StockModel]:
        """
        Hisse bilgisi al.
        
        Args:
            symbol: Hisse sembolü
            
        Returns:
            Optional[StockModel]: Hisse veya None
        """
        return await self.repo.find_by_symbol(symbol)
    
    async def get_stock_or_raise(self, symbol: str) -> StockModel:
        """
        Hisse bilgisi al veya hata fırlat.
        
        Args:
            symbol: Hisse sembolü
            
        Returns:
            StockModel: Hisse
            
        Raises:
            ValueError: Hisse bulunamadı
        """
        stock = await self.repo.find_by_symbol(symbol)
        if not stock:
            raise ValueError(f"Stock not found: {symbol}")
        return stock
    
    async def get_stocks_by_symbols(self, symbols: List[str]) -> List[StockModel]:
        """
        Birden fazla hisse al.
        
        Args:
            symbols: Sembol listesi
            
        Returns:
            List[StockModel]: Hisse listesi
        """
        return await self.repo.find_by_symbols(symbols)
    
    async def get_sector_stocks(self, sector: str) -> List[StockModel]:
        """
        Sektördeki hisseleri al.
        
        Args:
            sector: Sektör kodu
            
        Returns:
            List[StockModel]: Sektör hisseleri
        """
        return await self.repo.find_by_sector(sector.upper())
    
    async def search(
        self,
        query: str,
        limit: int = 10
    ) -> List[StockModel]:
        """
        Hisse ara.
        
        Sembol ve isimde arama yapar.
        
        Args:
            query: Arama terimi
            limit: Maksimum sonuç
            
        Returns:
            List[StockModel]: Bulunan hisseler
        """
        return await self.repo.search_stocks(query, limit=limit)
    
    # =========================================================================
    # MARKET DATA
    # =========================================================================
    
    async def get_market_overview(self) -> Dict[str, Any]:
        """
        Piyasa genel görünümü al.
        
        Returns:
            dict: Piyasa özeti
        """
        # Cache kontrol
        cache_key = CacheKeys.market_overview()
        cached = await cache.get_json(cache_key)
        if cached:
            return cached
        
        # İstatistikler
        stats = await self.repo.get_market_statistics()
        
        # Top movers
        gainers = await self.repo.get_top_gainers(limit=5)
        losers = await self.repo.get_top_losers(limit=5)
        active = await self.repo.get_most_active(limit=5)
        
        # Sektör performansları
        sectors = await self.repo.get_sector_summary()
        
        result = {
            "statistics": stats,
            "top_gainers": [self._stock_to_mover(s) for s in gainers],
            "top_losers": [self._stock_to_mover(s) for s in losers],
            "most_active": [self._stock_to_mover(s) for s in active],
            "sectors": sectors,
            "timestamp": datetime.utcnow().isoformat(),
        }
        
        # Cache'e kaydet
        await cache.set_json(cache_key, result, ttl=CacheTTL.SHORT)
        
        return result
    
    async def get_sector_analysis(self, sector: str) -> Dict[str, Any]:
        """
        Sektör analizi.
        
        Args:
            sector: Sektör kodu
            
        Returns:
            dict: Sektör analiz sonuçları
        """
        sector = sector.upper()
        
        if sector not in SECTOR_META:
            raise ValueError(f"Invalid sector: {sector}")
        
        meta = SECTOR_META[sector]
        stocks = await self.repo.find_by_sector(sector)
        
        if not stocks:
            return {
                "sector": sector,
                "name": meta["name"],
                "stocks": [],
                "statistics": {},
            }
        
        # İstatistikler hesapla
        total_market_cap = sum(float(s.market_cap or 0) for s in stocks)
        avg_change = sum(float(s.day_change_pct or 0) for s in stocks) / len(stocks)
        gainers = sum(1 for s in stocks if s.day_change_pct and s.day_change_pct > 0)
        losers = sum(1 for s in stocks if s.day_change_pct and s.day_change_pct < 0)
        
        # RSI dağılımı
        oversold = sum(1 for s in stocks if s.rsi and s.rsi < 30)
        overbought = sum(1 for s in stocks if s.rsi and s.rsi > 70)
        
        return {
            "sector": sector,
            "name": meta["name"],
            "emoji": meta["emoji"],
            "color": meta["color"],
            "stocks": [self._stock_to_summary(s) for s in stocks],
            "statistics": {
                "total_stocks": len(stocks),
                "total_market_cap": total_market_cap,
                "avg_change": round(avg_change, 2),
                "gainers": gainers,
                "losers": losers,
                "unchanged": len(stocks) - gainers - losers,
                "oversold_count": oversold,
                "overbought_count": overbought,
            },
        }
    
    # =========================================================================
    # TECHNICAL INDICATORS
    # =========================================================================
    
    async def get_oversold_stocks(
        self,
        threshold: float = 30.0,
        limit: int = 20
    ) -> List[StockModel]:
        """
        Aşırı satım bölgesindeki hisseleri al.
        
        Args:
            threshold: RSI threshold
            limit: Maksimum sonuç
            
        Returns:
            List[StockModel]: Oversold hisseler
        """
        return await self.repo.get_oversold_stocks(threshold=threshold, limit=limit)
    
    async def get_overbought_stocks(
        self,
        threshold: float = 70.0,
        limit: int = 20
    ) -> List[StockModel]:
        """
        Aşırı alım bölgesindeki hisseleri al.
        
        Args:
            threshold: RSI threshold
            limit: Maksimum sonuç
            
        Returns:
            List[StockModel]: Overbought hisseler
        """
        return await self.repo.get_overbought_stocks(threshold=threshold, limit=limit)
    
    async def get_large_caps(self, limit: int = 30) -> List[StockModel]:
        """
        Büyük piyasa değerli hisseleri al.
        
        Args:
            limit: Maksimum sonuç
            
        Returns:
            List[StockModel]: Large cap hisseler
        """
        return await self.repo.get_large_caps(limit=limit)
    
    # =========================================================================
    # PRICE OPERATIONS
    # =========================================================================
    
    async def update_prices(
        self,
        price_data: List[Dict[str, Any]]
    ) -> int:
        """
        Toplu fiyat güncelleme.
        
        Args:
            price_data: Fiyat verileri [{"symbol": "THYAO", "price": 150.5, ...}]
            
        Returns:
            int: Güncellenen hisse sayısı
        """
        count = await self.repo.bulk_update_prices(price_data)
        
        # Cache'i temizle
        await cache.delete_pattern(f"{CacheKeys.PREFIX_MARKET}:*")
        
        logger.info(f"Updated prices for {count} stocks")
        return count
    
    async def get_stale_stocks(
        self,
        hours: int = 24
    ) -> List[StockModel]:
        """
        Eski fiyatlı hisseleri al.
        
        Args:
            hours: Kaç saat önce güncellenmiş olmalı
            
        Returns:
            List[StockModel]: Stale hisseler
        """
        return await self.repo.get_stale_stocks(hours=hours)
    
    # =========================================================================
    # HELPERS
    # =========================================================================
    
    def _stock_to_mover(self, stock: StockModel) -> Dict[str, Any]:
        """Stock'u mover dict'e çevir."""
        return {
            "symbol": stock.symbol,
            "name": stock.name,
            "last_price": float(stock.last_price) if stock.last_price else 0,
            "day_change_pct": float(stock.day_change_pct) if stock.day_change_pct else 0,
            "volume": stock.last_volume,
        }
    
    def _stock_to_summary(self, stock: StockModel) -> Dict[str, Any]:
        """Stock'u summary dict'e çevir."""
        return {
            "symbol": stock.symbol,
            "name": stock.name,
            "sector": stock.sector,
            "last_price": float(stock.last_price) if stock.last_price else None,
            "day_change_pct": float(stock.day_change_pct) if stock.day_change_pct else None,
            "volume": stock.last_volume,
            "rsi": float(stock.rsi) if stock.rsi else None,
            "market_cap": float(stock.market_cap) if stock.market_cap else None,
        }
