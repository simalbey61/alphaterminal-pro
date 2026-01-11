"""
AlphaTerminal Pro - Stock Repository
====================================

Hisse senetleri için özelleştirilmiş repository.

Author: AlphaTerminal Team
Version: 1.0.0
"""

from __future__ import annotations

import logging
from decimal import Decimal
from typing import List, Optional, Sequence, Dict, Any
from datetime import datetime, timedelta

from sqlalchemy import select, func, and_, or_, desc, asc
from sqlalchemy.ext.asyncio import AsyncSession

from app.db.repositories.base import BaseRepository
from app.db.models import StockModel

logger = logging.getLogger(__name__)


class StockRepository(BaseRepository[StockModel]):
    """
    Hisse senetleri repository'si.
    
    Hisse senetleri için özelleştirilmiş sorgular ve
    işlemler sağlar.
    
    Example:
        ```python
        repo = StockRepository(session)
        
        # Sembol ile bul
        stock = await repo.find_by_symbol("THYAO")
        
        # Sektör hisselerini getir
        stocks = await repo.find_by_sector("BANKA")
        
        # En çok yükselenler
        movers = await repo.get_top_gainers(limit=10)
        ```
    """
    
    def __init__(self, session: AsyncSession):
        """Repository'yi initialize et."""
        super().__init__(StockModel, session)
    
    # =========================================================================
    # FIND BY SYMBOL
    # =========================================================================
    
    async def find_by_symbol(self, symbol: str) -> Optional[StockModel]:
        """
        Sembol ile hisse bul.
        
        Args:
            symbol: Hisse sembolü (örn: "THYAO" veya "THYAO.IS")
            
        Returns:
            Optional[StockModel]: Hisse modeli veya None
        """
        # .IS suffix'ini temizle
        clean_symbol = symbol.upper().replace(".IS", "")
        
        query = select(self.model).where(
            or_(
                self.model.symbol == clean_symbol,
                self.model.yahoo_symbol == f"{clean_symbol}.IS"
            )
        )
        
        result = await self.session.execute(query)
        return result.scalar_one_or_none()
    
    async def find_by_symbols(self, symbols: List[str]) -> Sequence[StockModel]:
        """
        Birden fazla sembol ile hisse bul.
        
        Args:
            symbols: Sembol listesi
            
        Returns:
            Sequence[StockModel]: Hisse modelleri
        """
        if not symbols:
            return []
        
        # Sembolleri temizle
        clean_symbols = [s.upper().replace(".IS", "") for s in symbols]
        
        query = select(self.model).where(
            self.model.symbol.in_(clean_symbols)
        )
        
        result = await self.session.execute(query)
        return result.scalars().all()
    
    # =========================================================================
    # SECTOR QUERIES
    # =========================================================================
    
    async def find_by_sector(
        self,
        sector: str,
        active_only: bool = True
    ) -> Sequence[StockModel]:
        """
        Sektöre göre hisseleri getir.
        
        Args:
            sector: Sektör kodu
            active_only: Sadece aktif hisseler
            
        Returns:
            Sequence[StockModel]: Sektör hisseleri
        """
        query = select(self.model).where(
            self.model.sector == sector.upper()
        )
        
        if active_only:
            query = query.where(self.model.is_active == True)
        
        query = query.order_by(self.model.symbol)
        
        result = await self.session.execute(query)
        return result.scalars().all()
    
    async def get_sector_summary(self) -> List[Dict[str, Any]]:
        """
        Sektör özetini getir.
        
        Returns:
            List[Dict]: Her sektör için özet bilgiler
        """
        query = select(
            self.model.sector,
            func.count(self.model.id).label("stock_count"),
            func.avg(self.model.day_change_pct).label("avg_change"),
            func.sum(self.model.market_cap).label("total_market_cap")
        ).where(
            and_(
                self.model.is_active == True,
                self.model.sector.isnot(None)
            )
        ).group_by(
            self.model.sector
        ).order_by(
            desc("avg_change")
        )
        
        result = await self.session.execute(query)
        rows = result.all()
        
        return [
            {
                "sector": row.sector,
                "stock_count": row.stock_count,
                "avg_change": float(row.avg_change) if row.avg_change else 0,
                "total_market_cap": float(row.total_market_cap) if row.total_market_cap else 0,
            }
            for row in rows
        ]
    
    # =========================================================================
    # MOVERS
    # =========================================================================
    
    async def get_top_gainers(
        self,
        limit: int = 10,
        min_volume: Optional[int] = None
    ) -> Sequence[StockModel]:
        """
        En çok yükselen hisseleri getir.
        
        Args:
            limit: Maksimum kayıt sayısı
            min_volume: Minimum hacim filtresi
            
        Returns:
            Sequence[StockModel]: Yükselen hisseler
        """
        query = select(self.model).where(
            and_(
                self.model.is_active == True,
                self.model.day_change_pct.isnot(None),
                self.model.day_change_pct > 0
            )
        )
        
        if min_volume:
            query = query.where(self.model.last_volume >= min_volume)
        
        query = query.order_by(desc(self.model.day_change_pct)).limit(limit)
        
        result = await self.session.execute(query)
        return result.scalars().all()
    
    async def get_top_losers(
        self,
        limit: int = 10,
        min_volume: Optional[int] = None
    ) -> Sequence[StockModel]:
        """
        En çok düşen hisseleri getir.
        
        Args:
            limit: Maksimum kayıt sayısı
            min_volume: Minimum hacim filtresi
            
        Returns:
            Sequence[StockModel]: Düşen hisseler
        """
        query = select(self.model).where(
            and_(
                self.model.is_active == True,
                self.model.day_change_pct.isnot(None),
                self.model.day_change_pct < 0
            )
        )
        
        if min_volume:
            query = query.where(self.model.last_volume >= min_volume)
        
        query = query.order_by(asc(self.model.day_change_pct)).limit(limit)
        
        result = await self.session.execute(query)
        return result.scalars().all()
    
    async def get_most_active(
        self,
        limit: int = 10
    ) -> Sequence[StockModel]:
        """
        En yüksek hacimli hisseleri getir.
        
        Args:
            limit: Maksimum kayıt sayısı
            
        Returns:
            Sequence[StockModel]: Yüksek hacimli hisseler
        """
        query = select(self.model).where(
            and_(
                self.model.is_active == True,
                self.model.last_volume.isnot(None)
            )
        ).order_by(
            desc(self.model.last_volume)
        ).limit(limit)
        
        result = await self.session.execute(query)
        return result.scalars().all()
    
    # =========================================================================
    # MARKET CAP
    # =========================================================================
    
    async def get_by_market_cap(
        self,
        min_cap: Optional[Decimal] = None,
        max_cap: Optional[Decimal] = None,
        limit: int = 50
    ) -> Sequence[StockModel]:
        """
        Piyasa değerine göre hisseleri getir.
        
        Args:
            min_cap: Minimum piyasa değeri
            max_cap: Maksimum piyasa değeri
            limit: Maksimum kayıt sayısı
            
        Returns:
            Sequence[StockModel]: Hisseler
        """
        conditions = [
            self.model.is_active == True,
            self.model.market_cap.isnot(None)
        ]
        
        if min_cap:
            conditions.append(self.model.market_cap >= min_cap)
        if max_cap:
            conditions.append(self.model.market_cap <= max_cap)
        
        query = select(self.model).where(
            and_(*conditions)
        ).order_by(
            desc(self.model.market_cap)
        ).limit(limit)
        
        result = await self.session.execute(query)
        return result.scalars().all()
    
    async def get_large_caps(self, limit: int = 30) -> Sequence[StockModel]:
        """
        Büyük piyasa değerli hisseleri getir (BIST30 benzeri).
        
        Args:
            limit: Maksimum kayıt sayısı
            
        Returns:
            Sequence[StockModel]: Büyük hisseler
        """
        return await self.get_by_market_cap(
            min_cap=Decimal("5000000000"),  # 5 Milyar TL
            limit=limit
        )
    
    # =========================================================================
    # SEARCH
    # =========================================================================
    
    async def search_stocks(
        self,
        term: str,
        limit: int = 20
    ) -> Sequence[StockModel]:
        """
        Hisse ara (sembol veya isim).
        
        Args:
            term: Arama terimi
            limit: Maksimum kayıt sayısı
            
        Returns:
            Sequence[StockModel]: Bulunan hisseler
        """
        term_upper = term.upper()
        
        query = select(self.model).where(
            and_(
                self.model.is_active == True,
                or_(
                    self.model.symbol.ilike(f"%{term_upper}%"),
                    self.model.name.ilike(f"%{term}%")
                )
            )
        ).order_by(
            # Tam eşleşme önce
            self.model.symbol == term_upper,
            self.model.symbol
        ).limit(limit)
        
        result = await self.session.execute(query)
        return result.scalars().all()
    
    # =========================================================================
    # STATISTICS
    # =========================================================================
    
    async def get_market_statistics(self) -> Dict[str, Any]:
        """
        Piyasa istatistiklerini getir.
        
        Returns:
            Dict: Piyasa istatistikleri
        """
        # Aktif hisse sayısı
        total_count = await self.count(is_active=True)
        
        # Yükselen/düşen sayısı
        gainers_query = select(func.count()).select_from(self.model).where(
            and_(
                self.model.is_active == True,
                self.model.day_change_pct > 0
            )
        )
        gainers_result = await self.session.execute(gainers_query)
        gainers_count = gainers_result.scalar_one()
        
        losers_query = select(func.count()).select_from(self.model).where(
            and_(
                self.model.is_active == True,
                self.model.day_change_pct < 0
            )
        )
        losers_result = await self.session.execute(losers_query)
        losers_count = losers_result.scalar_one()
        
        # Ortalama değişim
        avg_change = await self.avg("day_change_pct", is_active=True)
        
        # Toplam hacim
        total_volume = await self.sum("last_volume", is_active=True)
        
        # Toplam piyasa değeri
        total_market_cap = await self.sum("market_cap", is_active=True)
        
        return {
            "total_stocks": total_count,
            "gainers": gainers_count,
            "losers": losers_count,
            "unchanged": total_count - gainers_count - losers_count,
            "average_change": float(avg_change) if avg_change else 0,
            "total_volume": int(total_volume) if total_volume else 0,
            "total_market_cap": float(total_market_cap) if total_market_cap else 0,
            "breadth": gainers_count / total_count if total_count > 0 else 0,
        }
    
    # =========================================================================
    # PRICE UPDATES
    # =========================================================================
    
    async def bulk_update_prices(
        self,
        updates: List[Dict[str, Any]]
    ) -> int:
        """
        Toplu fiyat güncelleme.
        
        Args:
            updates: Güncelleme listesi
                     [{"symbol": "THYAO", "last_price": 100.5, "day_change_pct": 2.5}, ...]
            
        Returns:
            int: Güncellenen kayıt sayısı
        """
        updated = 0
        
        for update_data in updates:
            symbol = update_data.pop("symbol", None)
            if not symbol:
                continue
            
            stock = await self.find_by_symbol(symbol)
            if stock:
                update_data["last_updated_at"] = datetime.utcnow()
                
                for key, value in update_data.items():
                    if hasattr(stock, key):
                        setattr(stock, key, value)
                
                updated += 1
        
        await self.session.flush()
        logger.info(f"Bulk updated {updated} stock prices")
        
        return updated
    
    async def get_stale_stocks(
        self,
        max_age_minutes: int = 60
    ) -> Sequence[StockModel]:
        """
        Eski fiyatlı hisseleri getir.
        
        Args:
            max_age_minutes: Maksimum yaş (dakika)
            
        Returns:
            Sequence[StockModel]: Eski hisseler
        """
        cutoff = datetime.utcnow() - timedelta(minutes=max_age_minutes)
        
        query = select(self.model).where(
            and_(
                self.model.is_active == True,
                or_(
                    self.model.last_updated_at.is_(None),
                    self.model.last_updated_at < cutoff
                )
            )
        )
        
        result = await self.session.execute(query)
        return result.scalars().all()
    
    # =========================================================================
    # RSI / TECHNICAL FILTERS
    # =========================================================================
    
    async def get_oversold_stocks(
        self,
        rsi_threshold: float = 30,
        limit: int = 20
    ) -> Sequence[StockModel]:
        """
        Aşırı satılmış hisseleri getir (düşük RSI).
        
        Args:
            rsi_threshold: RSI eşiği
            limit: Maksimum kayıt sayısı
            
        Returns:
            Sequence[StockModel]: Aşırı satılmış hisseler
        """
        query = select(self.model).where(
            and_(
                self.model.is_active == True,
                self.model.rsi.isnot(None),
                self.model.rsi <= rsi_threshold
            )
        ).order_by(
            asc(self.model.rsi)
        ).limit(limit)
        
        result = await self.session.execute(query)
        return result.scalars().all()
    
    async def get_overbought_stocks(
        self,
        rsi_threshold: float = 70,
        limit: int = 20
    ) -> Sequence[StockModel]:
        """
        Aşırı alınmış hisseleri getir (yüksek RSI).
        
        Args:
            rsi_threshold: RSI eşiği
            limit: Maksimum kayıt sayısı
            
        Returns:
            Sequence[StockModel]: Aşırı alınmış hisseler
        """
        query = select(self.model).where(
            and_(
                self.model.is_active == True,
                self.model.rsi.isnot(None),
                self.model.rsi >= rsi_threshold
            )
        ).order_by(
            desc(self.model.rsi)
        ).limit(limit)
        
        result = await self.session.execute(query)
        return result.scalars().all()
