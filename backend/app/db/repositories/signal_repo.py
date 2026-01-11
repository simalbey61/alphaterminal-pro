"""
AlphaTerminal Pro - Signal Repository
=====================================

Trading sinyalleri için özelleştirilmiş repository.

Author: AlphaTerminal Team
Version: 1.0.0
"""

from __future__ import annotations

import uuid
import logging
from decimal import Decimal
from typing import List, Optional, Sequence, Dict, Any
from datetime import datetime, timedelta, date

from sqlalchemy import select, func, and_, or_, desc, asc, case
from sqlalchemy.ext.asyncio import AsyncSession

from app.db.repositories.base import BaseRepository
from app.db.models import SignalModel
from app.config import SignalStatus, SignalTier, SignalType

logger = logging.getLogger(__name__)


class SignalRepository(BaseRepository[SignalModel]):
    """
    Trading sinyalleri repository'si.
    
    Sinyaller için özelleştirilmiş sorgular ve işlemler sağlar.
    
    Example:
        ```python
        repo = SignalRepository(session)
        
        # Aktif sinyalleri getir
        signals = await repo.get_active_signals()
        
        # Hisse sinyallerini getir
        signals = await repo.find_by_symbol("THYAO")
        
        # Sinyal performansını al
        stats = await repo.get_performance_stats(strategy_id)
        ```
    """
    
    def __init__(self, session: AsyncSession):
        """Repository'yi initialize et."""
        super().__init__(SignalModel, session)
    
    # =========================================================================
    # ACTIVE SIGNALS
    # =========================================================================
    
    async def get_active_signals(
        self,
        tier: Optional[SignalTier] = None,
        signal_type: Optional[SignalType] = None,
        limit: int = 50
    ) -> Sequence[SignalModel]:
        """
        Aktif sinyalleri getir.
        
        Args:
            tier: Sinyal tier filtresi
            signal_type: Sinyal tipi filtresi
            limit: Maksimum kayıt sayısı
            
        Returns:
            Sequence[SignalModel]: Aktif sinyaller
        """
        conditions = [self.model.status == SignalStatus.ACTIVE.value]
        
        if tier:
            conditions.append(self.model.tier == tier.value)
        
        if signal_type:
            conditions.append(self.model.signal_type == signal_type.value)
        
        query = select(self.model).where(
            and_(*conditions)
        ).order_by(
            desc(self.model.total_score),
            desc(self.model.created_at)
        ).limit(limit)
        
        result = await self.session.execute(query)
        return result.scalars().all()
    
    async def get_active_by_symbol(self, symbol: str) -> Sequence[SignalModel]:
        """
        Hisse için aktif sinyalleri getir.
        
        Args:
            symbol: Hisse sembolü
            
        Returns:
            Sequence[SignalModel]: Aktif sinyaller
        """
        clean_symbol = symbol.upper().replace(".IS", "")
        
        query = select(self.model).where(
            and_(
                self.model.symbol == clean_symbol,
                self.model.status == SignalStatus.ACTIVE.value
            )
        ).order_by(desc(self.model.created_at))
        
        result = await self.session.execute(query)
        return result.scalars().all()
    
    # =========================================================================
    # FIND BY FILTERS
    # =========================================================================
    
    async def find_by_symbol(
        self,
        symbol: str,
        include_closed: bool = False,
        limit: int = 20
    ) -> Sequence[SignalModel]:
        """
        Hisse sinyallerini getir.
        
        Args:
            symbol: Hisse sembolü
            include_closed: Kapalı sinyalleri dahil et
            limit: Maksimum kayıt sayısı
            
        Returns:
            Sequence[SignalModel]: Sinyaller
        """
        clean_symbol = symbol.upper().replace(".IS", "")
        
        conditions = [self.model.symbol == clean_symbol]
        
        if not include_closed:
            conditions.append(self.model.status == SignalStatus.ACTIVE.value)
        
        query = select(self.model).where(
            and_(*conditions)
        ).order_by(
            desc(self.model.created_at)
        ).limit(limit)
        
        result = await self.session.execute(query)
        return result.scalars().all()
    
    async def find_by_strategy(
        self,
        strategy_id: uuid.UUID,
        status: Optional[SignalStatus] = None,
        limit: int = 100
    ) -> Sequence[SignalModel]:
        """
        Strateji sinyallerini getir.
        
        Args:
            strategy_id: Strateji UUID'si
            status: Durum filtresi
            limit: Maksimum kayıt sayısı
            
        Returns:
            Sequence[SignalModel]: Sinyaller
        """
        conditions = [self.model.strategy_id == strategy_id]
        
        if status:
            conditions.append(self.model.status == status.value)
        
        query = select(self.model).where(
            and_(*conditions)
        ).order_by(
            desc(self.model.created_at)
        ).limit(limit)
        
        result = await self.session.execute(query)
        return result.scalars().all()
    
    async def find_by_tier(
        self,
        tier: SignalTier,
        active_only: bool = True,
        limit: int = 50
    ) -> Sequence[SignalModel]:
        """
        Tier'a göre sinyalleri getir.
        
        Args:
            tier: Sinyal tier'ı
            active_only: Sadece aktif sinyaller
            limit: Maksimum kayıt sayısı
            
        Returns:
            Sequence[SignalModel]: Sinyaller
        """
        conditions = [self.model.tier == tier.value]
        
        if active_only:
            conditions.append(self.model.status == SignalStatus.ACTIVE.value)
        
        query = select(self.model).where(
            and_(*conditions)
        ).order_by(
            desc(self.model.total_score)
        ).limit(limit)
        
        result = await self.session.execute(query)
        return result.scalars().all()
    
    # =========================================================================
    # DATE RANGE QUERIES
    # =========================================================================
    
    async def find_by_date_range(
        self,
        start_date: datetime,
        end_date: Optional[datetime] = None,
        status: Optional[SignalStatus] = None
    ) -> Sequence[SignalModel]:
        """
        Tarih aralığında sinyalleri getir.
        
        Args:
            start_date: Başlangıç tarihi
            end_date: Bitiş tarihi
            status: Durum filtresi
            
        Returns:
            Sequence[SignalModel]: Sinyaller
        """
        conditions = [self.model.created_at >= start_date]
        
        if end_date:
            conditions.append(self.model.created_at <= end_date)
        
        if status:
            conditions.append(self.model.status == status.value)
        
        query = select(self.model).where(
            and_(*conditions)
        ).order_by(desc(self.model.created_at))
        
        result = await self.session.execute(query)
        return result.scalars().all()
    
    async def get_today_signals(
        self,
        tier: Optional[SignalTier] = None
    ) -> Sequence[SignalModel]:
        """
        Bugünün sinyallerini getir.
        
        Args:
            tier: Tier filtresi
            
        Returns:
            Sequence[SignalModel]: Bugünün sinyalleri
        """
        today_start = datetime.combine(date.today(), datetime.min.time())
        
        conditions = [self.model.created_at >= today_start]
        
        if tier:
            conditions.append(self.model.tier == tier.value)
        
        query = select(self.model).where(
            and_(*conditions)
        ).order_by(desc(self.model.created_at))
        
        result = await self.session.execute(query)
        return result.scalars().all()
    
    # =========================================================================
    # PERFORMANCE STATS
    # =========================================================================
    
    async def get_performance_stats(
        self,
        strategy_id: Optional[uuid.UUID] = None,
        days: int = 30
    ) -> Dict[str, Any]:
        """
        Sinyal performans istatistiklerini getir.
        
        Args:
            strategy_id: Strateji filtresi
            days: Gün sayısı
            
        Returns:
            Dict: Performans istatistikleri
        """
        cutoff = datetime.utcnow() - timedelta(days=days)
        
        conditions = [self.model.created_at >= cutoff]
        
        if strategy_id:
            conditions.append(self.model.strategy_id == strategy_id)
        
        # Toplam sinyal sayısı
        total_query = select(func.count()).select_from(self.model).where(
            and_(*conditions)
        )
        total_result = await self.session.execute(total_query)
        total = total_result.scalar_one()
        
        # Kapalı sinyal sayısı
        closed_conditions = conditions + [
            self.model.status != SignalStatus.ACTIVE.value
        ]
        closed_query = select(func.count()).select_from(self.model).where(
            and_(*closed_conditions)
        )
        closed_result = await self.session.execute(closed_query)
        closed = closed_result.scalar_one()
        
        # Karlı sinyal sayısı
        profitable_conditions = conditions + [
            self.model.result_pnl.isnot(None),
            self.model.result_pnl > 0
        ]
        profitable_query = select(func.count()).select_from(self.model).where(
            and_(*profitable_conditions)
        )
        profitable_result = await self.session.execute(profitable_query)
        profitable = profitable_result.scalar_one()
        
        # Stop'a takılan sayısı
        stopped_conditions = conditions + [
            self.model.status == SignalStatus.STOPPED.value
        ]
        stopped_query = select(func.count()).select_from(self.model).where(
            and_(*stopped_conditions)
        )
        stopped_result = await self.session.execute(stopped_query)
        stopped = stopped_result.scalar_one()
        
        # Ortalama P&L
        avg_pnl_query = select(func.avg(self.model.result_pnl_pct)).where(
            and_(*conditions, self.model.result_pnl_pct.isnot(None))
        )
        avg_pnl_result = await self.session.execute(avg_pnl_query)
        avg_pnl = avg_pnl_result.scalar_one()
        
        # Toplam P&L
        total_pnl_query = select(func.sum(self.model.result_pnl)).where(
            and_(*conditions, self.model.result_pnl.isnot(None))
        )
        total_pnl_result = await self.session.execute(total_pnl_query)
        total_pnl = total_pnl_result.scalar_one()
        
        # Win rate hesapla
        win_rate = profitable / closed if closed > 0 else 0
        
        return {
            "total_signals": total,
            "closed_signals": closed,
            "active_signals": total - closed,
            "profitable_signals": profitable,
            "stopped_signals": stopped,
            "win_rate": win_rate,
            "average_pnl_pct": float(avg_pnl) if avg_pnl else 0,
            "total_pnl": float(total_pnl) if total_pnl else 0,
            "period_days": days,
        }
    
    async def get_tier_performance(
        self,
        days: int = 30
    ) -> List[Dict[str, Any]]:
        """
        Tier bazlı performans getir.
        
        Args:
            days: Gün sayısı
            
        Returns:
            List[Dict]: Her tier için performans
        """
        cutoff = datetime.utcnow() - timedelta(days=days)
        
        query = select(
            self.model.tier,
            func.count(self.model.id).label("total"),
            func.sum(case((self.model.result_pnl > 0, 1), else_=0)).label("wins"),
            func.avg(self.model.result_pnl_pct).label("avg_pnl")
        ).where(
            and_(
                self.model.created_at >= cutoff,
                self.model.result_pnl.isnot(None)
            )
        ).group_by(self.model.tier)
        
        result = await self.session.execute(query)
        rows = result.all()
        
        return [
            {
                "tier": row.tier,
                "total": row.total,
                "wins": row.wins or 0,
                "win_rate": (row.wins or 0) / row.total if row.total > 0 else 0,
                "avg_pnl": float(row.avg_pnl) if row.avg_pnl else 0,
            }
            for row in rows
        ]
    
    # =========================================================================
    # SIGNAL MANAGEMENT
    # =========================================================================
    
    async def close_signal(
        self,
        signal_id: uuid.UUID,
        exit_price: Decimal,
        reason: str,
        pnl: Optional[Decimal] = None
    ) -> Optional[SignalModel]:
        """
        Sinyali kapat.
        
        Args:
            signal_id: Sinyal UUID'si
            exit_price: Çıkış fiyatı
            reason: Kapanış sebebi
            pnl: P&L (opsiyonel)
            
        Returns:
            Optional[SignalModel]: Kapatılan sinyal
        """
        signal = await self.get(signal_id)
        if signal is None:
            return None
        
        signal.close(exit_price, reason, pnl)
        await self.session.flush()
        await self.session.refresh(signal)
        
        logger.info(f"Closed signal {signal_id}: {reason} @ {exit_price}")
        return signal
    
    async def expire_old_signals(
        self,
        max_age_hours: int = 48
    ) -> int:
        """
        Eski sinyalleri expire et.
        
        Args:
            max_age_hours: Maksimum yaş (saat)
            
        Returns:
            int: Expire edilen sinyal sayısı
        """
        cutoff = datetime.utcnow() - timedelta(hours=max_age_hours)
        
        # Expire edilecek sinyalleri bul
        query = select(self.model).where(
            and_(
                self.model.status == SignalStatus.ACTIVE.value,
                self.model.created_at < cutoff
            )
        )
        
        result = await self.session.execute(query)
        signals = result.scalars().all()
        
        count = 0
        for signal in signals:
            signal.status = SignalStatus.EXPIRED.value
            signal.closed_at = datetime.utcnow()
            signal.closed_reason = "expired"
            count += 1
        
        await self.session.flush()
        logger.info(f"Expired {count} old signals")
        
        return count
    
    async def check_price_levels(
        self,
        symbol: str,
        current_price: Decimal
    ) -> List[Dict[str, Any]]:
        """
        Fiyat seviyelerini kontrol et.
        
        Aktif sinyallerin stop/tp seviyelerini kontrol eder.
        
        Args:
            symbol: Hisse sembolü
            current_price: Güncel fiyat
            
        Returns:
            List[Dict]: Tetiklenen seviyeler
        """
        clean_symbol = symbol.upper().replace(".IS", "")
        
        active_signals = await self.get_active_by_symbol(clean_symbol)
        triggered = []
        
        for signal in active_signals:
            if signal.signal_type == SignalType.LONG.value:
                # LONG için kontroller
                if current_price <= signal.stop_loss:
                    triggered.append({
                        "signal_id": signal.id,
                        "type": "stop_loss",
                        "price": current_price
                    })
                elif current_price >= signal.take_profit_1:
                    triggered.append({
                        "signal_id": signal.id,
                        "type": "take_profit_1",
                        "price": current_price
                    })
            else:
                # SHORT için kontroller
                if current_price >= signal.stop_loss:
                    triggered.append({
                        "signal_id": signal.id,
                        "type": "stop_loss",
                        "price": current_price
                    })
                elif current_price <= signal.take_profit_1:
                    triggered.append({
                        "signal_id": signal.id,
                        "type": "take_profit_1",
                        "price": current_price
                    })
        
        return triggered
    
    # =========================================================================
    # STATISTICS
    # =========================================================================
    
    async def get_signal_distribution(self) -> Dict[str, Any]:
        """
        Sinyal dağılımını getir.
        
        Returns:
            Dict: Sinyal dağılım istatistikleri
        """
        # Durum dağılımı
        status_query = select(
            self.model.status,
            func.count(self.model.id)
        ).group_by(self.model.status)
        
        status_result = await self.session.execute(status_query)
        status_dist = {row[0]: row[1] for row in status_result.all()}
        
        # Tier dağılımı
        tier_query = select(
            self.model.tier,
            func.count(self.model.id)
        ).group_by(self.model.tier)
        
        tier_result = await self.session.execute(tier_query)
        tier_dist = {row[0]: row[1] for row in tier_result.all()}
        
        # Type dağılımı
        type_query = select(
            self.model.signal_type,
            func.count(self.model.id)
        ).group_by(self.model.signal_type)
        
        type_result = await self.session.execute(type_query)
        type_dist = {row[0]: row[1] for row in type_result.all()}
        
        return {
            "by_status": status_dist,
            "by_tier": tier_dist,
            "by_type": type_dist,
        }
    
    async def get_top_performing_symbols(
        self,
        days: int = 30,
        limit: int = 10
    ) -> List[Dict[str, Any]]:
        """
        En iyi performans gösteren sembolleri getir.
        
        Args:
            days: Gün sayısı
            limit: Maksimum kayıt sayısı
            
        Returns:
            List[Dict]: Sembol bazlı performans
        """
        cutoff = datetime.utcnow() - timedelta(days=days)
        
        query = select(
            self.model.symbol,
            func.count(self.model.id).label("total"),
            func.sum(case((self.model.result_pnl > 0, 1), else_=0)).label("wins"),
            func.sum(self.model.result_pnl_pct).label("total_pnl")
        ).where(
            and_(
                self.model.created_at >= cutoff,
                self.model.result_pnl.isnot(None)
            )
        ).group_by(
            self.model.symbol
        ).having(
            func.count(self.model.id) >= 3  # En az 3 sinyal
        ).order_by(
            desc("total_pnl")
        ).limit(limit)
        
        result = await self.session.execute(query)
        rows = result.all()
        
        return [
            {
                "symbol": row.symbol,
                "total_signals": row.total,
                "wins": row.wins or 0,
                "win_rate": (row.wins or 0) / row.total if row.total > 0 else 0,
                "total_pnl_pct": float(row.total_pnl) if row.total_pnl else 0,
            }
            for row in rows
        ]
