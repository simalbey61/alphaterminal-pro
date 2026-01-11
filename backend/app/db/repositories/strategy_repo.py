"""
AlphaTerminal Pro - Strategy Repository
=======================================

AI stratejileri için özelleştirilmiş repository.

Author: AlphaTerminal Team
Version: 1.0.0
"""

from __future__ import annotations

import uuid
import logging
from decimal import Decimal
from typing import List, Optional, Sequence, Dict, Any
from datetime import datetime, timedelta

from sqlalchemy import select, func, and_, or_, desc, asc
from sqlalchemy.ext.asyncio import AsyncSession

from app.db.repositories.base import BaseRepository
from app.db.models import AIStrategyModel
from app.config import StrategyStatus

logger = logging.getLogger(__name__)


class StrategyRepository(BaseRepository[AIStrategyModel]):
    """
    AI stratejileri repository'si.
    
    Stratejiler için özelleştirilmiş sorgular ve işlemler sağlar.
    
    Example:
        ```python
        repo = StrategyRepository(session)
        
        # Aktif stratejileri getir
        strategies = await repo.get_active_strategies()
        
        # En iyi performanslı stratejiler
        top = await repo.get_top_performing(limit=10)
        
        # Strateji evrim geçmişi
        history = await repo.get_evolution_history(strategy_id)
        ```
    """
    
    def __init__(self, session: AsyncSession):
        """Repository'yi initialize et."""
        super().__init__(AIStrategyModel, session)
    
    # =========================================================================
    # STATUS QUERIES
    # =========================================================================
    
    async def get_active_strategies(
        self,
        discovery_method: Optional[str] = None
    ) -> Sequence[AIStrategyModel]:
        """
        Aktif stratejileri getir.
        
        Args:
            discovery_method: Keşif yöntemi filtresi
            
        Returns:
            Sequence[AIStrategyModel]: Aktif stratejiler
        """
        conditions = [self.model.status == StrategyStatus.ACTIVE.value]
        
        if discovery_method:
            conditions.append(self.model.discovery_method == discovery_method)
        
        query = select(self.model).where(
            and_(*conditions)
        ).order_by(
            desc(self.model.performance_score)
        )
        
        result = await self.session.execute(query)
        return result.scalars().all()
    
    async def get_pending_strategies(self) -> Sequence[AIStrategyModel]:
        """
        Onay bekleyen stratejileri getir.
        
        Returns:
            Sequence[AIStrategyModel]: Onay bekleyen stratejiler
        """
        query = select(self.model).where(
            self.model.status == StrategyStatus.PENDING_VALIDATION.value
        ).order_by(
            desc(self.model.confidence),
            desc(self.model.created_at)
        )
        
        result = await self.session.execute(query)
        return result.scalars().all()
    
    async def get_retired_strategies(
        self,
        days: int = 30
    ) -> Sequence[AIStrategyModel]:
        """
        Son N günde emekli edilen stratejileri getir.
        
        Args:
            days: Gün sayısı
            
        Returns:
            Sequence[AIStrategyModel]: Emekli stratejiler
        """
        cutoff = datetime.utcnow() - timedelta(days=days)
        
        query = select(self.model).where(
            and_(
                self.model.status == StrategyStatus.RETIRED.value,
                self.model.retired_at >= cutoff
            )
        ).order_by(desc(self.model.retired_at))
        
        result = await self.session.execute(query)
        return result.scalars().all()
    
    # =========================================================================
    # PERFORMANCE QUERIES
    # =========================================================================
    
    async def get_top_performing(
        self,
        limit: int = 10,
        min_trades: int = 10
    ) -> Sequence[AIStrategyModel]:
        """
        En iyi performanslı stratejileri getir.
        
        Args:
            limit: Maksimum kayıt sayısı
            min_trades: Minimum trade sayısı
            
        Returns:
            Sequence[AIStrategyModel]: En iyi stratejiler
        """
        query = select(self.model).where(
            and_(
                self.model.status == StrategyStatus.ACTIVE.value,
                self.model.live_total_trades >= min_trades
            )
        ).order_by(
            desc(self.model.performance_score)
        ).limit(limit)
        
        result = await self.session.execute(query)
        return result.scalars().all()
    
    async def get_by_win_rate(
        self,
        min_win_rate: float = 0.5,
        min_trades: int = 10
    ) -> Sequence[AIStrategyModel]:
        """
        Win rate'e göre stratejileri getir.
        
        Args:
            min_win_rate: Minimum win rate
            min_trades: Minimum trade sayısı
            
        Returns:
            Sequence[AIStrategyModel]: Stratejiler
        """
        query = select(self.model).where(
            and_(
                self.model.status == StrategyStatus.ACTIVE.value,
                self.model.live_total_trades >= min_trades,
                self.model.live_win_rate >= min_win_rate
            )
        ).order_by(
            desc(self.model.live_win_rate)
        )
        
        result = await self.session.execute(query)
        return result.scalars().all()
    
    async def get_underperforming(
        self,
        max_win_rate: float = 0.4,
        min_trades: int = 20,
        days_active: int = 30
    ) -> Sequence[AIStrategyModel]:
        """
        Düşük performanslı stratejileri getir (emeklilik adayları).
        
        Args:
            max_win_rate: Maksimum win rate
            min_trades: Minimum trade sayısı
            days_active: Minimum aktif gün
            
        Returns:
            Sequence[AIStrategyModel]: Düşük performanslı stratejiler
        """
        cutoff = datetime.utcnow() - timedelta(days=days_active)
        
        query = select(self.model).where(
            and_(
                self.model.status == StrategyStatus.ACTIVE.value,
                self.model.approved_at <= cutoff,
                self.model.live_total_trades >= min_trades,
                or_(
                    self.model.live_win_rate < max_win_rate,
                    self.model.live_win_rate.is_(None)
                )
            )
        ).order_by(
            asc(self.model.live_win_rate)
        )
        
        result = await self.session.execute(query)
        return result.scalars().all()
    
    # =========================================================================
    # DISCOVERY METHOD QUERIES
    # =========================================================================
    
    async def get_by_discovery_method(
        self,
        method: str,
        active_only: bool = True
    ) -> Sequence[AIStrategyModel]:
        """
        Keşif yöntemine göre stratejileri getir.
        
        Args:
            method: Keşif yöntemi
            active_only: Sadece aktif stratejiler
            
        Returns:
            Sequence[AIStrategyModel]: Stratejiler
        """
        conditions = [self.model.discovery_method == method]
        
        if active_only:
            conditions.append(self.model.status == StrategyStatus.ACTIVE.value)
        
        query = select(self.model).where(
            and_(*conditions)
        ).order_by(
            desc(self.model.performance_score)
        )
        
        result = await self.session.execute(query)
        return result.scalars().all()
    
    async def get_discovery_method_stats(self) -> List[Dict[str, Any]]:
        """
        Keşif yöntemi istatistiklerini getir.
        
        Returns:
            List[Dict]: Her yöntem için istatistikler
        """
        query = select(
            self.model.discovery_method,
            func.count(self.model.id).label("total"),
            func.count(self.model.id).filter(
                self.model.status == StrategyStatus.ACTIVE.value
            ).label("active"),
            func.avg(self.model.live_win_rate).label("avg_win_rate"),
            func.avg(self.model.performance_score).label("avg_score")
        ).group_by(
            self.model.discovery_method
        )
        
        result = await self.session.execute(query)
        rows = result.all()
        
        return [
            {
                "method": row.discovery_method,
                "total": row.total,
                "active": row.active,
                "avg_win_rate": float(row.avg_win_rate) if row.avg_win_rate else 0,
                "avg_score": float(row.avg_score) if row.avg_score else 0,
            }
            for row in rows
        ]
    
    # =========================================================================
    # GENETIC ALGORITHM QUERIES
    # =========================================================================
    
    async def get_by_generation(
        self,
        generation: int
    ) -> Sequence[AIStrategyModel]:
        """
        Jenerasyona göre stratejileri getir.
        
        Args:
            generation: Jenerasyon numarası
            
        Returns:
            Sequence[AIStrategyModel]: Stratejiler
        """
        query = select(self.model).where(
            self.model.generation == generation
        ).order_by(
            desc(self.model.performance_score)
        )
        
        result = await self.session.execute(query)
        return result.scalars().all()
    
    async def get_latest_generation(self) -> int:
        """
        En son jenerasyon numarasını al.
        
        Returns:
            int: Jenerasyon numarası
        """
        query = select(func.max(self.model.generation))
        result = await self.session.execute(query)
        max_gen = result.scalar_one_or_none()
        return max_gen or 1
    
    async def get_parent_strategies(
        self,
        strategy_id: uuid.UUID
    ) -> Sequence[AIStrategyModel]:
        """
        Strateji ebeveynlerini getir.
        
        Args:
            strategy_id: Strateji UUID'si
            
        Returns:
            Sequence[AIStrategyModel]: Ebeveyn stratejiler
        """
        strategy = await self.get(strategy_id)
        if not strategy or not strategy.parent_strategy_ids:
            return []
        
        query = select(self.model).where(
            self.model.id.in_(strategy.parent_strategy_ids)
        )
        
        result = await self.session.execute(query)
        return result.scalars().all()
    
    async def get_children_strategies(
        self,
        strategy_id: uuid.UUID
    ) -> Sequence[AIStrategyModel]:
        """
        Strateji çocuklarını getir.
        
        Args:
            strategy_id: Strateji UUID'si
            
        Returns:
            Sequence[AIStrategyModel]: Çocuk stratejiler
        """
        query = select(self.model).where(
            self.model.parent_strategy_ids.contains([strategy_id])
        )
        
        result = await self.session.execute(query)
        return result.scalars().all()
    
    # =========================================================================
    # STRATEGY MANAGEMENT
    # =========================================================================
    
    async def approve_strategy(
        self,
        strategy_id: uuid.UUID
    ) -> Optional[AIStrategyModel]:
        """
        Stratejiyi onayla.
        
        Args:
            strategy_id: Strateji UUID'si
            
        Returns:
            Optional[AIStrategyModel]: Onaylanan strateji
        """
        strategy = await self.get(strategy_id)
        if strategy is None:
            return None
        
        strategy.approve()
        await self.session.flush()
        await self.session.refresh(strategy)
        
        logger.info(f"Approved strategy: {strategy.name}")
        return strategy
    
    async def pause_strategy(
        self,
        strategy_id: uuid.UUID
    ) -> Optional[AIStrategyModel]:
        """
        Stratejiyi duraklat.
        
        Args:
            strategy_id: Strateji UUID'si
            
        Returns:
            Optional[AIStrategyModel]: Duraklatılan strateji
        """
        strategy = await self.get(strategy_id)
        if strategy is None:
            return None
        
        strategy.pause()
        await self.session.flush()
        await self.session.refresh(strategy)
        
        logger.info(f"Paused strategy: {strategy.name}")
        return strategy
    
    async def retire_strategy(
        self,
        strategy_id: uuid.UUID
    ) -> Optional[AIStrategyModel]:
        """
        Stratejiyi emekli et.
        
        Args:
            strategy_id: Strateji UUID'si
            
        Returns:
            Optional[AIStrategyModel]: Emekli edilen strateji
        """
        strategy = await self.get(strategy_id)
        if strategy is None:
            return None
        
        strategy.retire()
        await self.session.flush()
        await self.session.refresh(strategy)
        
        logger.info(f"Retired strategy: {strategy.name}")
        return strategy
    
    async def record_trade(
        self,
        strategy_id: uuid.UUID,
        is_win: bool,
        pnl: Decimal
    ) -> Optional[AIStrategyModel]:
        """
        Trade sonucu kaydet.
        
        Args:
            strategy_id: Strateji UUID'si
            is_win: Kazanç mı
            pnl: P&L miktarı
            
        Returns:
            Optional[AIStrategyModel]: Güncellenen strateji
        """
        strategy = await self.get(strategy_id)
        if strategy is None:
            return None
        
        strategy.record_trade_result(is_win, pnl)
        strategy.update_performance_score()
        strategy.last_signal_at = datetime.utcnow()
        
        await self.session.flush()
        await self.session.refresh(strategy)
        
        return strategy
    
    # =========================================================================
    # STATISTICS
    # =========================================================================
    
    async def get_statistics(self) -> Dict[str, Any]:
        """
        Strateji istatistiklerini getir.
        
        Returns:
            Dict: İstatistikler
        """
        # Toplam sayılar
        total = await self.count()
        active = await self.count(status=StrategyStatus.ACTIVE.value)
        pending = await self.count(status=StrategyStatus.PENDING_VALIDATION.value)
        retired = await self.count(status=StrategyStatus.RETIRED.value)
        paused = await self.count(status=StrategyStatus.PAUSED.value)
        
        # Ortalama performans (aktif stratejiler)
        avg_query = select(
            func.avg(self.model.live_win_rate).label("avg_win_rate"),
            func.avg(self.model.performance_score).label("avg_score"),
            func.sum(self.model.live_total_trades).label("total_trades"),
            func.sum(self.model.live_total_pnl).label("total_pnl")
        ).where(
            self.model.status == StrategyStatus.ACTIVE.value
        )
        
        avg_result = await self.session.execute(avg_query)
        avg_row = avg_result.one()
        
        return {
            "total": total,
            "active": active,
            "pending": pending,
            "retired": retired,
            "paused": paused,
            "avg_win_rate": float(avg_row.avg_win_rate) if avg_row.avg_win_rate else 0,
            "avg_performance_score": float(avg_row.avg_score) if avg_row.avg_score else 0,
            "total_trades": int(avg_row.total_trades) if avg_row.total_trades else 0,
            "total_pnl": float(avg_row.total_pnl) if avg_row.total_pnl else 0,
        }
    
    async def get_needs_evolution(
        self,
        check_interval_days: int = 7
    ) -> Sequence[AIStrategyModel]:
        """
        Evrim gerektiren stratejileri getir.
        
        Args:
            check_interval_days: Kontrol aralığı (gün)
            
        Returns:
            Sequence[AIStrategyModel]: Evrim gerektiren stratejiler
        """
        cutoff = datetime.utcnow() - timedelta(days=check_interval_days)
        
        query = select(self.model).where(
            and_(
                self.model.status == StrategyStatus.ACTIVE.value,
                or_(
                    self.model.last_evolution_at.is_(None),
                    self.model.last_evolution_at < cutoff
                )
            )
        )
        
        result = await self.session.execute(query)
        return result.scalars().all()
