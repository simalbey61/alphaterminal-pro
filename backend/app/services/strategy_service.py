"""
AlphaTerminal Pro - Strategy Service
====================================

AI strateji business logic servisi.

Author: AlphaTerminal Team
Version: 1.0.0
"""

from __future__ import annotations

import logging
from typing import Optional, List, Dict, Any
from decimal import Decimal
from datetime import datetime, timedelta
from uuid import UUID

from sqlalchemy.ext.asyncio import AsyncSession

from app.config import settings, StrategyStatus
from app.db.models import AIStrategyModel, EvolutionLogModel
from app.db.repositories import StrategyRepository, SignalRepository
from app.cache import cache, CacheKeys, CacheTTL

logger = logging.getLogger(__name__)


class StrategyService:
    """
    AI strateji servisi.
    
    Strateji yaşam döngüsü, performans takibi ve evrim yönetimi.
    
    Example:
        ```python
        service = StrategyService(session)
        
        # Aktif stratejileri al
        strategies = await service.get_active_strategies()
        
        # Strateji performansı
        perf = await service.get_strategy_performance(strategy_id)
        
        # Stratejiyi onayla
        await service.approve_strategy(strategy_id)
        ```
    """
    
    def __init__(self, session: AsyncSession):
        """
        Initialize strategy service.
        
        Args:
            session: Database session
        """
        self.session = session
        self.strategy_repo = StrategyRepository(session)
        self.signal_repo = SignalRepository(session)
    
    # =========================================================================
    # STRATEGY RETRIEVAL
    # =========================================================================
    
    async def get_strategy(self, strategy_id: UUID) -> Optional[AIStrategyModel]:
        """
        Strateji al.
        
        Args:
            strategy_id: Strateji ID
            
        Returns:
            Optional[AIStrategyModel]: Strateji
        """
        return await self.strategy_repo.get(strategy_id)
    
    async def get_active_strategies(
        self,
        discovery_method: Optional[str] = None
    ) -> List[AIStrategyModel]:
        """
        Aktif stratejileri al.
        
        Args:
            discovery_method: Keşif yöntemi filtresi
            
        Returns:
            List[AIStrategyModel]: Aktif stratejiler
        """
        return await self.strategy_repo.get_active_strategies(
            discovery_method=discovery_method
        )
    
    async def get_pending_strategies(self) -> List[AIStrategyModel]:
        """
        Onay bekleyen stratejileri al.
        
        Returns:
            List[AIStrategyModel]: Pending stratejiler
        """
        return await self.strategy_repo.get_pending_strategies()
    
    async def get_top_performers(
        self,
        limit: int = 10,
        min_trades: int = 10
    ) -> List[AIStrategyModel]:
        """
        En iyi performanslı stratejileri al.
        
        Args:
            limit: Maksimum sonuç
            min_trades: Minimum trade sayısı
            
        Returns:
            List[AIStrategyModel]: Top stratejiler
        """
        return await self.strategy_repo.get_top_performing(
            limit=limit,
            min_trades=min_trades
        )
    
    async def get_underperformers(
        self,
        max_win_rate: float = 0.4,
        min_trades: int = 20
    ) -> List[AIStrategyModel]:
        """
        Düşük performanslı stratejileri al.
        
        Args:
            max_win_rate: Maksimum win rate
            min_trades: Minimum trade sayısı
            
        Returns:
            List[AIStrategyModel]: Underperforming stratejiler
        """
        return await self.strategy_repo.get_underperforming(
            max_win_rate=max_win_rate,
            min_trades=min_trades
        )
    
    # =========================================================================
    # STRATEGY LIFECYCLE
    # =========================================================================
    
    async def create_strategy(
        self,
        name: str,
        conditions: List[Dict[str, Any]],
        confidence: Decimal,
        sample_size: int,
        discovery_method: str,
        **kwargs
    ) -> AIStrategyModel:
        """
        Yeni strateji oluştur.
        
        Args:
            name: Strateji adı
            conditions: Strateji koşulları
            confidence: Güven seviyesi
            sample_size: Örnek sayısı
            discovery_method: Keşif yöntemi
            **kwargs: Ek parametreler
            
        Returns:
            AIStrategyModel: Oluşturulan strateji
        """
        strategy = await self.strategy_repo.create(
            name=name,
            conditions=conditions,
            confidence=confidence,
            sample_size=sample_size,
            discovery_method=discovery_method,
            status=StrategyStatus.PENDING.value,
            **kwargs
        )
        
        await self._invalidate_strategy_cache()
        
        logger.info(f"Strategy created: {name} ({discovery_method})")
        
        return strategy
    
    async def approve_strategy(self, strategy_id: UUID) -> Optional[AIStrategyModel]:
        """
        Stratejiyi onayla ve aktifleştir.
        
        Args:
            strategy_id: Strateji ID
            
        Returns:
            Optional[AIStrategyModel]: Onaylanan strateji
        """
        strategy = await self.strategy_repo.approve_strategy(strategy_id)
        
        if strategy:
            await self._invalidate_strategy_cache()
            await self._log_evolution(
                strategy_id=strategy_id,
                change_type="status_change",
                old_value={"status": StrategyStatus.PENDING.value},
                new_value={"status": StrategyStatus.ACTIVE.value},
                reason="Strategy approved after validation"
            )
            logger.info(f"Strategy approved: {strategy.name}")
        
        return strategy
    
    async def pause_strategy(self, strategy_id: UUID) -> Optional[AIStrategyModel]:
        """
        Stratejiyi duraklat.
        
        Args:
            strategy_id: Strateji ID
            
        Returns:
            Optional[AIStrategyModel]: Duraklatılan strateji
        """
        strategy = await self.strategy_repo.pause_strategy(strategy_id)
        
        if strategy:
            await self._invalidate_strategy_cache()
            await self._log_evolution(
                strategy_id=strategy_id,
                change_type="status_change",
                old_value={"status": StrategyStatus.ACTIVE.value},
                new_value={"status": StrategyStatus.PAUSED.value},
                reason="Strategy paused"
            )
            logger.info(f"Strategy paused: {strategy.name}")
        
        return strategy
    
    async def retire_strategy(self, strategy_id: UUID) -> Optional[AIStrategyModel]:
        """
        Stratejiyi emekli et.
        
        Args:
            strategy_id: Strateji ID
            
        Returns:
            Optional[AIStrategyModel]: Emekli edilen strateji
        """
        strategy = await self.strategy_repo.retire_strategy(strategy_id)
        
        if strategy:
            await self._invalidate_strategy_cache()
            await self._log_evolution(
                strategy_id=strategy_id,
                change_type="status_change",
                old_value={"status": strategy.status},
                new_value={"status": StrategyStatus.RETIRED.value},
                reason="Strategy retired due to underperformance or obsolescence"
            )
            logger.info(f"Strategy retired: {strategy.name}")
        
        return strategy
    
    # =========================================================================
    # PERFORMANCE
    # =========================================================================
    
    async def get_strategy_performance(
        self,
        strategy_id: UUID,
        days: int = 30
    ) -> Dict[str, Any]:
        """
        Strateji performansını al.
        
        Args:
            strategy_id: Strateji ID
            days: Analiz süresi
            
        Returns:
            Dict: Performans metrikleri
        """
        strategy = await self.strategy_repo.get(strategy_id)
        if not strategy:
            raise ValueError(f"Strategy not found: {strategy_id}")
        
        # Sinyal istatistikleri
        signal_stats = await self.signal_repo.get_performance_stats(
            strategy_id=strategy_id,
            days=days
        )
        
        return {
            "strategy_id": str(strategy_id),
            "strategy_name": strategy.name,
            "status": strategy.status,
            
            # Backtest metrikleri
            "backtest": {
                "win_rate": float(strategy.backtest_win_rate) if strategy.backtest_win_rate else None,
                "profit_factor": float(strategy.backtest_profit_factor) if strategy.backtest_profit_factor else None,
                "sharpe_ratio": float(strategy.backtest_sharpe) if strategy.backtest_sharpe else None,
                "max_drawdown": float(strategy.backtest_max_drawdown) if strategy.backtest_max_drawdown else None,
                "total_trades": strategy.backtest_total_trades,
            },
            
            # Canlı metrikleri
            "live": {
                "win_rate": float(strategy.live_win_rate) if strategy.live_win_rate else None,
                "profit_factor": float(strategy.live_profit_factor) if strategy.live_profit_factor else None,
                "total_trades": strategy.live_total_trades,
                "total_pnl": float(strategy.live_total_pnl),
                "wins": strategy.live_wins,
                "losses": strategy.live_losses,
            },
            
            # Son N gün
            "recent": signal_stats,
            
            # Genel
            "performance_score": float(strategy.performance_score),
            "last_signal_at": strategy.last_signal_at.isoformat() if strategy.last_signal_at else None,
        }
    
    async def record_trade_result(
        self,
        strategy_id: UUID,
        is_win: bool,
        pnl: Decimal
    ) -> Optional[AIStrategyModel]:
        """
        Trade sonucunu kaydet.
        
        Args:
            strategy_id: Strateji ID
            is_win: Kazanç mı?
            pnl: P&L miktarı
            
        Returns:
            Optional[AIStrategyModel]: Güncellenen strateji
        """
        strategy = await self.strategy_repo.record_trade(
            strategy_id=strategy_id,
            is_win=is_win,
            pnl=pnl
        )
        
        if strategy:
            await self._invalidate_strategy_cache()
        
        return strategy
    
    # =========================================================================
    # STATISTICS
    # =========================================================================
    
    async def get_statistics(self) -> Dict[str, Any]:
        """
        Genel strateji istatistikleri.
        
        Returns:
            Dict: İstatistikler
        """
        return await self.strategy_repo.get_statistics()
    
    async def get_discovery_method_stats(self) -> List[Dict[str, Any]]:
        """
        Keşif yöntemi bazlı istatistikler.
        
        Returns:
            List[Dict]: Keşif yöntemi istatistikleri
        """
        return await self.strategy_repo.get_discovery_method_stats()
    
    # =========================================================================
    # VALIDATION
    # =========================================================================
    
    async def validate_for_approval(
        self,
        strategy_id: UUID,
        criteria: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Stratejiyi onay için değerlendir.
        
        Args:
            strategy_id: Strateji ID
            criteria: Onay kriterleri (opsiyonel)
            
        Returns:
            Dict: Değerlendirme sonucu
        """
        strategy = await self.strategy_repo.get(strategy_id)
        if not strategy:
            raise ValueError(f"Strategy not found: {strategy_id}")
        
        # Varsayılan kriterler
        if criteria is None:
            criteria = {
                "min_win_rate": settings.ai_strategy.min_win_rate,
                "min_profit_factor": settings.ai_strategy.min_profit_factor,
                "min_sharpe_ratio": settings.ai_strategy.min_sharpe_ratio,
                "max_drawdown": settings.ai_strategy.max_drawdown,
                "min_trades": 30,
                "min_consistency": 0.6,
            }
        
        results = {}
        failing = []
        recommendations = []
        
        # Win rate
        win_rate = float(strategy.backtest_win_rate or 0)
        results["min_win_rate"] = win_rate >= criteria["min_win_rate"]
        if not results["min_win_rate"]:
            failing.append("min_win_rate")
            recommendations.append(
                f"Win rate ({win_rate:.2%}) is below threshold ({criteria['min_win_rate']:.2%})"
            )
        
        # Profit factor
        pf = float(strategy.backtest_profit_factor or 0)
        results["min_profit_factor"] = pf >= criteria["min_profit_factor"]
        if not results["min_profit_factor"]:
            failing.append("min_profit_factor")
            recommendations.append(
                f"Profit factor ({pf:.2f}) is below threshold ({criteria['min_profit_factor']:.2f})"
            )
        
        # Sharpe ratio
        sharpe = float(strategy.backtest_sharpe or 0)
        results["min_sharpe_ratio"] = sharpe >= criteria["min_sharpe_ratio"]
        if not results["min_sharpe_ratio"]:
            failing.append("min_sharpe_ratio")
            recommendations.append(
                f"Sharpe ratio ({sharpe:.2f}) is below threshold ({criteria['min_sharpe_ratio']:.2f})"
            )
        
        # Max drawdown
        dd = float(strategy.backtest_max_drawdown or 0)
        results["max_drawdown"] = dd <= criteria["max_drawdown"]
        if not results["max_drawdown"]:
            failing.append("max_drawdown")
            recommendations.append(
                f"Max drawdown ({dd:.2%}) exceeds threshold ({criteria['max_drawdown']:.2%})"
            )
        
        # Trade count
        trades = strategy.backtest_total_trades or 0
        results["min_trades"] = trades >= criteria["min_trades"]
        if not results["min_trades"]:
            failing.append("min_trades")
            recommendations.append(
                f"Trade count ({trades}) is below threshold ({criteria['min_trades']})"
            )
        
        # Walk-forward consistency
        consistency = float(strategy.walkforward_consistency or 0)
        results["min_consistency"] = consistency >= criteria["min_consistency"]
        if not results["min_consistency"]:
            failing.append("min_consistency")
            recommendations.append(
                f"Walk-forward consistency ({consistency:.2%}) is below threshold ({criteria['min_consistency']:.2%})"
            )
        
        return {
            "strategy_id": str(strategy_id),
            "approved": len(failing) == 0,
            "criteria_results": results,
            "failing_criteria": failing,
            "recommendations": recommendations,
        }
    
    # =========================================================================
    # EVOLUTION & LINEAGE
    # =========================================================================
    
    async def get_evolution_history(
        self,
        strategy_id: UUID
    ) -> List[Dict[str, Any]]:
        """
        Strateji evrim geçmişi.
        
        Args:
            strategy_id: Strateji ID
            
        Returns:
            List[Dict]: Evrim logları
        """
        strategy = await self.strategy_repo.get_with_relations(
            strategy_id,
            "evolution_logs"
        )
        
        if not strategy:
            raise ValueError(f"Strategy not found: {strategy_id}")
        
        return [
            {
                "id": str(log.id),
                "change_type": log.change_type,
                "old_value": log.old_value,
                "new_value": log.new_value,
                "reason": log.reason,
                "impact_score": float(log.impact_score) if log.impact_score else None,
                "created_at": log.created_at.isoformat(),
            }
            for log in strategy.evolution_logs
        ]
    
    async def get_lineage(self, strategy_id: UUID) -> Dict[str, Any]:
        """
        Strateji soy ağacı.
        
        Args:
            strategy_id: Strateji ID
            
        Returns:
            Dict: Soy ağacı bilgileri
        """
        strategy = await self.strategy_repo.get(strategy_id)
        if not strategy:
            raise ValueError(f"Strategy not found: {strategy_id}")
        
        parents = await self.strategy_repo.get_parent_strategies(strategy_id)
        children = await self.strategy_repo.get_children_strategies(strategy_id)
        
        return {
            "strategy": {
                "id": str(strategy.id),
                "name": strategy.name,
                "generation": strategy.generation,
                "status": strategy.status,
            },
            "parents": [
                {"id": str(p.id), "name": p.name, "generation": p.generation}
                for p in parents
            ],
            "children": [
                {"id": str(c.id), "name": c.name, "generation": c.generation}
                for c in children
            ],
        }
    
    # =========================================================================
    # HELPERS
    # =========================================================================
    
    async def _log_evolution(
        self,
        strategy_id: UUID,
        change_type: str,
        old_value: Optional[Dict] = None,
        new_value: Optional[Dict] = None,
        reason: str = "",
        impact_score: Optional[Decimal] = None
    ) -> None:
        """Evrim logu kaydet."""
        log = EvolutionLogModel(
            strategy_id=strategy_id,
            change_type=change_type,
            old_value=old_value,
            new_value=new_value,
            reason=reason,
            impact_score=impact_score,
        )
        self.session.add(log)
        await self.session.commit()
    
    async def _invalidate_strategy_cache(self) -> None:
        """Strateji cache'ini temizle."""
        await cache.delete_pattern(f"{CacheKeys.PREFIX_STRATEGIES}:*")
