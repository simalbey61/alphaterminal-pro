"""
AlphaTerminal Pro - Strategies Endpoints
========================================

AI stratejileri CRUD ve yönetim endpoint'leri.

Author: AlphaTerminal Team
Version: 1.0.0
"""

from __future__ import annotations

import logging
from typing import Optional, List
from uuid import UUID
from datetime import datetime
from decimal import Decimal

from fastapi import APIRouter, Depends, HTTPException, status, Query, Path
from sqlalchemy.ext.asyncio import AsyncSession

from app.config import settings, StrategyStatus
from app.db.database import get_session
from app.db.repositories import StrategyRepository, SignalRepository
from app.db.models import AIStrategyModel
from app.api.dependencies import (
    get_strategy_repository,
    get_signal_repository,
    get_current_user,
    get_current_admin_user,
    get_current_premium_user,
    CurrentUser,
    CurrentAdmin,
    CurrentPremium,
    DbSession,
    StrategyRepo,
    SignalRepo,
    Pagination,
    PaginationParams,
    rate_limiter_default,
)
from app.schemas import (
    StrategyResponse,
    StrategyListResponse,
    StrategySummary,
    StrategyCreate,
    StrategyUpdate,
    StrategyStatistics,
    DiscoveryMethodStats,
    EvolutionLogResponse,
    StrategyEvolutionHistory,
    StrategyApprovalCriteria,
    StrategyApprovalResult,
    SignalResponse,
    SuccessResponse,
    ErrorResponse,
)
from app.cache import cache, CacheKeys, CacheTTL

logger = logging.getLogger(__name__)
router = APIRouter()


# =============================================================================
# LIST & FILTER
# =============================================================================

@router.get(
    "",
    response_model=StrategyListResponse,
    summary="List Strategies",
    description="AI stratejilerini listeler.",
    dependencies=[Depends(rate_limiter_default)],
)
async def list_strategies(
    repo: StrategyRepo,
    pagination: Pagination,
    status_filter: Optional[StrategyStatus] = Query(None, alias="status", description="Durum filtresi"),
    discovery_method: Optional[str] = Query(None, description="Keşif yöntemi filtresi"),
    min_win_rate: Optional[float] = Query(None, ge=0, le=1, description="Minimum win rate"),
    min_trades: Optional[int] = Query(None, ge=0, description="Minimum trade sayısı"),
    generation: Optional[int] = Query(None, ge=1, description="Jenerasyon numarası"),
) -> StrategyListResponse:
    """
    Strateji listesi.
    
    Args:
        repo: Strategy repository
        pagination: Sayfalama parametreleri
        status_filter: Durum filtresi
        discovery_method: Keşif yöntemi
        min_win_rate: Minimum win rate
        min_trades: Minimum trade sayısı
        generation: Jenerasyon numarası
        
    Returns:
        StrategyListResponse: Sayfalanmış strateji listesi
    """
    # Filtreleri oluştur
    filters = {}
    if status_filter:
        filters["status"] = status_filter.value
    if discovery_method:
        filters["discovery_method"] = discovery_method
    if generation:
        filters["generation"] = generation
    
    # Sayfalama
    result = await repo.paginate(
        page=pagination.page,
        per_page=pagination.per_page,
        filters=filters if filters else None,
        order_by=pagination.order_by or "performance_score",
        order_desc=True,
    )
    
    strategies = result["items"]
    
    # Ek filtreler (DB seviyesinde olmayan)
    if min_win_rate is not None:
        strategies = [s for s in strategies if s.live_win_rate and float(s.live_win_rate) >= min_win_rate]
    if min_trades is not None:
        strategies = [s for s in strategies if s.live_total_trades >= min_trades]
    
    # Response oluştur
    items = [StrategyResponse.model_validate(strategy) for strategy in strategies]
    
    return StrategyListResponse(
        items=items,
        total=result["total"],
        page=pagination.page,
        per_page=pagination.per_page,
        pages=result["pages"],
    )


@router.get(
    "/active",
    response_model=List[StrategyResponse],
    summary="Get Active Strategies",
    description="Aktif stratejileri getirir.",
)
async def get_active_strategies(
    repo: StrategyRepo,
    discovery_method: Optional[str] = Query(None, description="Keşif yöntemi filtresi"),
) -> List[StrategyResponse]:
    """
    Aktif stratejiler.
    
    Args:
        repo: Strategy repository
        discovery_method: Keşif yöntemi filtresi
        
    Returns:
        List[StrategyResponse]: Aktif stratejiler
    """
    # Cache kontrol
    cache_key = f"{CacheKeys.strategy_active()}:{discovery_method or 'all'}"
    cached = await cache.get_json(cache_key)
    if cached:
        return [StrategyResponse(**item) for item in cached]
    
    strategies = await repo.get_active_strategies(discovery_method=discovery_method)
    result = [StrategyResponse.model_validate(strategy) for strategy in strategies]
    
    # Cache'e kaydet
    await cache.set_json(cache_key, [item.model_dump(mode="json") for item in result], ttl=CacheTTL.STRATEGIES)
    
    return result


@router.get(
    "/pending",
    response_model=List[StrategyResponse],
    summary="Get Pending Strategies",
    description="Onay bekleyen stratejileri getirir (Admin only).",
)
async def get_pending_strategies(
    repo: StrategyRepo,
    admin: CurrentAdmin,
) -> List[StrategyResponse]:
    """
    Onay bekleyen stratejiler.
    
    Args:
        repo: Strategy repository
        admin: Admin kullanıcı
        
    Returns:
        List[StrategyResponse]: Onay bekleyen stratejiler
    """
    strategies = await repo.get_pending_strategies()
    return [StrategyResponse.model_validate(strategy) for strategy in strategies]


@router.get(
    "/top-performing",
    response_model=List[StrategyResponse],
    summary="Get Top Performing Strategies",
    description="En iyi performanslı stratejileri getirir.",
)
async def get_top_performing_strategies(
    repo: StrategyRepo,
    limit: int = Query(10, ge=1, le=50, description="Maksimum sonuç"),
    min_trades: int = Query(10, ge=0, description="Minimum trade sayısı"),
) -> List[StrategyResponse]:
    """
    En iyi performanslı stratejiler.
    
    Args:
        repo: Strategy repository
        limit: Maksimum sonuç
        min_trades: Minimum trade sayısı
        
    Returns:
        List[StrategyResponse]: Top stratejiler
    """
    strategies = await repo.get_top_performing(limit=limit, min_trades=min_trades)
    return [StrategyResponse.model_validate(strategy) for strategy in strategies]


@router.get(
    "/underperforming",
    response_model=List[StrategyResponse],
    summary="Get Underperforming Strategies",
    description="Düşük performanslı stratejileri getirir (emeklilik adayları).",
)
async def get_underperforming_strategies(
    repo: StrategyRepo,
    admin: CurrentAdmin,
    max_win_rate: float = Query(0.4, ge=0, le=1, description="Maksimum win rate"),
    min_trades: int = Query(20, ge=0, description="Minimum trade sayısı"),
) -> List[StrategyResponse]:
    """
    Düşük performanslı stratejiler.
    
    Args:
        repo: Strategy repository
        admin: Admin kullanıcı
        max_win_rate: Maksimum win rate
        min_trades: Minimum trade sayısı
        
    Returns:
        List[StrategyResponse]: Düşük performanslı stratejiler
    """
    strategies = await repo.get_underperforming(
        max_win_rate=max_win_rate,
        min_trades=min_trades
    )
    return [StrategyResponse.model_validate(strategy) for strategy in strategies]


# =============================================================================
# STATISTICS
# =============================================================================

@router.get(
    "/stats",
    response_model=StrategyStatistics,
    summary="Get Strategy Statistics",
    description="Strateji istatistiklerini getirir.",
)
async def get_strategy_stats(
    repo: StrategyRepo,
) -> StrategyStatistics:
    """
    Strateji istatistikleri.
    
    Args:
        repo: Strategy repository
        
    Returns:
        StrategyStatistics: İstatistikler
    """
    # Cache kontrol
    cache_key = CacheKeys.strategy_stats()
    cached = await cache.get_json(cache_key)
    if cached:
        return StrategyStatistics(**cached)
    
    stats = await repo.get_statistics()
    result = StrategyStatistics(**stats)
    
    # Cache'e kaydet
    await cache.set_json(cache_key, result.model_dump(), ttl=CacheTTL.STATS)
    
    return result


@router.get(
    "/stats/discovery-methods",
    response_model=List[DiscoveryMethodStats],
    summary="Get Discovery Method Statistics",
    description="Keşif yöntemi bazlı istatistikleri getirir.",
)
async def get_discovery_method_stats(
    repo: StrategyRepo,
) -> List[DiscoveryMethodStats]:
    """
    Keşif yöntemi istatistikleri.
    
    Args:
        repo: Strategy repository
        
    Returns:
        List[DiscoveryMethodStats]: Keşif yöntemi istatistikleri
    """
    stats = await repo.get_discovery_method_stats()
    return [DiscoveryMethodStats(**item) for item in stats]


# =============================================================================
# SINGLE STRATEGY
# =============================================================================

@router.get(
    "/{strategy_id}",
    response_model=StrategyResponse,
    summary="Get Strategy",
    description="Belirli bir stratejinin detaylarını getirir.",
)
async def get_strategy(
    strategy_id: UUID = Path(..., description="Strateji ID"),
    repo: StrategyRepo = Depends(get_strategy_repository),
) -> StrategyResponse:
    """
    Strateji detayları.
    
    Args:
        strategy_id: Strateji ID
        repo: Strategy repository
        
    Returns:
        StrategyResponse: Strateji detayları
        
    Raises:
        HTTPException: 404 - Strateji bulunamadı
    """
    strategy = await repo.get(strategy_id)
    
    if not strategy:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Strategy not found: {strategy_id}"
        )
    
    return StrategyResponse.model_validate(strategy)


@router.get(
    "/{strategy_id}/signals",
    response_model=List[SignalResponse],
    summary="Get Strategy Signals",
    description="Stratejinin ürettiği sinyalleri getirir.",
)
async def get_strategy_signals(
    strategy_id: UUID = Path(..., description="Strateji ID"),
    limit: int = Query(50, ge=1, le=200, description="Maksimum sonuç"),
    strategy_repo: StrategyRepo = Depends(get_strategy_repository),
    signal_repo: SignalRepo = Depends(get_signal_repository),
) -> List[SignalResponse]:
    """
    Strateji sinyalleri.
    
    Args:
        strategy_id: Strateji ID
        limit: Maksimum sonuç
        strategy_repo: Strategy repository
        signal_repo: Signal repository
        
    Returns:
        List[SignalResponse]: Strateji sinyalleri
    """
    # Strateji var mı kontrol
    strategy = await strategy_repo.get(strategy_id)
    if not strategy:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Strategy not found: {strategy_id}"
        )
    
    signals = await signal_repo.find_by_strategy(strategy_id=strategy_id, limit=limit)
    return [SignalResponse.model_validate(signal) for signal in signals]


@router.get(
    "/{strategy_id}/performance",
    summary="Get Strategy Performance",
    description="Stratejinin performans metriklerini getirir.",
)
async def get_strategy_performance(
    strategy_id: UUID = Path(..., description="Strateji ID"),
    days: int = Query(30, ge=1, le=365, description="Analiz süresi (gün)"),
    strategy_repo: StrategyRepo = Depends(get_strategy_repository),
    signal_repo: SignalRepo = Depends(get_signal_repository),
) -> dict:
    """
    Strateji performansı.
    
    Args:
        strategy_id: Strateji ID
        days: Analiz süresi
        strategy_repo: Strategy repository
        signal_repo: Signal repository
        
    Returns:
        dict: Performans metrikleri
    """
    # Strateji var mı kontrol
    strategy = await strategy_repo.get(strategy_id)
    if not strategy:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Strategy not found: {strategy_id}"
        )
    
    # Cache kontrol
    cache_key = CacheKeys.strategy_performance(str(strategy_id))
    cached = await cache.get_json(cache_key)
    if cached:
        return cached
    
    # Sinyal istatistikleri
    signal_stats = await signal_repo.get_performance_stats(strategy_id=strategy_id, days=days)
    
    result = {
        "strategy_id": str(strategy_id),
        "strategy_name": strategy.name,
        "strategy_version": strategy.version,
        "status": strategy.status,
        
        # Backtest metrikleri
        "backtest": {
            "win_rate": float(strategy.backtest_win_rate) if strategy.backtest_win_rate else None,
            "profit_factor": float(strategy.backtest_profit_factor) if strategy.backtest_profit_factor else None,
            "sharpe_ratio": float(strategy.backtest_sharpe) if strategy.backtest_sharpe else None,
            "max_drawdown": float(strategy.backtest_max_drawdown) if strategy.backtest_max_drawdown else None,
            "total_trades": strategy.backtest_total_trades,
            "period_days": strategy.backtest_period_days,
        },
        
        # Walk-forward metrikleri
        "walk_forward": {
            "consistency": float(strategy.walkforward_consistency) if strategy.walkforward_consistency else None,
            "windows_passed": strategy.walkforward_windows_passed,
        },
        
        # Canlı metrikleri
        "live": {
            "win_rate": float(strategy.live_win_rate) if strategy.live_win_rate else None,
            "profit_factor": float(strategy.live_profit_factor) if strategy.live_profit_factor else None,
            "sharpe_ratio": float(strategy.live_sharpe) if strategy.live_sharpe else None,
            "total_trades": strategy.live_total_trades,
            "total_pnl": float(strategy.live_total_pnl),
            "wins": strategy.live_wins,
            "losses": strategy.live_losses,
        },
        
        # Son N gün sinyal istatistikleri
        "recent_signals": signal_stats,
        
        # Meta
        "performance_score": float(strategy.performance_score),
        "last_signal_at": strategy.last_signal_at.isoformat() if strategy.last_signal_at else None,
        "approved_at": strategy.approved_at.isoformat() if strategy.approved_at else None,
    }
    
    # Cache'e kaydet
    await cache.set_json(cache_key, result, ttl=CacheTTL.STATS)
    
    return result


@router.get(
    "/{strategy_id}/evolution",
    response_model=StrategyEvolutionHistory,
    summary="Get Strategy Evolution History",
    description="Stratejinin evrim geçmişini getirir.",
)
async def get_strategy_evolution(
    strategy_id: UUID = Path(..., description="Strateji ID"),
    repo: StrategyRepo = Depends(get_strategy_repository),
    session: DbSession = Depends(get_session),
) -> StrategyEvolutionHistory:
    """
    Strateji evrim geçmişi.
    
    Args:
        strategy_id: Strateji ID
        repo: Strategy repository
        session: Database session
        
    Returns:
        StrategyEvolutionHistory: Evrim geçmişi
    """
    strategy = await repo.get_with_relations(strategy_id, "evolution_logs")
    
    if not strategy:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Strategy not found: {strategy_id}"
        )
    
    logs = [EvolutionLogResponse.model_validate(log) for log in strategy.evolution_logs]
    
    return StrategyEvolutionHistory(
        strategy_id=strategy_id,
        strategy_name=strategy.name,
        current_version=strategy.version,
        logs=logs,
        total_evolutions=len(logs),
    )


@router.get(
    "/{strategy_id}/lineage",
    summary="Get Strategy Lineage",
    description="Stratejinin genetik soy ağacını getirir.",
)
async def get_strategy_lineage(
    strategy_id: UUID = Path(..., description="Strateji ID"),
    repo: StrategyRepo = Depends(get_strategy_repository),
) -> dict:
    """
    Strateji soy ağacı.
    
    Args:
        strategy_id: Strateji ID
        repo: Strategy repository
        
    Returns:
        dict: Soy ağacı
    """
    strategy = await repo.get(strategy_id)
    
    if not strategy:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Strategy not found: {strategy_id}"
        )
    
    # Ebeveynleri al
    parents = await repo.get_parent_strategies(strategy_id)
    
    # Çocukları al
    children = await repo.get_children_strategies(strategy_id)
    
    return {
        "strategy": StrategySummary.model_validate(strategy),
        "generation": strategy.generation,
        "parents": [StrategySummary.model_validate(p) for p in parents],
        "children": [StrategySummary.model_validate(c) for c in children],
    }


# =============================================================================
# CREATE & UPDATE
# =============================================================================

@router.post(
    "",
    response_model=StrategyResponse,
    status_code=status.HTTP_201_CREATED,
    summary="Create Strategy",
    description="Yeni AI stratejisi oluşturur (Admin only).",
)
async def create_strategy(
    data: StrategyCreate,
    repo: StrategyRepo,
    admin: CurrentAdmin,
) -> StrategyResponse:
    """
    Yeni strateji oluştur.
    
    Args:
        data: Strateji verileri
        repo: Strategy repository
        admin: Admin kullanıcı
        
    Returns:
        StrategyResponse: Oluşturulan strateji
    """
    # Conditions'ı JSON'a çevir
    conditions_json = [cond.model_dump() for cond in data.conditions]
    
    strategy = await repo.create(
        name=data.name,
        description=data.description,
        conditions=conditions_json,
        confidence=data.confidence,
        sample_size=data.sample_size,
        discovery_method=data.discovery_method,
        stop_loss_atr=data.stop_loss_atr,
        take_profit_r=data.take_profit_r,
        position_size_pct=data.position_size_pct,
        parent_strategy_ids=data.parent_strategy_ids,
        generation=data.generation,
        metadata=data.metadata,
    )
    
    # Cache'i temizle
    await cache.delete_pattern(f"{CacheKeys.PREFIX_STRATEGIES}:*")
    
    logger.info(f"Strategy created by admin {admin.email}: {strategy.name}")
    
    return StrategyResponse.model_validate(strategy)


@router.put(
    "/{strategy_id}",
    response_model=StrategyResponse,
    summary="Update Strategy",
    description="Strateji bilgilerini günceller (Admin only).",
)
async def update_strategy(
    data: StrategyUpdate,
    strategy_id: UUID = Path(..., description="Strateji ID"),
    repo: StrategyRepo = Depends(get_strategy_repository),
    admin: CurrentAdmin = Depends(get_current_admin_user),
) -> StrategyResponse:
    """
    Strateji güncelle.
    
    Args:
        data: Güncellenecek veriler
        strategy_id: Strateji ID
        repo: Strategy repository
        admin: Admin kullanıcı
        
    Returns:
        StrategyResponse: Güncellenen strateji
    """
    strategy = await repo.get(strategy_id)
    if not strategy:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Strategy not found: {strategy_id}"
        )
    
    update_data = data.model_dump(exclude_unset=True)
    
    # Conditions varsa JSON'a çevir
    if "conditions" in update_data and update_data["conditions"]:
        update_data["conditions"] = [cond.model_dump() for cond in update_data["conditions"]]
    
    updated_strategy = await repo.update(strategy_id, **update_data)
    
    # Cache'i temizle
    await cache.delete_pattern(f"{CacheKeys.PREFIX_STRATEGIES}:*")
    
    logger.info(f"Strategy updated by admin {admin.email}: {strategy_id}")
    
    return StrategyResponse.model_validate(updated_strategy)


# =============================================================================
# LIFECYCLE MANAGEMENT
# =============================================================================

@router.post(
    "/{strategy_id}/approve",
    response_model=StrategyResponse,
    summary="Approve Strategy",
    description="Stratejiyi onaylar ve aktifleştirir (Admin only).",
)
async def approve_strategy(
    strategy_id: UUID = Path(..., description="Strateji ID"),
    repo: StrategyRepo = Depends(get_strategy_repository),
    admin: CurrentAdmin = Depends(get_current_admin_user),
) -> StrategyResponse:
    """
    Stratejiyi onayla.
    
    Args:
        strategy_id: Strateji ID
        repo: Strategy repository
        admin: Admin kullanıcı
        
    Returns:
        StrategyResponse: Onaylanan strateji
    """
    strategy = await repo.approve_strategy(strategy_id)
    
    if not strategy:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Strategy not found: {strategy_id}"
        )
    
    # Cache'i temizle
    await cache.delete_pattern(f"{CacheKeys.PREFIX_STRATEGIES}:*")
    
    logger.info(f"Strategy approved by admin {admin.email}: {strategy_id}")
    
    return StrategyResponse.model_validate(strategy)


@router.post(
    "/{strategy_id}/pause",
    response_model=StrategyResponse,
    summary="Pause Strategy",
    description="Stratejiyi duraklatır (Admin only).",
)
async def pause_strategy(
    strategy_id: UUID = Path(..., description="Strateji ID"),
    repo: StrategyRepo = Depends(get_strategy_repository),
    admin: CurrentAdmin = Depends(get_current_admin_user),
) -> StrategyResponse:
    """
    Stratejiyi duraklat.
    
    Args:
        strategy_id: Strateji ID
        repo: Strategy repository
        admin: Admin kullanıcı
        
    Returns:
        StrategyResponse: Duraklatılan strateji
    """
    strategy = await repo.pause_strategy(strategy_id)
    
    if not strategy:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Strategy not found: {strategy_id}"
        )
    
    # Cache'i temizle
    await cache.delete_pattern(f"{CacheKeys.PREFIX_STRATEGIES}:*")
    
    logger.info(f"Strategy paused by admin {admin.email}: {strategy_id}")
    
    return StrategyResponse.model_validate(strategy)


@router.post(
    "/{strategy_id}/activate",
    response_model=StrategyResponse,
    summary="Activate Strategy",
    description="Duraklatılmış stratejiyi aktifleştirir (Admin only).",
)
async def activate_strategy(
    strategy_id: UUID = Path(..., description="Strateji ID"),
    repo: StrategyRepo = Depends(get_strategy_repository),
    admin: CurrentAdmin = Depends(get_current_admin_user),
) -> StrategyResponse:
    """
    Stratejiyi aktifleştir.
    
    Args:
        strategy_id: Strateji ID
        repo: Strategy repository
        admin: Admin kullanıcı
        
    Returns:
        StrategyResponse: Aktifleştirilen strateji
    """
    strategy = await repo.get(strategy_id)
    
    if not strategy:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Strategy not found: {strategy_id}"
        )
    
    if strategy.status != StrategyStatus.PAUSED.value:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Only paused strategies can be activated"
        )
    
    strategy.activate()
    await repo.session.commit()
    await repo.session.refresh(strategy)
    
    # Cache'i temizle
    await cache.delete_pattern(f"{CacheKeys.PREFIX_STRATEGIES}:*")
    
    logger.info(f"Strategy activated by admin {admin.email}: {strategy_id}")
    
    return StrategyResponse.model_validate(strategy)


@router.post(
    "/{strategy_id}/retire",
    response_model=StrategyResponse,
    summary="Retire Strategy",
    description="Stratejiyi emekli eder (Admin only).",
)
async def retire_strategy(
    strategy_id: UUID = Path(..., description="Strateji ID"),
    repo: StrategyRepo = Depends(get_strategy_repository),
    admin: CurrentAdmin = Depends(get_current_admin_user),
) -> StrategyResponse:
    """
    Stratejiyi emekli et.
    
    Args:
        strategy_id: Strateji ID
        repo: Strategy repository
        admin: Admin kullanıcı
        
    Returns:
        StrategyResponse: Emekli edilen strateji
    """
    strategy = await repo.retire_strategy(strategy_id)
    
    if not strategy:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Strategy not found: {strategy_id}"
        )
    
    # Cache'i temizle
    await cache.delete_pattern(f"{CacheKeys.PREFIX_STRATEGIES}:*")
    
    logger.info(f"Strategy retired by admin {admin.email}: {strategy_id}")
    
    return StrategyResponse.model_validate(strategy)


# =============================================================================
# VALIDATION & APPROVAL
# =============================================================================

@router.post(
    "/{strategy_id}/validate",
    response_model=StrategyApprovalResult,
    summary="Validate Strategy",
    description="Stratejiyi onay kriterlerine göre değerlendirir.",
)
async def validate_strategy(
    criteria: StrategyApprovalCriteria,
    strategy_id: UUID = Path(..., description="Strateji ID"),
    repo: StrategyRepo = Depends(get_strategy_repository),
    admin: CurrentAdmin = Depends(get_current_admin_user),
) -> StrategyApprovalResult:
    """
    Strateji doğrulama.
    
    Args:
        criteria: Onay kriterleri
        strategy_id: Strateji ID
        repo: Strategy repository
        admin: Admin kullanıcı
        
    Returns:
        StrategyApprovalResult: Doğrulama sonucu
    """
    strategy = await repo.get(strategy_id)
    
    if not strategy:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Strategy not found: {strategy_id}"
        )
    
    # Kriterleri kontrol et
    criteria_results = {}
    failing_criteria = []
    recommendations = []
    
    # Win rate kontrolü
    win_rate = strategy.backtest_win_rate or Decimal("0")
    criteria_results["min_win_rate"] = win_rate >= criteria.min_win_rate
    if not criteria_results["min_win_rate"]:
        failing_criteria.append("min_win_rate")
        recommendations.append(f"Win rate ({float(win_rate):.2%}) is below threshold ({float(criteria.min_win_rate):.2%})")
    
    # Profit factor kontrolü
    pf = strategy.backtest_profit_factor or Decimal("0")
    criteria_results["min_profit_factor"] = pf >= criteria.min_profit_factor
    if not criteria_results["min_profit_factor"]:
        failing_criteria.append("min_profit_factor")
        recommendations.append(f"Profit factor ({float(pf):.2f}) is below threshold ({float(criteria.min_profit_factor):.2f})")
    
    # Sharpe ratio kontrolü
    sharpe = strategy.backtest_sharpe or Decimal("0")
    criteria_results["min_sharpe_ratio"] = sharpe >= criteria.min_sharpe_ratio
    if not criteria_results["min_sharpe_ratio"]:
        failing_criteria.append("min_sharpe_ratio")
        recommendations.append(f"Sharpe ratio ({float(sharpe):.2f}) is below threshold ({float(criteria.min_sharpe_ratio):.2f})")
    
    # Max drawdown kontrolü
    dd = strategy.backtest_max_drawdown or Decimal("0")
    criteria_results["max_drawdown"] = dd <= criteria.max_drawdown
    if not criteria_results["max_drawdown"]:
        failing_criteria.append("max_drawdown")
        recommendations.append(f"Max drawdown ({float(dd):.2%}) exceeds threshold ({float(criteria.max_drawdown):.2%})")
    
    # Trade sayısı kontrolü
    trades = strategy.backtest_total_trades or 0
    criteria_results["min_trades"] = trades >= criteria.min_trades
    if not criteria_results["min_trades"]:
        failing_criteria.append("min_trades")
        recommendations.append(f"Trade count ({trades}) is below threshold ({criteria.min_trades})")
    
    # Walk-forward consistency kontrolü
    consistency = strategy.walkforward_consistency or Decimal("0")
    criteria_results["min_consistency"] = consistency >= criteria.min_consistency
    if not criteria_results["min_consistency"]:
        failing_criteria.append("min_consistency")
        recommendations.append(f"Walk-forward consistency ({float(consistency):.2%}) is below threshold ({float(criteria.min_consistency):.2%})")
    
    # Genel onay durumu
    approved = len(failing_criteria) == 0
    
    return StrategyApprovalResult(
        strategy_id=strategy_id,
        approved=approved,
        criteria_results=criteria_results,
        failing_criteria=failing_criteria,
        recommendations=recommendations,
    )


# =============================================================================
# DELETE
# =============================================================================

@router.delete(
    "/{strategy_id}",
    response_model=SuccessResponse,
    summary="Delete Strategy",
    description="Stratejiyi siler (Admin only).",
)
async def delete_strategy(
    strategy_id: UUID = Path(..., description="Strateji ID"),
    repo: StrategyRepo = Depends(get_strategy_repository),
    admin: CurrentAdmin = Depends(get_current_admin_user),
) -> SuccessResponse:
    """
    Strateji sil.
    
    Args:
        strategy_id: Strateji ID
        repo: Strategy repository
        admin: Admin kullanıcı
        
    Returns:
        SuccessResponse: Silme sonucu
    """
    strategy = await repo.get(strategy_id)
    if not strategy:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Strategy not found: {strategy_id}"
        )
    
    await repo.delete(strategy_id)
    
    # Cache'i temizle
    await cache.delete_pattern(f"{CacheKeys.PREFIX_STRATEGIES}:*")
    
    logger.info(f"Strategy deleted by admin {admin.email}: {strategy_id}")
    
    return SuccessResponse(message=f"Strategy {strategy_id} deleted successfully")
