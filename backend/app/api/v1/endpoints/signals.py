"""
AlphaTerminal Pro - Signals Endpoints
=====================================

Trading sinyalleri CRUD ve yönetim endpoint'leri.

Author: AlphaTerminal Team
Version: 1.0.0
"""

from __future__ import annotations

import logging
from typing import Optional, List
from uuid import UUID
from datetime import datetime, timedelta
from decimal import Decimal

from fastapi import APIRouter, Depends, HTTPException, status, Query, Path, Body
from sqlalchemy.ext.asyncio import AsyncSession

from app.config import settings, SignalType, SignalTier, SignalStatus
from app.db.database import get_session
from app.db.repositories import SignalRepository, StockRepository
from app.db.models import SignalModel
from app.api.dependencies import (
    get_signal_repository,
    get_stock_repository,
    get_current_user,
    get_current_admin_user,
    CurrentUser,
    CurrentAdmin,
    DbSession,
    SignalRepo,
    StockRepo,
    Pagination,
    PaginationParams,
    rate_limiter_default,
)
from app.schemas import (
    SignalResponse,
    SignalListResponse,
    SignalSummary,
    SignalCreate,
    SignalUpdate,
    SignalClose,
    SignalFilter,
    SignalPerformanceStats,
    TierPerformance,
    SignalDistribution,
    SymbolPerformance,
    SignalAlert,
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
    response_model=SignalListResponse,
    summary="List Signals",
    description="Sinyalleri listeler. Çeşitli filtreler destekler.",
    dependencies=[Depends(rate_limiter_default)],
)
async def list_signals(
    repo: SignalRepo,
    pagination: Pagination,
    symbol: Optional[str] = Query(None, description="Hisse sembolü"),
    tier: Optional[SignalTier] = Query(None, description="Sinyal tier'ı"),
    signal_type: Optional[SignalType] = Query(None, description="Sinyal tipi (LONG/SHORT)"),
    status: Optional[SignalStatus] = Query(None, description="Sinyal durumu"),
    strategy_id: Optional[UUID] = Query(None, description="Strateji ID"),
    min_score: Optional[float] = Query(None, ge=0, le=100, description="Minimum skor"),
    start_date: Optional[datetime] = Query(None, description="Başlangıç tarihi"),
    end_date: Optional[datetime] = Query(None, description="Bitiş tarihi"),
    active_only: bool = Query(False, description="Sadece aktif sinyaller"),
) -> SignalListResponse:
    """
    Sinyal listesi.
    
    Args:
        repo: Signal repository
        pagination: Sayfalama parametreleri
        symbol: Hisse filtresi
        tier: Tier filtresi
        signal_type: Tip filtresi
        status: Durum filtresi
        strategy_id: Strateji filtresi
        min_score: Minimum skor
        start_date: Başlangıç tarihi
        end_date: Bitiş tarihi
        active_only: Sadece aktifler
        
    Returns:
        SignalListResponse: Sayfalanmış sinyal listesi
    """
    # Aktif sinyaller için özel sorgu
    if active_only:
        signals = await repo.get_active_signals(
            tier=tier,
            signal_type=signal_type,
            limit=pagination.per_page
        )
        total = len(signals)
        pages = 1
    else:
        # Filtreleri oluştur
        filters = {}
        if symbol:
            filters["symbol"] = symbol.upper().replace(".IS", "")
        if tier:
            filters["tier"] = tier.value
        if signal_type:
            filters["signal_type"] = signal_type.value
        if status:
            filters["status"] = status.value
        if strategy_id:
            filters["strategy_id"] = strategy_id
        
        # Tarih aralığı varsa özel sorgu
        if start_date or end_date:
            signals = await repo.find_by_date_range(
                start_date=start_date or datetime.min,
                end_date=end_date,
                status=status
            )
            # Diğer filtreleri uygula
            if filters:
                signals = [s for s in signals if all(
                    getattr(s, k, None) == v for k, v in filters.items()
                )]
            total = len(signals)
            # Pagination uygula
            start_idx = pagination.offset
            end_idx = start_idx + pagination.per_page
            signals = signals[start_idx:end_idx]
            pages = (total + pagination.per_page - 1) // pagination.per_page
        else:
            # Normal sayfalama
            result = await repo.paginate(
                page=pagination.page,
                per_page=pagination.per_page,
                filters=filters if filters else None,
                order_by=pagination.order_by or "created_at",
                order_desc=True,
            )
            signals = result["items"]
            total = result["total"]
            pages = result["pages"]
    
    # Min score filtresi
    if min_score is not None:
        signals = [s for s in signals if float(s.total_score) >= min_score]
    
    # Response oluştur
    items = [SignalResponse.model_validate(signal) for signal in signals]
    
    return SignalListResponse(
        items=items,
        total=total,
        page=pagination.page,
        per_page=pagination.per_page,
        pages=pages,
    )


@router.get(
    "/active",
    response_model=List[SignalResponse],
    summary="Get Active Signals",
    description="Tüm aktif sinyalleri getirir.",
)
async def get_active_signals(
    repo: SignalRepo,
    tier: Optional[SignalTier] = Query(None, description="Tier filtresi"),
    limit: int = Query(50, ge=1, le=100, description="Maksimum sonuç"),
) -> List[SignalResponse]:
    """
    Aktif sinyaller.
    
    Args:
        repo: Signal repository
        tier: Tier filtresi
        limit: Maksimum sonuç
        
    Returns:
        List[SignalResponse]: Aktif sinyaller
    """
    # Cache kontrol
    cache_key = f"{CacheKeys.signal_active()}:{tier.value if tier else 'all'}"
    cached = await cache.get_json(cache_key)
    if cached:
        return [SignalResponse(**item) for item in cached]
    
    signals = await repo.get_active_signals(tier=tier, limit=limit)
    result = [SignalResponse.model_validate(signal) for signal in signals]
    
    # Cache'e kaydet
    await cache.set_json(cache_key, [item.model_dump(mode="json") for item in result], ttl=CacheTTL.SHORT)
    
    return result


@router.get(
    "/today",
    response_model=List[SignalResponse],
    summary="Get Today's Signals",
    description="Bugün üretilen sinyalleri getirir.",
)
async def get_today_signals(
    repo: SignalRepo,
    tier: Optional[SignalTier] = Query(None, description="Tier filtresi"),
) -> List[SignalResponse]:
    """
    Bugünün sinyalleri.
    
    Args:
        repo: Signal repository
        tier: Tier filtresi
        
    Returns:
        List[SignalResponse]: Bugünün sinyalleri
    """
    signals = await repo.get_today_signals(tier=tier)
    return [SignalResponse.model_validate(signal) for signal in signals]


# =============================================================================
# STATISTICS
# =============================================================================

@router.get(
    "/stats",
    response_model=SignalPerformanceStats,
    summary="Get Signal Statistics",
    description="Sinyal performans istatistiklerini getirir.",
)
async def get_signal_stats(
    repo: SignalRepo,
    strategy_id: Optional[UUID] = Query(None, description="Strateji filtresi"),
    days: int = Query(30, ge=1, le=365, description="Analiz süresi (gün)"),
) -> SignalPerformanceStats:
    """
    Sinyal istatistikleri.
    
    Args:
        repo: Signal repository
        strategy_id: Strateji filtresi
        days: Analiz süresi
        
    Returns:
        SignalPerformanceStats: Performans istatistikleri
    """
    stats = await repo.get_performance_stats(strategy_id=strategy_id, days=days)
    return SignalPerformanceStats(**stats)


@router.get(
    "/stats/tiers",
    response_model=List[TierPerformance],
    summary="Get Tier Performance",
    description="Tier bazlı performans istatistiklerini getirir.",
)
async def get_tier_performance(
    repo: SignalRepo,
    days: int = Query(30, ge=1, le=365, description="Analiz süresi (gün)"),
) -> List[TierPerformance]:
    """
    Tier bazlı performans.
    
    Args:
        repo: Signal repository
        days: Analiz süresi
        
    Returns:
        List[TierPerformance]: Tier performansları
    """
    stats = await repo.get_tier_performance(days=days)
    return [TierPerformance(**item) for item in stats]


@router.get(
    "/stats/distribution",
    response_model=SignalDistribution,
    summary="Get Signal Distribution",
    description="Sinyal dağılımını getirir.",
)
async def get_signal_distribution(
    repo: SignalRepo,
) -> SignalDistribution:
    """
    Sinyal dağılımı.
    
    Args:
        repo: Signal repository
        
    Returns:
        SignalDistribution: Dağılım istatistikleri
    """
    dist = await repo.get_signal_distribution()
    return SignalDistribution(**dist)


@router.get(
    "/stats/symbols",
    response_model=List[SymbolPerformance],
    summary="Get Symbol Performance",
    description="Sembol bazlı performans istatistiklerini getirir.",
)
async def get_symbol_performance(
    repo: SignalRepo,
    days: int = Query(30, ge=1, le=365, description="Analiz süresi (gün)"),
    limit: int = Query(10, ge=1, le=50, description="Maksimum sonuç"),
) -> List[SymbolPerformance]:
    """
    Sembol bazlı performans.
    
    Args:
        repo: Signal repository
        days: Analiz süresi
        limit: Maksimum sonuç
        
    Returns:
        List[SymbolPerformance]: Sembol performansları
    """
    stats = await repo.get_top_performing_symbols(days=days, limit=limit)
    return [SymbolPerformance(**item) for item in stats]


# =============================================================================
# SINGLE SIGNAL
# =============================================================================

@router.get(
    "/{signal_id}",
    response_model=SignalResponse,
    summary="Get Signal",
    description="Belirli bir sinyalin detaylarını getirir.",
)
async def get_signal(
    signal_id: UUID = Path(..., description="Sinyal ID"),
    repo: SignalRepo = Depends(get_signal_repository),
) -> SignalResponse:
    """
    Sinyal detayları.
    
    Args:
        signal_id: Sinyal ID
        repo: Signal repository
        
    Returns:
        SignalResponse: Sinyal detayları
        
    Raises:
        HTTPException: 404 - Sinyal bulunamadı
    """
    signal = await repo.get(signal_id)
    
    if not signal:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Signal not found: {signal_id}"
        )
    
    return SignalResponse.model_validate(signal)


@router.get(
    "/symbol/{symbol}",
    response_model=List[SignalResponse],
    summary="Get Signals by Symbol",
    description="Belirli bir hissenin sinyallerini getirir.",
)
async def get_signals_by_symbol(
    symbol: str = Path(..., description="Hisse sembolü"),
    include_closed: bool = Query(False, description="Kapalı sinyalleri dahil et"),
    limit: int = Query(20, ge=1, le=100, description="Maksimum sonuç"),
    repo: SignalRepo = Depends(get_signal_repository),
) -> List[SignalResponse]:
    """
    Hisse sinyalleri.
    
    Args:
        symbol: Hisse sembolü
        include_closed: Kapalı sinyalleri dahil et
        limit: Maksimum sonuç
        repo: Signal repository
        
    Returns:
        List[SignalResponse]: Hisse sinyalleri
    """
    signals = await repo.find_by_symbol(
        symbol=symbol,
        include_closed=include_closed,
        limit=limit
    )
    return [SignalResponse.model_validate(signal) for signal in signals]


# =============================================================================
# CREATE & UPDATE
# =============================================================================

@router.post(
    "",
    response_model=SignalResponse,
    status_code=status.HTTP_201_CREATED,
    summary="Create Signal",
    description="Yeni sinyal oluşturur (Admin only).",
)
async def create_signal(
    data: SignalCreate,
    signal_repo: SignalRepo,
    stock_repo: StockRepo,
    admin: CurrentAdmin,
) -> SignalResponse:
    """
    Yeni sinyal oluştur.
    
    Args:
        data: Sinyal verileri
        signal_repo: Signal repository
        stock_repo: Stock repository
        admin: Admin kullanıcı
        
    Returns:
        SignalResponse: Oluşturulan sinyal
    """
    # Hisse kontrolü
    stock = await stock_repo.find_by_symbol(data.symbol)
    stock_id = stock.id if stock else None
    
    # Sinyal oluştur
    signal = await signal_repo.create(
        symbol=data.symbol.upper().replace(".IS", ""),
        stock_id=stock_id,
        strategy_id=data.strategy_id,
        signal_type=data.signal_type.value,
        tier=data.tier.value,
        status=SignalStatus.ACTIVE.value,
        entry_price=data.entry_price,
        stop_loss=data.stop_loss,
        take_profit_1=data.take_profit_1,
        take_profit_2=data.take_profit_2,
        take_profit_3=data.take_profit_3,
        total_score=data.total_score,
        smc_score=data.smc_score,
        orderflow_score=data.orderflow_score,
        alpha_score=data.alpha_score,
        ml_score=data.ml_score,
        mtf_score=data.mtf_score,
        confidence=data.confidence,
        risk_reward=data.risk_reward,
        setup_type=data.setup_type,
        timeframe=data.timeframe,
        reasoning=data.reasoning,
        smc_data=data.smc_data,
        orderflow_data=data.orderflow_data,
        valid_until=data.valid_until,
    )
    
    # Cache'i temizle
    await cache.delete_pattern(f"{CacheKeys.PREFIX_SIGNALS}:*")
    
    logger.info(f"Signal created by admin {admin.email}: {signal.symbol} {signal.signal_type}")
    
    return SignalResponse.model_validate(signal)


@router.put(
    "/{signal_id}",
    response_model=SignalResponse,
    summary="Update Signal",
    description="Sinyal bilgilerini günceller (Admin only).",
)
async def update_signal(
    data: SignalUpdate,
    signal_id: UUID = Path(..., description="Sinyal ID"),
    repo: SignalRepo = Depends(get_signal_repository),
    admin: CurrentAdmin = Depends(get_current_admin_user),
) -> SignalResponse:
    """
    Sinyal güncelle.
    
    Args:
        data: Güncellenecek veriler
        signal_id: Sinyal ID
        repo: Signal repository
        admin: Admin kullanıcı
        
    Returns:
        SignalResponse: Güncellenen sinyal
    """
    signal = await repo.get(signal_id)
    if not signal:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Signal not found: {signal_id}"
        )
    
    update_data = data.model_dump(exclude_unset=True)
    updated_signal = await repo.update(signal_id, **update_data)
    
    # Cache'i temizle
    await cache.delete(CacheKeys.signal_by_id(str(signal_id)))
    
    logger.info(f"Signal updated by admin {admin.email}: {signal_id}")
    
    return SignalResponse.model_validate(updated_signal)


@router.post(
    "/{signal_id}/close",
    response_model=SignalResponse,
    summary="Close Signal",
    description="Sinyali kapatır.",
)
async def close_signal(
    data: SignalClose,
    signal_id: UUID = Path(..., description="Sinyal ID"),
    repo: SignalRepo = Depends(get_signal_repository),
    admin: CurrentAdmin = Depends(get_current_admin_user),
) -> SignalResponse:
    """
    Sinyali kapat.
    
    Args:
        data: Kapanış verileri
        signal_id: Sinyal ID
        repo: Signal repository
        admin: Admin kullanıcı
        
    Returns:
        SignalResponse: Kapatılan sinyal
    """
    signal = await repo.close_signal(
        signal_id=signal_id,
        exit_price=data.exit_price,
        reason=data.reason,
        pnl=data.pnl
    )
    
    if not signal:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Signal not found: {signal_id}"
        )
    
    # Cache'i temizle
    await cache.delete_pattern(f"{CacheKeys.PREFIX_SIGNALS}:*")
    
    logger.info(f"Signal closed by admin {admin.email}: {signal_id} - {data.reason}")
    
    return SignalResponse.model_validate(signal)


# =============================================================================
# PRICE LEVEL CHECKS
# =============================================================================

@router.post(
    "/check-levels",
    response_model=List[SignalAlert],
    summary="Check Price Levels",
    description="Fiyat seviyelerini kontrol eder ve tetiklenen uyarıları döner.",
)
async def check_price_levels(
    prices: dict = Body(..., description="Sembol-fiyat eşleştirmesi"),
    repo: SignalRepo = Depends(get_signal_repository),
) -> List[SignalAlert]:
    """
    Fiyat seviyelerini kontrol et.
    
    Args:
        prices: Sembol-fiyat eşleştirmesi {"THYAO": 150.5, "GARAN": 45.2}
        repo: Signal repository
        
    Returns:
        List[SignalAlert]: Tetiklenen uyarılar
    """
    alerts = []
    
    for symbol, price in prices.items():
        triggered = await repo.check_price_levels(symbol, Decimal(str(price)))
        
        for t in triggered:
            signal = await repo.get(t["signal_id"])
            if signal:
                alert_type = t["type"]
                target_price = (
                    signal.stop_loss if alert_type == "stop_loss"
                    else signal.take_profit_1
                )
                
                alerts.append(SignalAlert(
                    signal_id=t["signal_id"],
                    symbol=symbol,
                    alert_type=alert_type,
                    current_price=Decimal(str(price)),
                    target_price=target_price,
                    message=f"{symbol} {alert_type.replace('_', ' ').title()} triggered at {price}",
                ))
    
    return alerts


# =============================================================================
# ADMIN OPERATIONS
# =============================================================================

@router.post(
    "/expire-old",
    response_model=SuccessResponse,
    summary="Expire Old Signals",
    description="Eski sinyalleri expire eder (Admin only).",
)
async def expire_old_signals(
    max_age_hours: int = Query(48, ge=1, le=168, description="Maksimum yaş (saat)"),
    repo: SignalRepo = Depends(get_signal_repository),
    admin: CurrentAdmin = Depends(get_current_admin_user),
) -> SuccessResponse:
    """
    Eski sinyalleri expire et.
    
    Args:
        max_age_hours: Maksimum yaş
        repo: Signal repository
        admin: Admin kullanıcı
        
    Returns:
        SuccessResponse: İşlem sonucu
    """
    count = await repo.expire_old_signals(max_age_hours=max_age_hours)
    
    # Cache'i temizle
    await cache.delete_pattern(f"{CacheKeys.PREFIX_SIGNALS}:*")
    
    logger.info(f"Expired {count} old signals by admin {admin.email}")
    
    return SuccessResponse(
        message=f"Successfully expired {count} old signals",
        data={"expired_count": count}
    )


@router.delete(
    "/{signal_id}",
    response_model=SuccessResponse,
    summary="Delete Signal",
    description="Sinyali siler (Admin only).",
)
async def delete_signal(
    signal_id: UUID = Path(..., description="Sinyal ID"),
    repo: SignalRepo = Depends(get_signal_repository),
    admin: CurrentAdmin = Depends(get_current_admin_user),
) -> SuccessResponse:
    """
    Sinyal sil.
    
    Args:
        signal_id: Sinyal ID
        repo: Signal repository
        admin: Admin kullanıcı
        
    Returns:
        SuccessResponse: Silme sonucu
    """
    signal = await repo.get(signal_id)
    if not signal:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Signal not found: {signal_id}"
        )
    
    await repo.delete(signal_id)
    
    # Cache'i temizle
    await cache.delete_pattern(f"{CacheKeys.PREFIX_SIGNALS}:*")
    
    logger.info(f"Signal deleted by admin {admin.email}: {signal_id}")
    
    return SuccessResponse(message=f"Signal {signal_id} deleted successfully")
