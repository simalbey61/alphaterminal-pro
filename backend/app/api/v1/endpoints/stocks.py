"""
AlphaTerminal Pro - Stocks Endpoints
====================================

Hisse senetleri CRUD ve analiz endpoint'leri.

Author: AlphaTerminal Team
Version: 1.0.0
"""

from __future__ import annotations

import logging
from typing import Optional, List
from uuid import UUID
from decimal import Decimal

from fastapi import APIRouter, Depends, HTTPException, status, Query, Path
from sqlalchemy.ext.asyncio import AsyncSession

from app.config import settings, SECTORS, SECTOR_META, get_sector_symbols, get_symbol_sector
from app.db.database import get_session
from app.db.repositories import StockRepository
from app.db.models import StockModel
from app.api.dependencies import (
    get_stock_repository,
    get_current_user_optional,
    get_current_admin_user,
    CurrentUser,
    CurrentUserOptional,
    CurrentAdmin,
    DbSession,
    StockRepo,
    Pagination,
    PaginationParams,
    rate_limiter_default,
)
from app.schemas import (
    StockResponse,
    StockListResponse,
    StockSummary,
    StockMover,
    StockCreate,
    StockUpdate,
    StockPriceUpdate,
    SectorSummary,
    SectorDetailResponse,
    MarketStatistics,
    SuccessResponse,
    ErrorResponse,
)
from app.cache import cache, CacheKeys, CacheTTL

logger = logging.getLogger(__name__)
router = APIRouter()


# =============================================================================
# LIST & SEARCH
# =============================================================================

@router.get(
    "",
    response_model=StockListResponse,
    summary="List Stocks",
    description="Tüm hisseleri listeler. Filtreleme ve sıralama destekler.",
    dependencies=[Depends(rate_limiter_default)],
)
async def list_stocks(
    repo: StockRepo,
    pagination: Pagination,
    sector: Optional[str] = Query(None, description="Sektör filtresi"),
    search: Optional[str] = Query(None, description="Arama terimi (sembol veya isim)"),
    min_change: Optional[float] = Query(None, description="Minimum günlük değişim (%)"),
    max_change: Optional[float] = Query(None, description="Maximum günlük değişim (%)"),
    active_only: bool = Query(True, description="Sadece aktif hisseler"),
) -> StockListResponse:
    """
    Hisse listesi.
    
    Args:
        repo: Stock repository
        pagination: Sayfalama parametreleri
        sector: Sektör filtresi
        search: Arama terimi
        min_change: Minimum değişim
        max_change: Maximum değişim
        active_only: Sadece aktifler
        
    Returns:
        StockListResponse: Sayfalanmış hisse listesi
    """
    # Filtreleri oluştur
    filters = {}
    if active_only:
        filters["is_active"] = True
    if sector:
        filters["sector"] = sector.upper()
    
    # Arama varsa özel sorgu
    if search:
        stocks = await repo.search_stocks(search, limit=pagination.per_page)
        total = len(stocks)
        pages = 1
    else:
        # Normal sayfalama
        result = await repo.paginate(
            page=pagination.page,
            per_page=pagination.per_page,
            filters=filters if filters else None,
            order_by=pagination.order_by or "symbol",
            order_desc=pagination.order_desc,
        )
        stocks = result["items"]
        total = result["total"]
        pages = result["pages"]
    
    # Response oluştur
    items = [StockResponse.model_validate(stock) for stock in stocks]
    
    return StockListResponse(
        items=items,
        total=total,
        page=pagination.page,
        per_page=pagination.per_page,
        pages=pages,
    )


@router.get(
    "/search",
    response_model=List[StockSummary],
    summary="Search Stocks",
    description="Hisse arama (autocomplete için optimize edilmiş).",
)
async def search_stocks(
    repo: StockRepo,
    q: str = Query(..., min_length=1, description="Arama terimi"),
    limit: int = Query(10, ge=1, le=50, description="Maksimum sonuç sayısı"),
) -> List[StockSummary]:
    """
    Hisse arama.
    
    Sembol veya şirket adında arama yapar.
    Autocomplete için optimize edilmiştir.
    
    Args:
        repo: Stock repository
        q: Arama terimi
        limit: Maksimum sonuç
        
    Returns:
        List[StockSummary]: Bulunan hisseler
    """
    stocks = await repo.search_stocks(q, limit=limit)
    return [StockSummary.model_validate(stock) for stock in stocks]


# =============================================================================
# MOVERS (Top Gainers/Losers)
# =============================================================================

@router.get(
    "/movers/gainers",
    response_model=List[StockMover],
    summary="Top Gainers",
    description="En çok yükselen hisseler.",
)
async def get_top_gainers(
    repo: StockRepo,
    limit: int = Query(10, ge=1, le=50, description="Maksimum sonuç sayısı"),
) -> List[StockMover]:
    """
    En çok yükselen hisseler.
    
    Args:
        repo: Stock repository
        limit: Maksimum sonuç
        
    Returns:
        List[StockMover]: Yükselen hisseler
    """
    # Cache kontrol
    cache_key = f"{CacheKeys.market_movers()}:gainers:{limit}"
    cached = await cache.get_json(cache_key)
    if cached:
        return [StockMover(**item) for item in cached]
    
    stocks = await repo.get_top_gainers(limit=limit)
    
    result = [
        StockMover(
            symbol=stock.symbol,
            name=stock.name,
            last_price=stock.last_price or Decimal("0"),
            day_change_pct=stock.day_change_pct or Decimal("0"),
            last_volume=stock.last_volume,
        )
        for stock in stocks
    ]
    
    # Cache'e kaydet
    await cache.set_json(cache_key, [item.model_dump() for item in result], ttl=CacheTTL.SHORT)
    
    return result


@router.get(
    "/movers/losers",
    response_model=List[StockMover],
    summary="Top Losers",
    description="En çok düşen hisseler.",
)
async def get_top_losers(
    repo: StockRepo,
    limit: int = Query(10, ge=1, le=50, description="Maksimum sonuç sayısı"),
) -> List[StockMover]:
    """
    En çok düşen hisseler.
    
    Args:
        repo: Stock repository
        limit: Maksimum sonuç
        
    Returns:
        List[StockMover]: Düşen hisseler
    """
    # Cache kontrol
    cache_key = f"{CacheKeys.market_movers()}:losers:{limit}"
    cached = await cache.get_json(cache_key)
    if cached:
        return [StockMover(**item) for item in cached]
    
    stocks = await repo.get_top_losers(limit=limit)
    
    result = [
        StockMover(
            symbol=stock.symbol,
            name=stock.name,
            last_price=stock.last_price or Decimal("0"),
            day_change_pct=stock.day_change_pct or Decimal("0"),
            last_volume=stock.last_volume,
        )
        for stock in stocks
    ]
    
    # Cache'e kaydet
    await cache.set_json(cache_key, [item.model_dump() for item in result], ttl=CacheTTL.SHORT)
    
    return result


@router.get(
    "/movers/active",
    response_model=List[StockMover],
    summary="Most Active",
    description="En yüksek hacimli hisseler.",
)
async def get_most_active(
    repo: StockRepo,
    limit: int = Query(10, ge=1, le=50, description="Maksimum sonuç sayısı"),
) -> List[StockMover]:
    """
    En yüksek hacimli hisseler.
    
    Args:
        repo: Stock repository
        limit: Maksimum sonuç
        
    Returns:
        List[StockMover]: Yüksek hacimli hisseler
    """
    stocks = await repo.get_most_active(limit=limit)
    
    return [
        StockMover(
            symbol=stock.symbol,
            name=stock.name,
            last_price=stock.last_price or Decimal("0"),
            day_change_pct=stock.day_change_pct or Decimal("0"),
            last_volume=stock.last_volume,
        )
        for stock in stocks
    ]


# =============================================================================
# SECTORS
# =============================================================================

@router.get(
    "/sectors",
    response_model=List[SectorSummary],
    summary="List Sectors",
    description="Tüm sektörleri ve performanslarını listeler.",
)
async def list_sectors(
    repo: StockRepo,
) -> List[SectorSummary]:
    """
    Sektör listesi ve performansları.
    
    Returns:
        List[SectorSummary]: Sektör özetleri
    """
    # Cache kontrol
    cache_key = CacheKeys.stats_sector_performance()
    cached = await cache.get_json(cache_key)
    if cached:
        return [SectorSummary(**item) for item in cached]
    
    # Veritabanından sektör istatistikleri
    sector_stats = await repo.get_sector_summary()
    
    result = []
    for stats in sector_stats:
        sector_code = stats["sector"]
        if sector_code and sector_code in SECTOR_META:
            meta = SECTOR_META[sector_code]
            result.append(SectorSummary(
                code=sector_code,
                name=meta["name"],
                emoji=meta["emoji"],
                color=meta["color"],
                stock_count=stats["stock_count"],
                avg_change=stats["avg_change"],
                total_market_cap=stats.get("total_market_cap"),
            ))
    
    # Değişime göre sırala
    result.sort(key=lambda x: x.avg_change, reverse=True)
    
    # Cache'e kaydet
    await cache.set_json(cache_key, [item.model_dump() for item in result], ttl=CacheTTL.STATS)
    
    return result


@router.get(
    "/sectors/{sector_code}",
    response_model=SectorDetailResponse,
    summary="Get Sector Detail",
    description="Belirli bir sektörün detayları ve hisseleri.",
)
async def get_sector_detail(
    sector_code: str = Path(..., description="Sektör kodu"),
    repo: StockRepo = Depends(get_stock_repository),
) -> SectorDetailResponse:
    """
    Sektör detayları.
    
    Args:
        sector_code: Sektör kodu
        repo: Stock repository
        
    Returns:
        SectorDetailResponse: Sektör detayları
        
    Raises:
        HTTPException: 404 - Sektör bulunamadı
    """
    sector_code = sector_code.upper()
    
    if sector_code not in SECTOR_META:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Sector not found: {sector_code}"
        )
    
    meta = SECTOR_META[sector_code]
    stocks = await repo.find_by_sector(sector_code)
    
    # İstatistikler
    total_stocks = len(stocks)
    gainers = sum(1 for s in stocks if s.day_change_pct and s.day_change_pct > 0)
    losers = sum(1 for s in stocks if s.day_change_pct and s.day_change_pct < 0)
    avg_change = sum(float(s.day_change_pct or 0) for s in stocks) / total_stocks if total_stocks > 0 else 0
    
    return SectorDetailResponse(
        code=sector_code,
        name=meta["name"],
        emoji=meta["emoji"],
        color=meta["color"],
        stocks=[StockSummary.model_validate(stock) for stock in stocks],
        statistics={
            "total_stocks": total_stocks,
            "gainers": gainers,
            "losers": losers,
            "unchanged": total_stocks - gainers - losers,
            "avg_change": round(avg_change, 2),
        }
    )


# =============================================================================
# SINGLE STOCK
# =============================================================================

@router.get(
    "/{symbol}",
    response_model=StockResponse,
    summary="Get Stock",
    description="Belirli bir hissenin detaylarını getirir.",
)
async def get_stock(
    symbol: str = Path(..., description="Hisse sembolü (örn: THYAO)"),
    repo: StockRepo = Depends(get_stock_repository),
) -> StockResponse:
    """
    Hisse detayları.
    
    Args:
        symbol: Hisse sembolü
        repo: Stock repository
        
    Returns:
        StockResponse: Hisse detayları
        
    Raises:
        HTTPException: 404 - Hisse bulunamadı
    """
    stock = await repo.find_by_symbol(symbol)
    
    if not stock:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Stock not found: {symbol}"
        )
    
    return StockResponse.model_validate(stock)


@router.get(
    "/{symbol}/history",
    summary="Get Stock Price History",
    description="Hisse fiyat geçmişini getirir.",
)
async def get_stock_history(
    symbol: str = Path(..., description="Hisse sembolü"),
    interval: str = Query("1d", description="Zaman dilimi (1h, 4h, 1d, 1w)"),
    period: str = Query("3mo", description="Dönem (1mo, 3mo, 6mo, 1y, 2y)"),
    repo: StockRepo = Depends(get_stock_repository),
) -> dict:
    """
    Hisse fiyat geçmişi.
    
    Args:
        symbol: Hisse sembolü
        interval: Zaman dilimi
        period: Dönem
        repo: Stock repository
        
    Returns:
        dict: OHLCV verisi
    """
    # Hisse var mı kontrol et
    stock = await repo.find_by_symbol(symbol)
    if not stock:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Stock not found: {symbol}"
        )
    
    # Cache kontrol
    cache_key = CacheKeys.market_data(symbol, interval)
    cached = await cache.get_json(cache_key)
    if cached:
        return cached
    
    # TODO: Data engine'den veri çek
    # from app.core import DataEngine
    # data = await DataEngine.get_ohlcv(symbol, interval, period)
    
    # Placeholder response
    result = {
        "symbol": symbol,
        "interval": interval,
        "period": period,
        "data": [],  # OHLCV verisi
        "message": "Historical data will be available when DataEngine is integrated"
    }
    
    return result


# =============================================================================
# ADMIN OPERATIONS
# =============================================================================

@router.post(
    "",
    response_model=StockResponse,
    status_code=status.HTTP_201_CREATED,
    summary="Create Stock",
    description="Yeni hisse ekler (Admin only).",
)
async def create_stock(
    data: StockCreate,
    repo: StockRepo,
    admin: CurrentAdmin,
) -> StockResponse:
    """
    Yeni hisse oluştur.
    
    Args:
        data: Hisse verileri
        repo: Stock repository
        admin: Admin kullanıcı
        
    Returns:
        StockResponse: Oluşturulan hisse
        
    Raises:
        HTTPException: 400 - Hisse zaten var
    """
    # Sembol kontrolü
    existing = await repo.find_by_symbol(data.symbol)
    if existing:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Stock already exists: {data.symbol}"
        )
    
    # Yahoo sembolü oluştur
    yahoo_symbol = data.yahoo_symbol or f"{data.symbol.upper()}.IS"
    
    stock = await repo.create(
        symbol=data.symbol.upper(),
        yahoo_symbol=yahoo_symbol,
        name=data.name,
        sector=data.sector,
        sub_sector=data.sub_sector,
        market_cap=data.market_cap,
        lot_size=data.lot_size,
    )
    
    logger.info(f"Stock created by admin {admin.email}: {stock.symbol}")
    
    return StockResponse.model_validate(stock)


@router.put(
    "/{symbol}",
    response_model=StockResponse,
    summary="Update Stock",
    description="Hisse bilgilerini günceller (Admin only).",
)
async def update_stock(
    data: StockUpdate,
    symbol: str = Path(..., description="Hisse sembolü"),
    repo: StockRepo = Depends(get_stock_repository),
    admin: CurrentAdmin = Depends(get_current_admin_user),
) -> StockResponse:
    """
    Hisse güncelle.
    
    Args:
        data: Güncellenecek veriler
        symbol: Hisse sembolü
        repo: Stock repository
        admin: Admin kullanıcı
        
    Returns:
        StockResponse: Güncellenen hisse
    """
    stock = await repo.find_by_symbol(symbol)
    if not stock:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Stock not found: {symbol}"
        )
    
    update_data = data.model_dump(exclude_unset=True)
    updated_stock = await repo.update(stock.id, **update_data)
    
    logger.info(f"Stock updated by admin {admin.email}: {symbol}")
    
    return StockResponse.model_validate(updated_stock)


@router.post(
    "/prices/bulk-update",
    response_model=SuccessResponse,
    summary="Bulk Update Prices",
    description="Toplu fiyat güncelleme (Admin only).",
)
async def bulk_update_prices(
    updates: List[StockPriceUpdate],
    repo: StockRepo,
    admin: CurrentAdmin,
) -> SuccessResponse:
    """
    Toplu fiyat güncelleme.
    
    Args:
        updates: Fiyat güncellemeleri
        repo: Stock repository
        admin: Admin kullanıcı
        
    Returns:
        SuccessResponse: Güncelleme sonucu
    """
    update_dicts = [u.model_dump() for u in updates]
    count = await repo.bulk_update_prices(update_dicts)
    
    # Cache'i temizle
    await cache.delete_pattern(f"{CacheKeys.PREFIX_MARKET}:*")
    
    logger.info(f"Bulk price update by admin {admin.email}: {count} stocks updated")
    
    return SuccessResponse(
        message=f"Successfully updated {count} stock prices",
        data={"updated_count": count}
    )


@router.delete(
    "/{symbol}",
    response_model=SuccessResponse,
    summary="Delete Stock",
    description="Hisse siler (Admin only, soft delete).",
)
async def delete_stock(
    symbol: str = Path(..., description="Hisse sembolü"),
    repo: StockRepo = Depends(get_stock_repository),
    admin: CurrentAdmin = Depends(get_current_admin_user),
) -> SuccessResponse:
    """
    Hisse sil (soft delete).
    
    Args:
        symbol: Hisse sembolü
        repo: Stock repository
        admin: Admin kullanıcı
        
    Returns:
        SuccessResponse: Silme sonucu
    """
    stock = await repo.find_by_symbol(symbol)
    if not stock:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Stock not found: {symbol}"
        )
    
    await repo.soft_delete(stock.id)
    
    logger.info(f"Stock deleted by admin {admin.email}: {symbol}")
    
    return SuccessResponse(message=f"Stock {symbol} deleted successfully")
