"""
AlphaTerminal Pro - Market Endpoints
====================================

Piyasa verileri ve genel görünüm endpoint'leri.

Author: AlphaTerminal Team
Version: 1.0.0
"""

from __future__ import annotations

import logging
from typing import Optional, List
from datetime import datetime, date
from decimal import Decimal

from fastapi import APIRouter, Depends, Query
from pydantic import BaseModel, Field

from app.config import settings, SECTOR_META
from app.db.models import MarketRegimeModel
from app.api.dependencies import (
    get_stock_repository,
    get_current_user_optional,
    CurrentUserOptional,
    StockRepo,
    DbSession,
    rate_limiter_default,
)
from app.schemas import MarketStatistics, MarketOverview, StockMover, SectorSummary
from app.cache import cache, CacheKeys, CacheTTL

logger = logging.getLogger(__name__)
router = APIRouter()


class MarketRegimeResponse(BaseModel):
    date: date
    trend_regime: str
    volatility_regime: str
    xu100_close: Optional[Decimal] = None
    xu100_change_pct: Optional[Decimal] = None
    advancing: int = 0
    declining: int = 0
    unchanged: int = 0
    is_bullish: bool = False
    is_bearish: bool = False


class MarketBreadth(BaseModel):
    advancing: int
    declining: int
    unchanged: int
    advance_decline_ratio: float


class IndexData(BaseModel):
    symbol: str
    name: str
    last_price: Decimal
    change: Decimal
    change_pct: Decimal
    timestamp: datetime


@router.get("/overview", response_model=MarketOverview, summary="Market Overview")
async def get_market_overview(repo: StockRepo, user: CurrentUserOptional = None) -> MarketOverview:
    """Piyasa genel görünümü."""
    cache_key = CacheKeys.market_overview()
    cached = await cache.get_json(cache_key)
    if cached:
        return MarketOverview(**cached)
    
    stats = await repo.get_market_statistics()
    statistics = MarketStatistics(**stats)
    
    gainers = await repo.get_top_gainers(limit=5)
    top_gainers = [StockMover(symbol=s.symbol, name=s.name, last_price=s.last_price or Decimal("0"), day_change_pct=s.day_change_pct or Decimal("0"), last_volume=s.last_volume) for s in gainers]
    
    losers = await repo.get_top_losers(limit=5)
    top_losers = [StockMover(symbol=s.symbol, name=s.name, last_price=s.last_price or Decimal("0"), day_change_pct=s.day_change_pct or Decimal("0"), last_volume=s.last_volume) for s in losers]
    
    active = await repo.get_most_active(limit=5)
    most_active = [StockMover(symbol=s.symbol, name=s.name, last_price=s.last_price or Decimal("0"), day_change_pct=s.day_change_pct or Decimal("0"), last_volume=s.last_volume) for s in active]
    
    sector_stats = await repo.get_sector_summary()
    sector_performance = []
    for s in sector_stats:
        code = s["sector"]
        if code and code in SECTOR_META:
            meta = SECTOR_META[code]
            sector_performance.append(SectorSummary(code=code, name=meta["name"], emoji=meta["emoji"], color=meta["color"], stock_count=s["stock_count"], avg_change=s["avg_change"]))
    
    result = MarketOverview(statistics=statistics, top_gainers=top_gainers, top_losers=top_losers, most_active=most_active, sector_performance=sector_performance)
    await cache.set_json(cache_key, result.model_dump(mode="json"), ttl=CacheTTL.SHORT)
    return result


@router.get("/statistics", response_model=MarketStatistics, summary="Market Statistics")
async def get_market_statistics(repo: StockRepo) -> MarketStatistics:
    """Piyasa istatistikleri."""
    stats = await repo.get_market_statistics()
    return MarketStatistics(**stats)


@router.get("/regime", response_model=MarketRegimeResponse, summary="Market Regime")
async def get_market_regime(session: DbSession) -> MarketRegimeResponse:
    """Piyasa rejimi."""
    from sqlalchemy import select, desc
    result = await session.execute(select(MarketRegimeModel).order_by(desc(MarketRegimeModel.date)).limit(1))
    regime = result.scalar_one_or_none()
    
    if not regime:
        return MarketRegimeResponse(date=date.today(), trend_regime="neutral", volatility_regime="normal")
    
    return MarketRegimeResponse(date=regime.date, trend_regime=regime.trend_regime, volatility_regime=regime.volatility_regime, xu100_close=regime.xu100_close, advancing=regime.advancing, declining=regime.declining, is_bullish=regime.is_bullish, is_bearish=regime.is_bearish)


@router.get("/breadth", response_model=MarketBreadth, summary="Market Breadth")
async def get_market_breadth(repo: StockRepo) -> MarketBreadth:
    """Piyasa genişliği."""
    stats = await repo.get_market_statistics()
    return MarketBreadth(advancing=stats["gainers"], declining=stats["losers"], unchanged=stats["unchanged"], advance_decline_ratio=stats["breadth"])


@router.get("/indices", response_model=List[IndexData], summary="Get Indices")
async def get_indices() -> List[IndexData]:
    """BIST endeksleri."""
    return [
        IndexData(symbol="XU100", name="BIST 100", last_price=Decimal("9500"), change=Decimal("120"), change_pct=Decimal("1.28"), timestamp=datetime.utcnow()),
        IndexData(symbol="XU030", name="BIST 30", last_price=Decimal("10200"), change=Decimal("150"), change_pct=Decimal("1.49"), timestamp=datetime.utcnow()),
    ]


@router.get("/screener", summary="Stock Screener")
async def stock_screener(
    repo: StockRepo,
    min_price: Optional[Decimal] = Query(None),
    max_price: Optional[Decimal] = Query(None),
    min_volume: Optional[int] = Query(None),
    sector: Optional[str] = Query(None),
    limit: int = Query(50, ge=1, le=200),
) -> List[dict]:
    """Hisse tarayıcı."""
    all_stocks = await repo.find_all()
    results = []
    
    for stock in all_stocks:
        if not stock.is_active:
            continue
        if min_price and (not stock.last_price or stock.last_price < min_price):
            continue
        if max_price and (not stock.last_price or stock.last_price > max_price):
            continue
        if min_volume and (not stock.last_volume or stock.last_volume < min_volume):
            continue
        if sector and stock.sector != sector.upper():
            continue
        
        results.append({"symbol": stock.symbol, "name": stock.name, "sector": stock.sector, "last_price": float(stock.last_price) if stock.last_price else None, "day_change_pct": float(stock.day_change_pct) if stock.day_change_pct else None})
        if len(results) >= limit:
            break
    
    return results
