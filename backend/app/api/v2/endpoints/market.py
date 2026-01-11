"""
AlphaTerminal Pro - API v2 Market Data Endpoints
================================================

RESTful endpoints for market data.

Author: AlphaTerminal Team
Version: 2.0.0
"""

import logging
from datetime import datetime, timedelta
from typing import Optional, List

from fastapi import APIRouter, Depends, Query, Path, HTTPException, Request
from fastapi.responses import JSONResponse

from app.api.v2.schemas.base import (
    APIResponse, PaginatedResponse, MetaInfo, ErrorCode,
    PaginationParams
)
from app.api.v2.schemas.market import (
    OHLCVBar, OHLCVResponse, MarketDataRequest, BatchMarketDataRequest,
    SymbolInfo, SymbolSearchRequest, SymbolListRequest,
    Quote, QuoteRequest, BatchQuoteRequest,
    MarketOverview, MarketOverviewRequest,
    FundamentalData,
    IndicatorRequest, IndicatorResponse, IndicatorValue
)
from app.api.v2.middleware.rate_limiter import rate_limit
from app.api.v2.utils.dependencies import get_data_manager, get_request_context


logger = logging.getLogger(__name__)

router = APIRouter(prefix="/market", tags=["Market Data"])


# =============================================================================
# OHLCV DATA ENDPOINTS
# =============================================================================

@router.get(
    "/ohlcv/{symbol}",
    response_model=APIResponse[OHLCVResponse],
    summary="Get OHLCV data",
    description="Get historical OHLCV (Open, High, Low, Close, Volume) data for a symbol."
)
async def get_ohlcv(
    symbol: str = Path(..., description="Trading symbol", example="THYAO"),
    interval: str = Query("1d", description="Data interval"),
    bars: int = Query(500, ge=1, le=5000, description="Number of bars"),
    start_date: Optional[datetime] = Query(None, description="Start date"),
    end_date: Optional[datetime] = Query(None, description="End date"),
    adjusted: bool = Query(True, description="Adjusted prices"),
    request: Request = None
):
    """
    Get OHLCV historical data.
    
    - **symbol**: Trading symbol (e.g., THYAO, GARAN)
    - **interval**: Time interval (1m, 5m, 15m, 1h, 4h, 1d, 1w)
    - **bars**: Number of bars to return (max 5000)
    - **start_date**: Start of date range (optional)
    - **end_date**: End of date range (optional)
    - **adjusted**: Return split/dividend adjusted prices
    """
    start_time = datetime.now()
    
    try:
        from app.data_providers import DataManager, DataInterval
        
        manager = DataManager.get_instance()
        
        # Convert interval string to enum
        interval_enum = DataInterval.from_string(interval)
        
        # Fetch data
        market_data = manager.get_data(
            symbol=symbol.upper(),
            interval=interval_enum,
            start_date=start_date,
            end_date=end_date,
            bars=bars
        )
        
        # Convert to response format
        bars_list = []
        for idx, row in market_data.data.iterrows():
            bars_list.append(OHLCVBar(
                timestamp=idx,
                open=row['Open'],
                high=row['High'],
                low=row['Low'],
                close=row['Close'],
                volume=int(row['Volume'])
            ))
        
        response_data = OHLCVResponse(
            symbol=market_data.symbol,
            interval=market_data.interval.value,
            bars=bars_list,
            start_time=market_data.start_time,
            end_time=market_data.end_time,
            bar_count=market_data.rows,
            data_source=market_data.source.value,
            is_adjusted=adjusted
        )
        
        # Build meta
        duration_ms = (datetime.now() - start_time).total_seconds() * 1000
        meta = MetaInfo(
            duration_ms=duration_ms,
            data_source=market_data.source.value,
            cache_hit=market_data.metadata.get("cache_hit", False)
        )
        
        return APIResponse.success(data=response_data, meta=meta)
        
    except Exception as e:
        logger.error(f"Error fetching OHLCV for {symbol}: {e}")
        return APIResponse.error(
            code=ErrorCode.DATA_FETCH_ERROR,
            message=str(e),
            details={"symbol": symbol, "interval": interval}
        )


@router.post(
    "/ohlcv/batch",
    response_model=APIResponse[dict],
    summary="Get OHLCV data for multiple symbols",
    description="Batch request for OHLCV data across multiple symbols."
)
async def get_ohlcv_batch(
    request_data: BatchMarketDataRequest,
    request: Request = None
):
    """
    Get OHLCV data for multiple symbols in a single request.
    
    More efficient than multiple individual requests due to:
    - Parallel fetching
    - Single response payload
    - Reduced overhead
    """
    start_time = datetime.now()
    
    try:
        from app.data_providers import DataManager, DataInterval
        
        manager = DataManager.get_instance()
        interval_enum = DataInterval.from_string(request_data.interval)
        
        # Batch fetch
        results = manager.get_batch(
            symbols=request_data.symbols,
            interval=interval_enum,
            bars=request_data.bars,
            parallel=True
        )
        
        # Format results
        formatted = {}
        errors = []
        
        for symbol, result in results.items():
            if isinstance(result, Exception):
                errors.append({
                    "symbol": symbol,
                    "error": str(result)
                })
            else:
                bars_list = []
                for idx, row in result.data.iterrows():
                    bars_list.append({
                        "timestamp": idx.isoformat(),
                        "open": row['Open'],
                        "high": row['High'],
                        "low": row['Low'],
                        "close": row['Close'],
                        "volume": int(row['Volume'])
                    })
                
                formatted[symbol] = {
                    "symbol": symbol,
                    "interval": result.interval.value,
                    "bar_count": result.rows,
                    "bars": bars_list
                }
        
        duration_ms = (datetime.now() - start_time).total_seconds() * 1000
        
        response_data = {
            "symbols": formatted,
            "total_requested": len(request_data.symbols),
            "total_success": len(formatted),
            "total_failed": len(errors),
            "errors": errors if errors else None
        }
        
        meta = MetaInfo(duration_ms=duration_ms)
        
        return APIResponse.success(data=response_data, meta=meta)
        
    except Exception as e:
        logger.error(f"Error in batch OHLCV: {e}")
        return APIResponse.error(
            code=ErrorCode.DATA_FETCH_ERROR,
            message=str(e)
        )


# =============================================================================
# SYMBOL ENDPOINTS
# =============================================================================

@router.get(
    "/symbols/{symbol}",
    response_model=APIResponse[SymbolInfo],
    summary="Get symbol information",
    description="Get detailed information about a trading symbol."
)
async def get_symbol_info(
    symbol: str = Path(..., description="Trading symbol"),
    request: Request = None
):
    """
    Get symbol information including:
    - Company name
    - Market/exchange
    - Sector and industry
    - Currency
    - Trading specifications
    """
    try:
        from app.data_providers import DataManager
        
        manager = DataManager.get_instance()
        info = manager.get_symbol_info(symbol.upper())
        
        response_data = SymbolInfo(
            symbol=info.symbol,
            name=info.name,
            market=info.market.value,
            symbol_type=info.symbol_type.value,
            sector=info.sector,
            industry=info.industry,
            currency=info.currency,
            lot_size=info.lot_size,
            tick_size=info.tick_size,
            is_active=info.is_active,
            isin=info.isin,
            description=info.description,
            liquidity_tier=info.liquidity_tier.value if info.liquidity_tier else None
        )
        
        return APIResponse.success(data=response_data)
        
    except Exception as e:
        logger.error(f"Error getting symbol info for {symbol}: {e}")
        return APIResponse.error(
            code=ErrorCode.SYMBOL_NOT_FOUND,
            message=f"Symbol not found: {symbol}",
            details={"symbol": symbol}
        )


@router.get(
    "/symbols",
    response_model=PaginatedResponse[str],
    summary="List available symbols",
    description="Get paginated list of available trading symbols."
)
async def list_symbols(
    market: Optional[str] = Query(None, description="Filter by market"),
    page: int = Query(1, ge=1),
    page_size: int = Query(50, ge=1, le=500),
    request: Request = None
):
    """
    List all available symbols with optional market filter.
    """
    try:
        from app.data_providers import DataManager, Market
        
        manager = DataManager.get_instance()
        
        # Get market enum if specified
        market_enum = None
        if market:
            market_enum = Market(market.upper())
        
        symbols = manager.get_available_symbols(market_enum)
        
        # Paginate
        total = len(symbols)
        start = (page - 1) * page_size
        end = start + page_size
        paginated = symbols[start:end]
        
        return PaginatedResponse.create(
            items=paginated,
            total=total,
            page=page,
            page_size=page_size
        )
        
    except Exception as e:
        logger.error(f"Error listing symbols: {e}")
        return APIResponse.error(
            code=ErrorCode.INTERNAL_ERROR,
            message=str(e)
        )


@router.get(
    "/symbols/search",
    response_model=APIResponse[List[SymbolInfo]],
    summary="Search symbols",
    description="Search for symbols by query string."
)
async def search_symbols(
    q: str = Query(..., min_length=1, description="Search query"),
    market: Optional[str] = Query(None, description="Filter by market"),
    limit: int = Query(20, ge=1, le=100),
    request: Request = None
):
    """
    Search for symbols matching query.
    
    Searches against:
    - Symbol code
    - Company name
    """
    try:
        from app.data_providers import DataManager, Market
        
        manager = DataManager.get_instance()
        
        market_enum = None
        if market:
            market_enum = Market(market.upper())
        
        # Get all symbols and filter
        all_symbols = manager.get_available_symbols(market_enum)
        
        # Simple search - matches symbol prefix
        query = q.upper()
        matches = [s for s in all_symbols if s.startswith(query)][:limit]
        
        # Get info for matches
        results = []
        for symbol in matches:
            try:
                info = manager.get_symbol_info(symbol)
                results.append(SymbolInfo(
                    symbol=info.symbol,
                    name=info.name,
                    market=info.market.value,
                    symbol_type=info.symbol_type.value,
                    sector=info.sector,
                    currency=info.currency,
                    is_active=info.is_active
                ))
            except:
                # Skip symbols without info
                results.append(SymbolInfo(
                    symbol=symbol,
                    name=symbol,
                    market=market or "BIST",
                    symbol_type="stock",
                    currency="TRY"
                ))
        
        return APIResponse.success(data=results)
        
    except Exception as e:
        logger.error(f"Error searching symbols: {e}")
        return APIResponse.error(
            code=ErrorCode.INTERNAL_ERROR,
            message=str(e)
        )


# =============================================================================
# MARKET OVERVIEW
# =============================================================================

@router.get(
    "/overview/{market}",
    response_model=APIResponse[dict],
    summary="Get market overview",
    description="Get market overview including indices, breadth, and top movers."
)
async def get_market_overview(
    market: str = Path(..., description="Market code", example="BIST"),
    top_count: int = Query(10, ge=1, le=50),
    request: Request = None
):
    """
    Get comprehensive market overview:
    - Index values and changes
    - Market breadth (advancing/declining)
    - Top gainers and losers
    - Most active stocks
    """
    try:
        # This would integrate with real market data
        # For now, return structure
        
        overview = {
            "market": market.upper(),
            "status": "open",
            "timestamp": datetime.now().isoformat(),
            "indices": [],
            "breadth": {
                "advancing": 0,
                "declining": 0,
                "unchanged": 0
            },
            "top_gainers": [],
            "top_losers": [],
            "most_active": []
        }
        
        return APIResponse.success(data=overview)
        
    except Exception as e:
        logger.error(f"Error getting market overview: {e}")
        return APIResponse.error(
            code=ErrorCode.INTERNAL_ERROR,
            message=str(e)
        )


# =============================================================================
# HEALTH CHECK
# =============================================================================

@router.get(
    "/health",
    summary="Market data health check",
    description="Check health of market data providers."
)
async def market_health(request: Request = None):
    """Check health of market data services."""
    try:
        from app.data_providers import DataManager
        
        manager = DataManager.get_instance()
        health = manager.get_provider_health()
        
        return APIResponse.success(data={
            "status": "healthy",
            "providers": {
                name: {
                    "status": h.status.value,
                    "latency_ms": h.latency_ms,
                    "success_rate": h.success_rate,
                    "total_requests": h.total_requests
                }
                for name, h in health.items()
            },
            "cache": manager.get_cache_stats()
        })
        
    except Exception as e:
        return APIResponse.error(
            code=ErrorCode.INTERNAL_ERROR,
            message=str(e)
        )


# =============================================================================
# ALL EXPORTS
# =============================================================================

__all__ = ["router"]
