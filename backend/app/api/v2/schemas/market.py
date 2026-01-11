"""
AlphaTerminal Pro - API v2 Market Data Schemas
==============================================

Schemas for market data endpoints.

Author: AlphaTerminal Team
Version: 2.0.0
"""

from datetime import datetime
from typing import Optional, Dict, Any, List
from pydantic import Field, field_validator

from app.api.v2.schemas.base import (
    BaseSchema, SymbolParams, MultiSymbolParams,
    IntervalParams, DateRangeParams, PaginationParams
)


# =============================================================================
# OHLCV DATA
# =============================================================================

class OHLCVBar(BaseSchema):
    """Single OHLCV bar."""
    timestamp: datetime
    open: float = Field(..., ge=0)
    high: float = Field(..., ge=0)
    low: float = Field(..., ge=0)
    close: float = Field(..., ge=0)
    volume: int = Field(..., ge=0)
    
    # Optional extended fields
    vwap: Optional[float] = None
    trades: Optional[int] = None
    turnover: Optional[float] = None


class OHLCVResponse(BaseSchema):
    """OHLCV data response."""
    symbol: str
    interval: str
    bars: List[OHLCVBar]
    
    # Metadata
    start_time: datetime
    end_time: datetime
    bar_count: int
    data_source: str
    is_adjusted: bool = True
    
    # Quality info
    has_gaps: bool = False
    quality_score: Optional[float] = None


# =============================================================================
# MARKET DATA REQUEST
# =============================================================================

class MarketDataRequest(SymbolParams, IntervalParams, DateRangeParams):
    """Request for market data."""
    bars: Optional[int] = Field(
        None,
        ge=1,
        le=10000,
        description="Number of bars (alternative to date range)"
    )
    adjusted: bool = Field(
        True,
        description="Return adjusted prices (splits/dividends)"
    )
    include_premarket: bool = Field(
        False,
        description="Include pre-market data"
    )


class BatchMarketDataRequest(MultiSymbolParams, IntervalParams):
    """Batch request for multiple symbols."""
    bars: int = Field(
        default=500,
        ge=1,
        le=5000,
        description="Number of bars per symbol"
    )


# =============================================================================
# SYMBOL INFO
# =============================================================================

class SymbolInfo(BaseSchema):
    """Symbol information."""
    symbol: str
    name: str
    market: str
    symbol_type: str
    sector: Optional[str] = None
    industry: Optional[str] = None
    currency: str
    lot_size: int = 1
    tick_size: float = 0.01
    is_active: bool = True
    
    # Additional info
    isin: Optional[str] = None
    description: Optional[str] = None
    liquidity_tier: Optional[str] = None


class SymbolSearchRequest(BaseSchema):
    """Symbol search request."""
    query: str = Field(
        ...,
        min_length=1,
        max_length=50,
        description="Search query"
    )
    market: Optional[str] = Field(
        None,
        description="Filter by market (BIST, NYSE, etc.)"
    )
    symbol_type: Optional[str] = Field(
        None,
        description="Filter by type (stock, etf, etc.)"
    )
    limit: int = Field(
        default=20,
        ge=1,
        le=100,
        description="Maximum results"
    )


class SymbolListRequest(PaginationParams):
    """Symbol list request with filters."""
    market: Optional[str] = None
    sector: Optional[str] = None
    index: Optional[str] = Field(
        None,
        description="Filter by index (XU030, XU100, etc.)"
    )
    is_active: bool = True


# =============================================================================
# QUOTES
# =============================================================================

class Quote(BaseSchema):
    """Real-time quote data."""
    symbol: str
    last_price: float
    change: float
    change_pct: float
    
    # OHLC for current session
    open: float
    high: float
    low: float
    close: float
    
    # Volume
    volume: int
    avg_volume: Optional[int] = None
    
    # Bid/Ask
    bid: Optional[float] = None
    bid_size: Optional[int] = None
    ask: Optional[float] = None
    ask_size: Optional[int] = None
    
    # Timestamps
    last_trade_time: Optional[datetime] = None
    market_status: str = "open"
    
    # Extended info
    previous_close: Optional[float] = None
    week_52_high: Optional[float] = None
    week_52_low: Optional[float] = None


class QuoteRequest(SymbolParams):
    """Single quote request."""
    pass


class BatchQuoteRequest(MultiSymbolParams):
    """Batch quote request."""
    pass


# =============================================================================
# MARKET OVERVIEW
# =============================================================================

class MarketIndexData(BaseSchema):
    """Market index data."""
    symbol: str
    name: str
    value: float
    change: float
    change_pct: float
    volume: Optional[int] = None
    timestamp: datetime


class MarketOverview(BaseSchema):
    """Market overview data."""
    market: str
    status: str
    timestamp: datetime
    
    # Indices
    indices: List[MarketIndexData]
    
    # Market breadth
    advancing: int
    declining: int
    unchanged: int
    
    # Volume
    total_volume: int
    total_turnover: Optional[float] = None
    
    # Top movers
    top_gainers: List[Dict[str, Any]]
    top_losers: List[Dict[str, Any]]
    most_active: List[Dict[str, Any]]


class MarketOverviewRequest(BaseSchema):
    """Market overview request."""
    market: str = Field(
        default="BIST",
        description="Market code"
    )
    top_count: int = Field(
        default=10,
        ge=1,
        le=50,
        description="Number of top movers to include"
    )


# =============================================================================
# FUNDAMENTALS
# =============================================================================

class FundamentalData(BaseSchema):
    """Fundamental data for a symbol."""
    symbol: str
    name: Optional[str] = None
    
    # Valuation
    market_cap: Optional[float] = None
    enterprise_value: Optional[float] = None
    pe_ratio: Optional[float] = None
    forward_pe: Optional[float] = None
    peg_ratio: Optional[float] = None
    price_to_book: Optional[float] = None
    price_to_sales: Optional[float] = None
    
    # Profitability
    profit_margin: Optional[float] = None
    operating_margin: Optional[float] = None
    return_on_equity: Optional[float] = None
    return_on_assets: Optional[float] = None
    
    # Growth
    revenue_growth: Optional[float] = None
    earnings_growth: Optional[float] = None
    
    # Dividends
    dividend_yield: Optional[float] = None
    payout_ratio: Optional[float] = None
    
    # Financial health
    current_ratio: Optional[float] = None
    debt_to_equity: Optional[float] = None
    
    # Stock info
    beta: Optional[float] = None
    shares_outstanding: Optional[int] = None
    float_shares: Optional[int] = None
    
    # Dates
    last_updated: Optional[datetime] = None


# =============================================================================
# TECHNICAL INDICATORS
# =============================================================================

class IndicatorRequest(SymbolParams, IntervalParams):
    """Technical indicator request."""
    indicator: str = Field(
        ...,
        description="Indicator name (sma, ema, rsi, macd, etc.)"
    )
    params: Optional[Dict[str, Any]] = Field(
        default={},
        description="Indicator parameters"
    )
    bars: int = Field(
        default=100,
        ge=10,
        le=1000
    )


class IndicatorValue(BaseSchema):
    """Single indicator value."""
    timestamp: datetime
    value: float
    signal: Optional[float] = None  # For MACD-like indicators
    histogram: Optional[float] = None


class IndicatorResponse(BaseSchema):
    """Technical indicator response."""
    symbol: str
    indicator: str
    interval: str
    params: Dict[str, Any]
    values: List[IndicatorValue]


class MultiIndicatorRequest(SymbolParams, IntervalParams):
    """Request multiple indicators."""
    indicators: List[Dict[str, Any]] = Field(
        ...,
        min_length=1,
        max_length=10,
        description="List of indicators with params"
    )
    bars: int = Field(default=100)


# =============================================================================
# ALL EXPORTS
# =============================================================================

__all__ = [
    # OHLCV
    "OHLCVBar",
    "OHLCVResponse",
    "MarketDataRequest",
    "BatchMarketDataRequest",
    
    # Symbol
    "SymbolInfo",
    "SymbolSearchRequest",
    "SymbolListRequest",
    
    # Quotes
    "Quote",
    "QuoteRequest",
    "BatchQuoteRequest",
    
    # Market
    "MarketIndexData",
    "MarketOverview",
    "MarketOverviewRequest",
    
    # Fundamentals
    "FundamentalData",
    
    # Indicators
    "IndicatorRequest",
    "IndicatorValue",
    "IndicatorResponse",
    "MultiIndicatorRequest",
]
