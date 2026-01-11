"""
AlphaTerminal Pro - Stock Schemas
=================================

Hisse senetleri için Pydantic schemas.

Author: AlphaTerminal Team
Version: 1.0.0
"""

from __future__ import annotations

from datetime import datetime
from decimal import Decimal
from typing import Optional, List
from uuid import UUID

from pydantic import BaseModel, Field, ConfigDict


# =============================================================================
# BASE SCHEMAS
# =============================================================================

class StockBase(BaseModel):
    """Stock base schema."""
    
    symbol: str = Field(..., description="Hisse sembolü", example="THYAO")
    name: Optional[str] = Field(None, description="Şirket adı")
    sector: Optional[str] = Field(None, description="Sektör kodu")


class StockCreate(StockBase):
    """Stock oluşturma schema."""
    
    yahoo_symbol: Optional[str] = Field(None, description="Yahoo Finance sembolü")
    sub_sector: Optional[str] = Field(None, description="Alt sektör")
    market_cap: Optional[Decimal] = Field(None, description="Piyasa değeri")
    lot_size: int = Field(default=1, description="Lot büyüklüğü")


class StockUpdate(BaseModel):
    """Stock güncelleme schema."""
    
    model_config = ConfigDict(extra="forbid")
    
    name: Optional[str] = None
    sector: Optional[str] = None
    sub_sector: Optional[str] = None
    market_cap: Optional[Decimal] = None
    is_active: Optional[bool] = None


class StockPriceUpdate(BaseModel):
    """Fiyat güncelleme schema."""
    
    symbol: str
    last_price: Decimal
    last_volume: Optional[int] = None
    day_change: Optional[Decimal] = None
    day_change_pct: Optional[Decimal] = None


# =============================================================================
# RESPONSE SCHEMAS
# =============================================================================

class StockResponse(StockBase):
    """Stock response schema."""
    
    model_config = ConfigDict(from_attributes=True)
    
    id: UUID
    yahoo_symbol: str
    sub_sector: Optional[str] = None
    market_cap: Optional[Decimal] = None
    free_float: Optional[Decimal] = None
    lot_size: int = 1
    
    last_price: Optional[Decimal] = None
    last_volume: Optional[int] = None
    last_updated_at: Optional[datetime] = None
    
    day_change: Optional[Decimal] = None
    day_change_pct: Optional[Decimal] = None
    week_change_pct: Optional[Decimal] = None
    month_change_pct: Optional[Decimal] = None
    year_change_pct: Optional[Decimal] = None
    
    atr: Optional[Decimal] = None
    rsi: Optional[Decimal] = None
    
    is_active: bool = True
    created_at: datetime
    updated_at: Optional[datetime] = None


class StockListResponse(BaseModel):
    """Stock listesi response schema."""
    
    items: List[StockResponse]
    total: int
    page: int = 1
    per_page: int = 20
    pages: int = 1


class StockSummary(BaseModel):
    """Stock özet schema (list view için)."""
    
    model_config = ConfigDict(from_attributes=True)
    
    id: UUID
    symbol: str
    name: Optional[str] = None
    sector: Optional[str] = None
    last_price: Optional[Decimal] = None
    day_change_pct: Optional[Decimal] = None
    last_volume: Optional[int] = None


class StockMover(BaseModel):
    """Top mover schema."""
    
    symbol: str
    name: Optional[str] = None
    last_price: Decimal
    day_change_pct: Decimal
    last_volume: Optional[int] = None


# =============================================================================
# SECTOR SCHEMAS
# =============================================================================

class SectorSummary(BaseModel):
    """Sektör özet schema."""
    
    code: str = Field(..., description="Sektör kodu")
    name: str = Field(..., description="Sektör adı")
    emoji: str = Field(..., description="Sektör emoji")
    color: str = Field(..., description="Sektör rengi")
    stock_count: int = Field(..., description="Hisse sayısı")
    avg_change: float = Field(..., description="Ortalama değişim")
    total_market_cap: Optional[float] = Field(None, description="Toplam piyasa değeri")


class SectorDetailResponse(BaseModel):
    """Sektör detay response schema."""
    
    code: str
    name: str
    emoji: str
    color: str
    stocks: List[StockSummary]
    statistics: dict


# =============================================================================
# MARKET SCHEMAS
# =============================================================================

class MarketStatistics(BaseModel):
    """Piyasa istatistikleri schema."""
    
    total_stocks: int
    gainers: int
    losers: int
    unchanged: int
    average_change: float
    total_volume: int
    total_market_cap: float
    breadth: float = Field(..., description="Yükselen/Toplam oranı")


class MarketOverview(BaseModel):
    """Piyasa genel görünümü schema."""
    
    statistics: MarketStatistics
    top_gainers: List[StockMover]
    top_losers: List[StockMover]
    most_active: List[StockMover]
    sector_performance: List[SectorSummary]
    timestamp: datetime = Field(default_factory=datetime.utcnow)
