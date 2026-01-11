"""
AlphaTerminal Pro - Signal Schemas
==================================

Trading sinyalleri için Pydantic schemas.

Author: AlphaTerminal Team
Version: 1.0.0
"""

from __future__ import annotations

from datetime import datetime
from decimal import Decimal
from typing import Optional, List, Dict, Any
from uuid import UUID

from pydantic import BaseModel, Field, ConfigDict, field_validator

from app.config import SignalType, SignalTier, SignalStatus


# =============================================================================
# BASE SCHEMAS
# =============================================================================

class SignalBase(BaseModel):
    """Signal base schema."""
    
    symbol: str = Field(..., description="Hisse sembolü")
    signal_type: SignalType = Field(..., description="Sinyal türü")
    entry_price: Decimal = Field(..., description="Giriş fiyatı", gt=0)
    stop_loss: Decimal = Field(..., description="Stop loss fiyatı", gt=0)
    take_profit_1: Decimal = Field(..., description="Take profit 1", gt=0)


class SignalCreate(SignalBase):
    """Signal oluşturma schema."""
    
    stock_id: Optional[UUID] = None
    strategy_id: Optional[UUID] = None
    tier: SignalTier = SignalTier.TIER3
    
    take_profit_2: Optional[Decimal] = Field(None, gt=0)
    take_profit_3: Optional[Decimal] = Field(None, gt=0)
    
    total_score: Decimal = Field(..., ge=0, le=100)
    smc_score: Optional[Decimal] = Field(None, ge=0, le=100)
    orderflow_score: Optional[Decimal] = Field(None, ge=0, le=100)
    alpha_score: Optional[Decimal] = Field(None, ge=0, le=100)
    ml_score: Optional[Decimal] = Field(None, ge=0, le=100)
    mtf_score: Optional[Decimal] = Field(None, ge=0, le=100)
    
    confidence: Decimal = Field(..., ge=0, le=1)
    risk_reward: Decimal = Field(..., gt=0)
    setup_type: str = Field(..., description="Setup türü")
    timeframe: str = Field(default="4h")
    
    reasoning: Optional[List[str]] = None
    smc_data: Optional[Dict[str, Any]] = None
    orderflow_data: Optional[Dict[str, Any]] = None
    
    valid_until: Optional[datetime] = None
    
    @field_validator("stop_loss")
    @classmethod
    def validate_stop_loss(cls, v: Decimal, info) -> Decimal:
        """Stop loss fiyatını doğrula."""
        entry = info.data.get("entry_price")
        signal_type = info.data.get("signal_type")
        
        if entry and signal_type:
            if signal_type == SignalType.LONG and v >= entry:
                raise ValueError("LONG sinyalinde stop loss entry'den düşük olmalı")
            if signal_type == SignalType.SHORT and v <= entry:
                raise ValueError("SHORT sinyalinde stop loss entry'den yüksek olmalı")
        
        return v


class SignalUpdate(BaseModel):
    """Signal güncelleme schema."""
    
    model_config = ConfigDict(extra="forbid")
    
    stop_loss: Optional[Decimal] = None
    take_profit_1: Optional[Decimal] = None
    take_profit_2: Optional[Decimal] = None
    take_profit_3: Optional[Decimal] = None
    valid_until: Optional[datetime] = None


class SignalClose(BaseModel):
    """Signal kapatma schema."""
    
    exit_price: Decimal = Field(..., gt=0)
    reason: str = Field(..., description="Kapanış sebebi")
    pnl: Optional[Decimal] = None


# =============================================================================
# RESPONSE SCHEMAS
# =============================================================================

class SignalResponse(SignalBase):
    """Signal response schema."""
    
    model_config = ConfigDict(from_attributes=True)
    
    id: UUID
    stock_id: Optional[UUID] = None
    strategy_id: Optional[UUID] = None
    
    tier: str
    status: str
    
    take_profit_2: Optional[Decimal] = None
    take_profit_3: Optional[Decimal] = None
    
    total_score: Decimal
    smc_score: Optional[Decimal] = None
    orderflow_score: Optional[Decimal] = None
    alpha_score: Optional[Decimal] = None
    ml_score: Optional[Decimal] = None
    mtf_score: Optional[Decimal] = None
    
    confidence: Decimal
    risk_reward: Decimal
    setup_type: str
    timeframe: str
    
    reasoning: Optional[List[str]] = None
    smc_data: Optional[Dict[str, Any]] = None
    orderflow_data: Optional[Dict[str, Any]] = None
    metadata: Optional[Dict[str, Any]] = None
    
    result_pnl: Optional[Decimal] = None
    result_pnl_pct: Optional[Decimal] = None
    exit_price: Optional[Decimal] = None
    closed_at: Optional[datetime] = None
    closed_reason: Optional[str] = None
    
    valid_until: Optional[datetime] = None
    created_at: datetime
    updated_at: Optional[datetime] = None


class SignalListResponse(BaseModel):
    """Signal listesi response schema."""
    
    items: List[SignalResponse]
    total: int
    page: int = 1
    per_page: int = 20
    pages: int = 1


class SignalSummary(BaseModel):
    """Signal özet schema (list view için)."""
    
    model_config = ConfigDict(from_attributes=True)
    
    id: UUID
    symbol: str
    signal_type: str
    tier: str
    status: str
    entry_price: Decimal
    total_score: Decimal
    risk_reward: Decimal
    created_at: datetime


# =============================================================================
# FILTER SCHEMAS
# =============================================================================

class SignalFilter(BaseModel):
    """Signal filtre schema."""
    
    symbol: Optional[str] = None
    signal_type: Optional[SignalType] = None
    tier: Optional[SignalTier] = None
    status: Optional[SignalStatus] = None
    strategy_id: Optional[UUID] = None
    min_score: Optional[float] = Field(None, ge=0, le=100)
    max_score: Optional[float] = Field(None, ge=0, le=100)
    start_date: Optional[datetime] = None
    end_date: Optional[datetime] = None


# =============================================================================
# STATISTICS SCHEMAS
# =============================================================================

class SignalPerformanceStats(BaseModel):
    """Sinyal performans istatistikleri schema."""
    
    total_signals: int
    closed_signals: int
    active_signals: int
    profitable_signals: int
    stopped_signals: int
    win_rate: float
    average_pnl_pct: float
    total_pnl: float
    period_days: int


class TierPerformance(BaseModel):
    """Tier bazlı performans schema."""
    
    tier: str
    total: int
    wins: int
    win_rate: float
    avg_pnl: float


class SignalDistribution(BaseModel):
    """Sinyal dağılımı schema."""
    
    by_status: Dict[str, int]
    by_tier: Dict[str, int]
    by_type: Dict[str, int]


class SymbolPerformance(BaseModel):
    """Sembol bazlı performans schema."""
    
    symbol: str
    total_signals: int
    wins: int
    win_rate: float
    total_pnl_pct: float


# =============================================================================
# TELEGRAM SCHEMAS
# =============================================================================

class SignalTelegramMessage(BaseModel):
    """Telegram mesaj schema."""
    
    signal_id: UUID
    message: str
    photo_url: Optional[str] = None
    buttons: Optional[List[Dict[str, str]]] = None


class SignalAlert(BaseModel):
    """Sinyal uyarı schema."""
    
    signal_id: UUID
    symbol: str
    alert_type: str = Field(..., description="stop_loss, take_profit_1, etc.")
    current_price: Decimal
    target_price: Decimal
    message: str
    timestamp: datetime = Field(default_factory=datetime.utcnow)
