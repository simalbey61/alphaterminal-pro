"""
AlphaTerminal Pro - Strategy Schemas
====================================

AI stratejileri için Pydantic schemas.

Author: AlphaTerminal Team
Version: 1.0.0
"""

from __future__ import annotations

from datetime import datetime
from decimal import Decimal
from typing import Optional, List, Dict, Any
from uuid import UUID

from pydantic import BaseModel, Field, ConfigDict

from app.config import StrategyStatus


# =============================================================================
# CONDITION SCHEMAS
# =============================================================================

class StrategyCondition(BaseModel):
    """Strateji koşulu schema."""
    
    indicator: str = Field(..., description="İndikatör adı")
    operator: str = Field(..., description="Operatör (>, <, ==, >=, <=, between)")
    value: Any = Field(..., description="Karşılaştırma değeri")
    timeframe: str = Field(default="1d", description="Zaman dilimi")
    
    # Opsiyonel parametreler
    period: Optional[int] = Field(None, description="İndikatör periyodu")
    secondary_value: Optional[Any] = Field(None, description="İkinci değer (between için)")


# =============================================================================
# BASE SCHEMAS
# =============================================================================

class StrategyBase(BaseModel):
    """Strategy base schema."""
    
    name: str = Field(..., description="Strateji adı", max_length=100)
    description: Optional[str] = Field(None, description="Strateji açıklaması")
    conditions: List[StrategyCondition] = Field(..., description="Strateji koşulları")


class StrategyCreate(StrategyBase):
    """Strategy oluşturma schema."""
    
    confidence: Decimal = Field(..., ge=0, le=1, description="Güven seviyesi")
    sample_size: int = Field(..., gt=0, description="Örnek sayısı")
    discovery_method: str = Field(..., description="Keşif yöntemi")
    
    stop_loss_atr: Decimal = Field(default=Decimal("1.5"), gt=0)
    take_profit_r: Decimal = Field(default=Decimal("2.0"), gt=0)
    position_size_pct: Decimal = Field(default=Decimal("0.02"), gt=0, le=1)
    
    parent_strategy_ids: Optional[List[UUID]] = None
    generation: int = Field(default=1, ge=1)
    
    metadata: Optional[Dict[str, Any]] = None


class StrategyUpdate(BaseModel):
    """Strategy güncelleme schema."""
    
    model_config = ConfigDict(extra="forbid")
    
    name: Optional[str] = None
    description: Optional[str] = None
    conditions: Optional[List[StrategyCondition]] = None
    
    stop_loss_atr: Optional[Decimal] = None
    take_profit_r: Optional[Decimal] = None
    position_size_pct: Optional[Decimal] = None


# =============================================================================
# RESPONSE SCHEMAS
# =============================================================================

class StrategyResponse(StrategyBase):
    """Strategy response schema."""
    
    model_config = ConfigDict(from_attributes=True)
    
    id: UUID
    version: str
    status: str
    
    confidence: Decimal
    sample_size: int
    discovery_method: str
    
    stop_loss_atr: Decimal
    take_profit_r: Decimal
    position_size_pct: Decimal
    
    parent_strategy_ids: Optional[List[UUID]] = None
    generation: int
    performance_score: Decimal
    
    # Backtest sonuçları
    backtest_win_rate: Optional[Decimal] = None
    backtest_profit_factor: Optional[Decimal] = None
    backtest_sharpe: Optional[Decimal] = None
    backtest_max_drawdown: Optional[Decimal] = None
    backtest_total_trades: Optional[int] = None
    backtest_period_days: Optional[int] = None
    
    # Walk-forward
    walkforward_consistency: Optional[Decimal] = None
    walkforward_windows_passed: Optional[int] = None
    
    # Canlı performans
    live_win_rate: Optional[Decimal] = None
    live_profit_factor: Optional[Decimal] = None
    live_sharpe: Optional[Decimal] = None
    live_total_trades: int
    live_total_pnl: Decimal
    live_wins: int
    live_losses: int
    
    # Tarihler
    approved_at: Optional[datetime] = None
    paused_at: Optional[datetime] = None
    retired_at: Optional[datetime] = None
    last_signal_at: Optional[datetime] = None
    last_evolution_at: Optional[datetime] = None
    
    created_at: datetime
    updated_at: Optional[datetime] = None
    
    metadata: Optional[Dict[str, Any]] = None


class StrategyListResponse(BaseModel):
    """Strategy listesi response schema."""
    
    items: List[StrategyResponse]
    total: int
    page: int = 1
    per_page: int = 20
    pages: int = 1


class StrategySummary(BaseModel):
    """Strategy özet schema (list view için)."""
    
    model_config = ConfigDict(from_attributes=True)
    
    id: UUID
    name: str
    version: str
    status: str
    discovery_method: str
    performance_score: Decimal
    live_win_rate: Optional[Decimal] = None
    live_total_trades: int
    created_at: datetime


# =============================================================================
# BACKTEST SCHEMAS
# =============================================================================

class BacktestRequest(BaseModel):
    """Backtest istek schema."""
    
    strategy_id: Optional[UUID] = None
    conditions: Optional[List[StrategyCondition]] = None
    
    symbols: Optional[List[str]] = Field(None, description="Test edilecek semboller")
    start_date: datetime = Field(..., description="Başlangıç tarihi")
    end_date: Optional[datetime] = Field(None, description="Bitiş tarihi")
    
    initial_capital: Decimal = Field(default=Decimal("100000"), gt=0)
    stop_loss_atr: Decimal = Field(default=Decimal("1.5"), gt=0)
    take_profit_r: Decimal = Field(default=Decimal("2.0"), gt=0)
    position_size_pct: Decimal = Field(default=Decimal("0.02"), gt=0, le=1)
    
    timeframe: str = Field(default="1d")
    commission_pct: Decimal = Field(default=Decimal("0.001"))


class BacktestTrade(BaseModel):
    """Backtest trade schema."""
    
    symbol: str
    entry_date: datetime
    exit_date: datetime
    direction: str
    entry_price: Decimal
    exit_price: Decimal
    quantity: int
    pnl: Decimal
    pnl_pct: Decimal
    exit_reason: str


class BacktestResult(BaseModel):
    """Backtest sonuç schema."""
    
    strategy_id: Optional[UUID] = None
    
    # Genel istatistikler
    total_trades: int
    winning_trades: int
    losing_trades: int
    win_rate: Decimal
    
    # Getiri
    total_return: Decimal
    total_return_pct: Decimal
    avg_return_per_trade: Decimal
    
    # Risk metrikleri
    profit_factor: Decimal
    sharpe_ratio: Decimal
    sortino_ratio: Optional[Decimal] = None
    max_drawdown: Decimal
    max_drawdown_duration: Optional[int] = None
    
    # Trade istatistikleri
    avg_win: Decimal
    avg_loss: Decimal
    largest_win: Decimal
    largest_loss: Decimal
    avg_holding_period: float  # Gün
    
    # Equity curve
    equity_curve: List[Dict[str, Any]]
    
    # Trade detayları
    trades: List[BacktestTrade]
    
    # Meta
    start_date: datetime
    end_date: datetime
    initial_capital: Decimal
    final_capital: Decimal
    execution_time_ms: int


# =============================================================================
# EVOLUTION SCHEMAS
# =============================================================================

class EvolutionLogResponse(BaseModel):
    """Evolution log response schema."""
    
    model_config = ConfigDict(from_attributes=True)
    
    id: UUID
    strategy_id: UUID
    change_type: str
    old_value: Optional[Dict[str, Any]] = None
    new_value: Optional[Dict[str, Any]] = None
    reason: str
    impact_score: Optional[Decimal] = None
    created_at: datetime


class StrategyEvolutionHistory(BaseModel):
    """Strateji evrim geçmişi schema."""
    
    strategy_id: UUID
    strategy_name: str
    current_version: str
    logs: List[EvolutionLogResponse]
    total_evolutions: int


# =============================================================================
# STATISTICS SCHEMAS
# =============================================================================

class StrategyStatistics(BaseModel):
    """Strateji istatistikleri schema."""
    
    total: int
    active: int
    pending: int
    retired: int
    paused: int
    avg_win_rate: float
    avg_performance_score: float
    total_trades: int
    total_pnl: float


class DiscoveryMethodStats(BaseModel):
    """Keşif yöntemi istatistikleri schema."""
    
    method: str
    total: int
    active: int
    avg_win_rate: float
    avg_score: float


class GenerationStats(BaseModel):
    """Jenerasyon istatistikleri schema."""
    
    generation: int
    strategy_count: int
    avg_performance: float
    best_strategy_id: UUID
    best_strategy_name: str
    best_strategy_score: float


# =============================================================================
# APPROVAL SCHEMAS
# =============================================================================

class StrategyApprovalCriteria(BaseModel):
    """Strateji onay kriterleri schema."""
    
    min_win_rate: Decimal = Field(default=Decimal("0.55"))
    min_profit_factor: Decimal = Field(default=Decimal("1.5"))
    min_sharpe_ratio: Decimal = Field(default=Decimal("1.0"))
    max_drawdown: Decimal = Field(default=Decimal("0.20"))
    min_trades: int = Field(default=30)
    min_consistency: Decimal = Field(default=Decimal("0.6"))


class StrategyApprovalResult(BaseModel):
    """Strateji onay sonucu schema."""
    
    strategy_id: UUID
    approved: bool
    criteria_results: Dict[str, bool]
    failing_criteria: List[str]
    recommendations: List[str]
