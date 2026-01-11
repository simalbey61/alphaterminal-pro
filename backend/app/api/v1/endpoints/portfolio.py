"""
AlphaTerminal Pro - Portfolio Endpoints
=======================================

Portfolio ve pozisyon yönetimi endpoint'leri.

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
from pydantic import BaseModel, Field
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.config import settings
from app.db.database import get_session
from app.db.models import PortfolioModel, PositionModel
from app.api.dependencies import (
    get_current_user,
    CurrentUser,
    DbSession,
    Pagination,
    rate_limiter_default,
)
from app.schemas import SuccessResponse

logger = logging.getLogger(__name__)
router = APIRouter()


# =============================================================================
# SCHEMAS
# =============================================================================

class PortfolioCreate(BaseModel):
    """Portfolio oluşturma schema."""
    
    name: str = Field(..., max_length=100)
    description: Optional[str] = None
    capital: Decimal = Field(..., gt=0)
    is_paper: bool = Field(default=True)
    max_positions: int = Field(default=5, ge=1, le=20)
    max_risk_per_trade: float = Field(default=0.02, gt=0, le=0.1)


class PortfolioUpdate(BaseModel):
    """Portfolio güncelleme schema."""
    
    name: Optional[str] = None
    description: Optional[str] = None
    max_positions: Optional[int] = None
    max_risk_per_trade: Optional[float] = None


class PortfolioResponse(BaseModel):
    """Portfolio response schema."""
    
    id: UUID
    name: str
    description: Optional[str] = None
    capital: Decimal
    current_value: Decimal
    available_capital: Decimal
    is_default: bool
    is_paper: bool
    max_positions: int
    max_risk_per_trade: Decimal
    total_pnl: Decimal
    total_pnl_pct: Decimal
    total_trades: int
    winning_trades: int
    losing_trades: int
    max_drawdown: Decimal
    sharpe_ratio: Optional[Decimal] = None
    win_rate: Optional[float] = None
    open_positions_count: int = 0
    created_at: datetime
    
    class Config:
        from_attributes = True


class PositionCreate(BaseModel):
    """Pozisyon açma schema."""
    
    symbol: str
    direction: str = Field(..., description="LONG or SHORT")
    quantity: int = Field(..., gt=0)
    entry_price: Decimal = Field(..., gt=0)
    stop_loss: Decimal = Field(..., gt=0)
    take_profit: Decimal = Field(..., gt=0)
    signal_id: Optional[UUID] = None
    notes: Optional[str] = None


class PositionUpdate(BaseModel):
    """Pozisyon güncelleme schema."""
    
    stop_loss: Optional[Decimal] = None
    take_profit: Optional[Decimal] = None
    trailing_stop: Optional[Decimal] = None
    notes: Optional[str] = None


class PositionClose(BaseModel):
    """Pozisyon kapatma schema."""
    
    exit_price: Decimal = Field(..., gt=0)
    reason: str = Field(..., description="Kapatma sebebi")


class PositionResponse(BaseModel):
    """Pozisyon response schema."""
    
    id: UUID
    portfolio_id: UUID
    signal_id: Optional[UUID] = None
    symbol: str
    direction: str
    quantity: int
    entry_price: Decimal
    current_price: Optional[Decimal] = None
    stop_loss: Decimal
    take_profit: Decimal
    trailing_stop: Optional[Decimal] = None
    unrealized_pnl: Decimal
    unrealized_pnl_pct: Decimal
    realized_pnl: Optional[Decimal] = None
    realized_pnl_pct: Optional[Decimal] = None
    position_value: Decimal
    risk_amount: Decimal
    status: str
    exit_price: Optional[Decimal] = None
    exit_reason: Optional[str] = None
    closed_at: Optional[datetime] = None
    holding_duration: Optional[int] = None
    notes: Optional[str] = None
    created_at: datetime
    
    class Config:
        from_attributes = True


class PortfolioSummary(BaseModel):
    """Portfolio özet schema."""
    
    total_value: Decimal
    total_pnl: Decimal
    total_pnl_pct: Decimal
    open_positions: int
    total_risk: Decimal
    remaining_risk_budget: Decimal
    day_pnl: Decimal
    week_pnl: Decimal
    month_pnl: Decimal


# =============================================================================
# PORTFOLIO ENDPOINTS
# =============================================================================

@router.get(
    "",
    response_model=List[PortfolioResponse],
    summary="List Portfolios",
    description="Kullanıcının portfolyolarını listeler.",
)
async def list_portfolios(
    user: CurrentUser,
    session: DbSession,
) -> List[PortfolioResponse]:
    """Portfolio listesi."""
    result = await session.execute(
        select(PortfolioModel)
        .where(PortfolioModel.user_id == user.id)
        .order_by(PortfolioModel.is_default.desc(), PortfolioModel.created_at)
    )
    portfolios = result.scalars().all()
    
    responses = []
    for p in portfolios:
        resp = PortfolioResponse.model_validate(p)
        resp.open_positions_count = len([pos for pos in p.positions if pos.status == "open"])
        resp.win_rate = float(p.winning_trades / p.total_trades) if p.total_trades > 0 else None
        responses.append(resp)
    
    return responses


@router.post(
    "",
    response_model=PortfolioResponse,
    status_code=status.HTTP_201_CREATED,
    summary="Create Portfolio",
    description="Yeni portfolio oluşturur.",
)
async def create_portfolio(
    data: PortfolioCreate,
    user: CurrentUser,
    session: DbSession,
) -> PortfolioResponse:
    """Portfolio oluştur."""
    existing = await session.execute(
        select(PortfolioModel).where(PortfolioModel.user_id == user.id)
    )
    is_first = existing.scalar_one_or_none() is None
    
    portfolio = PortfolioModel(
        user_id=user.id,
        name=data.name,
        description=data.description,
        capital=data.capital,
        current_value=data.capital,
        available_capital=data.capital,
        peak_value=data.capital,
        is_default=is_first,
        is_paper=data.is_paper,
        max_positions=data.max_positions,
        max_risk_per_trade=Decimal(str(data.max_risk_per_trade)),
    )
    
    session.add(portfolio)
    await session.commit()
    await session.refresh(portfolio)
    
    logger.info(f"Portfolio created for user {user.id}: {portfolio.name}")
    
    resp = PortfolioResponse.model_validate(portfolio)
    resp.open_positions_count = 0
    return resp


@router.get(
    "/{portfolio_id}",
    response_model=PortfolioResponse,
    summary="Get Portfolio",
)
async def get_portfolio(
    portfolio_id: UUID = Path(...),
    user: CurrentUser = Depends(get_current_user),
    session: DbSession = Depends(get_session),
) -> PortfolioResponse:
    """Portfolio detayları."""
    portfolio = await session.get(PortfolioModel, portfolio_id)
    
    if not portfolio or portfolio.user_id != user.id:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Portfolio not found")
    
    resp = PortfolioResponse.model_validate(portfolio)
    resp.open_positions_count = len([pos for pos in portfolio.positions if pos.status == "open"])
    resp.win_rate = float(portfolio.winning_trades / portfolio.total_trades) if portfolio.total_trades > 0 else None
    return resp


@router.put(
    "/{portfolio_id}",
    response_model=PortfolioResponse,
    summary="Update Portfolio",
)
async def update_portfolio(
    data: PortfolioUpdate,
    portfolio_id: UUID = Path(...),
    user: CurrentUser = Depends(get_current_user),
    session: DbSession = Depends(get_session),
) -> PortfolioResponse:
    """Portfolio güncelle."""
    portfolio = await session.get(PortfolioModel, portfolio_id)
    
    if not portfolio or portfolio.user_id != user.id:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Portfolio not found")
    
    update_data = data.model_dump(exclude_unset=True)
    for key, value in update_data.items():
        setattr(portfolio, key, value)
    
    await session.commit()
    await session.refresh(portfolio)
    
    resp = PortfolioResponse.model_validate(portfolio)
    resp.open_positions_count = len([pos for pos in portfolio.positions if pos.status == "open"])
    return resp


@router.delete(
    "/{portfolio_id}",
    response_model=SuccessResponse,
    summary="Delete Portfolio",
)
async def delete_portfolio(
    portfolio_id: UUID = Path(...),
    user: CurrentUser = Depends(get_current_user),
    session: DbSession = Depends(get_session),
) -> SuccessResponse:
    """Portfolio sil."""
    portfolio = await session.get(PortfolioModel, portfolio_id)
    
    if not portfolio or portfolio.user_id != user.id:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Portfolio not found")
    
    if portfolio.is_default:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Cannot delete default portfolio")
    
    open_positions = [pos for pos in portfolio.positions if pos.status == "open"]
    if open_positions:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Cannot delete portfolio with open positions")
    
    await session.delete(portfolio)
    await session.commit()
    
    return SuccessResponse(message="Portfolio deleted successfully")


@router.get(
    "/{portfolio_id}/summary",
    response_model=PortfolioSummary,
    summary="Get Portfolio Summary",
)
async def get_portfolio_summary(
    portfolio_id: UUID = Path(...),
    user: CurrentUser = Depends(get_current_user),
    session: DbSession = Depends(get_session),
) -> PortfolioSummary:
    """Portfolio özeti."""
    portfolio = await session.get(PortfolioModel, portfolio_id)
    
    if not portfolio or portfolio.user_id != user.id:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Portfolio not found")
    
    open_positions = [pos for pos in portfolio.positions if pos.status == "open"]
    total_risk = sum(pos.risk_amount for pos in open_positions)
    max_portfolio_risk = portfolio.capital * Decimal(str(settings.risk.max_portfolio_heat))
    
    return PortfolioSummary(
        total_value=portfolio.current_value,
        total_pnl=portfolio.total_pnl,
        total_pnl_pct=portfolio.total_pnl_pct,
        open_positions=len(open_positions),
        total_risk=total_risk,
        remaining_risk_budget=max_portfolio_risk - total_risk,
        day_pnl=Decimal("0"),
        week_pnl=Decimal("0"),
        month_pnl=Decimal("0"),
    )


# =============================================================================
# POSITION ENDPOINTS
# =============================================================================

@router.get(
    "/{portfolio_id}/positions",
    response_model=List[PositionResponse],
    summary="List Positions",
)
async def list_positions(
    portfolio_id: UUID = Path(...),
    status_filter: Optional[str] = Query(None, alias="status"),
    user: CurrentUser = Depends(get_current_user),
    session: DbSession = Depends(get_session),
) -> List[PositionResponse]:
    """Pozisyon listesi."""
    portfolio = await session.get(PortfolioModel, portfolio_id)
    
    if not portfolio or portfolio.user_id != user.id:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Portfolio not found")
    
    positions = portfolio.positions
    if status_filter:
        positions = [p for p in positions if p.status == status_filter]
    
    return [PositionResponse.model_validate(p) for p in positions]


@router.post(
    "/{portfolio_id}/positions",
    response_model=PositionResponse,
    status_code=status.HTTP_201_CREATED,
    summary="Open Position",
)
async def open_position(
    data: PositionCreate,
    portfolio_id: UUID = Path(...),
    user: CurrentUser = Depends(get_current_user),
    session: DbSession = Depends(get_session),
) -> PositionResponse:
    """Pozisyon aç."""
    portfolio = await session.get(PortfolioModel, portfolio_id)
    
    if not portfolio or portfolio.user_id != user.id:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Portfolio not found")
    
    open_count = len([p for p in portfolio.positions if p.status == "open"])
    if open_count >= portfolio.max_positions:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=f"Maximum positions ({portfolio.max_positions}) reached")
    
    position_value = data.entry_price * data.quantity
    
    if data.direction.upper() == "LONG":
        risk_per_share = data.entry_price - data.stop_loss
    else:
        risk_per_share = data.stop_loss - data.entry_price
    
    risk_amount = risk_per_share * data.quantity
    max_risk = portfolio.capital * portfolio.max_risk_per_trade
    
    if risk_amount > max_risk:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=f"Risk amount exceeds maximum")
    
    if position_value > portfolio.available_capital:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Insufficient available capital")
    
    position = PositionModel(
        portfolio_id=portfolio_id,
        signal_id=data.signal_id,
        symbol=data.symbol.upper(),
        direction=data.direction.upper(),
        quantity=data.quantity,
        entry_price=data.entry_price,
        current_price=data.entry_price,
        stop_loss=data.stop_loss,
        take_profit=data.take_profit,
        position_value=position_value,
        risk_amount=risk_amount,
        status="open",
        notes=data.notes,
    )
    
    portfolio.available_capital -= position_value
    session.add(position)
    await session.commit()
    await session.refresh(position)
    
    logger.info(f"Position opened: {data.symbol} {data.direction} x{data.quantity}")
    return PositionResponse.model_validate(position)


@router.post(
    "/{portfolio_id}/positions/{position_id}/close",
    response_model=PositionResponse,
    summary="Close Position",
)
async def close_position(
    data: PositionClose,
    portfolio_id: UUID = Path(...),
    position_id: UUID = Path(...),
    user: CurrentUser = Depends(get_current_user),
    session: DbSession = Depends(get_session),
) -> PositionResponse:
    """Pozisyon kapat."""
    portfolio = await session.get(PortfolioModel, portfolio_id)
    
    if not portfolio or portfolio.user_id != user.id:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Portfolio not found")
    
    position = await session.get(PositionModel, position_id)
    
    if not position or position.portfolio_id != portfolio_id:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Position not found")
    
    if position.status != "open":
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Position already closed")
    
    position.close(data.exit_price, data.reason)
    portfolio.available_capital += position.position_value + (position.realized_pnl or Decimal("0"))
    portfolio.record_trade(
        is_win=position.realized_pnl > 0 if position.realized_pnl else False,
        pnl=position.realized_pnl or Decimal("0")
    )
    
    await session.commit()
    await session.refresh(position)
    
    logger.info(f"Position closed: {position.symbol} P&L: {position.realized_pnl}")
    return PositionResponse.model_validate(position)
