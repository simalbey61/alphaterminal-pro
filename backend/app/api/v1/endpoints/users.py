"""
AlphaTerminal Pro - Users Endpoints
===================================

Kullanıcı yönetimi endpoint'leri.

Author: AlphaTerminal Team
Version: 1.0.0
"""

from __future__ import annotations

import logging
from typing import Optional, List
from uuid import UUID
from datetime import datetime

from fastapi import APIRouter, Depends, HTTPException, status, Query, Path
from pydantic import BaseModel, Field
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.db.database import get_session
from app.db.models import UserModel, WatchlistModel, NotificationModel
from app.api.dependencies import (
    get_current_user,
    get_current_admin_user,
    CurrentUser,
    CurrentAdmin,
    DbSession,
    Pagination,
)
from app.schemas import SuccessResponse, NotificationResponse

logger = logging.getLogger(__name__)
router = APIRouter()


# =============================================================================
# SCHEMAS
# =============================================================================

class UserProfileUpdate(BaseModel):
    """Kullanıcı profil güncelleme schema."""
    username: Optional[str] = Field(None, min_length=3, max_length=50)
    full_name: Optional[str] = Field(None, max_length=100)
    avatar_url: Optional[str] = None


class UserPreferencesUpdate(BaseModel):
    """Kullanıcı tercihleri güncelleme schema."""
    theme: Optional[str] = None
    language: Optional[str] = None
    notifications_enabled: Optional[bool] = None
    email_notifications: Optional[bool] = None


class WatchlistItemCreate(BaseModel):
    """Watchlist item oluşturma schema."""
    symbol: str = Field(..., max_length=20)
    notes: Optional[str] = None
    alert_price_above: Optional[float] = None
    alert_price_below: Optional[float] = None
    alert_enabled: bool = False


class WatchlistItemResponse(BaseModel):
    """Watchlist item response schema."""
    id: UUID
    symbol: str
    notes: Optional[str] = None
    alert_price_above: Optional[float] = None
    alert_price_below: Optional[float] = None
    alert_enabled: bool
    created_at: datetime
    
    class Config:
        from_attributes = True


class UserResponse(BaseModel):
    """Kullanıcı response schema."""
    id: UUID
    email: Optional[str] = None
    username: Optional[str] = None
    full_name: Optional[str] = None
    telegram_id: Optional[int] = None
    telegram_username: Optional[str] = None
    avatar_url: Optional[str] = None
    is_admin: bool = False
    is_premium: bool = False
    is_verified: bool = False
    is_active: bool = True
    last_login_at: Optional[datetime] = None
    created_at: datetime
    
    class Config:
        from_attributes = True


# =============================================================================
# PROFILE
# =============================================================================

@router.get("/me", response_model=UserResponse, summary="Get Current User Profile")
async def get_my_profile(user: CurrentUser) -> UserResponse:
    """Mevcut kullanıcı profili."""
    return UserResponse.model_validate(user)


@router.put("/me", response_model=UserResponse, summary="Update Profile")
async def update_my_profile(data: UserProfileUpdate, user: CurrentUser, session: DbSession) -> UserResponse:
    """Profil güncelle."""
    update_data = data.model_dump(exclude_unset=True)
    
    if "username" in update_data:
        existing = await session.execute(select(UserModel).where(UserModel.username == update_data["username"], UserModel.id != user.id))
        if existing.scalar_one_or_none():
            raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Username already taken")
    
    for key, value in update_data.items():
        setattr(user, key, value)
    
    await session.commit()
    await session.refresh(user)
    
    return UserResponse.model_validate(user)


@router.get("/me/preferences", summary="Get Preferences")
async def get_my_preferences(user: CurrentUser) -> dict:
    """Kullanıcı tercihlerini getir."""
    return user.preferences or {}


@router.put("/me/preferences", summary="Update Preferences")
async def update_my_preferences(data: UserPreferencesUpdate, user: CurrentUser, session: DbSession) -> dict:
    """Kullanıcı tercihlerini güncelle."""
    prefs = user.preferences or {}
    update_data = data.model_dump(exclude_unset=True)
    prefs.update(update_data)
    user.preferences = prefs
    
    await session.commit()
    return prefs


# =============================================================================
# WATCHLIST
# =============================================================================

@router.get("/me/watchlist", response_model=List[WatchlistItemResponse], summary="Get Watchlist")
async def get_my_watchlist(user: CurrentUser, session: DbSession) -> List[WatchlistItemResponse]:
    """Watchlist getir."""
    result = await session.execute(select(WatchlistModel).where(WatchlistModel.user_id == user.id).order_by(WatchlistModel.sort_order, WatchlistModel.created_at))
    items = result.scalars().all()
    return [WatchlistItemResponse.model_validate(item) for item in items]


@router.post("/me/watchlist", response_model=WatchlistItemResponse, status_code=status.HTTP_201_CREATED, summary="Add to Watchlist")
async def add_to_watchlist(data: WatchlistItemCreate, user: CurrentUser, session: DbSession) -> WatchlistItemResponse:
    """Watchlist'e ekle."""
    existing = await session.execute(select(WatchlistModel).where(WatchlistModel.user_id == user.id, WatchlistModel.symbol == data.symbol.upper()))
    if existing.scalar_one_or_none():
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Symbol already in watchlist")
    
    item = WatchlistModel(user_id=user.id, symbol=data.symbol.upper(), notes=data.notes, alert_price_above=data.alert_price_above, alert_price_below=data.alert_price_below, alert_enabled=data.alert_enabled)
    session.add(item)
    await session.commit()
    await session.refresh(item)
    
    return WatchlistItemResponse.model_validate(item)


@router.delete("/me/watchlist/{symbol}", response_model=SuccessResponse, summary="Remove from Watchlist")
async def remove_from_watchlist(symbol: str = Path(...), user: CurrentUser = Depends(get_current_user), session: DbSession = Depends(get_session)) -> SuccessResponse:
    """Watchlist'ten kaldır."""
    result = await session.execute(select(WatchlistModel).where(WatchlistModel.user_id == user.id, WatchlistModel.symbol == symbol.upper()))
    item = result.scalar_one_or_none()
    
    if not item:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Symbol not in watchlist")
    
    await session.delete(item)
    await session.commit()
    
    return SuccessResponse(message=f"{symbol} removed from watchlist")


# =============================================================================
# NOTIFICATIONS
# =============================================================================

@router.get("/me/notifications", response_model=List[NotificationResponse], summary="Get Notifications")
async def get_my_notifications(user: CurrentUser, session: DbSession, unread_only: bool = Query(False), limit: int = Query(50, ge=1, le=200)) -> List[NotificationResponse]:
    """Bildirimleri getir."""
    query = select(NotificationModel).where(NotificationModel.user_id == user.id)
    if unread_only:
        query = query.where(NotificationModel.is_read == False)
    query = query.order_by(NotificationModel.created_at.desc()).limit(limit)
    
    result = await session.execute(query)
    notifications = result.scalars().all()
    return [NotificationResponse.model_validate(n) for n in notifications]


@router.post("/me/notifications/{notification_id}/read", response_model=SuccessResponse, summary="Mark as Read")
async def mark_notification_read(notification_id: UUID = Path(...), user: CurrentUser = Depends(get_current_user), session: DbSession = Depends(get_session)) -> SuccessResponse:
    """Bildirimi okundu işaretle."""
    notification = await session.get(NotificationModel, notification_id)
    
    if not notification or notification.user_id != user.id:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Notification not found")
    
    notification.mark_as_read()
    await session.commit()
    
    return SuccessResponse(message="Notification marked as read")


@router.post("/me/notifications/read-all", response_model=SuccessResponse, summary="Mark All as Read")
async def mark_all_notifications_read(user: CurrentUser, session: DbSession) -> SuccessResponse:
    """Tüm bildirimleri okundu işaretle."""
    from sqlalchemy import update
    
    await session.execute(update(NotificationModel).where(NotificationModel.user_id == user.id, NotificationModel.is_read == False).values(is_read=True, read_at=datetime.utcnow()))
    await session.commit()
    
    return SuccessResponse(message="All notifications marked as read")


# =============================================================================
# ADMIN - USER MANAGEMENT
# =============================================================================

@router.get("", response_model=List[UserResponse], summary="List Users (Admin)")
async def list_users(admin: CurrentAdmin, session: DbSession, pagination: Pagination) -> List[UserResponse]:
    """Kullanıcı listesi (Admin)."""
    result = await session.execute(select(UserModel).order_by(UserModel.created_at.desc()).offset(pagination.offset).limit(pagination.per_page))
    users = result.scalars().all()
    return [UserResponse.model_validate(u) for u in users]


@router.get("/{user_id}", response_model=UserResponse, summary="Get User (Admin)")
async def get_user(user_id: UUID = Path(...), admin: CurrentAdmin = Depends(get_current_admin_user), session: DbSession = Depends(get_session)) -> UserResponse:
    """Kullanıcı detayları (Admin)."""
    user = await session.get(UserModel, user_id)
    if not user:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="User not found")
    return UserResponse.model_validate(user)


@router.post("/{user_id}/toggle-premium", response_model=UserResponse, summary="Toggle Premium (Admin)")
async def toggle_premium(user_id: UUID = Path(...), admin: CurrentAdmin = Depends(get_current_admin_user), session: DbSession = Depends(get_session)) -> UserResponse:
    """Premium durumunu değiştir (Admin)."""
    user = await session.get(UserModel, user_id)
    if not user:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="User not found")
    
    user.is_premium = not user.is_premium
    await session.commit()
    await session.refresh(user)
    
    logger.info(f"User {user_id} premium status changed to {user.is_premium} by admin {admin.id}")
    return UserResponse.model_validate(user)


@router.post("/{user_id}/deactivate", response_model=SuccessResponse, summary="Deactivate User (Admin)")
async def deactivate_user(user_id: UUID = Path(...), admin: CurrentAdmin = Depends(get_current_admin_user), session: DbSession = Depends(get_session)) -> SuccessResponse:
    """Kullanıcıyı deaktive et (Admin)."""
    user = await session.get(UserModel, user_id)
    if not user:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="User not found")
    
    if user.is_admin:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Cannot deactivate admin user")
    
    user.is_active = False
    await session.commit()
    
    logger.info(f"User {user_id} deactivated by admin {admin.id}")
    return SuccessResponse(message="User deactivated")
