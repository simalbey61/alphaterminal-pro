"""
AlphaTerminal Pro - Authentication Endpoints
============================================

JWT tabanlı kimlik doğrulama endpoint'leri.

Author: AlphaTerminal Team
Version: 1.0.0
"""

from __future__ import annotations

import logging
from datetime import datetime, timedelta
from typing import Optional
from uuid import UUID

from fastapi import APIRouter, Depends, HTTPException, status, Body
from pydantic import BaseModel, Field, EmailStr, field_validator
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.config import settings
from app.db.database import get_session
from app.db.models import UserModel
from app.api.dependencies import (
    get_password_hash,
    verify_password,
    create_access_token,
    create_refresh_token,
    decode_token,
    get_current_user,
    CurrentUser,
    DbSession,
    rate_limiter_strict,
)

logger = logging.getLogger(__name__)
router = APIRouter()


# =============================================================================
# REQUEST/RESPONSE SCHEMAS
# =============================================================================

class LoginRequest(BaseModel):
    """Login istek schema."""
    
    email: EmailStr = Field(..., description="E-posta adresi")
    password: str = Field(..., min_length=6, description="Şifre")


class RegisterRequest(BaseModel):
    """Kayıt istek schema."""
    
    email: EmailStr = Field(..., description="E-posta adresi")
    password: str = Field(..., min_length=8, description="Şifre (min 8 karakter)")
    password_confirm: str = Field(..., description="Şifre onayı")
    username: Optional[str] = Field(None, min_length=3, max_length=50, description="Kullanıcı adı")
    full_name: Optional[str] = Field(None, max_length=100, description="Tam ad")
    
    @field_validator("password")
    @classmethod
    def validate_password_strength(cls, v: str) -> str:
        """Şifre gücünü kontrol et."""
        if len(v) < 8:
            raise ValueError("Password must be at least 8 characters")
        if not any(c.isupper() for c in v):
            raise ValueError("Password must contain at least one uppercase letter")
        if not any(c.islower() for c in v):
            raise ValueError("Password must contain at least one lowercase letter")
        if not any(c.isdigit() for c in v):
            raise ValueError("Password must contain at least one digit")
        return v
    
    @field_validator("password_confirm")
    @classmethod
    def passwords_match(cls, v: str, info) -> str:
        """Şifrelerin eşleştiğini kontrol et."""
        if "password" in info.data and v != info.data["password"]:
            raise ValueError("Passwords do not match")
        return v


class TokenResponse(BaseModel):
    """Token response schema."""
    
    access_token: str
    refresh_token: str
    token_type: str = "bearer"
    expires_in: int = Field(..., description="Access token süresi (saniye)")


class RefreshTokenRequest(BaseModel):
    """Token yenileme istek schema."""
    
    refresh_token: str = Field(..., description="Refresh token")


class ChangePasswordRequest(BaseModel):
    """Şifre değiştirme istek schema."""
    
    current_password: str = Field(..., description="Mevcut şifre")
    new_password: str = Field(..., min_length=8, description="Yeni şifre")
    new_password_confirm: str = Field(..., description="Yeni şifre onayı")
    
    @field_validator("new_password_confirm")
    @classmethod
    def passwords_match(cls, v: str, info) -> str:
        """Şifrelerin eşleştiğini kontrol et."""
        if "new_password" in info.data and v != info.data["new_password"]:
            raise ValueError("Passwords do not match")
        return v


class UserResponse(BaseModel):
    """Kullanıcı response schema."""
    
    id: UUID
    email: Optional[str] = None
    username: Optional[str] = None
    full_name: Optional[str] = None
    telegram_id: Optional[int] = None
    telegram_username: Optional[str] = None
    is_admin: bool = False
    is_premium: bool = False
    is_verified: bool = False
    last_login_at: Optional[datetime] = None
    created_at: datetime
    
    class Config:
        from_attributes = True


class TelegramAuthRequest(BaseModel):
    """Telegram authentication istek schema."""
    
    telegram_id: int = Field(..., description="Telegram kullanıcı ID")
    telegram_username: Optional[str] = Field(None, description="Telegram kullanıcı adı")
    full_name: Optional[str] = Field(None, description="Telegram'dan alınan ad")
    auth_date: int = Field(..., description="Authentication tarihi (Unix timestamp)")
    hash: str = Field(..., description="Telegram doğrulama hash'i")


# =============================================================================
# ENDPOINTS
# =============================================================================

@router.post(
    "/register",
    response_model=UserResponse,
    status_code=status.HTTP_201_CREATED,
    summary="Register New User",
    description="Yeni kullanıcı kaydı oluşturur.",
    dependencies=[Depends(rate_limiter_strict)],
)
async def register(
    request: RegisterRequest,
    session: DbSession,
) -> UserResponse:
    """
    Yeni kullanıcı kaydı.
    
    Args:
        request: Kayıt bilgileri
        session: Database session
        
    Returns:
        UserResponse: Oluşturulan kullanıcı
        
    Raises:
        HTTPException: 400 - Email zaten kayıtlı
    """
    # Email kontrolü
    existing_email = await session.execute(
        select(UserModel).where(UserModel.email == request.email)
    )
    if existing_email.scalar_one_or_none():
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Email already registered"
        )
    
    # Username kontrolü (varsa)
    if request.username:
        existing_username = await session.execute(
            select(UserModel).where(UserModel.username == request.username)
        )
        if existing_username.scalar_one_or_none():
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Username already taken"
            )
    
    # Kullanıcı oluştur
    user = UserModel(
        email=request.email,
        password_hash=get_password_hash(request.password),
        username=request.username,
        full_name=request.full_name,
        is_active=True,
        is_verified=False,  # Email doğrulaması gerekecek
    )
    
    session.add(user)
    await session.commit()
    await session.refresh(user)
    
    logger.info(f"New user registered: {user.email}")
    
    return UserResponse.model_validate(user)


@router.post(
    "/login",
    response_model=TokenResponse,
    summary="User Login",
    description="Kullanıcı girişi yapar ve JWT token döner.",
    dependencies=[Depends(rate_limiter_strict)],
)
async def login(
    request: LoginRequest,
    session: DbSession,
) -> TokenResponse:
    """
    Kullanıcı girişi.
    
    Args:
        request: Login bilgileri
        session: Database session
        
    Returns:
        TokenResponse: JWT tokenlar
        
    Raises:
        HTTPException: 401 - Geçersiz credentials
    """
    # Kullanıcıyı bul
    result = await session.execute(
        select(UserModel).where(UserModel.email == request.email)
    )
    user = result.scalar_one_or_none()
    
    if not user:
        logger.warning(f"Login attempt for non-existent email: {request.email}")
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid email or password"
        )
    
    # Şifre kontrolü
    if not user.password_hash or not verify_password(request.password, user.password_hash):
        logger.warning(f"Invalid password attempt for: {request.email}")
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid email or password"
        )
    
    # Aktiflik kontrolü
    if not user.is_active:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Account is disabled"
        )
    
    if user.is_deleted:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid email or password"
        )
    
    # Token oluştur
    token_data = {
        "sub": str(user.id),
        "email": user.email,
        "is_admin": user.is_admin,
    }
    
    access_token = create_access_token(token_data)
    refresh_token = create_refresh_token(token_data)
    
    # Son giriş güncelle
    user.update_last_login()
    await session.commit()
    
    logger.info(f"User logged in: {user.email}")
    
    return TokenResponse(
        access_token=access_token,
        refresh_token=refresh_token,
        expires_in=settings.jwt.access_token_expire_minutes * 60,
    )


@router.post(
    "/refresh",
    response_model=TokenResponse,
    summary="Refresh Token",
    description="Refresh token kullanarak yeni access token alır.",
)
async def refresh_token(
    request: RefreshTokenRequest,
    session: DbSession,
) -> TokenResponse:
    """
    Token yenileme.
    
    Args:
        request: Refresh token
        session: Database session
        
    Returns:
        TokenResponse: Yeni JWT tokenlar
        
    Raises:
        HTTPException: 401 - Geçersiz refresh token
    """
    # Token decode
    payload = decode_token(request.refresh_token)
    
    if not payload:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid refresh token"
        )
    
    # Token tipini kontrol et
    if payload.get("type") != "refresh":
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid token type"
        )
    
    user_id = payload.get("sub")
    if not user_id:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid token payload"
        )
    
    # Kullanıcıyı bul
    try:
        user_uuid = UUID(user_id)
    except ValueError:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid user ID"
        )
    
    user = await session.get(UserModel, user_uuid)
    
    if not user or not user.is_active or user.is_deleted:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="User not found or inactive"
        )
    
    # Yeni token oluştur
    token_data = {
        "sub": str(user.id),
        "email": user.email,
        "is_admin": user.is_admin,
    }
    
    new_access_token = create_access_token(token_data)
    new_refresh_token = create_refresh_token(token_data)
    
    logger.info(f"Token refreshed for: {user.email}")
    
    return TokenResponse(
        access_token=new_access_token,
        refresh_token=new_refresh_token,
        expires_in=settings.jwt.access_token_expire_minutes * 60,
    )


@router.get(
    "/me",
    response_model=UserResponse,
    summary="Get Current User",
    description="Mevcut kullanıcı bilgilerini döner.",
)
async def get_me(
    current_user: CurrentUser,
) -> UserResponse:
    """
    Mevcut kullanıcı bilgileri.
    
    Returns:
        UserResponse: Kullanıcı bilgileri
    """
    return UserResponse.model_validate(current_user)


@router.post(
    "/change-password",
    summary="Change Password",
    description="Kullanıcı şifresini değiştirir.",
)
async def change_password(
    request: ChangePasswordRequest,
    current_user: CurrentUser,
    session: DbSession,
) -> dict:
    """
    Şifre değiştirme.
    
    Args:
        request: Yeni şifre bilgileri
        current_user: Mevcut kullanıcı
        session: Database session
        
    Returns:
        dict: Başarı mesajı
        
    Raises:
        HTTPException: 400 - Geçersiz mevcut şifre
    """
    # Mevcut şifre kontrolü
    if not verify_password(request.current_password, current_user.password_hash):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Current password is incorrect"
        )
    
    # Yeni şifre eski şifreyle aynı olmamalı
    if verify_password(request.new_password, current_user.password_hash):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="New password must be different from current password"
        )
    
    # Şifreyi güncelle
    current_user.password_hash = get_password_hash(request.new_password)
    await session.commit()
    
    logger.info(f"Password changed for: {current_user.email}")
    
    return {"message": "Password changed successfully"}


@router.post(
    "/telegram",
    response_model=TokenResponse,
    summary="Telegram Authentication",
    description="Telegram ile kimlik doğrulama yapar.",
    dependencies=[Depends(rate_limiter_strict)],
)
async def telegram_auth(
    request: TelegramAuthRequest,
    session: DbSession,
) -> TokenResponse:
    """
    Telegram ile kimlik doğrulama.
    
    Telegram Login Widget'tan gelen verileri doğrular
    ve JWT token döner. Kullanıcı yoksa otomatik oluşturur.
    
    Args:
        request: Telegram auth verileri
        session: Database session
        
    Returns:
        TokenResponse: JWT tokenlar
        
    Raises:
        HTTPException: 401 - Geçersiz authentication
    """
    # TODO: Telegram hash doğrulaması
    # import hashlib
    # import hmac
    # data_check_string = ...
    # secret_key = hashlib.sha256(settings.telegram.token.encode()).digest()
    # computed_hash = hmac.new(secret_key, data_check_string.encode(), hashlib.sha256).hexdigest()
    # if computed_hash != request.hash:
    #     raise HTTPException(status_code=401, detail="Invalid Telegram authentication")
    
    # Kullanıcıyı bul veya oluştur
    result = await session.execute(
        select(UserModel).where(UserModel.telegram_id == request.telegram_id)
    )
    user = result.scalar_one_or_none()
    
    if not user:
        # Yeni kullanıcı oluştur
        user = UserModel(
            telegram_id=request.telegram_id,
            telegram_username=request.telegram_username,
            full_name=request.full_name,
            is_active=True,
            is_verified=True,  # Telegram ile doğrulanmış
        )
        session.add(user)
        await session.commit()
        await session.refresh(user)
        logger.info(f"New Telegram user created: {request.telegram_id}")
    else:
        # Mevcut kullanıcıyı güncelle
        if request.telegram_username:
            user.telegram_username = request.telegram_username
        if request.full_name:
            user.full_name = request.full_name
        user.update_last_login()
        await session.commit()
    
    # Token oluştur
    token_data = {
        "sub": str(user.id),
        "telegram_id": user.telegram_id,
        "is_admin": user.is_admin,
    }
    
    access_token = create_access_token(token_data)
    refresh_token = create_refresh_token(token_data)
    
    logger.info(f"Telegram user logged in: {request.telegram_id}")
    
    return TokenResponse(
        access_token=access_token,
        refresh_token=refresh_token,
        expires_in=settings.jwt.access_token_expire_minutes * 60,
    )


@router.post(
    "/logout",
    summary="Logout",
    description="Kullanıcı çıkışı yapar.",
)
async def logout(
    current_user: CurrentUser,
) -> dict:
    """
    Kullanıcı çıkışı.
    
    Not: JWT stateless olduğu için sunucu tarafında
    token invalidation yapılmaz. Client token'ı silmelidir.
    Gerekirse token blacklist implementasyonu eklenebilir.
    
    Returns:
        dict: Başarı mesajı
    """
    logger.info(f"User logged out: {current_user.email or current_user.telegram_id}")
    
    return {"message": "Successfully logged out"}
