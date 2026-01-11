"""
AlphaTerminal Pro - API Dependencies
====================================

Kurumsal seviye dependency injection, authentication ve authorization.

Author: AlphaTerminal Team
Version: 1.0.0
"""

from __future__ import annotations

import logging
from datetime import datetime, timedelta
from typing import Annotated, Optional, AsyncGenerator
from uuid import UUID

from fastapi import Depends, HTTPException, status, Request, Header
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from jose import JWTError, jwt
from passlib.context import CryptContext
from sqlalchemy.ext.asyncio import AsyncSession

from app.config import settings
from app.db.database import get_session
from app.db.models import UserModel
from app.db.repositories import (
    StockRepository,
    SignalRepository,
    StrategyRepository,
)

logger = logging.getLogger(__name__)


# =============================================================================
# PASSWORD HASHING
# =============================================================================

pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")


def verify_password(plain_password: str, hashed_password: str) -> bool:
    """Şifre doğrula."""
    return pwd_context.verify(plain_password, hashed_password)


def get_password_hash(password: str) -> str:
    """Şifre hashle."""
    return pwd_context.hash(password)


# =============================================================================
# JWT TOKEN
# =============================================================================

class TokenData:
    """JWT token içeriği."""
    
    def __init__(
        self,
        user_id: UUID,
        email: Optional[str] = None,
        telegram_id: Optional[int] = None,
        is_admin: bool = False,
        scopes: list[str] = None,
    ):
        self.user_id = user_id
        self.email = email
        self.telegram_id = telegram_id
        self.is_admin = is_admin
        self.scopes = scopes or []


def create_access_token(
    data: dict,
    expires_delta: Optional[timedelta] = None
) -> str:
    """
    Access token oluştur.
    
    Args:
        data: Token payload
        expires_delta: Geçerlilik süresi
        
    Returns:
        str: JWT token
    """
    to_encode = data.copy()
    
    if expires_delta:
        expire = datetime.utcnow() + expires_delta
    else:
        expire = datetime.utcnow() + timedelta(
            minutes=settings.jwt.access_token_expire_minutes
        )
    
    to_encode.update({
        "exp": expire,
        "iat": datetime.utcnow(),
        "type": "access"
    })
    
    encoded_jwt = jwt.encode(
        to_encode,
        settings.jwt.secret_key,
        algorithm=settings.jwt.algorithm
    )
    
    return encoded_jwt


def create_refresh_token(data: dict) -> str:
    """
    Refresh token oluştur.
    
    Args:
        data: Token payload
        
    Returns:
        str: JWT refresh token
    """
    to_encode = data.copy()
    expire = datetime.utcnow() + timedelta(days=settings.jwt.refresh_token_expire_days)
    
    to_encode.update({
        "exp": expire,
        "iat": datetime.utcnow(),
        "type": "refresh"
    })
    
    encoded_jwt = jwt.encode(
        to_encode,
        settings.jwt.secret_key,
        algorithm=settings.jwt.algorithm
    )
    
    return encoded_jwt


def decode_token(token: str) -> Optional[dict]:
    """
    Token decode et.
    
    Args:
        token: JWT token
        
    Returns:
        Optional[dict]: Token payload veya None
    """
    try:
        payload = jwt.decode(
            token,
            settings.jwt.secret_key,
            algorithms=[settings.jwt.algorithm]
        )
        return payload
    except JWTError as e:
        logger.warning(f"JWT decode error: {e}")
        return None


# =============================================================================
# SECURITY SCHEMES
# =============================================================================

security_scheme = HTTPBearer(auto_error=False)


class OptionalHTTPBearer(HTTPBearer):
    """Opsiyonel authentication için HTTPBearer."""
    
    async def __call__(self, request: Request) -> Optional[HTTPAuthorizationCredentials]:
        try:
            return await super().__call__(request)
        except HTTPException:
            return None


optional_security = OptionalHTTPBearer(auto_error=False)


# =============================================================================
# AUTHENTICATION DEPENDENCIES
# =============================================================================

async def get_current_user_optional(
    credentials: Annotated[
        Optional[HTTPAuthorizationCredentials],
        Depends(optional_security)
    ],
    session: Annotated[AsyncSession, Depends(get_session)],
) -> Optional[UserModel]:
    """
    Mevcut kullanıcıyı al (opsiyonel).
    
    Authentication yoksa None döner, hata fırlatmaz.
    """
    if credentials is None:
        return None
    
    token = credentials.credentials
    payload = decode_token(token)
    
    if payload is None:
        return None
    
    # Token tipini kontrol et
    if payload.get("type") != "access":
        return None
    
    user_id = payload.get("sub")
    if user_id is None:
        return None
    
    try:
        user_uuid = UUID(user_id)
    except ValueError:
        return None
    
    # Kullanıcıyı veritabanından al
    user = await session.get(UserModel, user_uuid)
    
    if user is None or not user.is_active or user.is_deleted:
        return None
    
    return user


async def get_current_user(
    credentials: Annotated[
        HTTPAuthorizationCredentials,
        Depends(security_scheme)
    ],
    session: Annotated[AsyncSession, Depends(get_session)],
) -> UserModel:
    """
    Mevcut kullanıcıyı al (zorunlu).
    
    Authentication yoksa 401 hatası fırlatır.
    
    Raises:
        HTTPException: 401 Unauthorized
    """
    credentials_exception = HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Could not validate credentials",
        headers={"WWW-Authenticate": "Bearer"},
    )
    
    if credentials is None:
        raise credentials_exception
    
    token = credentials.credentials
    payload = decode_token(token)
    
    if payload is None:
        raise credentials_exception
    
    # Token tipini kontrol et
    if payload.get("type") != "access":
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid token type",
            headers={"WWW-Authenticate": "Bearer"},
        )
    
    user_id = payload.get("sub")
    if user_id is None:
        raise credentials_exception
    
    try:
        user_uuid = UUID(user_id)
    except ValueError:
        raise credentials_exception
    
    # Kullanıcıyı veritabanından al
    user = await session.get(UserModel, user_uuid)
    
    if user is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="User not found"
        )
    
    if not user.is_active:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Inactive user"
        )
    
    if user.is_deleted:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="User not found"
        )
    
    # Son aktivite güncelle
    user.update_last_activity()
    
    return user


async def get_current_active_user(
    current_user: Annotated[UserModel, Depends(get_current_user)]
) -> UserModel:
    """
    Aktif kullanıcıyı al.
    
    Ek aktiflik kontrolü yapar.
    """
    if not current_user.is_active:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Inactive user"
        )
    return current_user


async def get_current_admin_user(
    current_user: Annotated[UserModel, Depends(get_current_user)]
) -> UserModel:
    """
    Admin kullanıcıyı al.
    
    Admin değilse 403 hatası fırlatır.
    
    Raises:
        HTTPException: 403 Forbidden
    """
    if not current_user.is_admin:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Admin privileges required"
        )
    return current_user


async def get_current_premium_user(
    current_user: Annotated[UserModel, Depends(get_current_user)]
) -> UserModel:
    """
    Premium kullanıcıyı al.
    
    Premium değilse 403 hatası fırlatır.
    """
    if not current_user.is_premium and not current_user.is_admin:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Premium subscription required"
        )
    return current_user


# =============================================================================
# REPOSITORY DEPENDENCIES
# =============================================================================

async def get_stock_repository(
    session: Annotated[AsyncSession, Depends(get_session)]
) -> StockRepository:
    """Stock repository dependency."""
    return StockRepository(session)


async def get_signal_repository(
    session: Annotated[AsyncSession, Depends(get_session)]
) -> SignalRepository:
    """Signal repository dependency."""
    return SignalRepository(session)


async def get_strategy_repository(
    session: Annotated[AsyncSession, Depends(get_session)]
) -> StrategyRepository:
    """Strategy repository dependency."""
    return StrategyRepository(session)


# =============================================================================
# RATE LIMITING
# =============================================================================

class RateLimiter:
    """
    API rate limiter.
    
    Redis tabanlı sliding window rate limiting.
    """
    
    def __init__(
        self,
        times: int = 100,
        seconds: int = 60,
    ):
        """
        Initialize rate limiter.
        
        Args:
            times: Maksimum istek sayısı
            seconds: Zaman penceresi (saniye)
        """
        self.times = times
        self.seconds = seconds
    
    async def __call__(
        self,
        request: Request,
    ) -> None:
        """
        Rate limit kontrolü yap.
        
        Raises:
            HTTPException: 429 Too Many Requests
        """
        # Client identifier
        client_id = self._get_client_id(request)
        
        # Redis'ten cache import
        from app.cache import cache, CacheKeys
        
        key = CacheKeys.ratelimit_api(client_id)
        
        try:
            # Mevcut sayacı al
            current = await cache.get(key)
            current_count = int(current) if current else 0
            
            if current_count >= self.times:
                raise HTTPException(
                    status_code=status.HTTP_429_TOO_MANY_REQUESTS,
                    detail=f"Rate limit exceeded. Try again in {self.seconds} seconds.",
                    headers={"Retry-After": str(self.seconds)}
                )
            
            # Sayacı artır
            await cache.client.incr(key)
            
            # TTL ayarla (sadece ilk istek için)
            if current_count == 0:
                await cache.expire(key, self.seconds)
                
        except HTTPException:
            raise
        except Exception as e:
            # Redis hatası durumunda geç (fail open)
            logger.warning(f"Rate limiter error: {e}")
    
    def _get_client_id(self, request: Request) -> str:
        """Client identifier al."""
        # Önce X-Forwarded-For header'ını kontrol et (proxy arkasında)
        forwarded = request.headers.get("X-Forwarded-For")
        if forwarded:
            return forwarded.split(",")[0].strip()
        
        # Sonra X-Real-IP
        real_ip = request.headers.get("X-Real-IP")
        if real_ip:
            return real_ip
        
        # En son client host
        if request.client:
            return request.client.host
        
        return "unknown"


# Rate limiter instances
rate_limiter_default = RateLimiter(times=100, seconds=60)
rate_limiter_strict = RateLimiter(times=30, seconds=60)
rate_limiter_relaxed = RateLimiter(times=300, seconds=60)


# =============================================================================
# PAGINATION DEPENDENCY
# =============================================================================

class PaginationParams:
    """Pagination parametreleri dependency."""
    
    def __init__(
        self,
        page: int = 1,
        per_page: int = 20,
        order_by: Optional[str] = None,
        order_desc: bool = False,
    ):
        if page < 1:
            raise HTTPException(
                status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
                detail="Page must be >= 1"
            )
        if per_page < 1 or per_page > 100:
            raise HTTPException(
                status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
                detail="per_page must be between 1 and 100"
            )
        
        self.page = page
        self.per_page = per_page
        self.order_by = order_by
        self.order_desc = order_desc
        self.offset = (page - 1) * per_page


# =============================================================================
# API KEY AUTHENTICATION (Opsiyonel)
# =============================================================================

async def verify_api_key(
    x_api_key: Annotated[Optional[str], Header()] = None
) -> Optional[str]:
    """
    API key doğrula (opsiyonel endpoint'ler için).
    
    Returns:
        Optional[str]: API key veya None
    """
    if x_api_key is None:
        return None
    
    # Burada API key doğrulaması yapılabilir
    # Örneğin Redis'ten veya veritabanından kontrol
    
    return x_api_key


# =============================================================================
# TYPE ALIASES
# =============================================================================

# Dependency type aliases for cleaner code
CurrentUser = Annotated[UserModel, Depends(get_current_user)]
CurrentUserOptional = Annotated[Optional[UserModel], Depends(get_current_user_optional)]
CurrentAdmin = Annotated[UserModel, Depends(get_current_admin_user)]
CurrentPremium = Annotated[UserModel, Depends(get_current_premium_user)]
DbSession = Annotated[AsyncSession, Depends(get_session)]
StockRepo = Annotated[StockRepository, Depends(get_stock_repository)]
SignalRepo = Annotated[SignalRepository, Depends(get_signal_repository)]
StrategyRepo = Annotated[StrategyRepository, Depends(get_strategy_repository)]
Pagination = Annotated[PaginationParams, Depends()]
