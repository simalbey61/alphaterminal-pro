"""
AlphaTerminal Pro - Common Schemas
==================================

Ortak kullanılan Pydantic schemas.

Author: AlphaTerminal Team
Version: 1.0.0
"""

from __future__ import annotations

from datetime import datetime
from typing import Any, Dict, Generic, List, Optional, TypeVar
from uuid import UUID

from pydantic import BaseModel, Field


# =============================================================================
# GENERIC TYPE
# =============================================================================

T = TypeVar("T")


# =============================================================================
# PAGINATION
# =============================================================================

class PaginationParams(BaseModel):
    """Pagination parametreleri."""
    
    page: int = Field(default=1, ge=1, description="Sayfa numarası")
    per_page: int = Field(default=20, ge=1, le=100, description="Sayfa başına kayıt")
    order_by: Optional[str] = Field(None, description="Sıralama alanı")
    order_desc: bool = Field(default=False, description="Azalan sıralama")


class PaginatedResponse(BaseModel, Generic[T]):
    """Generic paginated response."""
    
    items: List[T]
    total: int
    page: int
    per_page: int
    pages: int
    has_next: bool
    has_prev: bool


# =============================================================================
# ERROR RESPONSES
# =============================================================================

class ErrorDetail(BaseModel):
    """Hata detay schema."""
    
    field: Optional[str] = None
    message: str
    code: Optional[str] = None


class ErrorResponse(BaseModel):
    """Hata response schema."""
    
    error: str
    message: str
    details: Optional[List[ErrorDetail]] = None
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    request_id: Optional[str] = None


class ValidationErrorResponse(BaseModel):
    """Validation hata response schema."""
    
    error: str = "validation_error"
    message: str = "Validation failed"
    details: List[ErrorDetail]


# =============================================================================
# SUCCESS RESPONSES
# =============================================================================

class SuccessResponse(BaseModel):
    """Başarı response schema."""
    
    success: bool = True
    message: str
    data: Optional[Dict[str, Any]] = None


class DeleteResponse(BaseModel):
    """Silme response schema."""
    
    success: bool = True
    message: str = "Successfully deleted"
    deleted_id: UUID


class BulkOperationResponse(BaseModel):
    """Toplu işlem response schema."""
    
    success: bool = True
    total: int
    successful: int
    failed: int
    errors: Optional[List[ErrorDetail]] = None


# =============================================================================
# HEALTH CHECK
# =============================================================================

class HealthStatus(BaseModel):
    """Sağlık durumu schema."""
    
    status: str = Field(..., description="healthy, unhealthy, degraded")
    version: str
    environment: str
    timestamp: datetime = Field(default_factory=datetime.utcnow)


class ServiceHealth(BaseModel):
    """Servis sağlık durumu."""
    
    name: str
    status: str
    latency_ms: Optional[float] = None
    message: Optional[str] = None


class DetailedHealthStatus(HealthStatus):
    """Detaylı sağlık durumu schema."""
    
    services: List[ServiceHealth]
    database: ServiceHealth
    redis: ServiceHealth
    uptime_seconds: float


# =============================================================================
# API INFO
# =============================================================================

class APIInfo(BaseModel):
    """API bilgi schema."""
    
    name: str = "AlphaTerminal Pro API"
    version: str
    description: str = "Kurumsal seviye BIST analiz ve trading platformu"
    docs_url: str = "/docs"
    redoc_url: str = "/redoc"
    environment: str


# =============================================================================
# WEBSOCKET
# =============================================================================

class WebSocketMessage(BaseModel):
    """WebSocket mesaj schema."""
    
    type: str = Field(..., description="Mesaj tipi")
    data: Dict[str, Any]
    timestamp: datetime = Field(default_factory=datetime.utcnow)


class WebSocketSubscription(BaseModel):
    """WebSocket subscription schema."""
    
    action: str = Field(..., description="subscribe, unsubscribe")
    channel: str = Field(..., description="Kanal adı")
    params: Optional[Dict[str, Any]] = None


# =============================================================================
# NOTIFICATION
# =============================================================================

class NotificationCreate(BaseModel):
    """Notification oluşturma schema."""
    
    type: str = Field(..., description="signal, alert, strategy, system")
    title: str = Field(..., max_length=200)
    message: str
    data: Optional[Dict[str, Any]] = None
    action_url: Optional[str] = None
    priority: str = Field(default="normal")
    expires_at: Optional[datetime] = None


class NotificationResponse(BaseModel):
    """Notification response schema."""
    
    id: UUID
    type: str
    title: str
    message: str
    data: Optional[Dict[str, Any]] = None
    action_url: Optional[str] = None
    priority: str
    is_read: bool
    read_at: Optional[datetime] = None
    expires_at: Optional[datetime] = None
    created_at: datetime


# =============================================================================
# FILTER & SORT
# =============================================================================

class DateRangeFilter(BaseModel):
    """Tarih aralığı filtresi."""
    
    start_date: Optional[datetime] = None
    end_date: Optional[datetime] = None


class NumericRangeFilter(BaseModel):
    """Sayısal aralık filtresi."""
    
    min_value: Optional[float] = None
    max_value: Optional[float] = None


# =============================================================================
# BATCH OPERATIONS
# =============================================================================

class BatchDeleteRequest(BaseModel):
    """Toplu silme istek schema."""
    
    ids: List[UUID] = Field(..., min_length=1, max_length=100)


class BatchUpdateRequest(BaseModel):
    """Toplu güncelleme istek schema."""
    
    ids: List[UUID] = Field(..., min_length=1, max_length=100)
    updates: Dict[str, Any]
