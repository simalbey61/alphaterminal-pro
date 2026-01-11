"""
AlphaTerminal Pro - Database Models
===================================

Tüm SQLAlchemy modellerinin merkezi export noktası.

Author: AlphaTerminal Team
Version: 1.0.0
"""

from app.db.models.base import (
    Base,
    BaseModel,
    SoftDeleteModel,
    FullModel,
    TimestampMixin,
    UUIDMixin,
    SoftDeleteMixin,
    ActiveMixin,
)

from app.db.models.user import UserModel
from app.db.models.stock import StockModel
from app.db.models.signal import SignalModel
from app.db.models.strategy import AIStrategyModel
from app.db.models.portfolio import PortfolioModel, PositionModel
from app.db.models.ai_models import (
    WinnerHistoryModel,
    DiscoveredPatternModel,
    EvolutionLogModel,
    MarketRegimeModel,
    StrategyPerformanceModel,
)
from app.db.models.user_models import WatchlistModel, NotificationModel

__all__ = [
    # Base classes
    "Base",
    "BaseModel",
    "SoftDeleteModel",
    "FullModel",
    "TimestampMixin",
    "UUIDMixin",
    "SoftDeleteMixin",
    "ActiveMixin",
    
    # User
    "UserModel",
    
    # Market
    "StockModel",
    "SignalModel",
    
    # AI Strategy
    "AIStrategyModel",
    "WinnerHistoryModel",
    "DiscoveredPatternModel",
    "EvolutionLogModel",
    "MarketRegimeModel",
    "StrategyPerformanceModel",
    
    # Portfolio
    "PortfolioModel",
    "PositionModel",
    
    # User related
    "WatchlistModel",
    "NotificationModel",
]
