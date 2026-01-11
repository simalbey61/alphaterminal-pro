"""
AlphaTerminal Pro - Database Module
===================================

Veritabanı modelleri, repository'ler ve bağlantı yönetimi.

Author: AlphaTerminal Team
Version: 1.0.0
"""

from app.db.database import (
    db,
    DatabaseManager,
    get_session,
    init_db,
    close_db,
)

from app.db.models import (
    Base,
    BaseModel,
    UserModel,
    StockModel,
    SignalModel,
    AIStrategyModel,
    PortfolioModel,
    PositionModel,
    WinnerHistoryModel,
    DiscoveredPatternModel,
    EvolutionLogModel,
    MarketRegimeModel,
    StrategyPerformanceModel,
    WatchlistModel,
    NotificationModel,
)

from app.db.repositories import (
    BaseRepository,
    StockRepository,
    SignalRepository,
    StrategyRepository,
)

__all__ = [
    # Database management
    "db",
    "DatabaseManager",
    "get_session",
    "init_db",
    "close_db",
    
    # Base
    "Base",
    "BaseModel",
    
    # Models
    "UserModel",
    "StockModel",
    "SignalModel",
    "AIStrategyModel",
    "PortfolioModel",
    "PositionModel",
    "WinnerHistoryModel",
    "DiscoveredPatternModel",
    "EvolutionLogModel",
    "MarketRegimeModel",
    "StrategyPerformanceModel",
    "WatchlistModel",
    "NotificationModel",
    
    # Repositories
    "BaseRepository",
    "StockRepository",
    "SignalRepository",
    "StrategyRepository",
]
