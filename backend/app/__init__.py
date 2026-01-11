"""
AlphaTerminal Pro - Backend Application
=======================================

Kurumsal seviye BIST analiz ve trading platformu.

Modules:
    - core: Mevcut analiz motorları (SMC, OrderFlow, Alpha, Risk)
    - api: REST API endpoints
    - db: Veritabanı modelleri ve repository'ler
    - services: İş mantığı servisleri
    - ai_strategy: 7 katmanlı AI strateji sistemi
    - telegram: Telegram bot entegrasyonu
    - websocket: Gerçek zamanlı iletişim
    - cache: Redis cache yönetimi
    - tasks: Arka plan görevleri
    - backtest: Kurumsal backtesting framework

Author: AlphaTerminal Team
Version: 1.0.0
License: Proprietary
"""

__version__ = "1.0.0"
__author__ = "AlphaTerminal Team"
__email__ = "support@alphaterminal.com"

# Lazy imports to avoid pydantic dependency issues
# These are only loaded when actually needed
_settings = None
_logger = None

def get_settings():
    """Lazy load settings."""
    global _settings
    if _settings is None:
        from app.config import settings
        _settings = settings
    return _settings

def get_logger():
    """Lazy load logger."""
    global _logger
    if _logger is None:
        from app.config import logger
        _logger = logger
    return _logger

# For backwards compatibility
try:
    from app.config import settings, logger
except ImportError:
    # pydantic not installed, use lazy loading
    settings = None
    logger = None

__all__ = [
    "__version__",
    "__author__",
    "__email__",
    "get_settings",
    "get_logger",
    "settings",
    "logger",
]
