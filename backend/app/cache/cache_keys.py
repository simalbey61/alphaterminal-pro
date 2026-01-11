"""
AlphaTerminal Pro - Cache Keys
==============================

Redis cache key pattern'leri.

Author: AlphaTerminal Team
Version: 1.0.0
"""

from __future__ import annotations

from typing import Optional


class CacheKeys:
    """
    Cache key pattern'leri.
    
    Tüm cache key'lerinin merkezi tanımı.
    Bu class'ı kullanarak tutarlı key isimlendirmesi sağlanır.
    
    Example:
        ```python
        key = CacheKeys.market_data("THYAO", "1h")
        # "market:data:THYAO:1h"
        
        key = CacheKeys.signal_active()
        # "signals:active"
        ```
    """
    
    # =========================================================================
    # PREFIXES
    # =========================================================================
    
    PREFIX_MARKET = "market"
    PREFIX_ANALYSIS = "analysis"
    PREFIX_FEATURES = "features"
    PREFIX_SIGNALS = "signals"
    PREFIX_STRATEGIES = "strategies"
    PREFIX_REALTIME = "realtime"
    PREFIX_SESSION = "session"
    PREFIX_RATELIMIT = "ratelimit"
    PREFIX_STATS = "stats"
    PREFIX_USER = "user"
    
    # =========================================================================
    # MARKET DATA
    # =========================================================================
    
    @staticmethod
    def market_data(symbol: str, interval: str) -> str:
        """Market data cache key."""
        return f"{CacheKeys.PREFIX_MARKET}:data:{symbol}:{interval}"
    
    @staticmethod
    def market_indicators(symbol: str) -> str:
        """Calculated indicators cache key."""
        return f"{CacheKeys.PREFIX_MARKET}:indicators:{symbol}"
    
    @staticmethod
    def market_overview() -> str:
        """Market overview cache key."""
        return f"{CacheKeys.PREFIX_MARKET}:overview"
    
    @staticmethod
    def market_movers() -> str:
        """Market movers cache key."""
        return f"{CacheKeys.PREFIX_MARKET}:movers"
    
    # =========================================================================
    # ANALYSIS
    # =========================================================================
    
    @staticmethod
    def analysis_smc(symbol: str, timeframe: str) -> str:
        """SMC analysis cache key."""
        return f"{CacheKeys.PREFIX_ANALYSIS}:smc:{symbol}:{timeframe}"
    
    @staticmethod
    def analysis_orderflow(symbol: str) -> str:
        """Order flow analysis cache key."""
        return f"{CacheKeys.PREFIX_ANALYSIS}:flow:{symbol}"
    
    @staticmethod
    def analysis_alpha(symbol: str) -> str:
        """Alpha analysis cache key."""
        return f"{CacheKeys.PREFIX_ANALYSIS}:alpha:{symbol}"
    
    @staticmethod
    def analysis_full(symbol: str, timeframe: str) -> str:
        """Full analysis cache key."""
        return f"{CacheKeys.PREFIX_ANALYSIS}:full:{symbol}:{timeframe}"
    
    # =========================================================================
    # FEATURES
    # =========================================================================
    
    @staticmethod
    def features(symbol: str, date: str) -> str:
        """Feature cache key."""
        return f"{CacheKeys.PREFIX_FEATURES}:{symbol}:{date}"
    
    @staticmethod
    def features_pattern(symbol: str) -> str:
        """Feature pattern (for deletion)."""
        return f"{CacheKeys.PREFIX_FEATURES}:{symbol}:*"
    
    # =========================================================================
    # SIGNALS
    # =========================================================================
    
    @staticmethod
    def signal_active() -> str:
        """Active signals list cache key."""
        return f"{CacheKeys.PREFIX_SIGNALS}:active"
    
    @staticmethod
    def signal_by_id(signal_id: str) -> str:
        """Signal by ID cache key."""
        return f"{CacheKeys.PREFIX_SIGNALS}:id:{signal_id}"
    
    @staticmethod
    def signal_by_symbol(symbol: str) -> str:
        """Signals by symbol cache key."""
        return f"{CacheKeys.PREFIX_SIGNALS}:symbol:{symbol}"
    
    @staticmethod
    def signal_by_tier(tier: str) -> str:
        """Signals by tier cache key."""
        return f"{CacheKeys.PREFIX_SIGNALS}:tier:{tier}"
    
    @staticmethod
    def signal_stats() -> str:
        """Signal statistics cache key."""
        return f"{CacheKeys.PREFIX_SIGNALS}:stats"
    
    # =========================================================================
    # STRATEGIES
    # =========================================================================
    
    @staticmethod
    def strategy_active() -> str:
        """Active strategies list cache key."""
        return f"{CacheKeys.PREFIX_STRATEGIES}:active"
    
    @staticmethod
    def strategy_by_id(strategy_id: str) -> str:
        """Strategy by ID cache key."""
        return f"{CacheKeys.PREFIX_STRATEGIES}:id:{strategy_id}"
    
    @staticmethod
    def strategy_performance(strategy_id: str) -> str:
        """Strategy performance cache key."""
        return f"{CacheKeys.PREFIX_STRATEGIES}:performance:{strategy_id}"
    
    @staticmethod
    def strategy_stats() -> str:
        """Strategy statistics cache key."""
        return f"{CacheKeys.PREFIX_STRATEGIES}:stats"
    
    # =========================================================================
    # REAL-TIME (Pub/Sub Channels)
    # =========================================================================
    
    @staticmethod
    def realtime_prices() -> str:
        """Real-time prices channel."""
        return f"{CacheKeys.PREFIX_REALTIME}:prices"
    
    @staticmethod
    def realtime_signals() -> str:
        """Real-time signals channel."""
        return f"{CacheKeys.PREFIX_REALTIME}:signals"
    
    @staticmethod
    def realtime_alerts() -> str:
        """Real-time alerts channel."""
        return f"{CacheKeys.PREFIX_REALTIME}:alerts"
    
    @staticmethod
    def realtime_symbol(symbol: str) -> str:
        """Real-time symbol updates channel."""
        return f"{CacheKeys.PREFIX_REALTIME}:symbol:{symbol}"
    
    # =========================================================================
    # SESSION & RATE LIMITING
    # =========================================================================
    
    @staticmethod
    def session(user_id: str) -> str:
        """User session cache key."""
        return f"{CacheKeys.PREFIX_SESSION}:{user_id}"
    
    @staticmethod
    def ratelimit_api(ip: str) -> str:
        """API rate limit cache key."""
        return f"{CacheKeys.PREFIX_RATELIMIT}:api:{ip}"
    
    @staticmethod
    def ratelimit_telegram(user_id: int) -> str:
        """Telegram rate limit cache key."""
        return f"{CacheKeys.PREFIX_RATELIMIT}:telegram:{user_id}"
    
    # =========================================================================
    # STATS & LEADERBOARDS
    # =========================================================================
    
    @staticmethod
    def stats_top_strategies() -> str:
        """Top strategies sorted set."""
        return f"{CacheKeys.PREFIX_STATS}:top_strategies"
    
    @staticmethod
    def stats_market_summary() -> str:
        """Market summary cache key."""
        return f"{CacheKeys.PREFIX_STATS}:market_summary"
    
    @staticmethod
    def stats_sector_performance() -> str:
        """Sector performance cache key."""
        return f"{CacheKeys.PREFIX_STATS}:sector_performance"
    
    # =========================================================================
    # USER
    # =========================================================================
    
    @staticmethod
    def user_watchlist(user_id: str) -> str:
        """User watchlist cache key."""
        return f"{CacheKeys.PREFIX_USER}:{user_id}:watchlist"
    
    @staticmethod
    def user_notifications(user_id: str) -> str:
        """User notifications cache key."""
        return f"{CacheKeys.PREFIX_USER}:{user_id}:notifications"
    
    @staticmethod
    def user_preferences(user_id: str) -> str:
        """User preferences cache key."""
        return f"{CacheKeys.PREFIX_USER}:{user_id}:preferences"


# TTL sabitleri (saniye)
class CacheTTL:
    """Cache TTL sabitleri."""
    
    # Kısa süreli (real-time data)
    VERY_SHORT = 30  # 30 saniye
    SHORT = 60  # 1 dakika
    
    # Orta süreli (market data)
    MARKET_DATA = 300  # 5 dakika
    ANALYSIS = 300  # 5 dakika
    FEATURES = 600  # 10 dakika
    
    # Uzun süreli (computed data)
    SIGNALS = 3600  # 1 saat
    STRATEGIES = 3600  # 1 saat
    STATS = 900  # 15 dakika
    
    # Çok uzun süreli
    SESSION = 86400  # 1 gün
    USER_DATA = 43200  # 12 saat
