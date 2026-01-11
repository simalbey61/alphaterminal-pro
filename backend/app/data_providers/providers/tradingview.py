"""
AlphaTerminal Pro - TradingView Data Provider
=============================================

TradingView data provider using tvDatafeed library.
Primary data source for BIST and other markets.

Author: AlphaTerminal Team
Version: 1.0.0
"""

import logging
from datetime import datetime, timedelta
from typing import Optional, Dict, List, Any
import time
import threading

import pandas as pd
import numpy as np

from app.data_providers.providers.base import (
    BaseDataProvider, register_provider
)
from app.data_providers.enums import (
    DataInterval, DataSource, Market, DataQuality,
    AdjustmentType, SymbolType, LiquidityTier,
    TRADINGVIEW_INTERVALS
)
from app.data_providers.models import (
    SymbolInfo, MarketData, DataRequest
)
from app.data_providers.exceptions import (
    SymbolNotFoundError, NoDataError, ConnectionError,
    ProviderUnavailableError, DataProviderException
)


logger = logging.getLogger(__name__)


# TradingView library import
TV_AVAILABLE = False
TvDatafeed = None
Interval = None

try:
    from tvDatafeed import TvDatafeed as _TvDatafeed, Interval as _Interval
    TvDatafeed = _TvDatafeed
    Interval = _Interval
    TV_AVAILABLE = True
    logger.info("tvDatafeed library loaded successfully")
except ImportError:
    logger.warning("tvDatafeed not installed. TradingView provider unavailable.")


# Interval mapping
def get_tv_interval(interval: DataInterval):
    """Convert DataInterval to TvDatafeed Interval."""
    if not TV_AVAILABLE:
        return None
    
    mapping = {
        DataInterval.M1: Interval.in_1_minute,
        DataInterval.M3: Interval.in_3_minute,
        DataInterval.M5: Interval.in_5_minute,
        DataInterval.M15: Interval.in_15_minute,
        DataInterval.M30: Interval.in_30_minute,
        DataInterval.M45: Interval.in_45_minute,
        DataInterval.H1: Interval.in_1_hour,
        DataInterval.H2: Interval.in_2_hour,
        DataInterval.H3: Interval.in_3_hour,
        DataInterval.H4: Interval.in_4_hour,
        DataInterval.D1: Interval.in_daily,
        DataInterval.W1: Interval.in_weekly,
        DataInterval.MN1: Interval.in_monthly,
    }
    return mapping.get(interval)


# BIST symbol lists
BIST_XU030 = [
    "AKBNK", "ARCLK", "ASELS", "BIMAS", "DOHOL", "EKGYO", "EREGL",
    "FROTO", "GARAN", "GUBRF", "HEKTS", "ISCTR", "KCHOL", "KOZAA",
    "KOZAL", "KRDMD", "MGROS", "ODAS", "PETKM", "PGSUS", "SAHOL",
    "SASA", "SISE", "SOKM", "TAVHL", "TCELL", "THYAO", "TKFEN",
    "TOASO", "TUPRS", "VESTL", "YKBNK"
]

BIST_XU100_EXTRA = [
    "AEFES", "AFYON", "AGESA", "AGHOL", "AHGAZ", "AKFGY", "AKSA",
    "AKSUE", "AKYHO", "ALARK", "ALBRK", "ALFAS", "ALKIM", "ASUZU",
    "AYDEM", "BAGFS", "BASGZ", "BERA", "BIENY", "BIZIM", "BOBET",
    "BRISA", "BRYAT", "BUCIM", "CCOLA", "CEMTS", "CIMSA", "CLEBI",
    "DEVA", "DOAS", "EGEEN", "ENKAI", "EUPWR", "FENER", "GENIL",
    "GESAN", "GLYHO", "GOLTS", "GOODY", "GSRAY", "HALKB", "IHLGM",
    "INDES", "IPEKE", "ISGYO", "ISMEN", "JANTS", "KARSN", "KARTN",
    "KAYSE", "KERVT", "KLSER", "KMPUR", "KONYA", "KONTR", "KORDS",
    "LOGO", "MAVI", "MPARK", "NETAS", "NUGYO", "OTKAR", "OZKGY",
    "PAPIL", "PENTA", "QUAGR", "RAYSG", "RUBNS", "SELEC", "SKBNK",
    "SMRTG", "TATGD", "TKNSA", "TMSN", "TRGYO", "TSKB", "TTKOM",
    "TTRAK", "TUKAS", "TURSG", "ULKER", "ULUUN", "VAKBN", "VERUS",
    "VESBE", "YATAS", "YEOTK", "ZOREN"
]


@register_provider(DataSource.TRADINGVIEW)
class TradingViewProvider(BaseDataProvider):
    """
    TradingView data provider.
    
    Features:
    - Real-time and historical data
    - Multiple timeframes
    - BIST, US, and other markets
    - Thread-safe operations
    - Connection pooling
    """
    
    name = "TradingView"
    source = DataSource.TRADINGVIEW
    supported_intervals = [
        DataInterval.M1, DataInterval.M3, DataInterval.M5,
        DataInterval.M15, DataInterval.M30, DataInterval.M45,
        DataInterval.H1, DataInterval.H2, DataInterval.H3, DataInterval.H4,
        DataInterval.D1, DataInterval.W1, DataInterval.MN1
    ]
    supported_markets = [Market.BIST, Market.NYSE, Market.NASDAQ, Market.CRYPTO]
    requires_auth = False
    is_realtime = True
    
    default_requests_per_minute = 30  # Conservative rate limit
    default_requests_per_day = 5000
    
    def __init__(
        self,
        username: Optional[str] = None,
        password: Optional[str] = None,
        max_bars: int = 5000,
        **kwargs
    ):
        """
        Initialize TradingView provider.
        
        Args:
            username: TradingView username (optional)
            password: TradingView password (optional)
            max_bars: Maximum bars per request
            **kwargs: Additional base provider arguments
        """
        super().__init__(**kwargs)
        
        self.username = username
        self.password = password
        self.max_bars = max_bars
        
        self._tv_client: Optional[Any] = None
        self._lock = threading.Lock()
        
        # Symbol cache
        self._symbol_cache: Dict[str, SymbolInfo] = {}
        self._bist_symbols: List[str] = []
    
    def _do_initialize(self):
        """Initialize TradingView connection."""
        if not TV_AVAILABLE:
            raise ProviderUnavailableError(
                self.name,
                reason="tvDatafeed library not installed"
            )
        
        try:
            # Create client
            if self.username and self.password:
                self._tv_client = TvDatafeed(self.username, self.password)
                logger.info(f"{self.name}: Authenticated as {self.username}")
            else:
                self._tv_client = TvDatafeed()
                logger.info(f"{self.name}: Using anonymous connection")
            
            # Test connection
            self._test_connection()
            
            # Load BIST symbols
            self._load_bist_symbols()
            
        except Exception as e:
            raise ProviderUnavailableError(
                self.name,
                reason=f"Failed to connect: {e}"
            )
    
    def _do_shutdown(self):
        """Cleanup TradingView connection."""
        self._tv_client = None
        self._symbol_cache.clear()
    
    def _test_connection(self):
        """Test connection by fetching a known symbol."""
        try:
            # Try to fetch a small amount of data
            df = self._tv_client.get_hist(
                symbol="THYAO",
                exchange="BIST",
                interval=Interval.in_daily,
                n_bars=5
            )
            
            if df is None or df.empty:
                raise ConnectionError(
                    "Connection test failed - no data returned",
                    provider=self.name
                )
            
            logger.debug(f"{self.name}: Connection test successful")
            
        except Exception as e:
            raise ConnectionError(
                f"Connection test failed: {e}",
                provider=self.name
            )
    
    def _load_bist_symbols(self):
        """Load BIST symbol list."""
        # Combine known symbol lists
        self._bist_symbols = sorted(set(BIST_XU030 + BIST_XU100_EXTRA))
        logger.info(f"{self.name}: Loaded {len(self._bist_symbols)} BIST symbols")
    
    def _fetch_data(self, request: DataRequest) -> MarketData:
        """Fetch data from TradingView."""
        symbol = request.symbols[0] if isinstance(request.symbol, list) else request.symbol
        
        # Determine exchange
        exchange = self._get_exchange(symbol)
        
        # Get interval
        tv_interval = get_tv_interval(request.interval)
        if tv_interval is None:
            raise DataProviderException(
                f"Unsupported interval: {request.interval.value}",
                provider=self.name
            )
        
        # Calculate bars needed
        n_bars = self._calculate_bars(request)
        
        # Fetch with thread safety
        with self._lock:
            try:
                df = self._tv_client.get_hist(
                    symbol=symbol,
                    exchange=exchange,
                    interval=tv_interval,
                    n_bars=min(n_bars, self.max_bars)
                )
            except Exception as e:
                if "not found" in str(e).lower() or "invalid" in str(e).lower():
                    raise SymbolNotFoundError(
                        symbol=symbol,
                        provider=self.name
                    )
                raise DataProviderException(
                    f"Fetch failed: {e}",
                    provider=self.name,
                    symbol=symbol
                )
        
        # Handle empty result
        if df is None or df.empty:
            raise NoDataError(
                symbol=symbol,
                start_date=request.start_date,
                end_date=request.end_date,
                interval=request.interval.value,
                provider=self.name
            )
        
        # Normalize DataFrame
        df = self._normalize_dataframe(df)
        
        # Filter by date range if specified
        if request.start_date:
            df = df[df.index >= request.start_date]
        if request.end_date:
            df = df[df.index <= request.end_date]
        
        # Assess quality
        quality = self._assess_data_quality(df)
        
        return MarketData(
            symbol=symbol,
            interval=request.interval,
            data=df,
            source=self.source,
            quality=quality,
            adjustment=request.adjustment,
            metadata={
                "exchange": exchange,
                "bars_requested": n_bars,
                "bars_received": len(df)
            }
        )
    
    def _get_symbol_info(self, symbol: str) -> SymbolInfo:
        """Get symbol information."""
        # Check cache
        if symbol in self._symbol_cache:
            return self._symbol_cache[symbol]
        
        # Determine market and type
        if symbol in self._bist_symbols or self._is_bist_symbol(symbol):
            market = Market.BIST
            
            # Determine liquidity tier
            if symbol in BIST_XU030:
                tier = LiquidityTier.TIER_1
            elif symbol in BIST_XU100_EXTRA:
                tier = LiquidityTier.TIER_2
            else:
                tier = LiquidityTier.TIER_3
            
            info = SymbolInfo(
                symbol=symbol,
                name=symbol,  # TradingView doesn't provide names easily
                market=market,
                symbol_type=SymbolType.STOCK,
                currency="TRY",
                liquidity_tier=tier
            )
        else:
            # Default to unknown/other
            info = SymbolInfo(
                symbol=symbol,
                name=symbol,
                market=Market.NYSE,  # Default
                symbol_type=SymbolType.STOCK,
                currency="USD"
            )
        
        # Cache and return
        self._symbol_cache[symbol] = info
        return info
    
    def _get_available_symbols(self, market: Optional[Market] = None) -> List[str]:
        """Get available symbols."""
        if market == Market.BIST or market is None:
            return self._bist_symbols.copy()
        return []
    
    def _get_exchange(self, symbol: str) -> str:
        """Determine exchange for symbol."""
        symbol = symbol.upper()
        
        # BIST symbols
        if symbol in self._bist_symbols or self._is_bist_symbol(symbol):
            return "BIST"
        
        # Crypto
        if symbol.endswith("USDT") or symbol.endswith("BTC"):
            return "BINANCE"
        
        # Default to NASDAQ for US stocks
        return "NASDAQ"
    
    def _is_bist_symbol(self, symbol: str) -> bool:
        """Check if symbol looks like a BIST symbol."""
        # BIST symbols are typically 3-6 uppercase letters
        symbol = symbol.upper()
        if len(symbol) < 3 or len(symbol) > 6:
            return False
        if not symbol.isalpha():
            return False
        return True
    
    def _calculate_bars(self, request: DataRequest) -> int:
        """Calculate number of bars needed for request."""
        if request.bars:
            return request.bars
        
        if request.start_date and request.end_date:
            # Calculate based on date range
            days = (request.end_date - request.start_date).days
            
            # Adjust for interval
            if request.interval == DataInterval.D1:
                return int(days * 0.7)  # ~70% trading days
            elif request.interval == DataInterval.W1:
                return days // 7 + 1
            elif request.interval == DataInterval.MN1:
                return days // 30 + 1
            elif request.interval.is_intraday:
                trading_days = int(days * 0.7)
                bars_per_day = request.interval.bars_per_day
                return int(trading_days * bars_per_day)
        
        # Default
        return 500
    
    def _normalize_dataframe(self, df: pd.DataFrame) -> pd.DataFrame:
        """Normalize TradingView DataFrame."""
        if df is None or df.empty:
            return pd.DataFrame()
        
        # TradingView column names are lowercase
        df = df.copy()
        
        # Rename columns
        column_map = {
            'open': 'Open',
            'high': 'High',
            'low': 'Low',
            'close': 'Close',
            'volume': 'Volume'
        }
        df.rename(columns=column_map, inplace=True)
        
        # Ensure datetime index
        if not isinstance(df.index, pd.DatetimeIndex):
            df.index = pd.to_datetime(df.index)
        
        # Sort and dedupe
        df.sort_index(inplace=True)
        df = df[~df.index.duplicated(keep='last')]
        
        return df
    
    # =========================================================================
    # ADDITIONAL METHODS
    # =========================================================================
    
    def fetch_multiple(
        self,
        symbols: List[str],
        interval: DataInterval = DataInterval.D1,
        bars: int = 500,
        parallel: bool = True,
        max_workers: int = 5
    ) -> Dict[str, MarketData]:
        """
        Fetch data for multiple symbols.
        
        Args:
            symbols: List of symbols
            interval: Data interval
            bars: Number of bars
            parallel: Use parallel fetching
            max_workers: Max parallel workers
            
        Returns:
            Dict mapping symbol to MarketData
        """
        results = {}
        errors = {}
        
        if parallel and len(symbols) > 1:
            import concurrent.futures
            
            def fetch_one(sym):
                try:
                    request = DataRequest(
                        symbol=sym,
                        interval=interval,
                        bars=bars
                    )
                    return sym, self.get_data(request)
                except Exception as e:
                    return sym, e
            
            with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
                futures = {executor.submit(fetch_one, s): s for s in symbols}
                
                for future in concurrent.futures.as_completed(futures):
                    sym, result = future.result()
                    if isinstance(result, Exception):
                        errors[sym] = result
                    else:
                        results[sym] = result
                    
                    # Small delay to avoid rate limiting
                    time.sleep(0.1)
        else:
            for sym in symbols:
                try:
                    request = DataRequest(
                        symbol=sym,
                        interval=interval,
                        bars=bars
                    )
                    results[sym] = self.get_data(request)
                except Exception as e:
                    errors[sym] = e
                
                time.sleep(0.2)  # Rate limit delay
        
        if errors:
            logger.warning(
                f"{self.name}: Failed to fetch {len(errors)} symbols: "
                f"{list(errors.keys())[:5]}..."
            )
        
        return results
    
    def get_bist_xu030(
        self,
        interval: DataInterval = DataInterval.D1,
        bars: int = 500
    ) -> Dict[str, MarketData]:
        """Fetch all XU030 stocks."""
        return self.fetch_multiple(BIST_XU030, interval, bars)
    
    def get_bist_xu100(
        self,
        interval: DataInterval = DataInterval.D1,
        bars: int = 500
    ) -> Dict[str, MarketData]:
        """Fetch all XU100 stocks."""
        all_symbols = sorted(set(BIST_XU030 + BIST_XU100_EXTRA))
        return self.fetch_multiple(all_symbols, interval, bars)


# =============================================================================
# ALL EXPORTS
# =============================================================================

__all__ = [
    "TradingViewProvider",
    "TV_AVAILABLE",
    "BIST_XU030",
    "BIST_XU100_EXTRA",
]
