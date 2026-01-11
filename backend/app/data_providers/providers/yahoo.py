"""
AlphaTerminal Pro - Yahoo Finance Data Provider
===============================================

Yahoo Finance data provider using yfinance library.
Fallback/secondary data source.

Author: AlphaTerminal Team
Version: 1.0.0
"""

import logging
from datetime import datetime, timedelta
from typing import Optional, Dict, List, Any
import threading

import pandas as pd
import numpy as np

from app.data_providers.providers.base import (
    BaseDataProvider, register_provider
)
from app.data_providers.enums import (
    DataInterval, DataSource, Market, DataQuality,
    AdjustmentType, SymbolType, LiquidityTier,
    YAHOO_INTERVALS
)
from app.data_providers.models import (
    SymbolInfo, MarketData, DataRequest
)
from app.data_providers.exceptions import (
    SymbolNotFoundError, NoDataError, ConnectionError,
    ProviderUnavailableError, DataProviderException,
    RateLimitError
)


logger = logging.getLogger(__name__)


# Yahoo Finance library import
YF_AVAILABLE = False
yf = None

try:
    import yfinance as _yf
    yf = _yf
    YF_AVAILABLE = True
    logger.info("yfinance library loaded successfully")
except ImportError:
    logger.warning("yfinance not installed. Yahoo Finance provider unavailable.")


def get_yahoo_interval(interval: DataInterval) -> Optional[str]:
    """Convert DataInterval to yfinance interval string."""
    return YAHOO_INTERVALS.get(interval)


def get_yahoo_period(days: int) -> str:
    """Convert days to yfinance period string."""
    if days <= 7:
        return "5d"
    elif days <= 30:
        return "1mo"
    elif days <= 90:
        return "3mo"
    elif days <= 180:
        return "6mo"
    elif days <= 365:
        return "1y"
    elif days <= 730:
        return "2y"
    elif days <= 1825:
        return "5y"
    else:
        return "max"


@register_provider(DataSource.YAHOO_FINANCE)
class YahooFinanceProvider(BaseDataProvider):
    """
    Yahoo Finance data provider.
    
    Features:
    - Historical and (limited) real-time data
    - Multiple markets globally
    - Fundamental data available
    - Free, no authentication required
    - Thread-safe operations
    
    Limitations:
    - Intraday data limited to recent periods
    - Rate limiting without explicit headers
    - Occasional data gaps
    """
    
    name = "Yahoo Finance"
    source = DataSource.YAHOO_FINANCE
    supported_intervals = [
        DataInterval.M1, DataInterval.M2, DataInterval.M5,
        DataInterval.M15, DataInterval.M30,
        DataInterval.H1,
        DataInterval.D1, DataInterval.W1, DataInterval.MN1
    ]
    supported_markets = [
        Market.BIST, Market.NYSE, Market.NASDAQ, 
        Market.CRYPTO, Market.FOREX
    ]
    requires_auth = False
    is_realtime = False
    
    default_requests_per_minute = 60
    default_requests_per_day = 2000
    
    # BIST suffix for Yahoo Finance
    BIST_SUFFIX = ".IS"
    
    def __init__(
        self,
        auto_adjust: bool = True,
        prepost: bool = False,
        **kwargs
    ):
        """
        Initialize Yahoo Finance provider.
        
        Args:
            auto_adjust: Auto-adjust OHLC for splits/dividends
            prepost: Include pre/post market data
            **kwargs: Additional base provider arguments
        """
        super().__init__(**kwargs)
        
        self.auto_adjust = auto_adjust
        self.prepost = prepost
        
        self._lock = threading.Lock()
        self._ticker_cache: Dict[str, Any] = {}
        self._symbol_cache: Dict[str, SymbolInfo] = {}
    
    def _do_initialize(self):
        """Initialize Yahoo Finance provider."""
        if not YF_AVAILABLE:
            raise ProviderUnavailableError(
                self.name,
                reason="yfinance library not installed"
            )
        
        # Test connection
        self._test_connection()
    
    def _do_shutdown(self):
        """Cleanup resources."""
        self._ticker_cache.clear()
        self._symbol_cache.clear()
    
    def _test_connection(self):
        """Test connection by fetching known symbol."""
        try:
            ticker = yf.Ticker("AAPL")
            hist = ticker.history(period="5d")
            
            if hist.empty:
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
    
    def _fetch_data(self, request: DataRequest) -> MarketData:
        """Fetch data from Yahoo Finance."""
        symbol = request.symbols[0] if isinstance(request.symbol, list) else request.symbol
        
        # Convert to Yahoo symbol format
        yahoo_symbol = self._to_yahoo_symbol(symbol)
        
        # Get interval
        yf_interval = get_yahoo_interval(request.interval)
        if yf_interval is None:
            raise DataProviderException(
                f"Unsupported interval: {request.interval.value}",
                provider=self.name
            )
        
        # Fetch data
        with self._lock:
            try:
                ticker = self._get_ticker(yahoo_symbol)
                
                if request.start_date and request.end_date:
                    df = ticker.history(
                        start=request.start_date,
                        end=request.end_date,
                        interval=yf_interval,
                        auto_adjust=self.auto_adjust,
                        prepost=self.prepost
                    )
                elif request.bars:
                    # Calculate period based on bars needed
                    period = self._calculate_period(request.bars, request.interval)
                    df = ticker.history(
                        period=period,
                        interval=yf_interval,
                        auto_adjust=self.auto_adjust,
                        prepost=self.prepost
                    )
                else:
                    df = ticker.history(
                        period="1y",
                        interval=yf_interval,
                        auto_adjust=self.auto_adjust,
                        prepost=self.prepost
                    )
                    
            except Exception as e:
                error_str = str(e).lower()
                if "no data" in error_str or "not found" in error_str:
                    raise SymbolNotFoundError(
                        symbol=symbol,
                        provider=self.name
                    )
                if "rate limit" in error_str or "too many" in error_str:
                    raise RateLimitError(
                        f"Yahoo Finance rate limited",
                        provider=self.name,
                        retry_after=60
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
        
        # Assess quality
        quality = self._assess_data_quality(df)
        
        return MarketData(
            symbol=symbol,
            interval=request.interval,
            data=df,
            source=self.source,
            quality=quality,
            adjustment=AdjustmentType.FULL if self.auto_adjust else AdjustmentType.NONE,
            metadata={
                "yahoo_symbol": yahoo_symbol,
                "bars_received": len(df)
            }
        )
    
    def _get_symbol_info(self, symbol: str) -> SymbolInfo:
        """Get symbol information from Yahoo Finance."""
        # Check cache
        if symbol in self._symbol_cache:
            return self._symbol_cache[symbol]
        
        yahoo_symbol = self._to_yahoo_symbol(symbol)
        
        try:
            ticker = self._get_ticker(yahoo_symbol)
            info = ticker.info
            
            # Determine market
            exchange = info.get("exchange", "").upper()
            if "IST" in exchange or symbol.endswith(".IS"):
                market = Market.BIST
                currency = "TRY"
            elif exchange in ["NYSE", "NYQ"]:
                market = Market.NYSE
                currency = "USD"
            elif exchange in ["NMS", "NASDAQ", "NGM"]:
                market = Market.NASDAQ
                currency = "USD"
            else:
                market = Market.NYSE
                currency = info.get("currency", "USD")
            
            # Determine symbol type
            quote_type = info.get("quoteType", "EQUITY").upper()
            if quote_type == "ETF":
                sym_type = SymbolType.ETF
            elif quote_type == "INDEX":
                sym_type = SymbolType.INDEX
            elif quote_type == "CRYPTOCURRENCY":
                sym_type = SymbolType.CRYPTO
            elif quote_type == "CURRENCY":
                sym_type = SymbolType.FOREX
            else:
                sym_type = SymbolType.STOCK
            
            symbol_info = SymbolInfo(
                symbol=symbol,
                name=info.get("longName") or info.get("shortName") or symbol,
                market=market,
                symbol_type=sym_type,
                sector=info.get("sector"),
                industry=info.get("industry"),
                currency=currency,
                description=info.get("longBusinessSummary"),
                metadata={
                    "yahoo_symbol": yahoo_symbol,
                    "market_cap": info.get("marketCap"),
                    "pe_ratio": info.get("trailingPE"),
                    "dividend_yield": info.get("dividendYield"),
                }
            )
            
            self._symbol_cache[symbol] = symbol_info
            return symbol_info
            
        except Exception as e:
            logger.warning(f"{self.name}: Failed to get info for {symbol}: {e}")
            
            # Return basic info
            return SymbolInfo(
                symbol=symbol,
                name=symbol,
                market=Market.BIST if self._is_bist_symbol(symbol) else Market.NYSE,
                symbol_type=SymbolType.STOCK
            )
    
    def _get_available_symbols(self, market: Optional[Market] = None) -> List[str]:
        """
        Get available symbols.
        
        Note: Yahoo Finance doesn't provide a symbol list API.
        Returns empty list - use with TradingView provider for symbol discovery.
        """
        return []
    
    def _get_ticker(self, yahoo_symbol: str) -> Any:
        """Get or create ticker object."""
        if yahoo_symbol not in self._ticker_cache:
            self._ticker_cache[yahoo_symbol] = yf.Ticker(yahoo_symbol)
        return self._ticker_cache[yahoo_symbol]
    
    def _to_yahoo_symbol(self, symbol: str) -> str:
        """Convert symbol to Yahoo Finance format."""
        symbol = symbol.upper().strip()
        
        # Already has suffix
        if "." in symbol:
            return symbol
        
        # Check if BIST symbol
        if self._is_bist_symbol(symbol):
            return f"{symbol}{self.BIST_SUFFIX}"
        
        return symbol
    
    def _from_yahoo_symbol(self, yahoo_symbol: str) -> str:
        """Convert Yahoo symbol back to standard format."""
        if yahoo_symbol.endswith(self.BIST_SUFFIX):
            return yahoo_symbol[:-len(self.BIST_SUFFIX)]
        return yahoo_symbol
    
    def _is_bist_symbol(self, symbol: str) -> bool:
        """Check if symbol is likely a BIST symbol."""
        symbol = symbol.upper()
        
        # Known BIST symbols
        bist_indicators = [
            "THYAO", "GARAN", "AKBNK", "EREGL", "ASELS",
            "BIMAS", "SAHOL", "KCHOL", "TCELL", "TUPRS"
        ]
        
        if symbol in bist_indicators:
            return True
        
        # BIST symbols are typically 3-6 uppercase letters
        if len(symbol) >= 3 and len(symbol) <= 6 and symbol.isalpha():
            return True
        
        return False
    
    def _calculate_period(self, bars: int, interval: DataInterval) -> str:
        """Calculate yfinance period string for requested bars."""
        # Estimate days needed
        if interval == DataInterval.D1:
            days = int(bars * 1.5)  # Account for weekends
        elif interval == DataInterval.W1:
            days = bars * 7
        elif interval == DataInterval.MN1:
            days = bars * 30
        elif interval.is_intraday:
            trading_minutes_per_day = 390  # 6.5 hours
            minutes_needed = bars * interval.minutes
            days = max(7, int(minutes_needed / trading_minutes_per_day * 1.5))
        else:
            days = bars * 2
        
        return get_yahoo_period(days)
    
    def _normalize_dataframe(self, df: pd.DataFrame) -> pd.DataFrame:
        """Normalize Yahoo Finance DataFrame."""
        if df is None or df.empty:
            return pd.DataFrame()
        
        df = df.copy()
        
        # Yahoo returns capitalized columns already
        # But may have extra columns
        standard_cols = ['Open', 'High', 'Low', 'Close', 'Volume']
        
        # Keep only standard columns that exist
        cols_to_keep = [c for c in standard_cols if c in df.columns]
        df = df[cols_to_keep]
        
        # Ensure datetime index
        if not isinstance(df.index, pd.DatetimeIndex):
            df.index = pd.to_datetime(df.index)
        
        # Remove timezone info for consistency
        if df.index.tz is not None:
            df.index = df.index.tz_localize(None)
        
        # Sort and dedupe
        df.sort_index(inplace=True)
        df = df[~df.index.duplicated(keep='last')]
        
        return df
    
    # =========================================================================
    # ADDITIONAL METHODS
    # =========================================================================
    
    def get_fundamentals(self, symbol: str) -> Dict[str, Any]:
        """
        Get fundamental data for symbol.
        
        Args:
            symbol: Trading symbol
            
        Returns:
            Dict with fundamental metrics
        """
        self._ensure_initialized()
        
        yahoo_symbol = self._to_yahoo_symbol(symbol)
        ticker = self._get_ticker(yahoo_symbol)
        
        try:
            info = ticker.info
            
            return {
                "symbol": symbol,
                "name": info.get("longName"),
                "sector": info.get("sector"),
                "industry": info.get("industry"),
                "market_cap": info.get("marketCap"),
                "enterprise_value": info.get("enterpriseValue"),
                "pe_ratio": info.get("trailingPE"),
                "forward_pe": info.get("forwardPE"),
                "peg_ratio": info.get("pegRatio"),
                "price_to_book": info.get("priceToBook"),
                "price_to_sales": info.get("priceToSalesTrailing12Months"),
                "dividend_yield": info.get("dividendYield"),
                "beta": info.get("beta"),
                "52_week_high": info.get("fiftyTwoWeekHigh"),
                "52_week_low": info.get("fiftyTwoWeekLow"),
                "avg_volume": info.get("averageVolume"),
                "shares_outstanding": info.get("sharesOutstanding"),
                "float_shares": info.get("floatShares"),
                "profit_margin": info.get("profitMargins"),
                "operating_margin": info.get("operatingMargins"),
                "return_on_equity": info.get("returnOnEquity"),
                "return_on_assets": info.get("returnOnAssets"),
                "revenue": info.get("totalRevenue"),
                "revenue_growth": info.get("revenueGrowth"),
                "earnings_growth": info.get("earningsGrowth"),
                "current_ratio": info.get("currentRatio"),
                "debt_to_equity": info.get("debtToEquity"),
            }
        except Exception as e:
            logger.error(f"{self.name}: Failed to get fundamentals for {symbol}: {e}")
            return {"symbol": symbol, "error": str(e)}
    
    def get_dividends(self, symbol: str) -> pd.DataFrame:
        """Get dividend history for symbol."""
        self._ensure_initialized()
        
        yahoo_symbol = self._to_yahoo_symbol(symbol)
        ticker = self._get_ticker(yahoo_symbol)
        
        return ticker.dividends
    
    def get_splits(self, symbol: str) -> pd.DataFrame:
        """Get stock split history for symbol."""
        self._ensure_initialized()
        
        yahoo_symbol = self._to_yahoo_symbol(symbol)
        ticker = self._get_ticker(yahoo_symbol)
        
        return ticker.splits


# =============================================================================
# ALL EXPORTS
# =============================================================================

__all__ = [
    "YahooFinanceProvider",
    "YF_AVAILABLE",
]
