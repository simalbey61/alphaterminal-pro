# core/data_engine.py - AlphaTerminal Pro v4.1
# Kurumsal Seviye Veri Yönetim Motoru
# TradingView + Yahoo Finance Hibrit Sistem

import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Optional, Dict, List, Tuple
import pickle
import os
import hashlib
import time

# ta kütüphanesi (pandas_ta yerine)
import ta
from ta.trend import EMAIndicator, SMAIndicator, ADXIndicator, MACD
from ta.momentum import RSIIndicator, StochasticOscillator
from ta.volatility import AverageTrueRange, BollingerBands
from ta.volume import OnBalanceVolumeIndicator, MFIIndicator

import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from app.core.config import (
    logger, CACHE_DIR, BIST_INDEX,
    DEFAULT_INTERVAL, DEFAULT_PERIOD, SUPPORTED_PERIODS
)

# TradingView modülünü import et
try:
    from app.core.bist_data_fetcher import BISTDataFetcher, TV_AVAILABLE

    TRADINGVIEW_ENABLED = TV_AVAILABLE
    logger.info("✅ TradingView modülü yüklendi")
except ImportError:
    TRADINGVIEW_ENABLED = False
    logger.warning("⚠️ TradingView modülü yüklenemedi, Yahoo Finance kullanılacak")


class DataEngine:
    """
    Kurumsal Seviye Veri Yönetim Motoru

    v4.1 Güncellemesi:
    - TradingView birincil veri kaynağı (hızlı, paralel)
    - Yahoo Finance yedek kaynak
    - Dinamik BIST hisse listesi (500+ hisse)
    - Akıllı önbellekleme sistemi

    Özellikler:
    - Multi-timeframe veri çekme
    - Akıllı önbellekleme (caching)
    - Veri temizleme ve standardizasyon
    - Teknik indikatör hesaplama
    - Batch veri çekme (API limit koruması)
    """

    def __init__(self, cache_ttl: int = 300):
        """
        Args:
            cache_ttl: Önbellek geçerlilik süresi (saniye)
        """
        self.cache_ttl = cache_ttl
        self.cache = {}
        self._rate_limit_delay = 0.2  # API rate limit koruması

        # TradingView entegrasyonu
        self.tv_enabled = TRADINGVIEW_ENABLED
        self.bist_fetcher = None

        if self.tv_enabled:
            try:
                self.bist_fetcher = BISTDataFetcher(use_cache=True, cache_ttl=600)
                logger.info("✅ TradingView veri kaynağı aktif")
            except Exception as e:
                logger.warning(f"⚠️ TradingView başlatılamadı: {e}")
                self.tv_enabled = False

        # Proxy endeks için likit hisseler (XU100 çalışmazsa kullanılır)
        self._proxy_symbols = ["THYAO", "GARAN", "AKBNK", "EREGL", "ASELS"]

    def _get_cache_key(self, symbol: str, interval: str, period: str) -> str:
        """Önbellek anahtarı oluştur"""
        return hashlib.md5(f"{symbol}_{interval}_{period}".encode()).hexdigest()

    def _is_cache_valid(self, cache_key: str) -> bool:
        """Önbellek geçerli mi kontrol et"""
        if cache_key not in self.cache:
            return False

        cache_time = self.cache[cache_key].get('timestamp', 0)
        return (time.time() - cache_time) < self.cache_ttl

    def _save_to_disk_cache(self, cache_key: str, data: pd.DataFrame) -> None:
        """Disk önbelleğine kaydet"""
        try:
            cache_path = os.path.join(CACHE_DIR, f"{cache_key}.pkl")
            with open(cache_path, 'wb') as f:
                pickle.dump({
                    'data': data,
                    'timestamp': time.time()
                }, f)
        except Exception as e:
            logger.warning(f"Disk cache yazma hatası: {e}")

    def _load_from_disk_cache(self, cache_key: str) -> Optional[pd.DataFrame]:
        """Disk önbelleğinden oku"""
        try:
            cache_path = os.path.join(CACHE_DIR, f"{cache_key}.pkl")
            if os.path.exists(cache_path):
                with open(cache_path, 'rb') as f:
                    cached = pickle.load(f)
                    if (time.time() - cached['timestamp']) < self.cache_ttl * 12:  # Disk cache daha uzun
                        return cached['data']
        except Exception as e:
            logger.warning(f"Disk cache okuma hatası: {e}")
        return None

    def fetch_data(
            self,
            symbol: str,
            interval: str = DEFAULT_INTERVAL,
            period: str = DEFAULT_PERIOD,
            use_cache: bool = True
    ) -> Optional[pd.DataFrame]:
        """
        Hisse verisi çek ve işle - TradingView birincil, Yahoo Finance yedek

        Args:
            symbol: Hisse kodu (örn: "THYAO" veya "THYAO.IS")
            interval: Zaman dilimi ("15m", "1h", "4h", "1d")
            period: Geçmiş dönem ("5d", "1mo", "3mo", "1y")
            use_cache: Önbellek kullan

        Returns:
            İşlenmiş DataFrame veya None
        """
        try:
            # Sembol temizleme
            clean_symbol = symbol.replace(".IS", "").upper()
            yahoo_symbol = f"{clean_symbol}.IS"

            cache_key = self._get_cache_key(yahoo_symbol, interval, period)

            # Önbellek kontrolü
            if use_cache:
                if self._is_cache_valid(cache_key):
                    logger.debug(f"✅ {clean_symbol} RAM önbellekten yüklendi")
                    return self.cache[cache_key]['data'].copy()

                disk_data = self._load_from_disk_cache(cache_key)
                if disk_data is not None:
                    self.cache[cache_key] = {'data': disk_data, 'timestamp': time.time()}
                    logger.debug(f"✅ {clean_symbol} disk önbellekten yüklendi")
                    return disk_data.copy()

            df = None

            # 1. TradingView ile dene (birincil kaynak)
            if self.tv_enabled and self.bist_fetcher:
                try:
                    # Period'u n_bars'a çevir
                    n_bars_map = {"5d": 100, "1mo": 300, "3mo": 500, "1y": 1000}
                    n_bars = n_bars_map.get(period, 500)

                    logger.info(f"📊 {clean_symbol} TradingView'den çekiliyor ({interval})...")
                    df = self.bist_fetcher.get_stock_data(clean_symbol, interval, n_bars)

                    if df is not None and len(df) >= 30:
                        logger.info(f"✅ {clean_symbol} TradingView'den alındı ({len(df)} bar)")
                except Exception as e:
                    logger.debug(f"TradingView hatası ({clean_symbol}): {e}")
                    df = None

            # 2. Yahoo Finance yedek
            if df is None or len(df) < 30:
                logger.info(f"📊 {yahoo_symbol} Yahoo Finance'dan çekiliyor ({interval}/{period})...")

                df = yf.download(
                    yahoo_symbol,
                    period=period,
                    interval=interval,
                    progress=False,
                    timeout=10
                )

                if df.empty or len(df) < 30:
                    logger.warning(f"⚠️ {clean_symbol} için yetersiz veri ({len(df) if not df.empty else 0} bar)")
                    return None

                logger.info(f"✅ {clean_symbol} Yahoo Finance'dan alındı ({len(df)} bar)")

            # Veri temizleme
            df = self._clean_data(df)

            # Teknik indikatörler
            df = self._add_indicators(df)

            # Önbelleğe kaydet
            self.cache[cache_key] = {'data': df, 'timestamp': time.time()}
            self._save_to_disk_cache(cache_key, df)

            return df.copy()

        except Exception as e:
            logger.error(f"❌ Veri hatası ({symbol}): {e}")
            return None

    def _clean_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """Veri temizleme ve standardizasyon"""
        df = df.copy()

        # Multi-index temizliği
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)

        # Timezone standardizasyonu
        if df.index.tz is not None:
            df.index = df.index.tz_localize(None)

        # Eksik veri doldurma (forward fill)
        df = df.ffill()

        # Sıfır veya negatif değerleri temizle
        for col in ['Open', 'High', 'Low', 'Close', 'Volume']:
            if col in df.columns:
                df[col] = df[col].replace(0, np.nan)
                df[col] = df[col].ffill()

        # OHLC tutarlılık kontrolü
        df['High'] = df[['Open', 'High', 'Low', 'Close']].max(axis=1)
        df['Low'] = df[['Open', 'High', 'Low', 'Close']].min(axis=1)

        return df.dropna()

    def _add_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """Teknik indikatörler ekle - ta kütüphanesi ile"""
        df = df.copy()

        # ═══════════════════════════════════════════════════════════════════
        # TREND İNDİKATÖRLERİ
        # ═══════════════════════════════════════════════════════════════════
        try:
            df['EMA9'] = EMAIndicator(df['Close'], window=9).ema_indicator()
            df['EMA20'] = EMAIndicator(df['Close'], window=20).ema_indicator()
            df['EMA50'] = EMAIndicator(df['Close'], window=50).ema_indicator()
            df['EMA200'] = EMAIndicator(df['Close'], window=200).ema_indicator()
            df['SMA20'] = SMAIndicator(df['Close'], window=20).sma_indicator()
            df['SMA50'] = SMAIndicator(df['Close'], window=50).sma_indicator()
        except Exception as e:
            logger.debug(f"EMA/SMA hesaplama hatası: {e}")
            # Manuel hesaplama
            df['EMA9'] = df['Close'].ewm(span=9, adjust=False).mean()
            df['EMA20'] = df['Close'].ewm(span=20, adjust=False).mean()
            df['EMA50'] = df['Close'].ewm(span=50, adjust=False).mean()
            df['EMA200'] = df['Close'].ewm(span=200, adjust=False).mean()
            df['SMA20'] = df['Close'].rolling(20).mean()
            df['SMA50'] = df['Close'].rolling(50).mean()

        # ═══════════════════════════════════════════════════════════════════
        # MOMENTUM İNDİKATÖRLERİ
        # ═══════════════════════════════════════════════════════════════════
        try:
            df['RSI'] = RSIIndicator(df['Close'], window=14).rsi()
        except Exception as e:
            logger.debug(f"RSI hesaplama hatası: {e}")
            # Manuel RSI
            delta = df['Close'].diff()
            gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
            rs = gain / (loss + 1e-10)
            df['RSI'] = 100 - (100 / (1 + rs))

        df['RSI_SMA'] = df['RSI'].rolling(14).mean() if 'RSI' in df.columns else 50

        # MACD
        try:
            macd_indicator = MACD(df['Close'], window_slow=26, window_fast=12, window_sign=9)
            df['MACD'] = macd_indicator.macd()
            df['MACD_Signal'] = macd_indicator.macd_signal()
            df['MACD_Hist'] = macd_indicator.macd_diff()
        except Exception as e:
            logger.debug(f"MACD hesaplama hatası: {e}")
            # Manuel MACD
            ema12 = df['Close'].ewm(span=12, adjust=False).mean()
            ema26 = df['Close'].ewm(span=26, adjust=False).mean()
            df['MACD'] = ema12 - ema26
            df['MACD_Signal'] = df['MACD'].ewm(span=9, adjust=False).mean()
            df['MACD_Hist'] = df['MACD'] - df['MACD_Signal']

        # Stochastic
        try:
            stoch = StochasticOscillator(df['High'], df['Low'], df['Close'], window=14, smooth_window=3)
            df['Stoch_K'] = stoch.stoch()
            df['Stoch_D'] = stoch.stoch_signal()
        except Exception as e:
            logger.debug(f"Stochastic hesaplama hatası: {e}")
            # Manuel Stochastic
            low_14 = df['Low'].rolling(14).min()
            high_14 = df['High'].rolling(14).max()
            df['Stoch_K'] = 100 * (df['Close'] - low_14) / (high_14 - low_14 + 1e-10)
            df['Stoch_D'] = df['Stoch_K'].rolling(3).mean()

        # ═══════════════════════════════════════════════════════════════════
        # VOLATİLİTE İNDİKATÖRLERİ
        # ═══════════════════════════════════════════════════════════════════
        try:
            df['ATR'] = AverageTrueRange(df['High'], df['Low'], df['Close'], window=14).average_true_range()
        except Exception as e:
            logger.debug(f"ATR hesaplama hatası: {e}")
            # Manuel ATR
            tr1 = df['High'] - df['Low']
            tr2 = abs(df['High'] - df['Close'].shift(1))
            tr3 = abs(df['Low'] - df['Close'].shift(1))
            tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
            df['ATR'] = tr.rolling(14).mean()

        df['ATR_Percent'] = (df['ATR'] / df['Close']) * 100

        # Bollinger Bands
        try:
            bb = BollingerBands(df['Close'], window=20, window_dev=2)
            df['BB_Upper'] = bb.bollinger_hband()
            df['BB_Middle'] = bb.bollinger_mavg()
            df['BB_Lower'] = bb.bollinger_lband()
            df['BB_Width'] = (df['BB_Upper'] - df['BB_Lower']) / df['BB_Middle']
        except Exception as e:
            logger.debug(f"BB hesaplama hatası: {e}")
            # Manuel Bollinger Bands
            df['BB_Middle'] = df['Close'].rolling(20).mean()
            bb_std = df['Close'].rolling(20).std()
            df['BB_Upper'] = df['BB_Middle'] + (bb_std * 2)
            df['BB_Lower'] = df['BB_Middle'] - (bb_std * 2)
            df['BB_Width'] = (df['BB_Upper'] - df['BB_Lower']) / df['BB_Middle']

        # ═══════════════════════════════════════════════════════════════════
        # HACİM İNDİKATÖRLERİ
        # ═══════════════════════════════════════════════════════════════════
        try:
            df['Volume_SMA'] = df['Volume'].rolling(20).mean()
            df['Volume_Ratio'] = df['Volume'] / (df['Volume_SMA'] + 1)
        except:
            df['Volume_SMA'] = df['Volume']
            df['Volume_Ratio'] = 1

        # OBV
        try:
            df['OBV'] = OnBalanceVolumeIndicator(df['Close'], df['Volume']).on_balance_volume()
        except:
            df['OBV'] = (np.sign(df['Close'].diff()) * df['Volume']).fillna(0).cumsum()

        # MFI
        try:
            df['MFI'] = MFIIndicator(df['High'], df['Low'], df['Close'], df['Volume'], window=14).money_flow_index()
        except:
            # Manuel MFI
            typical_price = (df['High'] + df['Low'] + df['Close']) / 3
            money_flow = typical_price * df['Volume']
            positive_flow = money_flow.where(typical_price > typical_price.shift(1), 0).rolling(14).sum()
            negative_flow = money_flow.where(typical_price < typical_price.shift(1), 0).rolling(14).sum()
            mfi_ratio = positive_flow / (negative_flow + 1e-10)
            df['MFI'] = 100 - (100 / (1 + mfi_ratio))

        # ═══════════════════════════════════════════════════════════════════
        # TREND GÜÇ İNDİKATÖRLERİ
        # ═══════════════════════════════════════════════════════════════════
        try:
            adx_indicator = ADXIndicator(df['High'], df['Low'], df['Close'], window=14)
            df['ADX'] = adx_indicator.adx()
            df['DI_Plus'] = adx_indicator.adx_pos()
            df['DI_Minus'] = adx_indicator.adx_neg()
        except Exception as e:
            logger.debug(f"ADX hesaplama hatası: {e}")
            df['ADX'] = 25
            df['DI_Plus'] = 25
            df['DI_Minus'] = 25

        # ═══════════════════════════════════════════════════════════════════
        # FİYAT KANALLARI
        # ═══════════════════════════════════════════════════════════════════
        # Donchian Channels
        df['DC_Upper'] = df['High'].rolling(20).max()
        df['DC_Lower'] = df['Low'].rolling(20).min()
        df['DC_Middle'] = (df['DC_Upper'] + df['DC_Lower']) / 2

        # ═══════════════════════════════════════════════════════════════════
        # MUM FORMASYONLARI
        # ═══════════════════════════════════════════════════════════════════
        df['Body'] = abs(df['Close'] - df['Open'])
        df['Upper_Wick'] = df['High'] - df[['Open', 'Close']].max(axis=1)
        df['Lower_Wick'] = df[['Open', 'Close']].min(axis=1) - df['Low']
        df['Range'] = df['High'] - df['Low']
        df['Body_Ratio'] = df['Body'] / df['Range'].replace(0, np.nan)

        # Mum yönü
        df['Bullish'] = (df['Close'] > df['Open']).astype(int)
        df['Bearish'] = (df['Close'] < df['Open']).astype(int)

        return df

    def fetch_multi_timeframe(
            self,
            symbol: str,
            timeframes: List[str] = None
    ) -> Dict[str, pd.DataFrame]:
        """
        Çoklu zaman diliminde veri çek

        Args:
            symbol: Hisse kodu
            timeframes: Zaman dilimleri listesi

        Returns:
            {timeframe: DataFrame} sözlüğü
        """
        if timeframes is None:
            timeframes = ["1h", "4h", "1d"]

        result = {}

        for tf in timeframes:
            period = SUPPORTED_PERIODS.get(tf, "3mo")
            df = self.fetch_data(symbol, interval=tf, period=period)
            if df is not None:
                result[tf] = df
            time.sleep(self._rate_limit_delay)

        return result

    def get_index_data(
            self,
            interval: str = "1d",
            period: str = "1y"
    ) -> Optional[pd.DataFrame]:
        """
        Endeks verisini çek - TradingView birincil, Proxy yedek

        XU100 verisi TradingView'den çekilir, başarısız olursa proxy endeks oluşturur.
        """
        # 1. TradingView ile dene (birincil)
        if self.tv_enabled and self.bist_fetcher:
            try:
                n_bars_map = {"5d": 100, "1mo": 300, "3mo": 500, "1y": 1000}
                n_bars = n_bars_map.get(period, 500)

                logger.info(f"📊 XU100 TradingView'den çekiliyor...")
                df = self.bist_fetcher.get_index_data("XU100", interval, n_bars)

                if df is not None and len(df) >= 30:
                    # Indikatörler ekle
                    df = self._clean_data(df)
                    df = self._add_indicators(df)
                    logger.info(f"✅ XU100 TradingView'den alındı ({len(df)} bar)")
                    return df
            except Exception as e:
                logger.debug(f"TradingView XU100 hatası: {e}")

        # 2. Yahoo Finance ile dene
        index_symbols = ["XU100.IS", "^XU100"]

        for idx_symbol in index_symbols:
            try:
                logger.info(f"📊 {idx_symbol} Yahoo Finance'dan çekiliyor...")
                df = yf.download(
                    idx_symbol,
                    period=period,
                    interval=interval,
                    progress=False,
                    timeout=10
                )

                if not df.empty and len(df) >= 30:
                    df = self._clean_data(df)
                    df = self._add_indicators(df)
                    logger.info(f"✅ Endeks verisi {idx_symbol} ile alındı ({len(df)} bar)")
                    return df
            except Exception as e:
                logger.debug(f"Endeks {idx_symbol} denemesi başarısız: {e}")
                continue

        # 3. Gerçek endeks alınamazsa proxy endeks oluştur
        logger.warning("⚠️ XU100 verisi alınamadı, proxy endeks oluşturuluyor...")
        return self._create_proxy_index(interval, period)

    def _create_proxy_index(
            self,
            interval: str = "1d",
            period: str = "1y"
    ) -> Optional[pd.DataFrame]:
        """
        Proxy endeks oluştur - En likit 5 hisseden ağırlıklı ortalama

        THYAO, GARAN, AKBNK, EREGL, ASELS hisselerinin ortalaması
        gerçek XU100'e çok yakın hareket eder.
        """
        dfs = []

        for symbol in self._proxy_symbols:
            try:
                df = self.fetch_data(f"{symbol}.IS", interval=interval, period=period)
                if df is not None and len(df) >= 30:
                    # Normalize et (ilk fiyatı 100 kabul et)
                    normalized = df['Close'] / df['Close'].iloc[0] * 100
                    dfs.append(normalized)
            except Exception as e:
                logger.debug(f"Proxy sembol {symbol} atlandı: {e}")
                continue
            time.sleep(0.1)

        if len(dfs) < 3:
            logger.error("❌ Proxy endeks için yeterli veri yok")
            return None

        # DataFrame'leri birleştir ve ortalama al
        combined = pd.concat(dfs, axis=1)
        proxy_close = combined.mean(axis=1)

        # Orijinal df yapısını koru (son başarılı df'den)
        last_df = self.fetch_data(f"{self._proxy_symbols[0]}.IS", interval=interval, period=period)
        if last_df is None:
            return None

        # Proxy değerleri yerleştir
        proxy_df = last_df.copy()

        # Close değerini proxy ile değiştir (10000 baz puan)
        scale_factor = 10000 / proxy_close.iloc[0]
        proxy_df['Close'] = proxy_close * scale_factor
        proxy_df['Open'] = proxy_df['Close'].shift(1).fillna(proxy_df['Close'].iloc[0])
        proxy_df['High'] = proxy_df['Close'] * 1.005
        proxy_df['Low'] = proxy_df['Close'] * 0.995

        logger.info(f"✅ Proxy endeks oluşturuldu ({len(dfs)} hisse ortalaması)")
        return proxy_df

    def get_bist_symbols(self) -> List[str]:
        """
        TradingView'den dinamik BIST hisse listesi çek - TÜM HİSSELER

        Returns:
            Tüm BIST sembolleri listesi (500+ hisse)
        """
        # TradingView kullanılabilirse ondan çek
        if self.tv_enabled and self.bist_fetcher:
            try:
                symbols = self.bist_fetcher.get_bist_symbols()
                if len(symbols) > 100:
                    logger.info(f"✅ TradingView'den {len(symbols)} BIST hissesi alındı")
                    return symbols
            except Exception as e:
                logger.warning(f"⚠️ TradingView sembol listesi hatası: {e}")

        # Yedek: Statik liste
        logger.info("📊 Statik BIST listesi kullanılıyor...")

        # Kapsamlı BIST hisse listesi
        static_symbols = [
            # BIST30
            "THYAO", "GARAN", "AKBNK", "YKBNK", "ISCTR", "EREGL", "BIMAS",
            "ASELS", "KCHOL", "TUPRS", "SISE", "SAHOL", "FROTO", "TOASO",
            "TCELL", "PGSUS", "ARCLK", "TAVHL", "PETKM", "SASA", "EKGYO",
            "HEKTS", "GUBRF", "KONTR", "ENKAI", "TKFEN", "TTKOM", "KRDMD",
            "SOKM", "MGROS",
            # BIST50 ek
            "DOAS", "MAVI", "VESTL", "OTKAR", "AEFES", "AKSA", "ALARK",
            "ANHYT", "ASTOR", "BERA", "BRISA", "CCOLA", "CEMTS", "DOHOL",
            "EGEEN", "ENJSA", "GESAN", "GLYHO", "GOLTS", "ISGYO", "KARSN",
            "OYAKC", "ANSGR", "AGHOL", "AKSEN", "ALBRK", "ALGYO", "ALKIM",
            "ASUZU", "AYDEM", "BAGFS", "BANVT", "BIENY", "BIZIM", "CANTE",
            "CWENE", "GWIND", "NATEN", "ODAS", "ZOREN", "LOGO", "INDES",
            # Banka ve Finans
            "HALKB", "VAKBN", "TSKB", "QNBFB", "SKBNK", "ICBCT", "KLNMA",
            "AGESA", "AKGRT", "TURSG", "RAYSG",
            # GYO
            "ADGYO", "AVGYO", "HLGYO", "ISGYO", "KLGYO", "KRGYO", "NUGYO",
            "OZGYO", "PAGYO", "SNGYO", "TRGYO", "VKGYO", "YGGYO", "ZRGYO",
            # Spor
            "FENER", "GSRAY", "BJKAS", "TSPOR",
            # İnşaat/Çimento
            "SMRTG", "BTCIM", "CIMSA", "AKCNS", "AFYON", "UNYEC", "BUCIM",
            "BOLUC", "KONYA", "ADANA", "MRDIN", "NUHCM", "BSOKE", "GOLTS",
            # Holding
            "KCHOL", "SAHOL", "DOHOL", "GSDHO", "NTHOL", "POLHO", "SISE",
            # Enerji
            "AKSEN", "AYDEM", "ENJSA", "AYEN", "CWENE", "GWIND", "NATEN",
            "ODAS", "ZOREN", "AKENR", "AKSA",
            # Teknoloji
            "LOGO", "INDES", "ARENA", "ARMDA", "DGATE", "ESCOM", "FONET",
            "KRONT", "LINK", "NETAS", "PAPIL", "SMART",
            # Aracı Kurumlar
            "OYAYO", "INFO", "ISMEN", "GEDIK", "GLBMD", "A1CAP", "GRNYO",
            # Perakende
            "BIMAS", "MGROS", "SOKM", "BIZIM", "MAVI", "VAKKO",
            # Otomotiv
            "FROTO", "TOASO", "DOAS", "OTKAR", "ASUZU", "BRISA", "GOODY",
            # Gıda
            "AEFES", "CCOLA", "ULKER", "BANVT", "PETUN", "TATGD", "TUKAS",
            # Demir Çelik
            "EREGL", "KRDMD", "KRDMA", "KRDMB", "BRSAN", "CELHA", "CEMTS",
            # Kimya
            "PETKM", "SASA", "BAGFS", "GUBRF", "HEKTS", "ALKIM",
            # Havacılık
            "THYAO", "PGSUS", "CLEBI", "TAVHL",
            # Savunma
            "ASELS",
            # Diğer popüler
            "MIATK", "YEOTK", "KLSER", "ADESE", "ARCLK", "VESBE", "VESTL"
        ]

        # Tekrarları kaldır
        return list(dict.fromkeys(static_symbols))

    def resample_data(
            self,
            df: pd.DataFrame,
            target_interval: str
    ) -> pd.DataFrame:
        """
        Veriyi farklı zaman dilimine dönüştür

        Args:
            df: Kaynak DataFrame
            target_interval: Hedef zaman dilimi ("4H", "1D" vb.)

        Returns:
            Resampled DataFrame
        """
        resample_map = {
            "15m": "15T", "30m": "30T", "1h": "1H",
            "4h": "4H", "1d": "1D", "1w": "1W"
        }

        rule = resample_map.get(target_interval.lower(), target_interval)

        resampled = df.resample(rule).agg({
            'Open': 'first',
            'High': 'max',
            'Low': 'min',
            'Close': 'last',
            'Volume': 'sum'
        }).dropna()

        # Yeni indikatörler hesapla
        resampled = self._add_indicators(resampled)

        return resampled

    def batch_fetch(
            self,
            symbols: List[str],
            interval: str = "1h",
            period: str = "1mo",
            batch_size: int = 10,
            delay: float = 0.5
    ) -> Dict[str, pd.DataFrame]:
        """
        Toplu veri çekme (API limit korumalı)

        Args:
            symbols: Hisse listesi
            interval: Zaman dilimi
            period: Dönem
            batch_size: Batch büyüklüğü
            delay: Batch arası bekleme

        Returns:
            {symbol: DataFrame} sözlüğü
        """
        result = {}

        for i in range(0, len(symbols), batch_size):
            batch = symbols[i:i + batch_size]

            for symbol in batch:
                full_symbol = f"{symbol}.IS" if not symbol.endswith(".IS") else symbol
                df = self.fetch_data(full_symbol, interval, period)
                if df is not None:
                    result[symbol.replace(".IS", "")] = df

            if i + batch_size < len(symbols):
                time.sleep(delay)

        return result

    def calculate_returns(
            self,
            df: pd.DataFrame,
            periods: List[int] = None
    ) -> pd.DataFrame:
        """
        Getiri hesaplamaları

        Args:
            df: OHLCV DataFrame
            periods: Hesaplanacak dönemler

        Returns:
            Getiri sütunları eklenmiş DataFrame
        """
        if periods is None:
            periods = [1, 5, 10, 20, 50]

        df = df.copy()

        for p in periods:
            # Basit getiri
            df[f'Return_{p}'] = df['Close'].pct_change(p) * 100
            # Logaritmik getiri
            df[f'LogReturn_{p}'] = np.log(df['Close'] / df['Close'].shift(p)) * 100

        return df

    def get_market_hours_filter(self, df: pd.DataFrame) -> pd.DataFrame:
        """Sadece piyasa saatlerini filtrele (BIST: 10:00-18:00)"""
        df = df.copy()

        if hasattr(df.index, 'hour'):
            mask = (df.index.hour >= 10) & (df.index.hour < 18)
            return df[mask]

        return df

    def clear_cache(self, symbol: str = None) -> None:
        """Önbelleği temizle"""
        if symbol:
            keys_to_remove = [k for k in self.cache.keys() if symbol in k]
            for k in keys_to_remove:
                del self.cache[k]
            logger.info(f"🗑️ {symbol} önbelleği temizlendi")
        else:
            self.cache.clear()
            logger.info("🗑️ Tüm önbellek temizlendi")


# Test
if __name__ == "__main__":
    engine = DataEngine()

    # Tek hisse testi
    df = engine.fetch_data("THYAO", interval="1h", period="1mo")
    if df is not None:
        print(f"\n📊 THYAO Veri Özeti:")
        print(f"   Toplam bar: {len(df)}")
        print(f"   Tarih aralığı: {df.index[0]} - {df.index[-1]}")
        print(f"   Kolonlar: {list(df.columns)[:10]}...")
        print(f"   Son RSI: {df['RSI'].iloc[-1]:.2f}")