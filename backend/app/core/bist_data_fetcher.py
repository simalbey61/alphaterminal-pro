"""
BIST VERİ ÇEKME MODÜLÜ - OPTIMIZED VERSION
==========================================
TradingView'den otomatik olarak tüm BIST hisselerini çeker.

YENİ ÖZELLİKLER:
- ✅ Paralel veri çekimi (Threading)
- ✅ Önbellek sistemi (Cache)
- ✅ 3-5x daha hızlı
- ✅ Akıllı rate limit yönetimi

Kurulum:
    pip install git+https://github.com/rongardF/tvdatafeed
"""

import subprocess
import sys
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Optional, List, Dict, Tuple
import warnings
import time
import pickle
from pathlib import Path
import concurrent.futures
import logging

warnings.filterwarnings('ignore')

logger = logging.getLogger(__name__)

# tvDatafeed kurulum ve import
TV_AVAILABLE = False


def install_tvdatafeed():
    """tvDatafeed'i GitHub'dan yükler"""
    global TV_AVAILABLE
    try:
        print("📦 tvDatafeed kuruluyor (GitHub - rongardF fork)...")
        subprocess.check_call([
            sys.executable, "-m", "pip", "install",
            "git+https://github.com/rongardF/tvdatafeed", "-q"
        ])
        print("✅ tvDatafeed kuruldu!")
        return True
    except Exception as e:
        print(f"❌ tvDatafeed kurulum hatası: {e}")
        return False


try:
    from tvDatafeed import TvDatafeed, Interval

    TV_AVAILABLE = True
    print("✅ tvDatafeed hazır")
except ImportError:
    print("⚠️ tvDatafeed yüklü değil. Yükleniyor...")
    if install_tvdatafeed():
        try:
            from tvDatafeed import TvDatafeed, Interval

            TV_AVAILABLE = True
        except ImportError:
            print("❌ tvDatafeed import edilemedi. Runtime restart gerekebilir.")
            TV_AVAILABLE = False
    else:
        TV_AVAILABLE = False

# Periyot eşleştirme
INTERVAL_MAP = {
    "1m": Interval.in_1_minute if TV_AVAILABLE else None,
    "5m": Interval.in_5_minute if TV_AVAILABLE else None,
    "15m": Interval.in_15_minute if TV_AVAILABLE else None,
    "30m": Interval.in_30_minute if TV_AVAILABLE else None,
    "1h": Interval.in_1_hour if TV_AVAILABLE else None,
    "4h": Interval.in_4_hour if TV_AVAILABLE else None,
    "1d": Interval.in_daily if TV_AVAILABLE else None,
    "1w": Interval.in_weekly if TV_AVAILABLE else None,
    "1M": Interval.in_monthly if TV_AVAILABLE else None,
}


# ═══════════════════════════════════════════════════════════════
# CACHE SİSTEMİ
# ═══════════════════════════════════════════════════════════════

class DataCache:
    """
    Veri önbellek sistemi.
    İlk taramada veriyi saklar, sonraki taramalarda hızlı okuma sağlar.
    """

    def __init__(self, cache_dir: str = 'cache', ttl: int = 600):
        """
        Args:
            cache_dir: Cache klasörü
            ttl: Time to live (saniye) - varsayılan 10 dakika
        """
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(exist_ok=True)
        self.ttl = ttl

    def get(self, key: str) -> Optional[Dict]:
        """
        Cache'ten oku.

        Args:
            key: Cache anahtarı

        Returns:
            Cached data veya None
        """
        cache_file = self.cache_dir / f"{key}.pkl"

        if not cache_file.exists():
            return None

        # TTL kontrolü
        file_age = time.time() - cache_file.stat().st_mtime
        if file_age > self.ttl:
            logger.info(f"📦 Cache expired: {key} ({file_age:.0f}s > {self.ttl}s)")
            cache_file.unlink()
            return None

        try:
            with open(cache_file, 'rb') as f:
                data = pickle.load(f)
            logger.info(f"✅ Cache hit: {key} ({len(data)} items)")
            return data
        except Exception as e:
            logger.error(f"❌ Cache read error: {e}")
            cache_file.unlink()
            return None

    def set(self, key: str, data: Dict) -> None:
        """
        Cache'e yaz.

        Args:
            key: Cache anahtarı
            data: Kaydedilecek veri
        """
        cache_file = self.cache_dir / f"{key}.pkl"
        try:
            with open(cache_file, 'wb') as f:
                pickle.dump(data, f)
            logger.info(f"💾 Cached: {key} ({len(data)} items)")
        except Exception as e:
            logger.error(f"❌ Cache write error: {e}")

    def clear(self) -> int:
        """
        Tüm cache'i temizle.

        Returns:
            Silinen dosya sayısı
        """
        count = 0
        for f in self.cache_dir.glob('*.pkl'):
            f.unlink()
            count += 1
        logger.info(f"🧹 Cache cleared: {count} files")
        return count


# ═══════════════════════════════════════════════════════════════
# BIST DATA FETCHER (OPTIMIZED)
# ═══════════════════════════════════════════════════════════════

class BISTDataFetcher:
    """BIST hisse verilerini çeken sınıf - OPTIMIZED VERSION"""

    def __init__(self, use_cache: bool = True, cache_ttl: int = 600):
        """
        Args:
            use_cache: Cache kullan
            cache_ttl: Cache geçerlilik süresi (saniye)
        """
        if TV_AVAILABLE:
            self.tv = TvDatafeed()
        else:
            self.tv = None

        self._bist_symbols = None

        # Cache sistemi
        self.use_cache = use_cache
        self.cache = DataCache(ttl=cache_ttl) if use_cache else None

    def get_bist_symbols(self) -> List[str]:
        """
        Tüm BIST hisselerini dinamik olarak çeker.
        """
        if self._bist_symbols is not None:
            return self._bist_symbols

        # Önce tradingview-screener ile dinamik liste dene
        try:
            from tradingview_screener import get_all_symbols
            all_symbols = get_all_symbols(market='turkey')
            bist_symbols = [s.replace('BIST:', '') for s in all_symbols if s.startswith('BIST:')]
            if len(bist_symbols) > 100:
                print(f"✅ {len(bist_symbols)} BIST hissesi dinamik olarak alındı")
                self._bist_symbols = bist_symbols
                return self._bist_symbols
        except ImportError:
            print("⚠️ tradingview-screener yüklü değil. Yükleniyor...")
            try:
                import subprocess
                subprocess.check_call([sys.executable, "-m", "pip", "install", "tradingview-screener==2.5.0", "-q"])
                from tradingview_screener import get_all_symbols
                all_symbols = get_all_symbols(market='turkey')
                bist_symbols = [s.replace('BIST:', '') for s in all_symbols if s.startswith('BIST:')]
                if len(bist_symbols) > 100:
                    print(f"✅ {len(bist_symbols)} BIST hissesi dinamik olarak alındı")
                    self._bist_symbols = bist_symbols
                    return self._bist_symbols
            except:
                pass
        except Exception as e:
            print(f"⚠️ Dinamik liste alınamadı: {e}. Statik liste kullanılacak.")

        # Statik BIST hisse listesi (yedek)
        bist_symbols = [
            # A
            "A1CAP", "ACSEL", "ADEL", "ADESE", "ADGYO", "AEFES", "AFYON", "AGESA",
            "AGHOL", "AGROT", "AGYO", "AHGAZ", "AHSGY", "AKBNK", "AKCNS", "AKENR",
            "AKFGY", "AKFIS", "AKFYE", "AKGRT", "AKMGY", "AKSA", "AKSEN", "AKSGY",
            "AKSUE", "AKYHO", "ALARK", "ALBRK", "ALCAR", "ALCTL", "ALFAS", "ALGYO",
            "ALKA", "ALKIM", "ALKLC", "ALMAD", "ALTIN", "ALTNY", "ALVES", "ANELE",
            "ANGEN", "ANHYT", "ANSGR", "ARASE", "ARCLK", "ARDYZ", "ARENA", "ARFYO",
            "ARMDA", "ARMGD", "ARSAN", "ARTMS", "ARZUM", "ASELS", "ASGYO", "ASTOR",
            "ASUZU", "ATAGY", "ATAKP", "ATATP", "ATEKS", "ATLAS", "ATSYH", "AVGYO",
            "AVHOL", "AVOD", "AVPGY", "AVTUR", "AYCES", "AYDEM", "AYEN", "AYES",
            "AYGAZ", "AZTEK",
            # B
            "BAGFS", "BAHKM", "BAKAB", "BALAT", "BANVT", "BARMA", "BASCM", "BASGZ",
            "BAYRK", "BEGYO", "BERA", "BEYAZ", "BFREN", "BIENY", "BIGCH", "BIGEN",
            "BIGTK", "BIMAS", "BINBN", "BINHO", "BIOEN", "BIZIM", "BJKAS", "BLCYT",
            "BMSCH", "BMSTL", "BNTAS", "BOBET", "BORLS", "BORSK", "BOSSA", "BRISA",
            "BRKO", "BRKSN", "BRKVY", "BRLSM", "BRMEN", "BRSAN", "BRYAT", "BSOKE",
            "BTCIM", "BUCIM", "BURCE", "BURVA", "BVSAN", "BYDNR",
            # C
            "CANTE", "CASA", "CATES", "CCOLA", "CELHA", "CEMAS", "CEMTS", "CEMZY",
            "CEOEM", "CGCAM", "CIMSA", "CLEBI", "CMBTN", "CMENT", "CONSE", "COSMO",
            "CRDFA", "CRFSA", "CUSAN", "CVKMD", "CWENE",
            # D
            "DAGI", "DAPGM", "DARDL", "DCTTR", "DENGE", "DERHL", "DERIM", "DESA",
            "DESPC", "DEVA", "DGATE", "DGGYO", "DGNMO", "DIRIT", "DITAS", "DJIST",
            "DMLKT", "DMRGD", "DMSAS", "DNISI", "DOAS", "DOCO", "DOFER", "DOGUB",
            "DOHOL", "DOKTA", "DURDO", "DURKN", "DYOBY", "DZGYO",
            # E
            "EBEBK", "ECILC", "ECZYT", "EDATA", "EDIP", "EFOR", "EGEEN", "EGEGY",
            "EGEPO", "EGGUB", "EGPRO", "EGSER", "EKGYO", "EKIZ", "EKOS", "EKSUN",
            "ELITE", "EMKEL", "EMNIS", "ENERY", "ENJSA", "ENKAI", "ENSRI", "ENTRA",
            "EPLAS", "ERBOS", "ERCB", "EREGL", "ERSU", "ESCAR", "ESCOM", "ESEN",
            "ETILR", "ETYAT", "EUHOL", "EUKYO", "EUPWR", "EUREN", "EUYO", "EYGYO",
            # F
            "FADE", "FENER", "FLAP", "FMIZP", "FONET", "FORMT", "FORTE", "FRIGO",
            "FROTO", "FZLGY",
            # G
            "GARAN", "GARFA", "GEDIK", "GEDZA", "GENIL", "GENTS", "GEREL", "GESAN",
            "GIPTA", "GLBMD", "GLCVY", "GLRYH", "GLYHO", "GMTAS", "GOKNR", "GOLTS",
            "GOODY", "GOZDE", "GRNYO", "GRSEL", "GRTHO", "GRTRK", "GSDDE", "GSDHO",
            "GSRAY", "GUBRF", "GUNDG", "GWIND", "GZNMI",
            # H
            "HALKB", "HATEK", "HATSN", "HDFGS", "HEDEF", "HEKTS", "HKTM", "HLGYO",
            "HOROZ", "HRKET", "HTTBT", "HUBVC", "HUNER", "HURGZ",
            # I-İ
            "ICBCT", "ICUGS", "IDGYO", "IEYHO", "IHAAS", "IHEVA", "IHGZT", "IHLAS",
            "IHLGM", "IHYAY", "IMASM", "INDES", "INFO", "INGRM", "INTEK", "INTEM",
            "INVEO", "INVES", "IPEKE", "ISATR", "ISBIR", "ISBTR", "ISCTR", "ISDMR",
            "ISFIN", "ISGSY", "ISGYO", "ISKPL", "ISKUR", "ISMEN", "ISSEN", "ISYAT",
            "IZENR", "IZFAS", "IZINV", "IZMDC",
            # J-K
            "JANTS", "KAPLM", "KAREL", "KARSN", "KARTN", "KATMR", "KAYSE", "KBORU",
            "KCAER", "KCHOL", "KENT", "KERVN", "KFEIN", "KGYO", "KIMMR", "KLGYO",
            "KLKIM", "KLMSN", "KLNMA", "KLRHO", "KLSER", "KLSYN", "KMPUR", "KNFRT",
            "KOCMT", "KONKA", "KONTR", "KONYA", "KOPOL", "KORDS", "KOTON", "KRDMA",
            "KRDMB", "KRDMD", "KRGYO", "KRONT", "KRPLS", "KRSTL", "KRTEK", "KRVGD",
            "KSTUR", "KTLEV", "KTSKR", "KUTPO", "KUVVA", "KUYAS", "KZBGY", "KZGYO",
            # L
            "LIDER", "LIDFA", "LILAK", "LINK", "LKMNH", "LMKDC", "LOGO", "LRSHO",
            "LUKSK", "LYDHO",
            # M
            "MAALT", "MACKO", "MAGEN", "MAKIM", "MAKTK", "MANAS", "MARBL", "MARKA",
            "MARTI", "MAVI", "MEDTR", "MEGAP", "MEGMT", "MEKAG", "MEPET", "MERCN",
            "MERIT", "MERKO", "METRO", "METUR", "MGROS", "MHRGY", "MIATK", "MNDRS",
            "MNDTR", "MOBTL", "MOGAN", "MOPAS", "MPARK", "MRGYO", "MRSHL", "MSGYO",
            "MTRKS", "MTRYO", "MZHLD",
            # N
            "NATEN", "NETAS", "NIBAS", "NTGAZ", "NTHOL", "NUGYO", "NUHCM",
            # O-Ö
            "OBAMS", "OBASE", "ODAS", "ODINE", "OFSYM", "ONCSM", "ONRYT", "ORCAY",
            "ORGE", "ORMA", "OSMEN", "OSTIM", "OTKAR", "OTTO", "OYAKC", "OYAYO",
            "OYLUM", "OYYAT", "OZGYO", "OZKGY", "OZRDN", "OZSUB", "OZYSR",
            # P
            "PAGYO", "PAMEL", "PAPIL", "PARSN", "PASEU", "PATEK", "PCILT", "PEKGY",
            "PENGD", "PENTA", "PETKM", "PETUN", "PGSUS", "PINSU", "PKART", "PKENT",
            "PLTUR", "PNLSN", "PNSUT", "POLHO", "POLTK", "PRDGS", "PRKAB", "PRKME",
            "PRZMA", "PSDTC", "PSGYO",
            # Q-R
            "QNBFB", "QNBFL", "QUAGR", "RALYH", "RAYSG", "REEDR", "RGYAS", "RNPOL",
            "RODRG", "ROYAL", "RTALB", "RUBNS", "RYGYO", "RYSAS",
            # S-Ş
            "SAFKR", "SAHOL", "SAMAT", "SANEL", "SANFM", "SANKO", "SARKY", "SASA",
            "SAYAS", "SDTTR", "SEGMN", "SEGYO", "SEKFK", "SEKUR", "SELEC", "SELGD",
            "SELVA", "SEYKM", "SILVR", "SISE", "SKBNK", "SKTAS", "SKYLP", "SKYMD",
            "SMART", "SMRTG", "SNGYO", "SNICA", "SNKRN", "SNPAM", "SODSN", "SOKE",
            "SOKM", "SONME", "SRVGY", "SUMAS", "SUNTK", "SURGY", "SUWEN",
            # T
            "TABGD", "TARKM", "TATEN", "TATGD", "TAVHL", "TBORG", "TCELL", "TCKRC",
            "TDGYO", "TEHOL", "TEKTU", "TERA", "TETMT", "TEZOL", "TGSAS", "THYAO",
            "TKFEN", "TKNSA", "TLMAN", "TMPOL", "TMSN", "TNZTP", "TOASO", "TRCAS",
            "TRGYO", "TRILC", "TSGYO", "TSKB", "TSPOR", "TTKOM", "TTRAK", "TUCLK",
            "TUKAS", "TUPRS", "TUREX", "TURGG", "TURSG",
            # U-Ü
            "UFUK", "ULAS", "ULKER", "ULUFA", "ULUSE", "ULUUN", "UNLU", "USAK",
            # V
            "VAKBN", "VAKFN", "VAKKO", "VANGD", "VBTYZ", "VERTU", "VERUS", "VESBE",
            "VESTL", "VKFYO", "VKGYO", "VKING", "VRGYO",
            # Y
            "YAPRK", "YATAS", "YAYLA", "YBTAS", "YEOTK", "YESIL", "YGGYO", "YGYO",
            "YIGIT", "YKBNK", "YKSLN", "YONGA", "YUNSA", "YYAPI", "YYLGD",
            # Z
            "ZEDUR", "ZOREN", "ZRGYO",
        ]

        self._bist_symbols = bist_symbols
        return self._bist_symbols

    def get_stock_data(
            self,
            symbol: str,
            interval: str = "1d",
            n_bars: int = 500
    ) -> Optional[pd.DataFrame]:
        """
        Tek bir hisse için veri çeker.

        Args:
            symbol: Hisse kodu (örn: "AKBNK")
            interval: Periyot ("1h", "4h", "1d", "1w")
            n_bars: Çekilecek bar sayısı

        Returns:
            DataFrame: OHLCV verileri veya None
        """
        if not TV_AVAILABLE or self.tv is None:
            logger.warning(f"❌ TvDatafeed bağlantısı yok!")
            return None

        tv_interval = INTERVAL_MAP.get(interval)
        if tv_interval is None:
            logger.warning(f"❌ Geçersiz periyot: {interval}")
            return None

        # Retry mekanizması
        max_retries = 3
        retry_delay = 2

        for attempt in range(max_retries):
            try:
                df = self.tv.get_hist(
                    symbol=symbol,
                    exchange="BIST",
                    interval=tv_interval,
                    n_bars=n_bars
                )

                if df is None or df.empty:
                    return None

                # Sütun isimlerini standartlaştır
                df = df.rename(columns={
                    'open': 'Open',
                    'high': 'High',
                    'low': 'Low',
                    'close': 'Close',
                    'volume': 'Volume'
                })

                # Index'i datetime yap
                if not isinstance(df.index, pd.DatetimeIndex):
                    df.index = pd.to_datetime(df.index)

                return df

            except Exception as e:
                error_msg = str(e).lower()
                if 'connection' in error_msg or 'lost' in error_msg:
                    if attempt < max_retries - 1:
                        time.sleep(retry_delay * (attempt + 1))
                        try:
                            self.tv = TvDatafeed()
                        except:
                            pass
                        continue
                return None

        return None

    def get_multiple_stocks_parallel(
            self,
            symbols: List[str] = None,
            interval: str = "1d",
            n_bars: int = 500,
            max_workers: int = 5,
            progress_callback=None
    ) -> Dict[str, pd.DataFrame]:
        """
        Birden fazla hisse için PARALEL veri çeker.
        Threading kullanarak 3-5x daha hızlı.

        Args:
            symbols: Hisse listesi (None ise tüm BIST)
            interval: Periyot
            n_bars: Bar sayısı
            max_workers: Aynı anda kaç thread (5-10 arası önerili)
            progress_callback: İlerleme bildirimi fonksiyonu

        Returns:
            Dict: {symbol: DataFrame} sözlüğü
        """
        if symbols is None:
            symbols = self.get_bist_symbols()

        data = {}
        total = len(symbols)
        success_count = 0
        error_count = 0

        logger.info(f"📊 {total} hisse paralel olarak çekiliyor (max_workers={max_workers})")

        def fetch_single(symbol):
            """Tek hisse çeker."""
            df = self.get_stock_data(symbol, interval, n_bars)
            return symbol, df

        # ThreadPoolExecutor ile paralel çekme
        with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
            # Tüm işleri başlat
            future_to_symbol = {
                executor.submit(fetch_single, symbol): symbol
                for symbol in symbols
            }

            # Tamamlananları topla
            for i, future in enumerate(concurrent.futures.as_completed(future_to_symbol), 1):
                symbol = future_to_symbol[future]

                try:
                    sym, df = future.result()

                    if df is not None and not df.empty:
                        data[sym] = df
                        success_count += 1
                    else:
                        error_count += 1

                except Exception as e:
                    logger.error(f"❌ {symbol} hata: {e}")
                    error_count += 1

                # İlerleme bildirimi
                if progress_callback:
                    progress_callback(i, total, symbol)
                elif i % 50 == 0:
                    logger.info(f"   📈 İlerleme: {i}/{total} (✅{success_count} ❌{error_count})")

        logger.info(f"✅ Paralel çekme tamamlandı: {success_count} başarılı, {error_count} hatalı")
        return data

    def get_multiple_stocks(
            self,
            symbols: List[str] = None,
            interval: str = "1d",
            n_bars: int = 500,
            progress_callback=None,
            batch_size: int = 50,
            batch_delay: float = 3.0,
            use_parallel: bool = True,
            max_workers: int = 5
    ) -> Dict[str, pd.DataFrame]:
        """
        Birden fazla hisse için veri çeker.

        Args:
            symbols: Hisse listesi (None ise tüm BIST)
            interval: Periyot
            n_bars: Bar sayısı
            progress_callback: İlerleme bildirimi
            batch_size: Her batch'teki hisse sayısı (paralel kapalıysa)
            batch_delay: Batch'ler arası bekleme (paralel kapalıysa)
            use_parallel: Paralel çekme kullan (ÖNERİLEN: True)
            max_workers: Paralel thread sayısı

        Returns:
            Dict: {symbol: DataFrame} sözlüğü
        """
        # Cache kontrolü
        if self.use_cache and self.cache:
            cache_key = f"bist_all_{interval}_{n_bars}"
            cached_data = self.cache.get(cache_key)

            if cached_data is not None:
                logger.info(f"✅ Cache'ten {len(cached_data)} hisse alındı!")
                return cached_data

        # Paralel çekme (ÖNERİLEN)
        if use_parallel:
            data = self.get_multiple_stocks_parallel(
                symbols=symbols,
                interval=interval,
                n_bars=n_bars,
                max_workers=max_workers,
                progress_callback=progress_callback
            )
        else:
            # Sıralı çekme (ESKİ YÖNTEİM)
            if symbols is None:
                symbols = self.get_bist_symbols()

            data = {}
            total = len(symbols)
            success_count = 0
            error_count = 0

            logger.info(f"📊 {total} hisse sıralı olarak çekiliyor...")

            for i, symbol in enumerate(symbols):
                # Her batch sonunda bekle
                if i > 0 and i % batch_size == 0:
                    logger.info(f"   ⏳ Rate limit için {batch_delay}s bekleniyor... ({i}/{total})")
                    time.sleep(batch_delay)
                    try:
                        self.tv = TvDatafeed()
                    except:
                        pass

                df = self.get_stock_data(symbol, interval, n_bars)

                if df is not None and not df.empty:
                    data[symbol] = df
                    success_count += 1
                else:
                    error_count += 1

                # İlerleme bildirimi
                if progress_callback:
                    progress_callback(i + 1, total, symbol)
                elif (i + 1) % 25 == 0:
                    logger.info(f"   📈 İlerleme: {i + 1}/{total} (✅{success_count} ❌{error_count})")

                # Her istek arası bekleme
                time.sleep(0.3)

            logger.info(f"✅ Sıralı çekme tamamlandı: {success_count} başarılı, {error_count} hatalı")

        # Cache'e kaydet
        if self.use_cache and self.cache and data:
            cache_key = f"bist_all_{interval}_{n_bars}"
            self.cache.set(cache_key, data)

        return data

    def get_index_data(
            self,
            symbol: str = "XU100",
            interval: str = "1d",
            n_bars: int = 500
    ) -> Optional[pd.DataFrame]:
        """
        Endeks verisi çeker (XU100, XU030, vb.)

        Args:
            symbol: Endeks sembolü (XU100, XU030)
            interval: Periyot
            n_bars: Bar sayısı

        Returns:
            DataFrame veya None
        """
        return self.get_stock_data(symbol, interval, n_bars)


# ═══════════════════════════════════════════════════════════════
# GLOBAL FONKSİYONLAR
# ═══════════════════════════════════════════════════════════════

# Singleton instance
_data_fetcher = None


def get_data_fetcher(use_cache: bool = True, cache_ttl: int = 600) -> BISTDataFetcher:
    """
    Global BISTDataFetcher instance'ı döndürür.

    Args:
        use_cache: Cache kullan (önerilen: True)
        cache_ttl: Cache geçerlilik süresi (saniye, varsayılan: 600 = 10 dakika)
    """
    global _data_fetcher
    if _data_fetcher is None:
        _data_fetcher = BISTDataFetcher(use_cache=use_cache, cache_ttl=cache_ttl)
    return _data_fetcher


def fetch_all_bist(
        interval: str = "1d",
        n_bars: int = 500,
        use_parallel: bool = True,
        use_cache: bool = True
) -> Dict[str, pd.DataFrame]:
    """
    Tüm BIST hisselerini çeker (kısayol fonksiyon).

    Args:
        interval: Periyot
        n_bars: Bar sayısı
        use_parallel: Paralel çekme kullan (ÖNERİLEN: True, 3-5x hızlı)
        use_cache: Cache kullan (ÖNERİLEN: True, 2. tarama <1sn)

    Returns:
        Dict: {symbol: DataFrame}

    Örnek:
        # Paralel + Cache (ÖNERİLEN)
        data = fetch_all_bist(interval='1d', use_parallel=True, use_cache=True)
        # İlk çalıştırma: ~30-60 saniye
        # Sonraki çalıştırmalar: <1 saniye (cache'ten)

        # Sadece paralel (cache yok)
        data = fetch_all_bist(interval='1d', use_parallel=True, use_cache=False)
        # Her seferinde: ~30-60 saniye

        # ESKİ yöntem (sıralı, yavaş)
        data = fetch_all_bist(interval='1d', use_parallel=False, use_cache=False)
        # Her seferinde: ~2-3 dakika
    """
    fetcher = get_data_fetcher(use_cache=use_cache)
    return fetcher.get_multiple_stocks(
        interval=interval,
        n_bars=n_bars,
        use_parallel=use_parallel
    )


def fetch_stock(symbol: str, interval: str = "1d", n_bars: int = 500) -> Optional[pd.DataFrame]:
    """
    Tek hisse verisi çeker (kısayol fonksiyon).
    """
    fetcher = get_data_fetcher()
    return fetcher.get_stock_data(symbol, interval, n_bars)


def clear_cache():
    """Tüm cache'i temizler."""
    fetcher = get_data_fetcher()
    if fetcher.cache:
        return fetcher.cache.clear()
    return 0


# ═══════════════════════════════════════════════════════════════
# TEST
# ═══════════════════════════════════════════════════════════════

if __name__ == "__main__":
    print("=" * 60)
    print("🧪 BIST Data Fetcher Test")
    print("=" * 60)

    fetcher = BISTDataFetcher(use_cache=True)

    # Sembol listesi testi
    symbols = fetcher.get_bist_symbols()
    print(f"\n📊 Toplam BIST Hissesi: {len(symbols)}")
    print(f"   İlk 10: {symbols[:10]}")

    # Tek hisse testi
    if TV_AVAILABLE:
        print("\n📈 THYAO verisi çekiliyor...")
        df = fetcher.get_stock_data("THYAO", "1d", 100)
        if df is not None:
            print(f"   ✅ {len(df)} bar alındı")
            print(f"   Son kapanış: {df['Close'].iloc[-1]:.2f}")
        else:
            print("   ❌ Veri alınamadı")

        # Endeks testi
        print("\n📊 XU100 endeks verisi çekiliyor...")
        xu100 = fetcher.get_index_data("XU100", "1d", 100)
        if xu100 is not None:
            print(f"   ✅ {len(xu100)} bar alındı")
            print(f"   Son değer: {xu100['Close'].iloc[-1]:.2f}")
        else:
            print("   ❌ Endeks verisi alınamadı")
    else:
        print("\n⚠️ TvDatafeed kullanılamıyor, veri testi atlandı")