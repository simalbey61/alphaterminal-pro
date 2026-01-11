"""
AlphaTerminal Pro - Signal Service
==================================

Trading sinyali business logic servisi.

Author: AlphaTerminal Team
Version: 1.0.0
"""

from __future__ import annotations

import logging
from typing import Optional, List, Dict, Any, Tuple
from decimal import Decimal
from datetime import datetime, timedelta
from uuid import UUID

from sqlalchemy.ext.asyncio import AsyncSession

from app.config import settings, SignalType, SignalTier, SignalStatus
from app.db.models import SignalModel, StockModel
from app.db.repositories import SignalRepository, StockRepository
from app.cache import cache, CacheKeys, CacheTTL

logger = logging.getLogger(__name__)


class SignalService:
    """
    Trading sinyal servisi.
    
    Sinyal üretimi, yönetimi ve performans takibi.
    
    Example:
        ```python
        service = SignalService(session)
        
        # Sinyal oluştur
        signal = await service.create_signal(signal_data)
        
        # Aktif sinyalleri al
        signals = await service.get_active_signals()
        
        # Fiyat seviyelerini kontrol et
        alerts = await service.check_price_levels({"THYAO": 150.5})
        ```
    """
    
    def __init__(self, session: AsyncSession):
        """
        Initialize signal service.
        
        Args:
            session: Database session
        """
        self.session = session
        self.signal_repo = SignalRepository(session)
        self.stock_repo = StockRepository(session)
    
    # =========================================================================
    # SIGNAL CREATION
    # =========================================================================
    
    async def create_signal(
        self,
        symbol: str,
        signal_type: SignalType,
        entry_price: Decimal,
        stop_loss: Decimal,
        take_profit_1: Decimal,
        total_score: Decimal,
        confidence: Decimal,
        risk_reward: Decimal,
        setup_type: str,
        **kwargs
    ) -> SignalModel:
        """
        Yeni sinyal oluştur.
        
        Args:
            symbol: Hisse sembolü
            signal_type: LONG veya SHORT
            entry_price: Giriş fiyatı
            stop_loss: Stop loss fiyatı
            take_profit_1: İlk take profit
            total_score: Toplam skor (0-100)
            confidence: Güven seviyesi (0-1)
            risk_reward: Risk/Reward oranı
            setup_type: Setup türü
            **kwargs: Ek parametreler
            
        Returns:
            SignalModel: Oluşturulan sinyal
        """
        # Validasyonlar
        self._validate_signal_params(
            signal_type=signal_type,
            entry_price=entry_price,
            stop_loss=stop_loss,
            take_profit_1=take_profit_1
        )
        
        # Hisse kontrolü ve ID'si
        stock = await self.stock_repo.find_by_symbol(symbol)
        stock_id = stock.id if stock else None
        
        # Tier belirleme
        tier = self._calculate_tier(total_score)
        
        # Sinyal oluştur
        signal = await self.signal_repo.create(
            symbol=symbol.upper().replace(".IS", ""),
            stock_id=stock_id,
            signal_type=signal_type.value,
            tier=tier.value,
            status=SignalStatus.ACTIVE.value,
            entry_price=entry_price,
            stop_loss=stop_loss,
            take_profit_1=take_profit_1,
            total_score=total_score,
            confidence=confidence,
            risk_reward=risk_reward,
            setup_type=setup_type,
            **kwargs
        )
        
        # Cache'i invalidate et
        await self._invalidate_signal_cache()
        
        logger.info(f"Signal created: {symbol} {signal_type.value} @ {entry_price} | Score: {total_score}")
        
        return signal
    
    async def create_signal_from_analysis(
        self,
        symbol: str,
        analysis_result: Dict[str, Any],
        strategy_id: Optional[UUID] = None
    ) -> Optional[SignalModel]:
        """
        Analiz sonucundan sinyal oluştur.
        
        Args:
            symbol: Hisse sembolü
            analysis_result: Analiz sonuçları
            strategy_id: Strateji ID (varsa)
            
        Returns:
            Optional[SignalModel]: Sinyal veya None (skor yetersizse)
        """
        total_score = analysis_result.get("total_score", 0)
        
        # Minimum skor kontrolü
        if total_score < settings.signal.min_signal_score:
            logger.debug(f"Score too low for signal: {symbol} ({total_score})")
            return None
        
        # Bias'a göre sinyal tipi
        bias = analysis_result.get("bias", "neutral")
        if bias == "bullish":
            signal_type = SignalType.LONG
        elif bias == "bearish":
            signal_type = SignalType.SHORT
        else:
            return None
        
        # Risk verilerini al
        risk_data = analysis_result.get("risk", {})
        
        return await self.create_signal(
            symbol=symbol,
            signal_type=signal_type,
            entry_price=risk_data.get("entry_price", Decimal("0")),
            stop_loss=risk_data.get("suggested_stop_loss", Decimal("0")),
            take_profit_1=risk_data.get("suggested_tp1", Decimal("0")),
            take_profit_2=risk_data.get("suggested_tp2"),
            take_profit_3=risk_data.get("suggested_tp3"),
            total_score=Decimal(str(total_score)),
            smc_score=analysis_result.get("smc", {}).get("smc_score"),
            orderflow_score=analysis_result.get("orderflow", {}).get("orderflow_score"),
            alpha_score=analysis_result.get("alpha", {}).get("alpha_score"),
            confidence=Decimal(str(analysis_result.get("confidence", 0.5))),
            risk_reward=Decimal(str(risk_data.get("risk_reward", 2.0))),
            setup_type=analysis_result.get("setup_type", "composite"),
            strategy_id=strategy_id,
            reasoning=analysis_result.get("bullish_factors", []) + analysis_result.get("bearish_factors", []),
            smc_data=analysis_result.get("smc"),
            orderflow_data=analysis_result.get("orderflow"),
        )
    
    # =========================================================================
    # SIGNAL RETRIEVAL
    # =========================================================================
    
    async def get_active_signals(
        self,
        tier: Optional[SignalTier] = None,
        signal_type: Optional[SignalType] = None,
        limit: int = 50
    ) -> List[SignalModel]:
        """
        Aktif sinyalleri al.
        
        Args:
            tier: Tier filtresi
            signal_type: Tip filtresi
            limit: Maksimum sonuç
            
        Returns:
            List[SignalModel]: Aktif sinyaller
        """
        return await self.signal_repo.get_active_signals(
            tier=tier,
            signal_type=signal_type,
            limit=limit
        )
    
    async def get_signals_by_symbol(
        self,
        symbol: str,
        include_closed: bool = False,
        limit: int = 20
    ) -> List[SignalModel]:
        """
        Hisse sinyallerini al.
        
        Args:
            symbol: Hisse sembolü
            include_closed: Kapalı sinyalleri dahil et
            limit: Maksimum sonuç
            
        Returns:
            List[SignalModel]: Sinyaller
        """
        return await self.signal_repo.find_by_symbol(
            symbol=symbol,
            include_closed=include_closed,
            limit=limit
        )
    
    async def get_today_signals(
        self,
        tier: Optional[SignalTier] = None
    ) -> List[SignalModel]:
        """
        Bugünün sinyallerini al.
        
        Args:
            tier: Tier filtresi
            
        Returns:
            List[SignalModel]: Bugünün sinyalleri
        """
        return await self.signal_repo.get_today_signals(tier=tier)
    
    # =========================================================================
    # SIGNAL MANAGEMENT
    # =========================================================================
    
    async def close_signal(
        self,
        signal_id: UUID,
        exit_price: Decimal,
        reason: str,
        pnl: Optional[Decimal] = None
    ) -> Optional[SignalModel]:
        """
        Sinyali kapat.
        
        Args:
            signal_id: Sinyal ID
            exit_price: Çıkış fiyatı
            reason: Kapanış sebebi
            pnl: P&L (opsiyonel, hesaplanabilir)
            
        Returns:
            Optional[SignalModel]: Kapatılan sinyal
        """
        signal = await self.signal_repo.close_signal(
            signal_id=signal_id,
            exit_price=exit_price,
            reason=reason,
            pnl=pnl
        )
        
        if signal:
            await self._invalidate_signal_cache()
            logger.info(f"Signal closed: {signal.symbol} | Reason: {reason} | PnL: {signal.result_pnl}")
        
        return signal
    
    async def update_signal(
        self,
        signal_id: UUID,
        **kwargs
    ) -> Optional[SignalModel]:
        """
        Sinyal güncelle.
        
        Args:
            signal_id: Sinyal ID
            **kwargs: Güncellenecek alanlar
            
        Returns:
            Optional[SignalModel]: Güncellenen sinyal
        """
        signal = await self.signal_repo.update(signal_id, **kwargs)
        
        if signal:
            await self._invalidate_signal_cache()
        
        return signal
    
    async def expire_old_signals(self, max_age_hours: int = 48) -> int:
        """
        Eski sinyalleri expire et.
        
        Args:
            max_age_hours: Maksimum yaş (saat)
            
        Returns:
            int: Expire edilen sinyal sayısı
        """
        count = await self.signal_repo.expire_old_signals(max_age_hours=max_age_hours)
        
        if count > 0:
            await self._invalidate_signal_cache()
            logger.info(f"Expired {count} old signals")
        
        return count
    
    # =========================================================================
    # PRICE LEVEL MONITORING
    # =========================================================================
    
    async def check_price_levels(
        self,
        prices: Dict[str, Decimal]
    ) -> List[Dict[str, Any]]:
        """
        Fiyat seviyelerini kontrol et.
        
        Aktif sinyallerin stop loss ve take profit seviyelerini
        mevcut fiyatlarla karşılaştırır.
        
        Args:
            prices: Sembol-fiyat eşleştirmesi
            
        Returns:
            List[Dict]: Tetiklenen uyarılar
        """
        alerts = []
        
        for symbol, price in prices.items():
            triggered = await self.signal_repo.check_price_levels(symbol, price)
            
            for t in triggered:
                signal = await self.signal_repo.get(t["signal_id"])
                if signal:
                    alert = {
                        "signal_id": str(t["signal_id"]),
                        "symbol": symbol,
                        "alert_type": t["type"],
                        "current_price": float(price),
                        "target_price": float(
                            signal.stop_loss if t["type"] == "stop_loss"
                            else getattr(signal, t["type"], signal.take_profit_1)
                        ),
                        "signal_type": signal.signal_type,
                        "entry_price": float(signal.entry_price),
                        "timestamp": datetime.utcnow().isoformat(),
                    }
                    alerts.append(alert)
                    
                    # Uyarı log'la
                    logger.warning(
                        f"Price alert: {symbol} {t['type']} triggered at {price}"
                    )
        
        return alerts
    
    async def check_all_active_signals(
        self,
        prices: Dict[str, Decimal]
    ) -> Tuple[List[Dict], List[SignalModel]]:
        """
        Tüm aktif sinyalleri kontrol et ve otomatik kapat.
        
        Args:
            prices: Güncel fiyatlar
            
        Returns:
            Tuple[alerts, closed_signals]
        """
        alerts = []
        closed_signals = []
        
        active_signals = await self.get_active_signals()
        
        for signal in active_signals:
            if signal.symbol not in prices:
                continue
            
            current_price = prices[signal.symbol]
            
            # Stop loss kontrolü
            if self._is_stop_loss_hit(signal, current_price):
                closed = await self.close_signal(
                    signal.id,
                    current_price,
                    "stop_loss_triggered"
                )
                if closed:
                    closed_signals.append(closed)
                    alerts.append({
                        "type": "stop_loss",
                        "signal": closed,
                        "price": current_price
                    })
            
            # Take profit kontrolü
            elif self._is_take_profit_hit(signal, current_price):
                closed = await self.close_signal(
                    signal.id,
                    current_price,
                    "take_profit_triggered"
                )
                if closed:
                    closed_signals.append(closed)
                    alerts.append({
                        "type": "take_profit",
                        "signal": closed,
                        "price": current_price
                    })
        
        return alerts, closed_signals
    
    # =========================================================================
    # PERFORMANCE STATISTICS
    # =========================================================================
    
    async def get_performance_stats(
        self,
        strategy_id: Optional[UUID] = None,
        days: int = 30
    ) -> Dict[str, Any]:
        """
        Performans istatistikleri.
        
        Args:
            strategy_id: Strateji filtresi
            days: Analiz süresi
            
        Returns:
            Dict: Performans metrikleri
        """
        return await self.signal_repo.get_performance_stats(
            strategy_id=strategy_id,
            days=days
        )
    
    async def get_tier_performance(self, days: int = 30) -> List[Dict[str, Any]]:
        """
        Tier bazlı performans.
        
        Args:
            days: Analiz süresi
            
        Returns:
            List[Dict]: Tier performansları
        """
        return await self.signal_repo.get_tier_performance(days=days)
    
    async def get_symbol_performance(
        self,
        days: int = 30,
        limit: int = 10
    ) -> List[Dict[str, Any]]:
        """
        Sembol bazlı performans.
        
        Args:
            days: Analiz süresi
            limit: Maksimum sonuç
            
        Returns:
            List[Dict]: Sembol performansları
        """
        return await self.signal_repo.get_top_performing_symbols(
            days=days,
            limit=limit
        )
    
    # =========================================================================
    # HELPERS
    # =========================================================================
    
    def _validate_signal_params(
        self,
        signal_type: SignalType,
        entry_price: Decimal,
        stop_loss: Decimal,
        take_profit_1: Decimal
    ) -> None:
        """Sinyal parametrelerini doğrula."""
        if signal_type == SignalType.LONG:
            if stop_loss >= entry_price:
                raise ValueError("LONG signal: stop_loss must be below entry_price")
            if take_profit_1 <= entry_price:
                raise ValueError("LONG signal: take_profit must be above entry_price")
        else:  # SHORT
            if stop_loss <= entry_price:
                raise ValueError("SHORT signal: stop_loss must be above entry_price")
            if take_profit_1 >= entry_price:
                raise ValueError("SHORT signal: take_profit must be below entry_price")
    
    def _calculate_tier(self, score: Decimal) -> SignalTier:
        """Skora göre tier hesapla."""
        score_float = float(score)
        
        if score_float >= settings.signal.tier1_threshold:
            return SignalTier.TIER1
        elif score_float >= settings.signal.tier2_threshold:
            return SignalTier.TIER2
        else:
            return SignalTier.TIER3
    
    def _is_stop_loss_hit(self, signal: SignalModel, price: Decimal) -> bool:
        """Stop loss tetiklendi mi?"""
        if signal.signal_type == SignalType.LONG.value:
            return price <= signal.stop_loss
        else:  # SHORT
            return price >= signal.stop_loss
    
    def _is_take_profit_hit(self, signal: SignalModel, price: Decimal) -> bool:
        """Take profit tetiklendi mi?"""
        if signal.signal_type == SignalType.LONG.value:
            return price >= signal.take_profit_1
        else:  # SHORT
            return price <= signal.take_profit_1
    
    async def _invalidate_signal_cache(self) -> None:
        """Sinyal cache'ini temizle."""
        await cache.delete_pattern(f"{CacheKeys.PREFIX_SIGNALS}:*")
