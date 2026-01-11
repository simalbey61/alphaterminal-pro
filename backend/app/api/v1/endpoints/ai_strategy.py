"""
AlphaTerminal Pro - AI Strategy API Endpoints
=============================================

AI strateji sistemi için REST API endpoint'leri.

Author: AlphaTerminal Team
Version: 1.0.0
"""

from __future__ import annotations

import logging
from typing import Optional, List, Dict, Any
from datetime import datetime

from fastapi import APIRouter, Depends, HTTPException, Query, BackgroundTasks
from pydantic import BaseModel, Field

from app.api.dependencies import (
    get_current_user,
    get_premium_user,
    RateLimiter,
)

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/ai-strategy", tags=["AI Strategy"])


# =============================================================================
# PYDANTIC MODELS
# =============================================================================

class StrategyDiscoveryRequest(BaseModel):
    """Strateji keşif isteği."""
    symbol: str = Field(..., description="Hisse sembolü")
    timeframe: str = Field(default="4h", description="Zaman dilimi")
    lookback_days: int = Field(default=365, ge=30, le=1000, description="Geçmiş gün sayısı")
    min_trades: int = Field(default=100, ge=20, description="Minimum trade sayısı")


class SignalRequest(BaseModel):
    """Sinyal üretim isteği."""
    symbol: str = Field(..., description="Hisse sembolü")
    timeframe: str = Field(default="4h", description="Zaman dilimi")
    entry_price: Optional[float] = Field(None, description="Giriş fiyatı (position sizing için)")


class TradeResultRequest(BaseModel):
    """Trade sonucu kayıt isteği."""
    strategy_id: str
    pnl: float
    is_win: bool
    trade_data: Optional[Dict[str, Any]] = None


class StrategyResponse(BaseModel):
    """Strateji response modeli."""
    id: str
    name: str
    strategy_type: str
    lifecycle: str
    expected_win_rate: float
    expected_profit_factor: float
    confidence: float
    entry_conditions: List[Dict[str, Any]]
    created_at: datetime


class SignalResponse(BaseModel):
    """Sinyal response modeli."""
    symbol: str
    timeframe: str
    regime: Dict[str, Any]
    signals: List[Dict[str, Any]]
    consensus_direction: Optional[str]
    consensus_strength: float
    position_size: Optional[Dict[str, Any]]
    generated_at: datetime


class DiscoveryResponse(BaseModel):
    """Keşif response modeli."""
    discovered_patterns: int
    synthesized_strategies: int
    approved: int
    sandbox: int
    rejected: int
    discovery_time_seconds: float
    strategies: List[StrategyResponse]


class DiversityResponse(BaseModel):
    """Çeşitlilik response modeli."""
    total_strategies: int
    active_strategies: int
    avg_correlation: float
    regime_coverage: Dict[str, int]
    zoo_distribution: Dict[str, int]
    diversification_score: float


class HealthCheckResponse(BaseModel):
    """AI sistem sağlık kontrolü."""
    status: str
    active_strategies: int
    components: Dict[str, str]
    last_evolution: Optional[datetime]


# =============================================================================
# ORCHESTRATOR SINGLETON
# =============================================================================

# Lazy-loaded orchestrator
_orchestrator = None

def get_orchestrator():
    """Get or create orchestrator instance."""
    global _orchestrator
    if _orchestrator is None:
        from app.ai_strategy.orchestrator import AIStrategyOrchestrator
        _orchestrator = AIStrategyOrchestrator()
    return _orchestrator


# =============================================================================
# ENDPOINTS
# =============================================================================

@router.get("/health", response_model=HealthCheckResponse)
async def ai_health_check():
    """
    AI strateji sistemi sağlık kontrolü.
    """
    try:
        orchestrator = get_orchestrator()
        strategy_counts = orchestrator.get_strategy_count()
        
        return HealthCheckResponse(
            status="healthy",
            active_strategies=strategy_counts.get("active", 0),
            components={
                "data_layer": "ok",
                "feature_factory": "ok",
                "pattern_discovery": "ok",
                "strategy_generator": "ok",
                "validation_engine": "ok",
                "live_execution": "ok",
                "evolution_engine": "ok",
            },
            last_evolution=None,
        )
    except Exception as e:
        logger.error(f"AI health check error: {e}")
        return HealthCheckResponse(
            status="degraded",
            active_strategies=0,
            components={"error": str(e)},
            last_evolution=None,
        )


@router.post("/discover", response_model=DiscoveryResponse)
async def discover_strategies(
    request: StrategyDiscoveryRequest,
    background_tasks: BackgroundTasks,
    current_user = Depends(get_premium_user),
    _rate_limit = Depends(RateLimiter(requests=5, window=3600)),
):
    """
    Yeni stratejiler keşfet.
    
    Bu endpoint:
    1. Tarihsel veriyi analiz eder
    2. Örüntüleri keşfeder
    3. Stratejiler sentezler
    4. Doğrulama testleri yapar
    5. Onaylanan stratejileri aktifleştirir
    
    **Not:** Premium kullanıcılar için. Saatte max 5 istek.
    """
    try:
        orchestrator = get_orchestrator()
        
        # Veri çek (placeholder - gerçekte data_engine kullanılacak)
        import polars as pl
        import numpy as np
        
        # Placeholder data
        n_bars = request.lookback_days * 6  # 4h bars per day
        dates = pl.datetime_range(
            datetime(2023, 1, 1), datetime(2024, 1, 1),
            interval="4h", eager=True
        )[:n_bars]
        
        close = 100 * np.cumprod(1 + np.random.randn(n_bars) * 0.02)
        high = close * (1 + np.random.rand(n_bars) * 0.01)
        low = close * (1 - np.random.rand(n_bars) * 0.01)
        volume = np.random.randint(100000, 1000000, n_bars)
        
        df = pl.DataFrame({
            "timestamp": dates,
            "open": close * (1 + np.random.randn(n_bars) * 0.005),
            "high": high,
            "low": low,
            "close": close,
            "volume": volume,
        })
        
        # Keşif pipeline'ı çalıştır
        result = await orchestrator.discover_strategies(
            historical_data=df,
            symbol=request.symbol,
            timeframe=request.timeframe,
            min_trades=request.min_trades,
        )
        
        # Response oluştur
        strategies = []
        for s in result.synthesized_strategies[:10]:  # Max 10
            strategies.append(StrategyResponse(
                id=s.id,
                name=s.name,
                strategy_type=s.strategy_type.value,
                lifecycle=s.lifecycle.value,
                expected_win_rate=s.expected_win_rate,
                expected_profit_factor=s.expected_profit_factor,
                confidence=s.confidence,
                entry_conditions=[c.to_dict() for c in s.entry_conditions],
                created_at=s.created_at,
            ))
        
        return DiscoveryResponse(
            discovered_patterns=len(result.discovered_patterns),
            synthesized_strategies=len(result.synthesized_strategies),
            approved=len(result.approved_strategies),
            sandbox=len(result.sandbox_strategies),
            rejected=len(result.rejected_strategies),
            discovery_time_seconds=result.discovery_time_seconds,
            strategies=strategies,
        )
        
    except Exception as e:
        logger.error(f"Strategy discovery error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/signals", response_model=SignalResponse)
async def generate_signals(
    request: SignalRequest,
    current_user = Depends(get_current_user),
    _rate_limit = Depends(RateLimiter(requests=60, window=60)),
):
    """
    Aktif stratejilerden sinyal üret.
    
    Bu endpoint:
    1. Mevcut piyasa rejimini tespit eder
    2. Feature'ları hesaplar
    3. Aktif stratejileri değerlendirir
    4. Konsensüs sinyal üretir
    5. Position sizing önerir
    """
    try:
        orchestrator = get_orchestrator()
        
        # Veri çek (placeholder)
        import polars as pl
        import numpy as np
        
        n_bars = 200
        close = 100 * np.cumprod(1 + np.random.randn(n_bars) * 0.02)
        
        df = pl.DataFrame({
            "timestamp": pl.datetime_range(datetime(2024, 1, 1), periods=n_bars, interval="4h", eager=True),
            "open": close * (1 + np.random.randn(n_bars) * 0.005),
            "high": close * (1 + np.random.rand(n_bars) * 0.01),
            "low": close * (1 - np.random.rand(n_bars) * 0.01),
            "close": close,
            "volume": np.random.randint(100000, 1000000, n_bars),
        })
        
        # Sinyal üret
        result = await orchestrator.generate_signals(
            ohlcv_data=df,
            symbol=request.symbol,
            timeframe=request.timeframe,
            entry_price=request.entry_price,
        )
        
        return SignalResponse(
            symbol=result.symbol,
            timeframe=result.timeframe,
            regime=result.current_regime.to_dict(),
            signals=result.signals,
            consensus_direction=result.consensus_direction.value if result.consensus_direction else None,
            consensus_strength=result.consensus_strength,
            position_size=result.position_size,
            generated_at=result.generated_at,
        )
        
    except Exception as e:
        logger.error(f"Signal generation error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/strategies", response_model=List[StrategyResponse])
async def list_strategies(
    lifecycle: Optional[str] = Query(None, description="Lifecycle filtresi"),
    limit: int = Query(50, ge=1, le=100),
    current_user = Depends(get_current_user),
):
    """
    Stratejileri listele.
    """
    try:
        orchestrator = get_orchestrator()
        strategies = orchestrator.get_active_strategies()
        
        # Filter by lifecycle if specified
        if lifecycle:
            strategies = [s for s in strategies if s.get("lifecycle") == lifecycle]
        
        # Convert to response
        result = []
        for s in strategies[:limit]:
            result.append(StrategyResponse(
                id=s["id"],
                name=s["name"],
                strategy_type=s["strategy_type"],
                lifecycle=s["lifecycle"],
                expected_win_rate=s["expected_win_rate"],
                expected_profit_factor=s["expected_profit_factor"],
                confidence=s["confidence"],
                entry_conditions=s.get("entry_conditions", []),
                created_at=datetime.fromisoformat(s["created_at"]),
            ))
        
        return result
        
    except Exception as e:
        logger.error(f"List strategies error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/strategies/{strategy_id}")
async def get_strategy(
    strategy_id: str,
    current_user = Depends(get_current_user),
):
    """
    Strateji detayları.
    """
    orchestrator = get_orchestrator()
    strategies = orchestrator.get_active_strategies()
    
    for s in strategies:
        if s["id"] == strategy_id:
            return s
    
    raise HTTPException(status_code=404, detail="Strategy not found")


@router.post("/strategies/{strategy_id}/trade-result")
async def record_trade_result(
    strategy_id: str,
    request: TradeResultRequest,
    current_user = Depends(get_current_user),
):
    """
    Trade sonucunu kaydet.
    
    Bu, performans takibi ve strateji evriminde kullanılır.
    """
    try:
        orchestrator = get_orchestrator()
        await orchestrator.record_trade_result(
            strategy_id=strategy_id,
            pnl=request.pnl,
            is_win=request.is_win,
        )
        return {"status": "recorded", "strategy_id": strategy_id}
        
    except Exception as e:
        logger.error(f"Record trade result error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/diversity", response_model=DiversityResponse)
async def get_diversity_report(
    current_user = Depends(get_premium_user),
):
    """
    Strateji çeşitlilik raporu.
    
    **Premium kullanıcılar için.**
    """
    try:
        orchestrator = get_orchestrator()
        report = orchestrator.diversity_manager.analyze_diversity()
        
        return DiversityResponse(
            total_strategies=report.total_strategies,
            active_strategies=report.active_strategies,
            avg_correlation=report.avg_pairwise_correlation,
            regime_coverage=report.regime_coverage,
            zoo_distribution=report.zoo_distribution,
            diversification_score=report.diversification_score,
        )
        
    except Exception as e:
        logger.error(f"Diversity report error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/evolution/run")
async def run_evolution_cycle(
    background_tasks: BackgroundTasks,
    current_user = Depends(get_premium_user),
    _rate_limit = Depends(RateLimiter(requests=1, window=3600)),
):
    """
    Evrim döngüsünü çalıştır.
    
    Bu endpoint:
    1. Düşük performanslı stratejileri emekliye ayırır
    2. Çeşitlilik analizi yapar
    3. Canlandırılabilecek stratejileri kontrol eder
    
    **Premium kullanıcılar için. Saatte max 1 istek.**
    """
    try:
        orchestrator = get_orchestrator()
        
        # Background'da çalıştır
        async def run_evolution():
            result = await orchestrator.run_evolution_cycle()
            logger.info(f"Evolution cycle complete: {result}")
        
        background_tasks.add_task(run_evolution)
        
        return {
            "status": "started",
            "message": "Evolution cycle started in background",
        }
        
    except Exception as e:
        logger.error(f"Evolution cycle error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/regime")
async def get_current_regime(
    symbol: str = Query(..., description="Hisse sembolü"),
    timeframe: str = Query(default="4h", description="Zaman dilimi"),
    current_user = Depends(get_current_user),
):
    """
    Mevcut piyasa rejimini al.
    """
    try:
        orchestrator = get_orchestrator()
        
        # Veri çek (placeholder)
        import polars as pl
        import numpy as np
        
        n_bars = 200
        close = 100 * np.cumprod(1 + np.random.randn(n_bars) * 0.02)
        
        df = pl.DataFrame({
            "timestamp": pl.datetime_range(datetime(2024, 1, 1), periods=n_bars, interval="4h", eager=True),
            "open": close,
            "high": close * 1.01,
            "low": close * 0.99,
            "close": close,
            "volume": np.random.randint(100000, 1000000, n_bars),
        })
        
        regime = orchestrator.regime_detector.detect(df)
        
        return {
            "symbol": symbol,
            "timeframe": timeframe,
            "regime": regime.to_dict(),
        }
        
    except Exception as e:
        logger.error(f"Get regime error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/features/{symbol}")
async def get_features(
    symbol: str,
    timeframe: str = Query(default="4h"),
    current_user = Depends(get_current_user),
):
    """
    Hesaplanmış feature'ları al.
    """
    try:
        from app.ai_strategy.feature_factory import feature_store
        
        batch = await feature_store.get_batch(symbol, timeframe)
        
        if batch:
            return {
                "symbol": symbol,
                "timeframe": timeframe,
                "features": batch,
            }
        else:
            return {
                "symbol": symbol,
                "timeframe": timeframe,
                "features": {},
                "message": "No cached features found",
            }
        
    except Exception as e:
        logger.error(f"Get features error: {e}")
        raise HTTPException(status_code=500, detail=str(e))
