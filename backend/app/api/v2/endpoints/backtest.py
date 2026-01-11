"""
AlphaTerminal Pro - API v2 Backtest Endpoints
=============================================

RESTful endpoints for backtesting operations.

Author: AlphaTerminal Team
Version: 2.0.0
"""

import logging
import uuid
from datetime import datetime
from typing import Optional, List, Dict, Any

from fastapi import APIRouter, Depends, Query, Path, HTTPException, Request, BackgroundTasks
from fastapi.responses import JSONResponse

from app.api.v2.schemas.base import (
    APIResponse, PaginatedResponse, MetaInfo, ErrorCode
)
from app.api.v2.schemas.backtest import (
    BacktestRequest, MultiSymbolBacktestRequest, StrategyOptimizationRequest,
    BacktestResponse, MultiSymbolBacktestResponse, StrategyOptimizationResponse,
    BacktestJobStatus, BacktestStatus, StrategyType, StrategyConfig,
    BacktestConfig, PerformanceMetrics, TradeResult, EquityCurvePoint,
    MonthlyReturn
)


logger = logging.getLogger(__name__)

router = APIRouter(prefix="/backtest", tags=["Backtesting"])

# In-memory job storage (use Redis in production)
_jobs: Dict[str, Dict[str, Any]] = {}


# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

def _strategy_type_to_class(strategy_type: StrategyType):
    """Convert strategy type enum to strategy class."""
    try:
        from app.backtest.strategies import (
            SMACrossoverStrategy,
            DualSMACrossoverStrategy,
            RSIMeanReversionStrategy,
            RSIExtremesStrategy
        )
        
        mapping = {
            StrategyType.SMA_CROSSOVER: SMACrossoverStrategy,
            StrategyType.DUAL_SMA_CROSSOVER: DualSMACrossoverStrategy,
            StrategyType.RSI_MEAN_REVERSION: RSIMeanReversionStrategy,
            StrategyType.RSI_EXTREMES: RSIExtremesStrategy,
        }
        
        return mapping.get(strategy_type)
    except ImportError:
        return None


def _convert_backtest_result(result, symbol: str, strategy_config: StrategyConfig) -> BacktestResponse:
    """Convert internal backtest result to API response."""
    
    # Convert trades
    trades = []
    if hasattr(result, 'trades'):
        for trade in result.trades:
            trades.append(TradeResult(
                trade_id=getattr(trade, 'trade_id', str(uuid.uuid4())),
                symbol=getattr(trade, 'symbol', symbol),
                direction=getattr(trade, 'direction', 'long'),
                entry_time=trade.entry_time,
                entry_price=trade.entry_price,
                quantity=trade.quantity,
                exit_time=trade.exit_time,
                exit_price=trade.exit_price,
                exit_reason=str(getattr(trade, 'exit_reason', 'unknown')),
                gross_pnl=getattr(trade, 'gross_pnl', 0),
                net_pnl=getattr(trade, 'net_pnl', 0),
                pnl_pct=getattr(trade, 'pnl_pct', 0),
                initial_stop_loss=getattr(trade, 'initial_stop_loss', None),
                initial_take_profit=getattr(trade, 'initial_take_profit', None),
                r_multiple=getattr(trade, 'r_multiple', None),
                bars_held=getattr(trade, 'bars_held', 0),
                holding_hours=getattr(trade, 'holding_hours', 0),
                max_favorable_excursion=getattr(trade, 'max_favorable_excursion', None),
                max_adverse_excursion=getattr(trade, 'max_adverse_excursion', None)
            ))
    
    # Convert equity curve
    equity_curve = None
    if hasattr(result, 'equity_curve') and result.equity_curve is not None:
        equity_curve = []
        for timestamp, equity in result.equity_curve.items():
            equity_curve.append(EquityCurvePoint(
                timestamp=timestamp,
                equity=equity,
                cash=equity,
                positions_value=0,
                drawdown=0
            ))
    
    # Build metrics
    metrics = PerformanceMetrics(
        total_return=getattr(result, 'total_return', 0),
        total_return_pct=getattr(result, 'total_return_pct', 0),
        annualized_return=getattr(result, 'annualized_return', 0),
        volatility=getattr(result, 'volatility', 0),
        max_drawdown=getattr(result, 'max_drawdown', 0),
        sharpe_ratio=getattr(result, 'sharpe_ratio', 0),
        sortino_ratio=getattr(result, 'sortino_ratio', 0),
        calmar_ratio=getattr(result, 'calmar_ratio', 0),
        total_trades=getattr(result, 'total_trades', 0),
        winning_trades=getattr(result, 'winning_trades', 0),
        losing_trades=getattr(result, 'losing_trades', 0),
        win_rate=getattr(result, 'win_rate', 0),
        profit_factor=getattr(result, 'profit_factor', 0),
        avg_trade_pnl=getattr(result, 'avg_trade', 0),
        avg_winner=getattr(result, 'avg_winner', 0),
        avg_loser=getattr(result, 'avg_loser', 0),
        largest_winner=getattr(result, 'largest_winner', 0),
        largest_loser=getattr(result, 'largest_loser', 0),
        avg_win_loss_ratio=getattr(result, 'avg_win_loss_ratio', 0),
        expectancy=getattr(result, 'expectancy', 0),
        avg_holding_period_hours=getattr(result, 'avg_holding_period_hours', 0)
    )
    
    return BacktestResponse(
        backtest_id=str(uuid.uuid4()),
        status=BacktestStatus.COMPLETED,
        symbol=symbol,
        interval=getattr(result, 'timeframe', '1d'),
        strategy_type=strategy_config.strategy_type.value,
        strategy_params=strategy_config.params,
        start_date=getattr(result, 'start_date', datetime.now()),
        end_date=getattr(result, 'end_date', datetime.now()),
        total_bars=getattr(result, 'total_bars', 0),
        config=BacktestConfig(),
        metrics=metrics,
        trades=trades,
        equity_curve=equity_curve,
        execution_time_seconds=getattr(result, 'execution_time_seconds', 0),
        data_source="tradingview"
    )


# =============================================================================
# SYNCHRONOUS BACKTEST ENDPOINTS
# =============================================================================

@router.post(
    "/run",
    response_model=APIResponse[BacktestResponse],
    summary="Run backtest",
    description="Run a backtest synchronously and return results."
)
async def run_backtest(
    request_data: BacktestRequest,
    request: Request = None
):
    """
    Execute a backtest with the specified strategy and configuration.
    
    This is a synchronous endpoint - waits for backtest to complete.
    For long-running backtests, use the async job endpoint.
    
    **Supported Strategies:**
    - `sma_crossover`: Simple Moving Average Crossover
    - `dual_sma_crossover`: Dual SMA with trend filter
    - `rsi_mean_reversion`: RSI Mean Reversion
    - `rsi_extremes`: RSI Extremes strategy
    
    **Strategy Parameters:**
    - SMA Crossover: `fast_period`, `slow_period`, `atr_multiplier`, `risk_reward`
    - RSI Mean Reversion: `rsi_period`, `oversold`, `overbought`, `use_trend_filter`
    """
    start_time = datetime.now()
    
    try:
        from app.backtest import BacktestEngine, BacktestConfig as BTConfig
        from app.data_providers import DataManager, DataInterval
        
        # Get strategy class
        strategy_class = _strategy_type_to_class(request_data.strategy.strategy_type)
        if not strategy_class:
            return APIResponse.error(
                code=ErrorCode.INVALID_REQUEST,
                message=f"Unsupported strategy type: {request_data.strategy.strategy_type}"
            )
        
        # Create strategy instance
        strategy = strategy_class(**request_data.strategy.params)
        
        # Create backtest config
        bt_config = BTConfig(
            initial_capital=request_data.config.initial_capital,
            commission_rate=request_data.config.commission_rate,
            slippage_rate=request_data.config.slippage_rate,
            max_position_size=request_data.config.max_position_size,
            max_positions=request_data.config.max_positions,
            risk_per_trade=request_data.config.risk_per_trade,
            allow_shorting=request_data.config.allow_shorting,
            log_trades=request_data.log_trades
        )
        
        # Get market data
        manager = DataManager.get_instance()
        interval = DataInterval.from_string(request_data.interval)
        
        market_data = manager.get_data(
            symbol=request_data.symbol,
            interval=interval,
            start_date=request_data.start_date,
            end_date=request_data.end_date,
            bars=1000
        )
        
        # Run backtest
        engine = BacktestEngine(config=bt_config)
        result = engine.run(
            data=market_data.data,
            strategy=strategy,
            symbol=request_data.symbol,
            timeframe=request_data.interval
        )
        
        # Convert to response
        response_data = _convert_backtest_result(
            result,
            request_data.symbol,
            request_data.strategy
        )
        
        duration_ms = (datetime.now() - start_time).total_seconds() * 1000
        meta = MetaInfo(duration_ms=duration_ms)
        
        return APIResponse.success(data=response_data, meta=meta)
        
    except Exception as e:
        logger.error(f"Backtest error: {e}", exc_info=True)
        return APIResponse.error(
            code=ErrorCode.INTERNAL_ERROR,
            message=str(e),
            details={"symbol": request_data.symbol}
        )


@router.post(
    "/run/batch",
    response_model=APIResponse[dict],
    summary="Run multi-symbol backtest",
    description="Run backtest across multiple symbols."
)
async def run_batch_backtest(
    request_data: MultiSymbolBacktestRequest,
    request: Request = None
):
    """
    Run the same strategy across multiple symbols.
    
    Results include individual symbol performance and aggregate metrics.
    """
    start_time = datetime.now()
    
    try:
        from app.backtest import BacktestEngine, BacktestConfig as BTConfig
        from app.data_providers import DataManager, DataInterval
        
        strategy_class = _strategy_type_to_class(request_data.strategy.strategy_type)
        if not strategy_class:
            return APIResponse.error(
                code=ErrorCode.INVALID_REQUEST,
                message=f"Unsupported strategy type: {request_data.strategy.strategy_type}"
            )
        
        manager = DataManager.get_instance()
        interval = DataInterval.from_string(request_data.interval)
        
        bt_config = BTConfig(
            initial_capital=request_data.config.initial_capital,
            commission_rate=request_data.config.commission_rate,
            slippage_rate=request_data.config.slippage_rate,
            max_position_size=request_data.config.max_position_size,
            log_trades=False
        )
        
        results = {}
        errors = []
        
        for symbol in request_data.symbols:
            try:
                # Fresh strategy instance
                strategy = strategy_class(**request_data.strategy.params)
                
                # Get data
                market_data = manager.get_data(
                    symbol=symbol,
                    interval=interval,
                    start_date=request_data.start_date,
                    end_date=request_data.end_date,
                    bars=500
                )
                
                # Run backtest
                engine = BacktestEngine(config=bt_config)
                result = engine.run(
                    data=market_data.data,
                    strategy=strategy,
                    symbol=symbol,
                    timeframe=request_data.interval
                )
                
                results[symbol] = {
                    "total_return_pct": result.total_return_pct,
                    "sharpe_ratio": result.sharpe_ratio,
                    "max_drawdown": result.max_drawdown,
                    "total_trades": result.total_trades,
                    "win_rate": result.win_rate,
                    "profit_factor": result.profit_factor
                }
                
            except Exception as e:
                errors.append({"symbol": symbol, "error": str(e)})
        
        # Calculate aggregate metrics
        aggregate = {}
        if results:
            aggregate = {
                "avg_return": sum(r["total_return_pct"] for r in results.values()) / len(results),
                "avg_sharpe": sum(r["sharpe_ratio"] for r in results.values()) / len(results),
                "avg_max_drawdown": sum(r["max_drawdown"] for r in results.values()) / len(results),
                "total_trades": sum(r["total_trades"] for r in results.values()),
                "avg_win_rate": sum(r["win_rate"] for r in results.values()) / len(results),
                "best_symbol": max(results.keys(), key=lambda k: results[k]["total_return_pct"]),
                "worst_symbol": min(results.keys(), key=lambda k: results[k]["total_return_pct"])
            }
        
        duration_ms = (datetime.now() - start_time).total_seconds() * 1000
        
        response_data = {
            "results": results,
            "aggregate": aggregate,
            "total_symbols": len(request_data.symbols),
            "successful": len(results),
            "failed": len(errors),
            "errors": errors if errors else None
        }
        
        meta = MetaInfo(duration_ms=duration_ms)
        return APIResponse.success(data=response_data, meta=meta)
        
    except Exception as e:
        logger.error(f"Batch backtest error: {e}", exc_info=True)
        return APIResponse.error(
            code=ErrorCode.INTERNAL_ERROR,
            message=str(e)
        )


# =============================================================================
# ASYNC JOB ENDPOINTS
# =============================================================================

@router.post(
    "/jobs",
    response_model=APIResponse[BacktestJobStatus],
    summary="Submit backtest job",
    description="Submit a backtest job for async processing."
)
async def submit_backtest_job(
    request_data: BacktestRequest,
    background_tasks: BackgroundTasks,
    request: Request = None
):
    """
    Submit a backtest job for asynchronous execution.
    
    Returns a job ID that can be used to:
    - Check job status
    - Retrieve results when complete
    - Cancel the job
    
    Use this for long-running backtests or when you don't want to wait.
    """
    job_id = str(uuid.uuid4())
    
    # Create job record
    _jobs[job_id] = {
        "status": BacktestStatus.PENDING,
        "progress": 0,
        "message": "Job queued",
        "created_at": datetime.now(),
        "request": request_data.model_dump(),
        "result": None
    }
    
    # Add background task
    async def run_job():
        try:
            _jobs[job_id]["status"] = BacktestStatus.RUNNING
            _jobs[job_id]["started_at"] = datetime.now()
            _jobs[job_id]["message"] = "Running backtest..."
            
            # Run backtest (similar to sync endpoint)
            from app.backtest import BacktestEngine, BacktestConfig as BTConfig
            from app.data_providers import DataManager, DataInterval
            
            strategy_class = _strategy_type_to_class(request_data.strategy.strategy_type)
            strategy = strategy_class(**request_data.strategy.params)
            
            bt_config = BTConfig(
                initial_capital=request_data.config.initial_capital,
                commission_rate=request_data.config.commission_rate,
                log_trades=request_data.log_trades
            )
            
            manager = DataManager.get_instance()
            interval = DataInterval.from_string(request_data.interval)
            
            _jobs[job_id]["progress"] = 20
            _jobs[job_id]["message"] = "Fetching market data..."
            
            market_data = manager.get_data(
                symbol=request_data.symbol,
                interval=interval,
                start_date=request_data.start_date,
                end_date=request_data.end_date,
                bars=1000
            )
            
            _jobs[job_id]["progress"] = 50
            _jobs[job_id]["message"] = "Running backtest..."
            
            engine = BacktestEngine(config=bt_config)
            result = engine.run(
                data=market_data.data,
                strategy=strategy,
                symbol=request_data.symbol,
                timeframe=request_data.interval
            )
            
            _jobs[job_id]["progress"] = 100
            _jobs[job_id]["status"] = BacktestStatus.COMPLETED
            _jobs[job_id]["completed_at"] = datetime.now()
            _jobs[job_id]["message"] = "Completed"
            _jobs[job_id]["result"] = _convert_backtest_result(
                result, request_data.symbol, request_data.strategy
            ).model_dump()
            
        except Exception as e:
            _jobs[job_id]["status"] = BacktestStatus.FAILED
            _jobs[job_id]["message"] = str(e)
            _jobs[job_id]["completed_at"] = datetime.now()
    
    background_tasks.add_task(run_job)
    
    job_status = BacktestJobStatus(
        job_id=job_id,
        status=BacktestStatus.PENDING,
        progress=0,
        message="Job queued",
        created_at=datetime.now()
    )
    
    return APIResponse.success(data=job_status)


@router.get(
    "/jobs/{job_id}",
    response_model=APIResponse[BacktestJobStatus],
    summary="Get job status",
    description="Get the status of a backtest job."
)
async def get_job_status(
    job_id: str = Path(..., description="Job ID"),
    request: Request = None
):
    """
    Check the status of a submitted backtest job.
    
    Returns progress percentage and status message.
    """
    if job_id not in _jobs:
        return APIResponse.error(
            code=ErrorCode.NOT_FOUND,
            message=f"Job not found: {job_id}"
        )
    
    job = _jobs[job_id]
    
    job_status = BacktestJobStatus(
        job_id=job_id,
        status=job["status"],
        progress=job["progress"],
        message=job.get("message"),
        created_at=job["created_at"],
        started_at=job.get("started_at"),
        completed_at=job.get("completed_at"),
        result_url=f"/api/v2/backtest/jobs/{job_id}/result" if job["status"] == BacktestStatus.COMPLETED else None
    )
    
    return APIResponse.success(data=job_status)


@router.get(
    "/jobs/{job_id}/result",
    response_model=APIResponse[BacktestResponse],
    summary="Get job result",
    description="Get the result of a completed backtest job."
)
async def get_job_result(
    job_id: str = Path(..., description="Job ID"),
    request: Request = None
):
    """
    Retrieve the results of a completed backtest job.
    
    Returns full backtest results including metrics, trades, and equity curve.
    """
    if job_id not in _jobs:
        return APIResponse.error(
            code=ErrorCode.NOT_FOUND,
            message=f"Job not found: {job_id}"
        )
    
    job = _jobs[job_id]
    
    if job["status"] != BacktestStatus.COMPLETED:
        return APIResponse.error(
            code=ErrorCode.INVALID_REQUEST,
            message=f"Job not completed. Status: {job['status']}"
        )
    
    if not job.get("result"):
        return APIResponse.error(
            code=ErrorCode.INTERNAL_ERROR,
            message="Result not available"
        )
    
    return APIResponse.success(data=BacktestResponse(**job["result"]))


@router.delete(
    "/jobs/{job_id}",
    response_model=APIResponse[dict],
    summary="Cancel job",
    description="Cancel a pending or running backtest job."
)
async def cancel_job(
    job_id: str = Path(..., description="Job ID"),
    request: Request = None
):
    """
    Cancel a backtest job.
    
    Only pending or running jobs can be cancelled.
    """
    if job_id not in _jobs:
        return APIResponse.error(
            code=ErrorCode.NOT_FOUND,
            message=f"Job not found: {job_id}"
        )
    
    job = _jobs[job_id]
    
    if job["status"] in {BacktestStatus.COMPLETED, BacktestStatus.FAILED, BacktestStatus.CANCELLED}:
        return APIResponse.error(
            code=ErrorCode.INVALID_REQUEST,
            message=f"Cannot cancel job with status: {job['status']}"
        )
    
    job["status"] = BacktestStatus.CANCELLED
    job["message"] = "Cancelled by user"
    job["completed_at"] = datetime.now()
    
    return APIResponse.success(data={"job_id": job_id, "cancelled": True})


# =============================================================================
# STRATEGY INFO ENDPOINTS
# =============================================================================

@router.get(
    "/strategies",
    response_model=APIResponse[List[dict]],
    summary="List available strategies",
    description="Get list of available backtest strategies with their parameters."
)
async def list_strategies(request: Request = None):
    """
    List all available backtest strategies.
    
    Returns strategy names, descriptions, and parameter specifications.
    """
    strategies = [
        {
            "type": "sma_crossover",
            "name": "SMA Crossover",
            "description": "Simple Moving Average crossover strategy",
            "parameters": {
                "fast_period": {"type": "int", "default": 10, "min": 2, "max": 50},
                "slow_period": {"type": "int", "default": 30, "min": 10, "max": 200},
                "atr_multiplier": {"type": "float", "default": 2.0, "min": 0.5, "max": 5.0},
                "risk_reward": {"type": "float", "default": 2.0, "min": 1.0, "max": 5.0}
            }
        },
        {
            "type": "dual_sma_crossover",
            "name": "Dual SMA Crossover",
            "description": "Dual SMA crossover with trend filter",
            "parameters": {
                "fast_period": {"type": "int", "default": 10, "min": 2, "max": 50},
                "slow_period": {"type": "int", "default": 30, "min": 10, "max": 200},
                "trend_period": {"type": "int", "default": 100, "min": 50, "max": 300}
            }
        },
        {
            "type": "rsi_mean_reversion",
            "name": "RSI Mean Reversion",
            "description": "RSI-based mean reversion strategy",
            "parameters": {
                "rsi_period": {"type": "int", "default": 14, "min": 5, "max": 30},
                "oversold": {"type": "int", "default": 30, "min": 10, "max": 40},
                "overbought": {"type": "int", "default": 70, "min": 60, "max": 90},
                "use_trend_filter": {"type": "bool", "default": True}
            }
        },
        {
            "type": "rsi_extremes",
            "name": "RSI Extremes",
            "description": "RSI extreme levels strategy",
            "parameters": {
                "rsi_period": {"type": "int", "default": 7, "min": 3, "max": 20},
                "oversold": {"type": "int", "default": 20, "min": 5, "max": 30},
                "overbought": {"type": "int", "default": 80, "min": 70, "max": 95}
            }
        }
    ]
    
    return APIResponse.success(data=strategies)


# =============================================================================
# ALL EXPORTS
# =============================================================================

__all__ = ["router"]
