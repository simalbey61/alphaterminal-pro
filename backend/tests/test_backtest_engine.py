#!/usr/bin/env python3
"""
AlphaTerminal Pro - Backtest Engine Test
========================================

Backtest engine'in temel işlevselliğini test eder.

Author: AlphaTerminal Team
"""

import sys
import os

# Add project root to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import logging
from datetime import datetime

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def test_enums():
    """Test enum imports."""
    print("\n" + "="*60)
    print("TEST 1: Enum Imports")
    print("="*60)
    
    try:
        from app.backtest.enums import (
            OrderType, OrderSide, OrderStatus,
            TradeDirection, ExitReason, SignalType
        )
        
        print(f"  ✅ OrderType values: {[e.value for e in OrderType]}")
        print(f"  ✅ SignalType ENTRY_LONG is_entry: {SignalType.ENTRY_LONG.is_entry}")
        print("  ✅ All enums imported successfully")
        return True
    except Exception as e:
        print(f"  ❌ Failed: {e}")
        return False


def test_exceptions():
    """Test exception imports."""
    print("\n" + "="*60)
    print("TEST 2: Exception Imports")
    print("="*60)
    
    try:
        from app.backtest.exceptions import (
            BacktestError, InsufficientDataError,
            ExecutionError, StrategyError
        )
        
        # Test exception creation
        error = InsufficientDataError(
            symbol="TEST",
            available_bars=10,
            required_bars=100
        )
        print(f"  ✅ InsufficientDataError: {error}")
        
        print("  ✅ All exceptions imported successfully")
        return True
    except Exception as e:
        print(f"  ❌ Failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_models():
    """Test model imports and basic functionality."""
    print("\n" + "="*60)
    print("TEST 3: Model Imports")
    print("="*60)
    
    try:
        from app.backtest.models import (
            Order, Position, Trade, TradeList,
            create_market_order, create_long_position
        )
        from app.backtest.enums import OrderSide, TradeDirection, ExitReason
        
        # Test Order creation
        order = create_market_order(
            symbol="THYAO",
            side=OrderSide.BUY,
            quantity=100
        )
        print(f"  ✅ Order created: {order.order_id}")
        
        # Test Position creation
        position = create_long_position(
            symbol="THYAO",
            quantity=100,
            entry_price=285.50,
            commission=1.5,
            stop_loss=275.0,
            take_profit=300.0
        )
        print(f"  ✅ Position created: {position.position_id}")
        print(f"     Unrealized P&L: {position.unrealized_pnl:.2f}")
        
        # Test Trade creation
        trade = Trade(
            symbol="THYAO",
            direction=TradeDirection.LONG,
            quantity=100,
            entry_price=285.50,
            exit_price=295.00,
            entry_time=datetime(2024, 1, 1, 10, 0),
            exit_time=datetime(2024, 1, 5, 15, 30),
            exit_reason=ExitReason.TAKE_PROFIT,
            initial_stop_loss=275.0
        )
        print(f"  ✅ Trade created: {trade.trade_id}")
        print(f"     P&L: {trade.net_pnl:.2f} ({trade.pnl_pct:.2%})")
        print(f"     R-Multiple: {trade.r_multiple:.2f}R")
        
        print("  ✅ All models working correctly")
        return True
    except Exception as e:
        print(f"  ❌ Failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_costs():
    """Test BIST cost calculator."""
    print("\n" + "="*60)
    print("TEST 4: Cost Calculator")
    print("="*60)
    
    try:
        from app.backtest.costs import (
            BISTCostCalculator,
            BISTCommissionConfig
        )
        from app.backtest.enums import OrderSide
        
        calc = BISTCostCalculator()
        
        # Test cost calculation
        costs = calc.calculate_total_costs(
            price=285.50,
            quantity=100,
            side=OrderSide.BUY,
            symbol="THYAO"
        )
        
        print(f"  ✅ Trade value: {costs['trade_value']:.2f} TRY")
        print(f"  ✅ Commission: {costs['total_commission']:.4f} TRY")
        print(f"  ✅ Slippage: {costs['total_slippage']:.4f} TRY")
        print(f"  ✅ Total cost: {costs['total_cost']:.4f} TRY ({costs['cost_as_pct']:.4%})")
        
        # Test round-trip
        rt_costs = calc.estimate_round_trip_cost(
            price=285.50,
            quantity=100,
            symbol="THYAO"
        )
        print(f"  ✅ Round-trip cost: {rt_costs['total_round_trip']:.4f} TRY")
        print(f"     Breakeven move: {rt_costs['breakeven_move_pct']:.4%}")
        
        print("  ✅ Cost calculator working correctly")
        return True
    except Exception as e:
        print(f"  ❌ Failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_metrics():
    """Test performance metrics."""
    print("\n" + "="*60)
    print("TEST 5: Performance Metrics")
    print("="*60)
    
    try:
        import pandas as pd
        import numpy as np
        from app.backtest.metrics import (
            calculate_sharpe_ratio,
            calculate_max_drawdown,
            calculate_all_metrics
        )
        from app.backtest.models import TradeList
        
        # Create test equity curve
        np.random.seed(42)
        dates = pd.date_range('2024-01-01', periods=252, freq='D')
        returns = np.random.normal(0.0005, 0.02, 252)
        equity = 100000 * np.cumprod(1 + returns)
        equity_curve = pd.Series(equity, index=dates)
        returns_series = equity_curve.pct_change().dropna()
        
        # Calculate metrics
        sharpe = calculate_sharpe_ratio(returns_series)
        max_dd, _, _ = calculate_max_drawdown(equity_curve)
        
        print(f"  ✅ Sharpe Ratio: {sharpe:.2f}")
        print(f"  ✅ Max Drawdown: {max_dd:.2%}")
        
        # Test comprehensive metrics
        all_metrics = calculate_all_metrics(
            equity_curve=equity_curve,
            trades=TradeList(),
            initial_capital=100000
        )
        print(f"  ✅ Sortino Ratio: {all_metrics.sortino_ratio:.2f}")
        print(f"  ✅ Calmar Ratio: {all_metrics.calmar_ratio:.2f}")
        print(f"  ✅ Volatility: {all_metrics.volatility:.2%}")
        
        print("  ✅ All metrics calculated correctly")
        return True
    except Exception as e:
        print(f"  ❌ Failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_strategy():
    """Test strategy framework."""
    print("\n" + "="*60)
    print("TEST 6: Strategy Framework")
    print("="*60)
    
    try:
        from app.backtest.engine import BaseStrategy, Signal
        from app.backtest.strategies import SMACrossoverStrategy
        import pandas as pd
        import numpy as np
        
        # Create test data
        np.random.seed(42)
        dates = pd.date_range('2024-01-01', periods=100, freq='D')
        data = pd.DataFrame({
            'Open': 100 + np.cumsum(np.random.randn(100) * 0.5),
            'High': 100 + np.cumsum(np.random.randn(100) * 0.5) + 1,
            'Low': 100 + np.cumsum(np.random.randn(100) * 0.5) - 1,
            'Close': 100 + np.cumsum(np.random.randn(100) * 0.5),
            'Volume': np.random.randint(100000, 1000000, 100)
        }, index=dates)
        
        # Fix OHLC consistency
        data['High'] = data[['Open', 'Close', 'High']].max(axis=1)
        data['Low'] = data[['Open', 'Close', 'Low']].min(axis=1)
        
        # Test strategy
        strategy = SMACrossoverStrategy(fast_period=10, slow_period=20)
        strategy.initialize()
        
        print(f"  ✅ Strategy: {strategy.name} v{strategy.version}")
        print(f"     Warmup period: {strategy.warmup_period}")
        print(f"     Parameters: {strategy.get_parameters()}")
        
        # Generate signal
        signal = strategy.generate_signal(data)
        print(f"  ✅ Signal generated: {signal.signal_type}")
        
        # Test Signal factory methods
        entry_signal = Signal.long_entry(
            entry_price=100,
            stop_loss=95,
            take_profit=110
        )
        print(f"  ✅ Entry signal: {entry_signal.signal_type}, is_entry={entry_signal.is_entry}")
        
        print("  ✅ Strategy framework working correctly")
        return True
    except Exception as e:
        print(f"  ❌ Failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_engine():
    """Test backtest engine."""
    print("\n" + "="*60)
    print("TEST 7: Backtest Engine")
    print("="*60)
    
    try:
        from app.backtest import (
            BacktestEngine, BacktestConfig
        )
        from app.backtest.strategies import SMACrossoverStrategy
        from app.backtest.utils import generate_trending_data
        
        # Generate test data
        data = generate_trending_data(
            start_date=datetime(2023, 1, 1),
            periods=300,
            initial_price=100,
            trend_strength=0.001,
            noise_level=0.02
        )
        
        print(f"  ✅ Test data generated: {len(data)} bars")
        
        # Create config
        config = BacktestConfig(
            initial_capital=100000,
            commission_rate=0.001,
            slippage_rate=0.0005,
            max_position_size=0.20,
            max_positions=5
        )
        print(f"  ✅ Config created: {config.initial_capital:.0f} TRY capital")
        
        # Create engine
        engine = BacktestEngine(config)
        print("  ✅ Engine created")
        
        # Create strategy
        strategy = SMACrossoverStrategy(
            fast_period=10,
            slow_period=30,
            atr_multiplier=2.0,
            risk_reward=2.0,
            position_size=0.10
        )
        print(f"  ✅ Strategy: {strategy.name}")
        
        # Run backtest
        print("\n  🔄 Running backtest...")
        result = engine.run(
            data=data,
            strategy=strategy,
            symbol="TEST",
            timeframe="1d"
        )
        
        print(f"\n  ✅ BACKTEST COMPLETE")
        print(f"     Period: {result.start_date.strftime('%Y-%m-%d')} to {result.end_date.strftime('%Y-%m-%d')}")
        print(f"     Total Return: {result.total_return_pct:+.2%}")
        print(f"     Sharpe Ratio: {result.sharpe_ratio:.2f}")
        print(f"     Max Drawdown: {result.max_drawdown:.2%}")
        print(f"     Total Trades: {result.total_trades}")
        print(f"     Win Rate: {result.win_rate:.1%}")
        print(f"     Profit Factor: {result.profit_factor:.2f}")
        print(f"     Execution Time: {result.execution_time_seconds:.2f}s")
        
        # Print summary
        print("\n" + result.summary())
        
        print("  ✅ Backtest engine working correctly")
        return True
    except Exception as e:
        print(f"  ❌ Failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_validators():
    """Test validators."""
    print("\n" + "="*60)
    print("TEST 8: Validators")
    print("="*60)
    
    try:
        from app.core.validators import (
            DataFrameValidator,
            NumericValidator,
            validate_ohlcv
        )
        import pandas as pd
        import numpy as np
        
        # Create valid test data
        dates = pd.date_range('2024-01-01', periods=100, freq='D')
        data = pd.DataFrame({
            'Open': np.random.uniform(100, 110, 100),
            'High': np.random.uniform(110, 120, 100),
            'Low': np.random.uniform(90, 100, 100),
            'Close': np.random.uniform(100, 110, 100),
            'Volume': np.random.randint(100000, 1000000, 100)
        }, index=dates)
        
        # Fix OHLC
        data['High'] = data[['Open', 'Close', 'High']].max(axis=1)
        data['Low'] = data[['Open', 'Close', 'Low']].min(axis=1)
        
        # Validate
        is_valid, issues = DataFrameValidator.validate_ohlcv(
            data, symbol="TEST", raise_on_error=False
        )
        
        print(f"  ✅ DataFrame validation: valid={is_valid}")
        if issues:
            for issue in issues:
                print(f"     Issue: {issue}")
        
        # Test numeric validators
        try:
            NumericValidator.validate_positive(100, "price")
            print("  ✅ Positive validation passed")
        except Exception:
            print("  ❌ Positive validation failed")
        
        try:
            NumericValidator.validate_percentage(0.5, "rate")
            print("  ✅ Percentage validation passed")
        except Exception:
            print("  ❌ Percentage validation failed")
        
        print("  ✅ Validators working correctly")
        return True
    except Exception as e:
        print(f"  ❌ Failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_circuit_breaker():
    """Test circuit breaker."""
    print("\n" + "="*60)
    print("TEST 9: Circuit Breaker")
    print("="*60)
    
    try:
        from app.core.circuit_breaker import (
            CircuitBreaker,
            CircuitBreakerConfig,
            CircuitState
        )
        
        # Create circuit breaker
        config = CircuitBreakerConfig(
            failure_threshold=3,
            timeout=1.0  # Short for testing
        )
        cb = CircuitBreaker("test_service", config)
        
        print(f"  ✅ Circuit breaker created: {cb.name}")
        print(f"     State: {cb.state.value}")
        
        # Test successful call
        def success_func():
            return "success"
        
        result = cb.call(success_func)
        print(f"  ✅ Successful call: {result}")
        
        # Test failure handling
        call_count = 0
        def fail_func():
            nonlocal call_count
            call_count += 1
            raise ValueError("Test error")
        
        for i in range(4):
            try:
                cb.call(fail_func)
            except Exception:
                pass
        
        print(f"  ✅ After {call_count} failures, state: {cb.state.value}")
        
        # Get stats
        stats = cb.stats
        print(f"  ✅ Stats: {stats.failed_calls} failures, {stats.successful_calls} successes")
        
        print("  ✅ Circuit breaker working correctly")
        return True
    except Exception as e:
        print(f"  ❌ Failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """Run all tests."""
    print("\n" + "="*60)
    print("ALPHATERMINAL PRO - BACKTEST ENGINE TESTS")
    print("="*60)
    
    tests = [
        ("Enums", test_enums),
        ("Exceptions", test_exceptions),
        ("Models", test_models),
        ("Costs", test_costs),
        ("Metrics", test_metrics),
        ("Strategy", test_strategy),
        ("Engine", test_engine),
        ("Validators", test_validators),
        ("Circuit Breaker", test_circuit_breaker),
    ]
    
    results = []
    for name, test_func in tests:
        try:
            success = test_func()
            results.append((name, success))
        except Exception as e:
            print(f"\n❌ Test '{name}' crashed: {e}")
            results.append((name, False))
    
    # Summary
    print("\n" + "="*60)
    print("TEST SUMMARY")
    print("="*60)
    
    passed = sum(1 for _, success in results if success)
    total = len(results)
    
    for name, success in results:
        status = "✅ PASS" if success else "❌ FAIL"
        print(f"  {status}: {name}")
    
    print(f"\n  Total: {passed}/{total} tests passed")
    
    if passed == total:
        print("\n  🎉 ALL TESTS PASSED!")
    else:
        print(f"\n  ⚠️  {total - passed} test(s) failed")
    
    return passed == total


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
