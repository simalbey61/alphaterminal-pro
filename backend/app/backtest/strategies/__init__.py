"""
AlphaTerminal Pro - Trading Strategies
======================================

Strategy framework and examples.

Usage:
    ```python
    from app.backtest.strategies import SMACrossoverStrategy, RSIMeanReversionStrategy
    
    strategy = SMACrossoverStrategy(fast_period=10, slow_period=30)
    result = engine.run(data, strategy, "THYAO")
    ```
"""

from app.backtest.engine import BaseStrategy, Signal

from app.backtest.strategies.examples import (
    SMACrossoverStrategy,
    DualSMACrossoverStrategy,
    RSIMeanReversionStrategy,
    RSIExtremesStrategy
)

__all__ = [
    # Base
    "BaseStrategy",
    "Signal",
    
    # SMA Strategies
    "SMACrossoverStrategy",
    "DualSMACrossoverStrategy",
    
    # RSI Strategies
    "RSIMeanReversionStrategy",
    "RSIExtremesStrategy"
]
