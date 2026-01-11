"""
AlphaTerminal Pro - Example Strategies
======================================

Ready-to-use strategy examples.
"""

from app.backtest.strategies.examples.sma_crossover import (
    SMACrossoverStrategy,
    DualSMACrossoverStrategy
)

from app.backtest.strategies.examples.rsi_reversal import (
    RSIMeanReversionStrategy,
    RSIExtremesStrategy
)

__all__ = [
    "SMACrossoverStrategy",
    "DualSMACrossoverStrategy",
    "RSIMeanReversionStrategy",
    "RSIExtremesStrategy"
]
