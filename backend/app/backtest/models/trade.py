"""
AlphaTerminal Pro - Trade Model
===============================

Comprehensive model for completed trades.

Features:
- Full trade lifecycle tracking
- Detailed P&L breakdown
- Performance metrics per trade
- MFE/MAE analysis
- Trade classification

Author: AlphaTerminal Team
Version: 1.0.0
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional
from uuid import uuid4

from app.backtest.enums import (
    TradeDirection, ExitReason, PositionSide, SignalStrength
)


def generate_trade_id() -> str:
    """Generate a unique trade ID."""
    return f"TRD-{uuid4().hex[:12].upper()}"


@dataclass
class Trade:
    """
    Comprehensive completed trade model.
    
    Represents a fully closed trade with all metrics calculated.
    
    Attributes:
        trade_id: Unique identifier
        symbol: Trading symbol
        direction: Long or Short
        quantity: Number of shares traded
        entry_price: Average entry price
        exit_price: Average exit price
        
    Example:
        ```python
        trade = Trade(
            symbol="THYAO",
            direction=TradeDirection.LONG,
            quantity=100,
            entry_price=285.50,
            exit_price=295.00,
            entry_time=datetime(2024, 1, 1, 10, 0),
            exit_time=datetime(2024, 1, 5, 15, 30),
            exit_reason=ExitReason.TAKE_PROFIT
        )
        
        print(f"P&L: {trade.pnl:.2f} ({trade.pnl_pct:.2%})")
        print(f"R-Multiple: {trade.r_multiple:.2f}")
        ```
    """
    
    # Core trade info
    symbol: str
    direction: TradeDirection
    quantity: int
    
    # Prices
    entry_price: float
    exit_price: float
    
    # Times
    entry_time: datetime
    exit_time: datetime
    
    # Exit info
    exit_reason: ExitReason
    
    # Auto-generated
    trade_id: str = field(default_factory=generate_trade_id)
    
    # Costs
    entry_commission: float = 0.0
    exit_commission: float = 0.0
    entry_slippage: float = 0.0
    exit_slippage: float = 0.0
    
    # Stop/Target info
    initial_stop_loss: Optional[float] = None
    initial_take_profit: Optional[float] = None
    actual_stop_price: Optional[float] = None  # Where stop was actually hit
    
    # Price excursions
    highest_price: float = 0.0      # Highest price during trade
    lowest_price: float = 0.0       # Lowest price during trade
    max_favorable_excursion: float = 0.0   # Best unrealized P&L
    max_adverse_excursion: float = 0.0     # Worst unrealized P&L
    
    # Signal info
    signal_strength: Optional[SignalStrength] = None
    signal_score: float = 0.0
    
    # Strategy reference
    strategy_id: Optional[str] = None
    strategy_name: Optional[str] = None
    position_id: Optional[str] = None
    
    # Additional entries/exits (for scaled trades)
    entry_count: int = 1
    exit_count: int = 1
    
    # Holding period
    bars_held: int = 0
    
    # Tags
    tags: Dict[str, Any] = field(default_factory=dict)
    notes: str = ""
    
    # Calculated in __post_init__
    gross_pnl: float = field(init=False)
    net_pnl: float = field(init=False)
    pnl_pct: float = field(init=False)
    total_commission: float = field(init=False)
    total_slippage: float = field(init=False)
    total_costs: float = field(init=False)
    holding_period: timedelta = field(init=False)
    
    def __post_init__(self):
        """Calculate derived fields."""
        self._calculate_pnl()
        self._calculate_costs()
        self._calculate_holding_period()
        self._set_price_defaults()
    
    def _calculate_pnl(self) -> None:
        """Calculate P&L values."""
        price_diff = self.exit_price - self.entry_price
        
        if self.direction == TradeDirection.LONG:
            self.gross_pnl = price_diff * self.quantity
        else:
            self.gross_pnl = -price_diff * self.quantity
    
    def _calculate_costs(self) -> None:
        """Calculate cost values."""
        self.total_commission = self.entry_commission + self.exit_commission
        self.total_slippage = self.entry_slippage + self.exit_slippage
        self.total_costs = self.total_commission + self.total_slippage
        
        self.net_pnl = self.gross_pnl - self.total_costs
        
        # P&L percentage based on entry value
        entry_value = self.entry_price * self.quantity
        self.pnl_pct = self.net_pnl / entry_value if entry_value > 0 else 0.0
    
    def _calculate_holding_period(self) -> None:
        """Calculate holding period."""
        self.holding_period = self.exit_time - self.entry_time
    
    def _set_price_defaults(self) -> None:
        """Set default values for price tracking."""
        if self.highest_price == 0.0:
            self.highest_price = max(self.entry_price, self.exit_price)
        if self.lowest_price == 0.0:
            self.lowest_price = min(self.entry_price, self.exit_price)
    
    # =========================================================================
    # PROPERTIES
    # =========================================================================
    
    @property
    def is_winner(self) -> bool:
        """Check if trade was profitable."""
        return self.net_pnl > 0
    
    @property
    def is_loser(self) -> bool:
        """Check if trade was unprofitable."""
        return self.net_pnl < 0
    
    @property
    def is_breakeven(self) -> bool:
        """Check if trade was breakeven."""
        return abs(self.net_pnl) < 0.01  # Within 1 cent
    
    @property
    def entry_value(self) -> float:
        """Total entry value."""
        return self.entry_price * self.quantity
    
    @property
    def exit_value(self) -> float:
        """Total exit value."""
        return self.exit_price * self.quantity
    
    @property
    def r_multiple(self) -> Optional[float]:
        """
        R-Multiple (return in units of initial risk).
        
        R = 1.0 means you made what you risked.
        R = 2.0 means you made twice what you risked.
        R = -1.0 means you lost your full risk amount.
        """
        if self.initial_stop_loss is None:
            return None
        
        initial_risk = abs(self.entry_price - self.initial_stop_loss) * self.quantity
        
        if initial_risk == 0:
            return None
        
        return self.net_pnl / initial_risk
    
    @property
    def risk_amount(self) -> Optional[float]:
        """Initial risk amount in currency."""
        if self.initial_stop_loss is None:
            return None
        
        return abs(self.entry_price - self.initial_stop_loss) * self.quantity
    
    @property
    def reward_risk_ratio(self) -> Optional[float]:
        """Actual reward/risk ratio achieved."""
        if self.initial_stop_loss is None:
            return None
        
        risk = abs(self.entry_price - self.initial_stop_loss)
        reward = abs(self.exit_price - self.entry_price)
        
        return reward / risk if risk > 0 else None
    
    @property
    def planned_reward_risk(self) -> Optional[float]:
        """Planned reward/risk ratio from initial setup."""
        if self.initial_stop_loss is None or self.initial_take_profit is None:
            return None
        
        risk = abs(self.entry_price - self.initial_stop_loss)
        reward = abs(self.initial_take_profit - self.entry_price)
        
        return reward / risk if risk > 0 else None
    
    @property
    def mfe_pct(self) -> float:
        """Maximum Favorable Excursion as percentage."""
        entry_value = self.entry_price * self.quantity
        return self.max_favorable_excursion / entry_value if entry_value > 0 else 0.0
    
    @property
    def mae_pct(self) -> float:
        """Maximum Adverse Excursion as percentage."""
        entry_value = self.entry_price * self.quantity
        return self.max_adverse_excursion / entry_value if entry_value > 0 else 0.0
    
    @property
    def efficiency(self) -> Optional[float]:
        """
        Trade efficiency: how much of MFE was captured.
        
        100% = exited at the best possible price
        50% = captured half of the maximum potential
        """
        if self.max_favorable_excursion <= 0:
            return None
        
        return self.net_pnl / self.max_favorable_excursion
    
    @property
    def holding_days(self) -> float:
        """Holding period in days."""
        return self.holding_period.total_seconds() / 86400
    
    @property
    def holding_hours(self) -> float:
        """Holding period in hours."""
        return self.holding_period.total_seconds() / 3600
    
    @property
    def is_intraday(self) -> bool:
        """Check if trade was opened and closed same day."""
        return self.entry_time.date() == self.exit_time.date()
    
    @property
    def exit_type(self) -> str:
        """Categorize exit type."""
        if self.exit_reason in [ExitReason.STOP_LOSS, ExitReason.TRAILING_STOP]:
            return "STOPPED"
        elif self.exit_reason == ExitReason.TAKE_PROFIT:
            return "TARGET"
        elif self.exit_reason in [ExitReason.SIGNAL, ExitReason.SIGNAL_REVERSAL]:
            return "SIGNAL"
        elif self.exit_reason == ExitReason.TIME_STOP:
            return "TIME"
        else:
            return "OTHER"
    
    # =========================================================================
    # CLASSIFICATION
    # =========================================================================
    
    def classify_outcome(self) -> str:
        """
        Classify the trade outcome.
        
        Returns:
            Classification string
        """
        if self.is_breakeven:
            return "BREAKEVEN"
        
        r = self.r_multiple
        
        if r is not None:
            if r >= 3.0:
                return "BIG_WINNER"
            elif r >= 2.0:
                return "GOOD_WINNER"
            elif r >= 1.0:
                return "WINNER"
            elif r >= 0:
                return "SMALL_WINNER"
            elif r >= -0.5:
                return "SMALL_LOSER"
            elif r >= -1.0:
                return "LOSER"
            else:
                return "BIG_LOSER"
        else:
            if self.pnl_pct >= 0.10:
                return "BIG_WINNER"
            elif self.pnl_pct >= 0.05:
                return "GOOD_WINNER"
            elif self.pnl_pct >= 0:
                return "WINNER"
            elif self.pnl_pct >= -0.05:
                return "LOSER"
            else:
                return "BIG_LOSER"
    
    def classify_duration(self) -> str:
        """
        Classify trade duration.
        
        Returns:
            Duration classification
        """
        hours = self.holding_hours
        
        if hours < 1:
            return "SCALP"
        elif hours < 24:
            return "INTRADAY"
        elif hours < 24 * 5:
            return "SWING"
        elif hours < 24 * 20:
            return "POSITION"
        else:
            return "LONG_TERM"
    
    # =========================================================================
    # SERIALIZATION
    # =========================================================================
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert trade to dictionary."""
        return {
            "trade_id": self.trade_id,
            "symbol": self.symbol,
            "direction": self.direction.value,
            "quantity": self.quantity,
            
            # Prices
            "entry_price": round(self.entry_price, 4),
            "exit_price": round(self.exit_price, 4),
            
            # Times
            "entry_time": self.entry_time.isoformat(),
            "exit_time": self.exit_time.isoformat(),
            "holding_period_hours": round(self.holding_hours, 2),
            "holding_period_days": round(self.holding_days, 2),
            "bars_held": self.bars_held,
            
            # Exit
            "exit_reason": self.exit_reason.value,
            "exit_type": self.exit_type,
            
            # P&L
            "gross_pnl": round(self.gross_pnl, 2),
            "net_pnl": round(self.net_pnl, 2),
            "pnl_pct": round(self.pnl_pct, 4),
            "is_winner": self.is_winner,
            
            # Costs
            "entry_commission": round(self.entry_commission, 4),
            "exit_commission": round(self.exit_commission, 4),
            "total_commission": round(self.total_commission, 4),
            "entry_slippage": round(self.entry_slippage, 4),
            "exit_slippage": round(self.exit_slippage, 4),
            "total_slippage": round(self.total_slippage, 4),
            "total_costs": round(self.total_costs, 4),
            
            # Risk metrics
            "initial_stop_loss": self.initial_stop_loss,
            "initial_take_profit": self.initial_take_profit,
            "risk_amount": round(self.risk_amount, 2) if self.risk_amount else None,
            "r_multiple": round(self.r_multiple, 2) if self.r_multiple else None,
            "reward_risk_ratio": round(self.reward_risk_ratio, 2) if self.reward_risk_ratio else None,
            
            # Excursions
            "highest_price": round(self.highest_price, 4),
            "lowest_price": round(self.lowest_price, 4),
            "max_favorable_excursion": round(self.max_favorable_excursion, 2),
            "max_adverse_excursion": round(self.max_adverse_excursion, 2),
            "mfe_pct": round(self.mfe_pct, 4),
            "mae_pct": round(self.mae_pct, 4),
            "efficiency": round(self.efficiency, 2) if self.efficiency else None,
            
            # Classification
            "outcome": self.classify_outcome(),
            "duration_class": self.classify_duration(),
            
            # References
            "strategy_id": self.strategy_id,
            "strategy_name": self.strategy_name,
            "position_id": self.position_id,
            "signal_strength": self.signal_strength.value if self.signal_strength else None,
            "signal_score": round(self.signal_score, 2),
            
            # Scaling
            "entry_count": self.entry_count,
            "exit_count": self.exit_count,
            
            # Extra
            "tags": self.tags,
            "notes": self.notes
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Trade":
        """Create trade from dictionary."""
        return cls(
            trade_id=data.get("trade_id", generate_trade_id()),
            symbol=data["symbol"],
            direction=TradeDirection(data["direction"]),
            quantity=data["quantity"],
            entry_price=data["entry_price"],
            exit_price=data["exit_price"],
            entry_time=datetime.fromisoformat(data["entry_time"]),
            exit_time=datetime.fromisoformat(data["exit_time"]),
            exit_reason=ExitReason(data["exit_reason"]),
            entry_commission=data.get("entry_commission", 0.0),
            exit_commission=data.get("exit_commission", 0.0),
            entry_slippage=data.get("entry_slippage", 0.0),
            exit_slippage=data.get("exit_slippage", 0.0),
            initial_stop_loss=data.get("initial_stop_loss"),
            initial_take_profit=data.get("initial_take_profit"),
            highest_price=data.get("highest_price", 0.0),
            lowest_price=data.get("lowest_price", 0.0),
            max_favorable_excursion=data.get("max_favorable_excursion", 0.0),
            max_adverse_excursion=data.get("max_adverse_excursion", 0.0),
            bars_held=data.get("bars_held", 0),
            strategy_id=data.get("strategy_id"),
            strategy_name=data.get("strategy_name"),
            position_id=data.get("position_id"),
            tags=data.get("tags", {}),
            notes=data.get("notes", "")
        )
    
    def to_summary_dict(self) -> Dict[str, Any]:
        """Convert to compact summary dictionary."""
        return {
            "trade_id": self.trade_id,
            "symbol": self.symbol,
            "direction": self.direction.value,
            "entry": f"{self.entry_price:.2f}",
            "exit": f"{self.exit_price:.2f}",
            "pnl": f"{self.net_pnl:+.2f}",
            "pnl_pct": f"{self.pnl_pct:+.2%}",
            "r": f"{self.r_multiple:+.2f}R" if self.r_multiple else "N/A",
            "duration": f"{self.holding_hours:.1f}h",
            "exit_reason": self.exit_reason.value,
            "outcome": self.classify_outcome()
        }
    
    # =========================================================================
    # STRING REPRESENTATION
    # =========================================================================
    
    def __str__(self) -> str:
        """Human-readable string representation."""
        pnl_sign = "+" if self.net_pnl >= 0 else ""
        r_str = f", {self.r_multiple:+.2f}R" if self.r_multiple else ""
        
        return (
            f"Trade({self.trade_id}): {self.direction.value.upper()} {self.quantity} {self.symbol} "
            f"@ {self.entry_price:.2f} -> {self.exit_price:.2f} "
            f"[{pnl_sign}{self.net_pnl:.2f} ({self.pnl_pct:+.2%}){r_str}] "
            f"- {self.exit_reason.value} after {self.holding_hours:.1f}h"
        )
    
    def __repr__(self) -> str:
        """Detailed string representation."""
        return (
            f"Trade(id='{self.trade_id}', symbol='{self.symbol}', "
            f"dir={self.direction}, qty={self.quantity}, "
            f"pnl={self.net_pnl:.2f})"
        )


# =============================================================================
# TRADE LIST UTILITIES
# =============================================================================

@dataclass
class TradeList:
    """
    Collection of trades with aggregate statistics.
    """
    trades: List[Trade] = field(default_factory=list)
    
    def __len__(self) -> int:
        return len(self.trades)
    
    def __iter__(self):
        return iter(self.trades)
    
    def __getitem__(self, index):
        return self.trades[index]
    
    def append(self, trade: Trade) -> None:
        """Add a trade to the list."""
        self.trades.append(trade)
    
    def extend(self, trades: List[Trade]) -> None:
        """Add multiple trades."""
        self.trades.extend(trades)
    
    @property
    def winners(self) -> List[Trade]:
        """Get winning trades."""
        return [t for t in self.trades if t.is_winner]
    
    @property
    def losers(self) -> List[Trade]:
        """Get losing trades."""
        return [t for t in self.trades if t.is_loser]
    
    @property
    def total_pnl(self) -> float:
        """Total P&L of all trades."""
        return sum(t.net_pnl for t in self.trades)
    
    @property
    def win_rate(self) -> float:
        """Win rate."""
        if not self.trades:
            return 0.0
        return len(self.winners) / len(self.trades)
    
    @property
    def avg_winner(self) -> float:
        """Average winning trade P&L."""
        winners = self.winners
        if not winners:
            return 0.0
        return sum(t.net_pnl for t in winners) / len(winners)
    
    @property
    def avg_loser(self) -> float:
        """Average losing trade P&L (negative value)."""
        losers = self.losers
        if not losers:
            return 0.0
        return sum(t.net_pnl for t in losers) / len(losers)
    
    @property
    def profit_factor(self) -> float:
        """Profit factor (gross profit / gross loss)."""
        gross_profit = sum(t.net_pnl for t in self.winners)
        gross_loss = abs(sum(t.net_pnl for t in self.losers))
        
        if gross_loss == 0:
            return float('inf') if gross_profit > 0 else 0.0
        return gross_profit / gross_loss
    
    @property
    def expectancy(self) -> float:
        """Expected value per trade."""
        if not self.trades:
            return 0.0
        return self.total_pnl / len(self.trades)
    
    def filter_by_symbol(self, symbol: str) -> "TradeList":
        """Get trades for specific symbol."""
        return TradeList([t for t in self.trades if t.symbol == symbol])
    
    def filter_by_direction(self, direction: TradeDirection) -> "TradeList":
        """Get trades for specific direction."""
        return TradeList([t for t in self.trades if t.direction == direction])
    
    def filter_by_date_range(
        self,
        start: datetime,
        end: datetime
    ) -> "TradeList":
        """Get trades within date range."""
        return TradeList([
            t for t in self.trades
            if start <= t.entry_time <= end
        ])
    
    def to_dict_list(self) -> List[Dict[str, Any]]:
        """Convert all trades to list of dictionaries."""
        return [t.to_dict() for t in self.trades]
    
    def summary(self) -> Dict[str, Any]:
        """Get summary statistics."""
        return {
            "total_trades": len(self.trades),
            "winners": len(self.winners),
            "losers": len(self.losers),
            "win_rate": round(self.win_rate, 4),
            "total_pnl": round(self.total_pnl, 2),
            "avg_winner": round(self.avg_winner, 2),
            "avg_loser": round(self.avg_loser, 2),
            "profit_factor": round(self.profit_factor, 2),
            "expectancy": round(self.expectancy, 2)
        }
