"""
AlphaTerminal Pro - Position Model
==================================

Comprehensive position model for tracking open positions.

Features:
- Real-time P&L tracking
- Stop loss / Take profit management
- Trailing stop support
- Position scaling (add/reduce)
- Maximum adverse/favorable excursion tracking
- Cost basis calculation

Author: AlphaTerminal Team
Version: 1.0.0
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple
from uuid import uuid4

from app.backtest.enums import PositionSide, OrderSide, ExitReason


def generate_position_id() -> str:
    """Generate a unique position ID."""
    return f"POS-{uuid4().hex[:12].upper()}"


@dataclass
class PositionEntry:
    """
    Represents a single entry into a position.
    
    Used for tracking cost basis with multiple entries (scaling in).
    """
    entry_id: str
    quantity: int
    price: float
    commission: float
    timestamp: datetime
    order_id: Optional[str] = None
    
    @property
    def cost_basis(self) -> float:
        """Total cost including commission."""
        return (self.quantity * self.price) + self.commission
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "entry_id": self.entry_id,
            "quantity": self.quantity,
            "price": self.price,
            "commission": self.commission,
            "cost_basis": self.cost_basis,
            "timestamp": self.timestamp.isoformat(),
            "order_id": self.order_id
        }


@dataclass
class Position:
    """
    Comprehensive position model for backtesting.
    
    Tracks an open position with full P&L, cost basis, and exit level management.
    
    Attributes:
        position_id: Unique identifier
        symbol: Trading symbol
        side: Long or Short
        quantity: Current position size (positive)
        avg_entry_price: Weighted average entry price
        
    Example:
        ```python
        # Create long position
        pos = Position(
            symbol="THYAO",
            side=PositionSide.LONG,
            quantity=100,
            avg_entry_price=285.50
        )
        
        # Update with current price
        pos.update_price(290.00)
        print(f"Unrealized P&L: {pos.unrealized_pnl:.2f}")
        
        # Check stops
        if pos.should_stop_out(low=280.00):
            print("Stop loss triggered!")
        ```
    """
    
    # Core fields
    symbol: str
    side: PositionSide
    quantity: int
    avg_entry_price: float
    
    # Auto-generated
    position_id: str = field(default_factory=generate_position_id)
    
    # Entry tracking
    entries: List[PositionEntry] = field(default_factory=list)
    total_entry_commission: float = 0.0
    
    # Current state
    current_price: float = 0.0
    current_timestamp: Optional[datetime] = None
    
    # P&L tracking
    unrealized_pnl: float = 0.0
    unrealized_pnl_pct: float = 0.0
    realized_pnl: float = 0.0  # From partial exits
    
    # Excursion tracking (MFE/MAE)
    max_favorable_excursion: float = 0.0  # Best unrealized P&L
    max_adverse_excursion: float = 0.0    # Worst unrealized P&L
    highest_price: float = 0.0            # Highest price seen
    lowest_price: float = 0.0             # Lowest price seen
    
    # Exit levels
    stop_loss: Optional[float] = None
    take_profit: Optional[float] = None
    trailing_stop_distance: Optional[float] = None
    trailing_stop_price: Optional[float] = None
    time_stop_bars: Optional[int] = None
    
    # Timestamps
    opened_at: datetime = field(default_factory=datetime.utcnow)
    last_updated_at: Optional[datetime] = None
    
    # Tracking
    bars_held: int = 0
    updates_count: int = 0
    
    # Strategy reference
    strategy_id: Optional[str] = None
    signal_id: Optional[str] = None
    
    # Tags
    tags: Dict[str, Any] = field(default_factory=dict)
    
    def __post_init__(self):
        """Initialize derived values."""
        if self.quantity <= 0:
            raise ValueError(f"Position quantity must be positive, got {self.quantity}")
        
        # Initialize price tracking
        if self.current_price == 0.0:
            self.current_price = self.avg_entry_price
        
        self.highest_price = self.current_price
        self.lowest_price = self.current_price
        
        # Create initial entry if none provided
        if not self.entries:
            self.entries.append(PositionEntry(
                entry_id=f"{self.position_id}-E1",
                quantity=self.quantity,
                price=self.avg_entry_price,
                commission=self.total_entry_commission,
                timestamp=self.opened_at
            ))
    
    # =========================================================================
    # PROPERTIES
    # =========================================================================
    
    @property
    def market_value(self) -> float:
        """Current market value of position."""
        return self.quantity * self.current_price
    
    @property
    def cost_basis(self) -> float:
        """Total cost basis including commissions."""
        return (self.quantity * self.avg_entry_price) + self.total_entry_commission
    
    @property
    def is_long(self) -> bool:
        """Check if position is long."""
        return self.side == PositionSide.LONG
    
    @property
    def is_short(self) -> bool:
        """Check if position is short."""
        return self.side == PositionSide.SHORT
    
    @property
    def direction_sign(self) -> int:
        """Get direction sign (+1 for long, -1 for short)."""
        return 1 if self.is_long else -1
    
    @property
    def total_pnl(self) -> float:
        """Total P&L (realized + unrealized)."""
        return self.realized_pnl + self.unrealized_pnl
    
    @property
    def total_pnl_pct(self) -> float:
        """Total P&L as percentage of cost basis."""
        if self.cost_basis == 0:
            return 0.0
        return self.total_pnl / self.cost_basis
    
    @property
    def risk_reward_current(self) -> Optional[float]:
        """Current risk/reward if stop loss is set."""
        if self.stop_loss is None or self.take_profit is None:
            return None
        
        risk = abs(self.avg_entry_price - self.stop_loss)
        reward = abs(self.take_profit - self.avg_entry_price)
        
        return reward / risk if risk > 0 else None
    
    @property
    def effective_stop(self) -> Optional[float]:
        """Get the effective stop price (trailing or fixed)."""
        if self.trailing_stop_price is not None:
            if self.stop_loss is not None:
                # Use the more protective stop
                if self.is_long:
                    return max(self.trailing_stop_price, self.stop_loss)
                return min(self.trailing_stop_price, self.stop_loss)
            return self.trailing_stop_price
        return self.stop_loss
    
    # =========================================================================
    # PRICE UPDATES
    # =========================================================================
    
    def update_price(
        self,
        price: float,
        high: Optional[float] = None,
        low: Optional[float] = None,
        timestamp: Optional[datetime] = None
    ) -> None:
        """
        Update position with new price data.
        
        Args:
            price: Current/close price
            high: Bar high (for stop checking)
            low: Bar low (for stop checking)
            timestamp: Current timestamp
        """
        self.current_price = price
        self.current_timestamp = timestamp
        self.last_updated_at = timestamp or datetime.utcnow()
        self.updates_count += 1
        
        # Update price extremes
        bar_high = high or price
        bar_low = low or price
        
        self.highest_price = max(self.highest_price, bar_high)
        self.lowest_price = min(self.lowest_price, bar_low)
        
        # Calculate unrealized P&L
        self._update_pnl()
        
        # Update trailing stop
        self._update_trailing_stop()
        
        # Update excursions
        self._update_excursions()
    
    def _update_pnl(self) -> None:
        """Recalculate unrealized P&L."""
        price_change = self.current_price - self.avg_entry_price
        
        if self.is_long:
            self.unrealized_pnl = (price_change * self.quantity) - self.total_entry_commission
        else:
            self.unrealized_pnl = (-price_change * self.quantity) - self.total_entry_commission
        
        if self.cost_basis > 0:
            self.unrealized_pnl_pct = self.unrealized_pnl / self.cost_basis
    
    def _update_trailing_stop(self) -> None:
        """Update trailing stop price if enabled."""
        if self.trailing_stop_distance is None:
            return
        
        if self.is_long:
            # Trail below the highest price
            new_stop = self.highest_price - self.trailing_stop_distance
            if self.trailing_stop_price is None or new_stop > self.trailing_stop_price:
                self.trailing_stop_price = new_stop
        else:
            # Trail above the lowest price
            new_stop = self.lowest_price + self.trailing_stop_distance
            if self.trailing_stop_price is None or new_stop < self.trailing_stop_price:
                self.trailing_stop_price = new_stop
    
    def _update_excursions(self) -> None:
        """Update MFE and MAE."""
        self.max_favorable_excursion = max(self.max_favorable_excursion, self.unrealized_pnl)
        self.max_adverse_excursion = min(self.max_adverse_excursion, self.unrealized_pnl)
    
    def increment_bars(self, count: int = 1) -> None:
        """Increment bars held counter."""
        self.bars_held += count
    
    # =========================================================================
    # STOP/TARGET MANAGEMENT
    # =========================================================================
    
    def set_stop_loss(self, price: float) -> None:
        """Set stop loss price."""
        if self.is_long and price >= self.current_price:
            raise ValueError(f"Long stop loss ({price}) must be below current price ({self.current_price})")
        if self.is_short and price <= self.current_price:
            raise ValueError(f"Short stop loss ({price}) must be above current price ({self.current_price})")
        
        self.stop_loss = price
    
    def set_take_profit(self, price: float) -> None:
        """Set take profit price."""
        if self.is_long and price <= self.current_price:
            raise ValueError(f"Long take profit ({price}) must be above current price ({self.current_price})")
        if self.is_short and price >= self.current_price:
            raise ValueError(f"Short take profit ({price}) must be below current price ({self.current_price})")
        
        self.take_profit = price
    
    def set_trailing_stop(self, distance: float) -> None:
        """
        Enable trailing stop.
        
        Args:
            distance: Distance from price to trail (in price units)
        """
        if distance <= 0:
            raise ValueError(f"Trailing stop distance must be positive, got {distance}")
        
        self.trailing_stop_distance = distance
        self._update_trailing_stop()
    
    def set_time_stop(self, bars: int) -> None:
        """
        Set time-based exit.
        
        Args:
            bars: Number of bars before forced exit
        """
        if bars <= 0:
            raise ValueError(f"Time stop bars must be positive, got {bars}")
        
        self.time_stop_bars = bars
    
    # =========================================================================
    # STOP CHECKING
    # =========================================================================
    
    def should_stop_out(
        self,
        high: Optional[float] = None,
        low: Optional[float] = None
    ) -> Tuple[bool, Optional[ExitReason], Optional[float]]:
        """
        Check if any stop condition is triggered.
        
        Args:
            high: Bar high price
            low: Bar low price
            
        Returns:
            Tuple of (triggered, reason, exit_price)
        """
        bar_high = high or self.current_price
        bar_low = low or self.current_price
        
        effective_stop = self.effective_stop
        
        if self.is_long:
            # Check stop loss
            if effective_stop is not None and bar_low <= effective_stop:
                reason = (ExitReason.TRAILING_STOP 
                         if self.trailing_stop_price and effective_stop == self.trailing_stop_price 
                         else ExitReason.STOP_LOSS)
                return True, reason, effective_stop
            
            # Check take profit
            if self.take_profit is not None and bar_high >= self.take_profit:
                return True, ExitReason.TAKE_PROFIT, self.take_profit
        else:
            # Short position
            if effective_stop is not None and bar_high >= effective_stop:
                reason = (ExitReason.TRAILING_STOP 
                         if self.trailing_stop_price and effective_stop == self.trailing_stop_price 
                         else ExitReason.STOP_LOSS)
                return True, reason, effective_stop
            
            if self.take_profit is not None and bar_low <= self.take_profit:
                return True, ExitReason.TAKE_PROFIT, self.take_profit
        
        # Check time stop
        if self.time_stop_bars is not None and self.bars_held >= self.time_stop_bars:
            return True, ExitReason.TIME_STOP, self.current_price
        
        return False, None, None
    
    # =========================================================================
    # POSITION SCALING
    # =========================================================================
    
    def add_to_position(
        self,
        quantity: int,
        price: float,
        commission: float = 0.0,
        timestamp: Optional[datetime] = None,
        order_id: Optional[str] = None
    ) -> None:
        """
        Add to existing position (scale in).
        
        Args:
            quantity: Shares to add
            price: Entry price
            commission: Commission for this entry
            timestamp: Entry timestamp
            order_id: Associated order ID
        """
        if quantity <= 0:
            raise ValueError(f"Add quantity must be positive, got {quantity}")
        
        entry_timestamp = timestamp or datetime.utcnow()
        entry_id = f"{self.position_id}-E{len(self.entries) + 1}"
        
        entry = PositionEntry(
            entry_id=entry_id,
            quantity=quantity,
            price=price,
            commission=commission,
            timestamp=entry_timestamp,
            order_id=order_id
        )
        self.entries.append(entry)
        
        # Recalculate average entry price
        old_value = self.quantity * self.avg_entry_price
        new_value = quantity * price
        self.quantity += quantity
        self.avg_entry_price = (old_value + new_value) / self.quantity
        
        # Update commission tracking
        self.total_entry_commission += commission
        
        # Update P&L
        self._update_pnl()
        
        self.last_updated_at = entry_timestamp
    
    def reduce_position(
        self,
        quantity: int,
        price: float,
        commission: float = 0.0
    ) -> float:
        """
        Reduce position size (scale out / partial exit).
        
        Args:
            quantity: Shares to reduce
            price: Exit price
            commission: Exit commission
            
        Returns:
            Realized P&L for the reduced portion
        """
        if quantity <= 0:
            raise ValueError(f"Reduce quantity must be positive, got {quantity}")
        
        if quantity > self.quantity:
            raise ValueError(f"Cannot reduce by {quantity}, only {self.quantity} shares in position")
        
        # Calculate realized P&L for reduced portion
        price_change = price - self.avg_entry_price
        
        if self.is_long:
            realized = (price_change * quantity) - commission
        else:
            realized = (-price_change * quantity) - commission
        
        # Proportional entry commission allocation
        commission_portion = self.total_entry_commission * (quantity / self.quantity)
        realized -= commission_portion
        
        # Update position
        self.quantity -= quantity
        self.realized_pnl += realized
        self.total_entry_commission -= commission_portion
        
        # Update P&L
        if self.quantity > 0:
            self._update_pnl()
        else:
            self.unrealized_pnl = 0.0
            self.unrealized_pnl_pct = 0.0
        
        return realized
    
    # =========================================================================
    # SERIALIZATION
    # =========================================================================
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert position to dictionary."""
        return {
            "position_id": self.position_id,
            "symbol": self.symbol,
            "side": self.side.value,
            "quantity": self.quantity,
            "avg_entry_price": round(self.avg_entry_price, 4),
            "current_price": round(self.current_price, 4),
            "market_value": round(self.market_value, 2),
            "cost_basis": round(self.cost_basis, 2),
            "unrealized_pnl": round(self.unrealized_pnl, 2),
            "unrealized_pnl_pct": round(self.unrealized_pnl_pct, 4),
            "realized_pnl": round(self.realized_pnl, 2),
            "total_pnl": round(self.total_pnl, 2),
            "max_favorable_excursion": round(self.max_favorable_excursion, 2),
            "max_adverse_excursion": round(self.max_adverse_excursion, 2),
            "highest_price": round(self.highest_price, 4),
            "lowest_price": round(self.lowest_price, 4),
            "stop_loss": self.stop_loss,
            "take_profit": self.take_profit,
            "trailing_stop_price": self.trailing_stop_price,
            "effective_stop": self.effective_stop,
            "bars_held": self.bars_held,
            "entries": [e.to_dict() for e in self.entries],
            "opened_at": self.opened_at.isoformat(),
            "last_updated_at": self.last_updated_at.isoformat() if self.last_updated_at else None,
            "strategy_id": self.strategy_id,
            "tags": self.tags
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Position":
        """Create position from dictionary."""
        pos = cls(
            symbol=data["symbol"],
            side=PositionSide(data["side"]),
            quantity=data["quantity"],
            avg_entry_price=data["avg_entry_price"],
            position_id=data.get("position_id", generate_position_id()),
            current_price=data.get("current_price", data["avg_entry_price"]),
            stop_loss=data.get("stop_loss"),
            take_profit=data.get("take_profit"),
            strategy_id=data.get("strategy_id"),
            tags=data.get("tags", {})
        )
        
        # Restore entries
        pos.entries = []
        for entry_data in data.get("entries", []):
            pos.entries.append(PositionEntry(
                entry_id=entry_data["entry_id"],
                quantity=entry_data["quantity"],
                price=entry_data["price"],
                commission=entry_data["commission"],
                timestamp=datetime.fromisoformat(entry_data["timestamp"]),
                order_id=entry_data.get("order_id")
            ))
        
        # Restore tracking
        pos.unrealized_pnl = data.get("unrealized_pnl", 0.0)
        pos.realized_pnl = data.get("realized_pnl", 0.0)
        pos.bars_held = data.get("bars_held", 0)
        
        return pos
    
    # =========================================================================
    # STRING REPRESENTATION
    # =========================================================================
    
    def __str__(self) -> str:
        """Human-readable string representation."""
        pnl_str = f"+{self.unrealized_pnl:.2f}" if self.unrealized_pnl >= 0 else f"{self.unrealized_pnl:.2f}"
        pnl_pct = f"+{self.unrealized_pnl_pct:.2%}" if self.unrealized_pnl_pct >= 0 else f"{self.unrealized_pnl_pct:.2%}"
        
        stop_str = ""
        if self.effective_stop:
            stop_str = f", SL: {self.effective_stop:.2f}"
        if self.take_profit:
            stop_str += f", TP: {self.take_profit:.2f}"
        
        return (
            f"Position({self.position_id}): {self.side.value.upper()} {self.quantity} {self.symbol} "
            f"@ {self.avg_entry_price:.2f} -> {self.current_price:.2f} "
            f"[P&L: {pnl_str} ({pnl_pct}){stop_str}]"
        )
    
    def __repr__(self) -> str:
        """Detailed string representation."""
        return (
            f"Position(id='{self.position_id}', symbol='{self.symbol}', "
            f"side={self.side}, qty={self.quantity}, "
            f"entry={self.avg_entry_price:.2f}, current={self.current_price:.2f})"
        )


# =============================================================================
# FACTORY FUNCTIONS
# =============================================================================

def create_long_position(
    symbol: str,
    quantity: int,
    entry_price: float,
    commission: float = 0.0,
    stop_loss: Optional[float] = None,
    take_profit: Optional[float] = None,
    **kwargs
) -> Position:
    """
    Create a long position.
    
    Args:
        symbol: Trading symbol
        quantity: Number of shares
        entry_price: Entry price
        commission: Entry commission
        stop_loss: Stop loss price
        take_profit: Take profit price
        **kwargs: Additional position parameters
        
    Returns:
        Long position
    """
    pos = Position(
        symbol=symbol,
        side=PositionSide.LONG,
        quantity=quantity,
        avg_entry_price=entry_price,
        total_entry_commission=commission,
        stop_loss=stop_loss,
        take_profit=take_profit,
        **kwargs
    )
    pos.update_price(entry_price)
    return pos


def create_short_position(
    symbol: str,
    quantity: int,
    entry_price: float,
    commission: float = 0.0,
    stop_loss: Optional[float] = None,
    take_profit: Optional[float] = None,
    **kwargs
) -> Position:
    """
    Create a short position.
    
    Args:
        symbol: Trading symbol
        quantity: Number of shares
        entry_price: Entry price
        commission: Entry commission
        stop_loss: Stop loss price
        take_profit: Take profit price
        **kwargs: Additional position parameters
        
    Returns:
        Short position
    """
    pos = Position(
        symbol=symbol,
        side=PositionSide.SHORT,
        quantity=quantity,
        avg_entry_price=entry_price,
        total_entry_commission=commission,
        stop_loss=stop_loss,
        take_profit=take_profit,
        **kwargs
    )
    pos.update_price(entry_price)
    return pos
