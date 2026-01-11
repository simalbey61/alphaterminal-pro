"""
AlphaTerminal Pro - Order Model
===============================

Comprehensive order model for backtesting.

Features:
- Full order lifecycle tracking
- Support for all order types
- Cost tracking (commission, slippage)
- Audit trail with timestamps
- Serialization support

Author: AlphaTerminal Team
Version: 1.0.0
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, List, Optional
from uuid import uuid4
from decimal import Decimal, ROUND_HALF_UP

from app.backtest.enums import (
    OrderType, OrderSide, OrderStatus, OrderTimeInForce
)


def generate_order_id() -> str:
    """Generate a unique order ID."""
    return f"ORD-{uuid4().hex[:12].upper()}"


@dataclass
class OrderFill:
    """
    Represents a single fill (execution) of an order.
    
    An order may have multiple fills (partial executions).
    """
    fill_id: str
    quantity: int
    price: float
    commission: float
    slippage: float
    timestamp: datetime
    
    # Calculated fields
    gross_value: float = field(init=False)
    net_value: float = field(init=False)
    
    def __post_init__(self):
        """Calculate derived fields after initialization."""
        self.gross_value = self.quantity * self.price
        self.net_value = self.gross_value + self.commission + self.slippage
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert fill to dictionary."""
        return {
            "fill_id": self.fill_id,
            "quantity": self.quantity,
            "price": self.price,
            "commission": self.commission,
            "slippage": self.slippage,
            "gross_value": self.gross_value,
            "net_value": self.net_value,
            "timestamp": self.timestamp.isoformat()
        }


@dataclass
class Order:
    """
    Comprehensive order model for backtesting.
    
    Supports:
    - All order types (market, limit, stop, stop-limit)
    - Partial fills
    - Cost tracking
    - Full audit trail
    
    Attributes:
        order_id: Unique identifier
        symbol: Trading symbol (e.g., "THYAO")
        side: Buy or Sell
        order_type: Market, Limit, Stop, Stop-Limit
        quantity: Number of shares to trade
        price: Limit price (for limit/stop-limit orders)
        stop_price: Stop trigger price (for stop/stop-limit orders)
        time_in_force: How long the order remains active
        
    Example:
        ```python
        # Market order
        order = Order(
            symbol="THYAO",
            side=OrderSide.BUY,
            order_type=OrderType.MARKET,
            quantity=100
        )
        
        # Limit order
        order = Order(
            symbol="GARAN",
            side=OrderSide.SELL,
            order_type=OrderType.LIMIT,
            quantity=200,
            price=15.50
        )
        
        # Stop-loss order
        order = Order(
            symbol="AKBNK",
            side=OrderSide.SELL,
            order_type=OrderType.STOP,
            quantity=150,
            stop_price=8.20
        )
        ```
    """
    
    # Required fields
    symbol: str
    side: OrderSide
    order_type: OrderType
    quantity: int
    
    # Optional price fields
    price: Optional[float] = None           # Limit price
    stop_price: Optional[float] = None      # Stop trigger price
    
    # Order settings
    time_in_force: OrderTimeInForce = OrderTimeInForce.DAY
    
    # Auto-generated fields
    order_id: str = field(default_factory=generate_order_id)
    status: OrderStatus = OrderStatus.PENDING
    
    # Fill tracking
    filled_quantity: int = 0
    fills: List[OrderFill] = field(default_factory=list)
    
    # Cost tracking
    total_commission: float = 0.0
    total_slippage: float = 0.0
    
    # Calculated average fill price
    avg_fill_price: float = 0.0
    
    # Timestamps
    created_at: datetime = field(default_factory=datetime.utcnow)
    submitted_at: Optional[datetime] = None
    last_updated_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    
    # Rejection/cancellation info
    reject_reason: Optional[str] = None
    cancel_reason: Optional[str] = None
    
    # Strategy reference
    strategy_id: Optional[str] = None
    signal_id: Optional[str] = None
    
    # Tags for filtering/analysis
    tags: Dict[str, Any] = field(default_factory=dict)
    
    def __post_init__(self):
        """Validate order after initialization."""
        self._validate()
    
    def _validate(self) -> None:
        """Validate order parameters."""
        if self.quantity <= 0:
            raise ValueError(f"Order quantity must be positive, got {self.quantity}")
        
        if self.order_type in [OrderType.LIMIT, OrderType.STOP_LIMIT]:
            if self.price is None or self.price <= 0:
                raise ValueError(f"Limit orders require a positive price")
        
        if self.order_type in [OrderType.STOP, OrderType.STOP_LIMIT]:
            if self.stop_price is None or self.stop_price <= 0:
                raise ValueError(f"Stop orders require a positive stop price")
    
    # =========================================================================
    # PROPERTIES
    # =========================================================================
    
    @property
    def remaining_quantity(self) -> int:
        """Get the unfilled quantity."""
        return self.quantity - self.filled_quantity
    
    @property
    def fill_ratio(self) -> float:
        """Get the fill ratio (0.0 to 1.0)."""
        return self.filled_quantity / self.quantity if self.quantity > 0 else 0.0
    
    @property
    def is_filled(self) -> bool:
        """Check if order is completely filled."""
        return self.filled_quantity >= self.quantity
    
    @property
    def is_partially_filled(self) -> bool:
        """Check if order is partially filled."""
        return 0 < self.filled_quantity < self.quantity
    
    @property
    def is_active(self) -> bool:
        """Check if order is still active."""
        return self.status.is_active
    
    @property
    def is_terminal(self) -> bool:
        """Check if order is in terminal state."""
        return self.status.is_terminal
    
    @property
    def gross_value(self) -> float:
        """Get the gross value of filled quantity."""
        return self.filled_quantity * self.avg_fill_price
    
    @property
    def total_cost(self) -> float:
        """Get the total cost including commissions and slippage."""
        if self.side == OrderSide.BUY:
            return self.gross_value + self.total_commission + self.total_slippage
        else:
            return self.gross_value - self.total_commission - self.total_slippage
    
    @property
    def effective_price(self) -> float:
        """Get the effective price per share including costs."""
        if self.filled_quantity == 0:
            return 0.0
        total_costs = self.total_commission + self.total_slippage
        if self.side == OrderSide.BUY:
            return (self.gross_value + total_costs) / self.filled_quantity
        else:
            return (self.gross_value - total_costs) / self.filled_quantity
    
    # =========================================================================
    # STATE TRANSITIONS
    # =========================================================================
    
    def submit(self, timestamp: Optional[datetime] = None) -> None:
        """Mark order as submitted."""
        if self.status != OrderStatus.PENDING:
            raise ValueError(f"Cannot submit order in {self.status} status")
        
        self.status = OrderStatus.SUBMITTED
        self.submitted_at = timestamp or datetime.utcnow()
        self.last_updated_at = self.submitted_at
    
    def accept(self, timestamp: Optional[datetime] = None) -> None:
        """Mark order as accepted by market."""
        if self.status not in [OrderStatus.PENDING, OrderStatus.SUBMITTED]:
            raise ValueError(f"Cannot accept order in {self.status} status")
        
        self.status = OrderStatus.ACCEPTED
        self.last_updated_at = timestamp or datetime.utcnow()
    
    def add_fill(
        self,
        quantity: int,
        price: float,
        commission: float = 0.0,
        slippage: float = 0.0,
        timestamp: Optional[datetime] = None
    ) -> OrderFill:
        """
        Add a fill to the order.
        
        Args:
            quantity: Number of shares filled
            price: Fill price
            commission: Commission for this fill
            slippage: Slippage for this fill
            timestamp: Fill timestamp
            
        Returns:
            The created OrderFill
        """
        if not self.is_active and self.status != OrderStatus.ACCEPTED:
            raise ValueError(f"Cannot fill order in {self.status} status")
        
        if quantity <= 0:
            raise ValueError(f"Fill quantity must be positive, got {quantity}")
        
        if quantity > self.remaining_quantity:
            raise ValueError(
                f"Fill quantity ({quantity}) exceeds remaining ({self.remaining_quantity})"
            )
        
        fill_timestamp = timestamp or datetime.utcnow()
        fill_id = f"{self.order_id}-F{len(self.fills) + 1}"
        
        fill = OrderFill(
            fill_id=fill_id,
            quantity=quantity,
            price=price,
            commission=commission,
            slippage=slippage,
            timestamp=fill_timestamp
        )
        
        self.fills.append(fill)
        
        # Update tracking
        old_filled_value = self.filled_quantity * self.avg_fill_price
        self.filled_quantity += quantity
        self.total_commission += commission
        self.total_slippage += slippage
        
        # Recalculate average fill price
        new_filled_value = old_filled_value + (quantity * price)
        self.avg_fill_price = new_filled_value / self.filled_quantity
        
        # Update status
        if self.is_filled:
            self.status = OrderStatus.FILLED
            self.completed_at = fill_timestamp
        else:
            self.status = OrderStatus.PARTIALLY_FILLED
        
        self.last_updated_at = fill_timestamp
        
        return fill
    
    def fill_complete(
        self,
        price: float,
        commission: float = 0.0,
        slippage: float = 0.0,
        timestamp: Optional[datetime] = None
    ) -> OrderFill:
        """
        Fill the entire remaining quantity.
        
        Convenience method for complete fills.
        """
        return self.add_fill(
            quantity=self.remaining_quantity,
            price=price,
            commission=commission,
            slippage=slippage,
            timestamp=timestamp
        )
    
    def cancel(
        self,
        reason: Optional[str] = None,
        timestamp: Optional[datetime] = None
    ) -> None:
        """Cancel the order."""
        if self.is_terminal:
            raise ValueError(f"Cannot cancel order in {self.status} status")
        
        self.status = OrderStatus.CANCELLED
        self.cancel_reason = reason
        self.completed_at = timestamp or datetime.utcnow()
        self.last_updated_at = self.completed_at
    
    def reject(
        self,
        reason: str,
        timestamp: Optional[datetime] = None
    ) -> None:
        """Reject the order."""
        if self.status not in [OrderStatus.PENDING, OrderStatus.SUBMITTED]:
            raise ValueError(f"Cannot reject order in {self.status} status")
        
        self.status = OrderStatus.REJECTED
        self.reject_reason = reason
        self.completed_at = timestamp or datetime.utcnow()
        self.last_updated_at = self.completed_at
    
    def expire(self, timestamp: Optional[datetime] = None) -> None:
        """Mark order as expired."""
        if self.is_terminal:
            raise ValueError(f"Cannot expire order in {self.status} status")
        
        self.status = OrderStatus.EXPIRED
        self.completed_at = timestamp or datetime.utcnow()
        self.last_updated_at = self.completed_at
    
    # =========================================================================
    # PRICE CHECKS
    # =========================================================================
    
    def should_trigger_stop(self, high: float, low: float) -> bool:
        """
        Check if stop price is triggered.
        
        Args:
            high: Current bar high
            low: Current bar low
            
        Returns:
            True if stop should trigger
        """
        if self.stop_price is None:
            return False
        
        if self.order_type not in [OrderType.STOP, OrderType.STOP_LIMIT]:
            return False
        
        # Buy stop: triggers when price goes above stop
        if self.side == OrderSide.BUY:
            return high >= self.stop_price
        
        # Sell stop: triggers when price goes below stop
        return low <= self.stop_price
    
    def can_fill_at_price(self, price: float) -> bool:
        """
        Check if order can be filled at given price.
        
        Args:
            price: The proposed fill price
            
        Returns:
            True if order can be filled at this price
        """
        if self.order_type == OrderType.MARKET:
            return True
        
        if self.order_type == OrderType.LIMIT:
            if self.side == OrderSide.BUY:
                return price <= self.price  # Buy at limit or better
            return price >= self.price  # Sell at limit or better
        
        # For stop orders, assume stop has already triggered
        if self.order_type == OrderType.STOP:
            return True
        
        if self.order_type == OrderType.STOP_LIMIT:
            if self.side == OrderSide.BUY:
                return price <= self.price
            return price >= self.price
        
        return False
    
    # =========================================================================
    # SERIALIZATION
    # =========================================================================
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert order to dictionary."""
        return {
            "order_id": self.order_id,
            "symbol": self.symbol,
            "side": self.side.value,
            "order_type": self.order_type.value,
            "quantity": self.quantity,
            "price": self.price,
            "stop_price": self.stop_price,
            "time_in_force": self.time_in_force.value,
            "status": self.status.value,
            "filled_quantity": self.filled_quantity,
            "remaining_quantity": self.remaining_quantity,
            "avg_fill_price": round(self.avg_fill_price, 4),
            "total_commission": round(self.total_commission, 4),
            "total_slippage": round(self.total_slippage, 4),
            "gross_value": round(self.gross_value, 2),
            "total_cost": round(self.total_cost, 2),
            "effective_price": round(self.effective_price, 4),
            "fills": [f.to_dict() for f in self.fills],
            "created_at": self.created_at.isoformat(),
            "submitted_at": self.submitted_at.isoformat() if self.submitted_at else None,
            "completed_at": self.completed_at.isoformat() if self.completed_at else None,
            "reject_reason": self.reject_reason,
            "cancel_reason": self.cancel_reason,
            "strategy_id": self.strategy_id,
            "signal_id": self.signal_id,
            "tags": self.tags
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Order":
        """Create order from dictionary."""
        order = cls(
            symbol=data["symbol"],
            side=OrderSide(data["side"]),
            order_type=OrderType(data["order_type"]),
            quantity=data["quantity"],
            price=data.get("price"),
            stop_price=data.get("stop_price"),
            time_in_force=OrderTimeInForce(data.get("time_in_force", "day")),
            order_id=data.get("order_id", generate_order_id()),
            status=OrderStatus(data.get("status", "pending")),
            strategy_id=data.get("strategy_id"),
            signal_id=data.get("signal_id"),
            tags=data.get("tags", {})
        )
        
        # Restore fills if present
        for fill_data in data.get("fills", []):
            order.fills.append(OrderFill(
                fill_id=fill_data["fill_id"],
                quantity=fill_data["quantity"],
                price=fill_data["price"],
                commission=fill_data["commission"],
                slippage=fill_data["slippage"],
                timestamp=datetime.fromisoformat(fill_data["timestamp"])
            ))
        
        # Restore tracking
        order.filled_quantity = data.get("filled_quantity", 0)
        order.avg_fill_price = data.get("avg_fill_price", 0.0)
        order.total_commission = data.get("total_commission", 0.0)
        order.total_slippage = data.get("total_slippage", 0.0)
        
        return order
    
    # =========================================================================
    # STRING REPRESENTATION
    # =========================================================================
    
    def __str__(self) -> str:
        """Human-readable string representation."""
        price_str = ""
        if self.price:
            price_str = f" @ {self.price:.2f}"
        if self.stop_price:
            price_str += f" (stop: {self.stop_price:.2f})"
        
        fill_str = ""
        if self.filled_quantity > 0:
            fill_str = f" [{self.filled_quantity}/{self.quantity} filled @ {self.avg_fill_price:.2f}]"
        
        return (
            f"Order({self.order_id}): {self.side.value.upper()} {self.quantity} "
            f"{self.symbol} {self.order_type.value}{price_str} - {self.status.value}{fill_str}"
        )
    
    def __repr__(self) -> str:
        """Detailed string representation."""
        return (
            f"Order(order_id='{self.order_id}', symbol='{self.symbol}', "
            f"side={self.side}, type={self.order_type}, qty={self.quantity}, "
            f"status={self.status})"
        )


# =============================================================================
# FACTORY FUNCTIONS
# =============================================================================

def create_market_order(
    symbol: str,
    side: OrderSide,
    quantity: int,
    **kwargs
) -> Order:
    """
    Create a market order.
    
    Args:
        symbol: Trading symbol
        side: Buy or Sell
        quantity: Number of shares
        **kwargs: Additional order parameters
        
    Returns:
        Market order
    """
    return Order(
        symbol=symbol,
        side=side,
        order_type=OrderType.MARKET,
        quantity=quantity,
        **kwargs
    )


def create_limit_order(
    symbol: str,
    side: OrderSide,
    quantity: int,
    price: float,
    **kwargs
) -> Order:
    """
    Create a limit order.
    
    Args:
        symbol: Trading symbol
        side: Buy or Sell
        quantity: Number of shares
        price: Limit price
        **kwargs: Additional order parameters
        
    Returns:
        Limit order
    """
    return Order(
        symbol=symbol,
        side=side,
        order_type=OrderType.LIMIT,
        quantity=quantity,
        price=price,
        **kwargs
    )


def create_stop_order(
    symbol: str,
    side: OrderSide,
    quantity: int,
    stop_price: float,
    **kwargs
) -> Order:
    """
    Create a stop order.
    
    Args:
        symbol: Trading symbol
        side: Buy or Sell
        quantity: Number of shares
        stop_price: Stop trigger price
        **kwargs: Additional order parameters
        
    Returns:
        Stop order
    """
    return Order(
        symbol=symbol,
        side=side,
        order_type=OrderType.STOP,
        quantity=quantity,
        stop_price=stop_price,
        **kwargs
    )


def create_stop_limit_order(
    symbol: str,
    side: OrderSide,
    quantity: int,
    stop_price: float,
    limit_price: float,
    **kwargs
) -> Order:
    """
    Create a stop-limit order.
    
    Args:
        symbol: Trading symbol
        side: Buy or Sell
        quantity: Number of shares
        stop_price: Stop trigger price
        limit_price: Limit price after stop triggers
        **kwargs: Additional order parameters
        
    Returns:
        Stop-limit order
    """
    return Order(
        symbol=symbol,
        side=side,
        order_type=OrderType.STOP_LIMIT,
        quantity=quantity,
        stop_price=stop_price,
        price=limit_price,
        **kwargs
    )
