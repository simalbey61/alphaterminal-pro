"""
AlphaTerminal Pro - BIST Transaction Costs
==========================================

BIST (Borsa Istanbul) specific transaction cost models.

Includes:
- Commission rates by broker type
- Stamp tax (Damga Vergisi)
- BSMV (Banking Insurance Transaction Tax)
- Slippage models based on liquidity
- Market impact estimation

Current Rates (2024):
- Commission: 0.05% - 0.15% (broker dependent)
- Stamp Tax: Exempt for equities
- BSMV: 5% of commission

Author: AlphaTerminal Team
Version: 1.0.0
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, time
from typing import Any, Dict, Optional, Tuple
from enum import Enum
import math

from app.backtest.enums import OrderSide, OrderType


class BrokerType(str, Enum):
    """BIST broker types with typical commission rates."""
    DISCOUNT = "discount"       # Online-only discount broker
    STANDARD = "standard"       # Standard retail broker
    PREMIUM = "premium"         # Full-service broker
    INSTITUTIONAL = "institutional"  # Institutional rates


class LiquidityTier(str, Enum):
    """Stock liquidity classification."""
    VERY_HIGH = "very_high"     # BIST30 stocks
    HIGH = "high"               # BIST50 stocks
    MEDIUM = "medium"           # BIST100 stocks
    LOW = "low"                 # Other stocks
    VERY_LOW = "very_low"       # Illiquid stocks


@dataclass
class BISTCommissionConfig:
    """
    BIST commission configuration.
    
    Attributes:
        broker_type: Type of broker
        base_rate: Base commission rate (e.g., 0.001 = 0.1%)
        min_commission: Minimum commission per trade (TRY)
        bsmv_rate: BSMV rate on commission (5%)
        volume_discount: Volume-based discount brackets
    """
    broker_type: BrokerType = BrokerType.STANDARD
    base_rate: float = 0.001  # 0.1% default
    min_commission: float = 1.0  # Minimum 1 TRY
    bsmv_rate: float = 0.05  # 5% BSMV on commission
    
    # Volume discount brackets: (min_volume, rate_multiplier)
    volume_discount: Dict[float, float] = field(default_factory=lambda: {
        0: 1.0,           # Up to 100k: full rate
        100_000: 0.95,    # 100k-500k: 5% discount
        500_000: 0.90,    # 500k-1M: 10% discount
        1_000_000: 0.85,  # 1M+: 15% discount
    })
    
    @classmethod
    def discount_broker(cls) -> "BISTCommissionConfig":
        """Discount broker configuration (lowest rates)."""
        return cls(
            broker_type=BrokerType.DISCOUNT,
            base_rate=0.0005,  # 0.05%
            min_commission=0.5,
            volume_discount={
                0: 1.0,
                50_000: 0.90,
                200_000: 0.80,
                500_000: 0.70
            }
        )
    
    @classmethod
    def standard_broker(cls) -> "BISTCommissionConfig":
        """Standard retail broker configuration."""
        return cls(
            broker_type=BrokerType.STANDARD,
            base_rate=0.001,  # 0.1%
            min_commission=1.0
        )
    
    @classmethod
    def premium_broker(cls) -> "BISTCommissionConfig":
        """Premium/full-service broker configuration."""
        return cls(
            broker_type=BrokerType.PREMIUM,
            base_rate=0.0015,  # 0.15%
            min_commission=2.0
        )
    
    @classmethod
    def institutional(cls) -> "BISTCommissionConfig":
        """Institutional rates configuration."""
        return cls(
            broker_type=BrokerType.INSTITUTIONAL,
            base_rate=0.0003,  # 0.03%
            min_commission=0.0,
            volume_discount={
                0: 1.0,
                1_000_000: 0.80,
                10_000_000: 0.60,
                100_000_000: 0.40
            }
        )


@dataclass
class BISTSlippageConfig:
    """
    BIST slippage configuration.
    
    Attributes:
        base_slippage: Base slippage rate
        liquidity_multipliers: Multipliers by liquidity tier
        volatility_factor: How much volatility affects slippage
        order_size_factor: How much order size affects slippage
    """
    base_slippage: float = 0.0005  # 0.05% base
    
    liquidity_multipliers: Dict[LiquidityTier, float] = field(default_factory=lambda: {
        LiquidityTier.VERY_HIGH: 0.5,   # BIST30: low slippage
        LiquidityTier.HIGH: 0.75,       # BIST50
        LiquidityTier.MEDIUM: 1.0,      # BIST100
        LiquidityTier.LOW: 1.5,         # Others
        LiquidityTier.VERY_LOW: 2.5     # Illiquid
    })
    
    volatility_factor: float = 0.5  # Slippage scales with volatility
    order_size_factor: float = 0.1  # Additional slippage per 1% of ADV
    
    # Time-based adjustments
    opening_minutes: int = 30        # Minutes after open
    closing_minutes: int = 30        # Minutes before close
    auction_multiplier: float = 0.5  # Lower slippage in auctions
    high_volatility_multiplier: float = 1.5  # During high volatility


class BISTCommissionCalculator:
    """
    Calculate BIST trading commissions.
    
    Example:
        ```python
        calc = BISTCommissionCalculator(BISTCommissionConfig.standard_broker())
        
        commission = calc.calculate(
            trade_value=50000,
            side=OrderSide.BUY
        )
        print(f"Commission: {commission.total:.2f} TRY")
        ```
    """
    
    def __init__(self, config: BISTCommissionConfig = None):
        self.config = config or BISTCommissionConfig.standard_broker()
    
    def calculate(
        self,
        trade_value: float,
        side: OrderSide,
        cumulative_volume: float = 0.0
    ) -> "CommissionResult":
        """
        Calculate commission for a trade.
        
        Args:
            trade_value: Trade value in TRY
            side: Buy or Sell
            cumulative_volume: Cumulative daily volume for discount
            
        Returns:
            CommissionResult with breakdown
        """
        # Get volume-adjusted rate
        rate = self._get_volume_rate(cumulative_volume + trade_value)
        
        # Calculate base commission
        base_commission = trade_value * rate
        
        # Apply minimum
        base_commission = max(base_commission, self.config.min_commission)
        
        # Calculate BSMV (on commission, not trade value)
        bsmv = base_commission * self.config.bsmv_rate
        
        # Total commission
        total = base_commission + bsmv
        
        return CommissionResult(
            trade_value=trade_value,
            base_commission=base_commission,
            bsmv=bsmv,
            total=total,
            effective_rate=total / trade_value if trade_value > 0 else 0,
            rate_used=rate
        )
    
    def _get_volume_rate(self, volume: float) -> float:
        """Get commission rate based on volume."""
        base_rate = self.config.base_rate
        
        # Find applicable discount
        multiplier = 1.0
        for threshold, mult in sorted(self.config.volume_discount.items(), reverse=True):
            if volume >= threshold:
                multiplier = mult
                break
        
        return base_rate * multiplier


@dataclass
class CommissionResult:
    """Commission calculation result."""
    trade_value: float
    base_commission: float
    bsmv: float
    total: float
    effective_rate: float
    rate_used: float
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "trade_value": round(self.trade_value, 2),
            "base_commission": round(self.base_commission, 4),
            "bsmv": round(self.bsmv, 4),
            "total": round(self.total, 4),
            "effective_rate": round(self.effective_rate, 6),
            "rate_used": round(self.rate_used, 6)
        }


class BISTSlippageCalculator:
    """
    Calculate BIST trading slippage.
    
    Models realistic slippage based on:
    - Stock liquidity
    - Order size relative to ADV
    - Time of day
    - Current volatility
    
    Example:
        ```python
        calc = BISTSlippageCalculator()
        
        slippage = calc.calculate(
            price=100.0,
            quantity=1000,
            side=OrderSide.BUY,
            avg_daily_volume=500000,
            liquidity_tier=LiquidityTier.HIGH
        )
        print(f"Slippage: {slippage.slippage_amount:.4f}")
        ```
    """
    
    def __init__(self, config: BISTSlippageConfig = None):
        self.config = config or BISTSlippageConfig()
    
    def calculate(
        self,
        price: float,
        quantity: int,
        side: OrderSide,
        avg_daily_volume: float = 1_000_000,
        liquidity_tier: LiquidityTier = LiquidityTier.MEDIUM,
        current_volatility: float = 0.02,  # 2% daily volatility
        time_of_day: Optional[time] = None,
        spread_pct: float = 0.001  # 0.1% bid-ask spread
    ) -> "SlippageResult":
        """
        Calculate slippage for a trade.
        
        Args:
            price: Current price
            quantity: Order quantity
            side: Buy or Sell
            avg_daily_volume: Average daily volume
            liquidity_tier: Stock liquidity classification
            current_volatility: Current volatility (daily)
            time_of_day: Time of trade
            spread_pct: Current bid-ask spread as percentage
            
        Returns:
            SlippageResult with breakdown
        """
        # Base slippage
        base = self.config.base_slippage
        
        # Liquidity adjustment
        liquidity_mult = self.config.liquidity_multipliers.get(
            liquidity_tier, 1.0
        )
        
        # Order size impact
        order_value = price * quantity
        order_pct_of_adv = (quantity / avg_daily_volume) if avg_daily_volume > 0 else 0.1
        size_impact = order_pct_of_adv * self.config.order_size_factor
        
        # Volatility adjustment
        # Higher volatility = more slippage
        vol_adjustment = current_volatility * self.config.volatility_factor
        
        # Time of day adjustment
        time_mult = self._get_time_multiplier(time_of_day)
        
        # Spread contribution (half spread for market orders)
        spread_contribution = spread_pct / 2
        
        # Total slippage rate
        slippage_rate = (
            (base * liquidity_mult + size_impact + vol_adjustment) * time_mult
            + spread_contribution
        )
        
        # Slippage amount
        slippage_amount = price * slippage_rate
        
        # Direction adjustment (buy = worse price = higher, sell = lower)
        if side == OrderSide.BUY:
            effective_price = price + slippage_amount
        else:
            effective_price = price - slippage_amount
        
        return SlippageResult(
            original_price=price,
            effective_price=effective_price,
            slippage_amount=slippage_amount,
            slippage_rate=slippage_rate,
            components={
                "base": base * liquidity_mult,
                "size_impact": size_impact,
                "volatility": vol_adjustment,
                "spread": spread_contribution,
                "time_multiplier": time_mult
            }
        )
    
    def _get_time_multiplier(self, time_of_day: Optional[time]) -> float:
        """Get time-based slippage multiplier."""
        if time_of_day is None:
            return 1.0
        
        # BIST trading hours: 10:00 - 18:00
        market_open = time(10, 0)
        market_close = time(18, 0)
        
        # Opening period (10:00 - 10:30): higher slippage
        opening_end = time(10, 30)
        if market_open <= time_of_day < opening_end:
            return 1.3
        
        # Closing period (17:30 - 18:00): higher slippage
        closing_start = time(17, 30)
        if closing_start <= time_of_day <= market_close:
            return 1.2
        
        # Lunch period (12:30 - 14:00): slightly higher
        lunch_start = time(12, 30)
        lunch_end = time(14, 0)
        if lunch_start <= time_of_day < lunch_end:
            return 1.1
        
        return 1.0


@dataclass
class SlippageResult:
    """Slippage calculation result."""
    original_price: float
    effective_price: float
    slippage_amount: float
    slippage_rate: float
    components: Dict[str, float]
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "original_price": round(self.original_price, 4),
            "effective_price": round(self.effective_price, 4),
            "slippage_amount": round(self.slippage_amount, 4),
            "slippage_rate": round(self.slippage_rate, 6),
            "components": {k: round(v, 6) for k, v in self.components.items()}
        }


class BISTCostCalculator:
    """
    Combined BIST cost calculator.
    
    Calculates total transaction costs including:
    - Commission
    - BSMV
    - Slippage
    - Market impact (for large orders)
    
    Example:
        ```python
        calc = BISTCostCalculator()
        
        costs = calc.calculate_total_costs(
            price=100.0,
            quantity=1000,
            side=OrderSide.BUY,
            symbol="THYAO"
        )
        print(f"Total costs: {costs['total_cost']:.2f} TRY")
        ```
    """
    
    # BIST30 stocks (high liquidity)
    BIST30_SYMBOLS = {
        "AKBNK", "ARCLK", "ASELS", "BIMAS", "DOHOL", "EKGYO", "ENKAI",
        "EREGL", "FROTO", "GARAN", "GUBRF", "HEKTS", "ISCTR", "KCHOL",
        "KOZAA", "KOZAL", "KRDMD", "MGROS", "ODAS", "PETKM", "PGSUS",
        "SAHOL", "SASA", "SISE", "TAVHL", "TCELL", "THYAO", "TKFEN",
        "TOASO", "TUPRS", "VAKBN", "YKBNK"
    }
    
    # BIST50 additional stocks
    BIST50_ADDITIONAL = {
        "AEFES", "AFYON", "AKSA", "AKSEN", "ALGYO", "ALKIM", "ANHYT",
        "AYDEM", "CCOLA", "CIMSA", "DOAS", "EGEEN", "ENJSA", "GESAN",
        "HALKB", "IPEKE", "KONTR", "LOGO", "OTKAR", "SKBNK", "SOKM",
        "TATGD", "TSKB", "TTKOM", "TTRAK", "ULKER", "VESTL", "YEOTK"
    }
    
    def __init__(
        self,
        commission_config: BISTCommissionConfig = None,
        slippage_config: BISTSlippageConfig = None
    ):
        self.commission_calc = BISTCommissionCalculator(commission_config)
        self.slippage_calc = BISTSlippageCalculator(slippage_config)
    
    def get_liquidity_tier(self, symbol: str) -> LiquidityTier:
        """Determine liquidity tier for a symbol."""
        if symbol in self.BIST30_SYMBOLS:
            return LiquidityTier.VERY_HIGH
        elif symbol in self.BIST50_ADDITIONAL:
            return LiquidityTier.HIGH
        else:
            # Could be enhanced with actual volume data
            return LiquidityTier.MEDIUM
    
    def calculate_total_costs(
        self,
        price: float,
        quantity: int,
        side: OrderSide,
        symbol: str,
        avg_daily_volume: float = 1_000_000,
        current_volatility: float = 0.02,
        cumulative_volume: float = 0.0,
        time_of_day: Optional[time] = None
    ) -> Dict[str, Any]:
        """
        Calculate total transaction costs.
        
        Args:
            price: Trade price
            quantity: Number of shares
            side: Buy or Sell
            symbol: Stock symbol
            avg_daily_volume: Average daily volume
            current_volatility: Current volatility
            cumulative_volume: Cumulative daily volume (for commission discount)
            time_of_day: Time of trade
            
        Returns:
            Dictionary with cost breakdown
        """
        trade_value = price * quantity
        liquidity_tier = self.get_liquidity_tier(symbol)
        
        # Calculate commission
        commission = self.commission_calc.calculate(
            trade_value=trade_value,
            side=side,
            cumulative_volume=cumulative_volume
        )
        
        # Calculate slippage
        slippage = self.slippage_calc.calculate(
            price=price,
            quantity=quantity,
            side=side,
            avg_daily_volume=avg_daily_volume,
            liquidity_tier=liquidity_tier,
            current_volatility=current_volatility,
            time_of_day=time_of_day
        )
        
        # Total cost
        total_cost = commission.total + (slippage.slippage_amount * quantity)
        
        return {
            "symbol": symbol,
            "side": side.value,
            "price": price,
            "quantity": quantity,
            "trade_value": trade_value,
            "liquidity_tier": liquidity_tier.value,
            
            # Commission breakdown
            "commission": commission.to_dict(),
            
            # Slippage breakdown
            "slippage": slippage.to_dict(),
            
            # Totals
            "total_commission": commission.total,
            "total_slippage": slippage.slippage_amount * quantity,
            "total_cost": total_cost,
            "effective_price": slippage.effective_price,
            "cost_as_pct": total_cost / trade_value if trade_value > 0 else 0
        }
    
    def estimate_round_trip_cost(
        self,
        price: float,
        quantity: int,
        symbol: str,
        holding_days: int = 1,
        avg_daily_volume: float = 1_000_000
    ) -> Dict[str, Any]:
        """
        Estimate round-trip (buy + sell) costs.
        
        Args:
            price: Entry price
            quantity: Position size
            symbol: Stock symbol
            holding_days: Expected holding period
            avg_daily_volume: Average daily volume
            
        Returns:
            Round-trip cost estimate
        """
        # Entry costs
        entry_costs = self.calculate_total_costs(
            price=price,
            quantity=quantity,
            side=OrderSide.BUY,
            symbol=symbol,
            avg_daily_volume=avg_daily_volume
        )
        
        # Exit costs (assume same price for estimation)
        exit_costs = self.calculate_total_costs(
            price=price,
            quantity=quantity,
            side=OrderSide.SELL,
            symbol=symbol,
            avg_daily_volume=avg_daily_volume
        )
        
        total_round_trip = entry_costs["total_cost"] + exit_costs["total_cost"]
        trade_value = price * quantity
        
        return {
            "entry_costs": entry_costs["total_cost"],
            "exit_costs": exit_costs["total_cost"],
            "total_round_trip": total_round_trip,
            "round_trip_pct": total_round_trip / trade_value if trade_value > 0 else 0,
            "breakeven_move_pct": total_round_trip / trade_value if trade_value > 0 else 0
        }
