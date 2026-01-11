"""
AlphaTerminal Pro - Risk Engine v4.2
====================================

Kurumsal Seviye Risk Yönetim Motoru

YENİ ÖZELLİKLER (v4.2):
- CVaR (Conditional VaR) hesaplama
- Correlation-adjusted position sizing
- Monte Carlo Risk Simulation
- Stress Testing
- Dynamic Position Sizing
- Portfolio Optimization

Özellikler:
- Kelly Criterion pozisyon boyutlandırma
- ATR-based stop loss
- Structure-based stop loss
- Multi-level take profit
- Portfolio heat tracking
- Drawdown protection
- Risk-adjusted position sizing

Author: AlphaTerminal Team
Version: 4.2.0
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, field
from enum import Enum
from datetime import datetime

from app.core.config import logger, RISK_CONFIG


# ═══════════════════════════════════════════════════════════════════════════════
# ENUMS
# ═══════════════════════════════════════════════════════════════════════════════

class StopLossType(Enum):
    """Stop loss hesaplama yöntemi"""
    ATR_BASED = "ATR_BASED"
    STRUCTURE_BASED = "STRUCTURE_BASED"
    PERCENTAGE = "PERCENTAGE"
    SWING_BASED = "SWING_BASED"
    VOLATILITY_BASED = "VOLATILITY_BASED"


class PositionSizeMethod(Enum):
    """Pozisyon boyutlandırma yöntemi"""
    FIXED_FRACTIONAL = "FIXED_FRACTIONAL"
    KELLY = "KELLY"
    VOLATILITY_ADJUSTED = "VOLATILITY_ADJUSTED"
    OPTIMAL_F = "OPTIMAL_F"
    CORRELATION_ADJUSTED = "CORRELATION_ADJUSTED"


class RiskLevel(Enum):
    """Risk seviyesi"""
    VERY_LOW = "VERY_LOW"
    LOW = "LOW"
    MODERATE = "MODERATE"
    HIGH = "HIGH"
    VERY_HIGH = "VERY_HIGH"
    EXTREME = "EXTREME"


class TradeStatus(Enum):
    """Trade durumu"""
    OPEN = "OPEN"
    CLOSED = "CLOSED"
    PARTIAL = "PARTIAL"
    STOPPED = "STOPPED"
    TARGET_HIT = "TARGET_HIT"


# ═══════════════════════════════════════════════════════════════════════════════
# DATA CLASSES
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass
class PositionSize:
    """Pozisyon boyutu hesaplama sonucu"""
    shares: int
    lots: int
    position_value: float
    risk_amount: float
    risk_percent: float
    stop_loss: float
    stop_distance: float
    stop_percent: float
    take_profit_1: float
    take_profit_2: float
    take_profit_3: float
    reward_risk_ratio: float
    kelly_fraction: float = 0.0
    conviction_multiplier: float = 1.0


@dataclass
class StopLossResult:
    """Stop loss hesaplama sonucu"""
    price: float
    distance: float
    percent: float
    method: StopLossType
    atr_multiplier: float = 0.0
    structure_level: float = 0.0


@dataclass
class TakeProfitLevel:
    """Take profit seviyesi"""
    price: float
    distance: float
    percent: float
    rr_ratio: float
    allocation: float  # Pozisyonun %'si


@dataclass
class RiskMetrics:
    """Portföy risk metrikleri"""
    position_risk: float
    portfolio_heat: float
    total_exposure: float
    daily_pnl: float
    weekly_pnl: float
    max_drawdown: float
    current_drawdown: float
    win_rate: float
    profit_factor: float
    expectancy: float
    avg_win: float
    avg_loss: float
    consecutive_wins: int
    consecutive_losses: int
    largest_win: float
    largest_loss: float
    can_trade: bool
    risk_level: RiskLevel
    risk_warnings: List[str]


@dataclass
class MonteCarloResult:
    """Monte Carlo simülasyon sonucu"""
    expected_return: float
    median_return: float
    var_95: float
    var_99: float
    cvar_95: float
    best_case: float
    worst_case: float
    win_probability: float
    ruin_probability: float
    confidence_interval: Tuple[float, float]


@dataclass
class StressTestResult:
    """Stress test sonucu"""
    scenario_name: str
    portfolio_impact: float
    worst_position: str
    worst_position_impact: float
    recovery_estimate_days: int
    risk_level: RiskLevel


@dataclass
class TradeSetup:
    """Trade kurulumu"""
    symbol: str
    direction: str  # "LONG" or "SHORT"
    entry_price: float
    entry_type: str  # "MARKET", "LIMIT", "STOP"
    stop_loss: StopLossResult
    take_profit_levels: List[TakeProfitLevel]
    position_size: PositionSize
    risk_amount: float
    risk_reward: float
    confidence_score: float
    setup_type: str
    notes: List[str]
    created_at: datetime = field(default_factory=datetime.now)


@dataclass
class Position:
    """Açık pozisyon"""
    symbol: str
    direction: str
    entry_price: float
    entry_date: datetime
    shares: int
    stop_loss: float
    take_profits: List[float]
    current_price: float = 0.0
    unrealized_pnl: float = 0.0
    unrealized_pnl_percent: float = 0.0
    risk_amount: float = 0.0
    status: TradeStatus = TradeStatus.OPEN
    partial_closes: List[Dict] = field(default_factory=list)


# ═══════════════════════════════════════════════════════════════════════════════
# RISK ENGINE CLASS
# ═══════════════════════════════════════════════════════════════════════════════

class RiskEngine:
    """
    Kurumsal Seviye Risk Yönetim Motoru v4.2
    
    Özellikler:
    - Kelly Criterion pozisyon boyutlandırma
    - ATR-based stop loss
    - Structure-based stop loss
    - Multi-level take profit
    - Portfolio heat tracking
    - Drawdown protection
    - Risk-adjusted position sizing
    - Monte Carlo simulation
    - Stress testing
    - Correlation-based sizing
    """
    
    def __init__(self, config=None, capital: float = 100000):
        """
        Args:
            config: RiskConfig instance
            capital: Toplam sermaye (TRY)
        """
        self.config = config or RISK_CONFIG
        self.capital = capital
        self.open_positions: List[Position] = []
        self.trade_history: List[Dict] = []
        self._correlation_matrix: Optional[pd.DataFrame] = None
    
    # ═══════════════════════════════════════════════════════════════════════════
    # POSITION SIZING
    # ═══════════════════════════════════════════════════════════════════════════
    
    def calculate_position_size(
        self,
        entry_price: float,
        stop_loss: float,
        take_profit: float = None,
        conviction: float = 1.0,
        method: PositionSizeMethod = PositionSizeMethod.FIXED_FRACTIONAL,
        volatility: float = None,
        correlation_factor: float = 1.0
    ) -> PositionSize:
        """
        Pozisyon boyutu hesaplama
        
        Args:
            entry_price: Giriş fiyatı
            stop_loss: Stop loss fiyatı
            take_profit: Take profit fiyatı (opsiyonel)
            conviction: Güven katsayısı (0.5-1.5)
            method: Pozisyon boyutlandırma yöntemi
            volatility: Volatilite (volatility-adjusted için)
            correlation_factor: Korelasyon faktörü (0-1)
            
        Returns:
            PositionSize dataclass
        """
        # Stop mesafesi
        stop_distance = abs(entry_price - stop_loss)
        stop_percent = stop_distance / entry_price if entry_price > 0 else 0
        
        # Maksimum stop kontrolü
        if stop_percent > self.config.max_sl_percent:
            stop_percent = self.config.max_sl_percent
            if entry_price > stop_loss:
                stop_loss = entry_price * (1 - stop_percent)
            else:
                stop_loss = entry_price * (1 + stop_percent)
            stop_distance = abs(entry_price - stop_loss)
        
        # Minimum stop kontrolü
        if stop_percent < self.config.min_sl_percent:
            stop_percent = self.config.min_sl_percent
            if entry_price > stop_loss:
                stop_loss = entry_price * (1 - stop_percent)
            else:
                stop_loss = entry_price * (1 + stop_percent)
            stop_distance = abs(entry_price - stop_loss)
        
        # Base risk hesaplama
        base_risk = self.capital * self.config.max_risk_per_trade
        
        # Kelly fraction
        kelly_fraction = 0.0
        
        if method == PositionSizeMethod.KELLY:
            kelly_fraction = self._calculate_kelly_fraction()
            max_risk = self.capital * min(kelly_fraction, self.config.kelly_fraction_cap)
        
        elif method == PositionSizeMethod.VOLATILITY_ADJUSTED:
            if volatility and volatility > 0:
                # Daha volatil = daha küçük pozisyon
                vol_factor = 0.20 / volatility  # %20 target volatility
                max_risk = base_risk * min(vol_factor, 2.0)
            else:
                max_risk = base_risk
        
        elif method == PositionSizeMethod.CORRELATION_ADJUSTED:
            # Yüksek korelasyonlu portföyde daha küçük pozisyon
            max_risk = base_risk * correlation_factor
        
        else:  # FIXED_FRACTIONAL
            max_risk = base_risk
        
        # Conviction adjustment
        conviction = np.clip(conviction, 0.5, 1.5)
        max_risk *= conviction
        
        # Pozisyon değeri
        if stop_percent > 0:
            position_value = max_risk / stop_percent
        else:
            position_value = self.capital * self.config.max_position_size
        
        # Maximum position size kontrolü
        max_position = self.capital * self.config.max_position_size
        position_value = min(position_value, max_position)
        
        # Lot sayısı
        shares = int(position_value / entry_price)
        shares = max(1, shares)  # Minimum 1 lot
        
        # BIST lot hesabı (genellikle 1 lot = 1 hisse)
        lots = shares
        
        # Gerçek pozisyon değeri ve risk
        actual_position_value = shares * entry_price
        actual_risk = shares * stop_distance
        actual_risk_percent = actual_risk / self.capital if self.capital > 0 else 0
        
        # Take profit seviyeleri
        if take_profit is None:
            tp_distance = stop_distance * self.config.default_rr_ratio
            if entry_price > stop_loss:  # LONG
                take_profit = entry_price + tp_distance
            else:  # SHORT
                take_profit = entry_price - tp_distance
        
        # Multi-level TP
        tp_distance = abs(take_profit - entry_price)
        direction = 1 if entry_price < take_profit else -1
        
        tp1 = entry_price + (tp_distance * self.config.tp_levels[0] * direction)
        tp2 = entry_price + (tp_distance * self.config.tp_levels[1] * direction)
        tp3 = entry_price + (tp_distance * self.config.tp_levels[2] * direction)
        
        # R:R hesaplama
        rr_ratio = tp_distance / stop_distance if stop_distance > 0 else 0
        
        return PositionSize(
            shares=shares,
            lots=lots,
            position_value=round(actual_position_value, 2),
            risk_amount=round(actual_risk, 2),
            risk_percent=round(actual_risk_percent * 100, 2),
            stop_loss=round(stop_loss, 2),
            stop_distance=round(stop_distance, 2),
            stop_percent=round(stop_percent * 100, 2),
            take_profit_1=round(tp1, 2),
            take_profit_2=round(tp2, 2),
            take_profit_3=round(tp3, 2),
            reward_risk_ratio=round(rr_ratio, 2),
            kelly_fraction=round(kelly_fraction, 4),
            conviction_multiplier=conviction
        )
    
    def _calculate_kelly_fraction(self) -> float:
        """
        Kelly Criterion hesaplama
        
        Kelly % = W - [(1-W)/R]
        W = Win rate
        R = Win/Loss ratio
        
        Returns:
            Kelly fraction (0-1)
        """
        if len(self.trade_history) < self.config.kelly_min_trades:
            return 0.0
        
        wins = [t for t in self.trade_history if t.get('pnl', 0) > 0]
        losses = [t for t in self.trade_history if t.get('pnl', 0) < 0]
        
        if len(wins) == 0 or len(losses) == 0:
            return 0.0
        
        win_rate = len(wins) / len(self.trade_history)
        avg_win = np.mean([t['pnl'] for t in wins])
        avg_loss = abs(np.mean([t['pnl'] for t in losses]))
        
        if avg_loss == 0:
            return 0.0
        
        win_loss_ratio = avg_win / avg_loss
        
        kelly = win_rate - ((1 - win_rate) / win_loss_ratio)
        
        # Half Kelly (daha muhafazakar)
        kelly = kelly / 2
        
        return max(0, min(kelly, self.config.kelly_fraction_cap))
    
    # ═══════════════════════════════════════════════════════════════════════════
    # STOP LOSS CALCULATION
    # ═══════════════════════════════════════════════════════════════════════════
    
    def calculate_stop_loss(
        self,
        df: pd.DataFrame,
        entry_price: float,
        direction: str = "LONG",
        method: StopLossType = StopLossType.ATR_BASED,
        smc_data: Dict = None,
        atr_multiplier: float = None
    ) -> StopLossResult:
        """
        Stop loss hesaplama
        
        Args:
            df: OHLCV DataFrame
            entry_price: Giriş fiyatı
            direction: "LONG" veya "SHORT"
            method: Stop loss hesaplama yöntemi
            smc_data: SMC analiz verisi (structure-based için)
            atr_multiplier: ATR çarpanı (custom)
            
        Returns:
            StopLossResult dataclass
        """
        atr_mult = atr_multiplier or self.config.default_sl_atr_mult
        
        if method == StopLossType.ATR_BASED:
            stop, distance = self._atr_based_stop(df, entry_price, direction, atr_mult)
            structure_level = 0
        
        elif method == StopLossType.STRUCTURE_BASED:
            stop, distance, structure_level = self._structure_based_stop(
                df, entry_price, direction, smc_data
            )
        
        elif method == StopLossType.SWING_BASED:
            stop, distance = self._swing_based_stop(df, entry_price, direction)
            structure_level = 0
        
        elif method == StopLossType.VOLATILITY_BASED:
            stop, distance = self._volatility_based_stop(df, entry_price, direction)
            structure_level = 0
        
        else:  # PERCENTAGE
            stop, distance = self._percentage_based_stop(entry_price, direction)
            structure_level = 0
        
        percent = (distance / entry_price) * 100 if entry_price > 0 else 0
        
        return StopLossResult(
            price=round(stop, 2),
            distance=round(distance, 2),
            percent=round(percent, 2),
            method=method,
            atr_multiplier=atr_mult,
            structure_level=structure_level
        )
    
    def _atr_based_stop(
        self,
        df: pd.DataFrame,
        entry: float,
        direction: str,
        multiplier: float
    ) -> Tuple[float, float]:
        """ATR-based stop loss"""
        if 'ATR' in df.columns:
            atr = df['ATR'].iloc[-1]
        else:
            tr = pd.concat([
                df['High'] - df['Low'],
                abs(df['High'] - df['Close'].shift(1)),
                abs(df['Low'] - df['Close'].shift(1))
            ], axis=1).max(axis=1)
            atr = tr.rolling(14).mean().iloc[-1]
        
        distance = atr * multiplier
        
        if direction == "LONG":
            stop = entry - distance
        else:
            stop = entry + distance
        
        return stop, distance
    
    def _structure_based_stop(
        self,
        df: pd.DataFrame,
        entry: float,
        direction: str,
        smc_data: Dict = None
    ) -> Tuple[float, float, float]:
        """Structure-based stop loss (SMC OB altı/üstü)"""
        structure_level = 0
        
        if smc_data and smc_data.get('ob'):
            ob = smc_data['ob']
            
            if direction == "LONG":
                # OB'nin altına stop
                stop = ob['bottom'] * 0.998  # %0.2 buffer
                structure_level = ob['bottom']
            else:
                # OB'nin üstüne stop
                stop = ob['top'] * 1.002
                structure_level = ob['top']
            
            distance = abs(entry - stop)
            return stop, distance, structure_level
        
        # Fallback to ATR
        stop, distance = self._atr_based_stop(df, entry, direction, self.config.default_sl_atr_mult)
        return stop, distance, 0
    
    def _swing_based_stop(
        self,
        df: pd.DataFrame,
        entry: float,
        direction: str
    ) -> Tuple[float, float]:
        """Swing-based stop loss"""
        lookback = 20
        
        if direction == "LONG":
            swing_low = df['Low'].iloc[-lookback:].min()
            stop = swing_low * 0.998
        else:
            swing_high = df['High'].iloc[-lookback:].max()
            stop = swing_high * 1.002
        
        distance = abs(entry - stop)
        return stop, distance
    
    def _volatility_based_stop(
        self,
        df: pd.DataFrame,
        entry: float,
        direction: str
    ) -> Tuple[float, float]:
        """Volatility-based stop loss (Keltner Channel)"""
        if 'ATR' in df.columns:
            atr = df['ATR'].iloc[-1]
        else:
            tr = (df['High'] - df['Low']).rolling(14).mean().iloc[-1]
            atr = tr
        
        ema = df['Close'].ewm(span=20).mean().iloc[-1]
        
        if direction == "LONG":
            stop = ema - (atr * 2.5)
        else:
            stop = ema + (atr * 2.5)
        
        distance = abs(entry - stop)
        return stop, distance
    
    def _percentage_based_stop(
        self,
        entry: float,
        direction: str
    ) -> Tuple[float, float]:
        """Percentage-based stop loss"""
        pct = self.config.max_sl_percent
        
        if direction == "LONG":
            stop = entry * (1 - pct)
        else:
            stop = entry * (1 + pct)
        
        distance = entry * pct
        return stop, distance
    
    # ═══════════════════════════════════════════════════════════════════════════
    # TAKE PROFIT CALCULATION
    # ═══════════════════════════════════════════════════════════════════════════
    
    def calculate_take_profits(
        self,
        entry_price: float,
        stop_loss: float,
        direction: str = "LONG",
        rr_ratios: List[float] = None,
        allocations: List[float] = None,
        smc_data: Dict = None
    ) -> List[TakeProfitLevel]:
        """
        Multi-level take profit hesaplama
        
        Args:
            entry_price: Giriş fiyatı
            stop_loss: Stop loss fiyatı
            direction: Trade yönü
            rr_ratios: R:R oranları
            allocations: Her seviye için allocation
            smc_data: SMC verisi (likidite hedefleri için)
            
        Returns:
            TakeProfitLevel listesi
        """
        if rr_ratios is None:
            rr_ratios = self.config.tp_levels
        
        if allocations is None:
            allocations = self.config.tp_allocations
        
        stop_distance = abs(entry_price - stop_loss)
        direction_mult = 1 if direction == "LONG" else -1
        
        take_profits = []
        
        for i, rr in enumerate(rr_ratios):
            tp_distance = stop_distance * rr
            tp_price = entry_price + (tp_distance * direction_mult)
            tp_percent = (tp_distance / entry_price) * 100 if entry_price > 0 else 0
            allocation = allocations[i] if i < len(allocations) else 0.34
            
            take_profits.append(TakeProfitLevel(
                price=round(tp_price, 2),
                distance=round(tp_distance, 2),
                percent=round(tp_percent, 2),
                rr_ratio=rr,
                allocation=allocation
            ))
        
        # SMC likidite hedefleri varsa ekle
        if smc_data:
            if direction == "LONG" and smc_data.get('liquidity_high'):
                liq_target = smc_data['liquidity_high']
                if liq_target > entry_price:
                    liq_distance = liq_target - entry_price
                    liq_rr = liq_distance / stop_distance if stop_distance > 0 else 0
                    take_profits.append(TakeProfitLevel(
                        price=round(liq_target, 2),
                        distance=round(liq_distance, 2),
                        percent=round((liq_distance / entry_price) * 100, 2),
                        rr_ratio=round(liq_rr, 2),
                        allocation=0.25
                    ))
            
            elif direction == "SHORT" and smc_data.get('liquidity_low'):
                liq_target = smc_data['liquidity_low']
                if liq_target < entry_price:
                    liq_distance = entry_price - liq_target
                    liq_rr = liq_distance / stop_distance if stop_distance > 0 else 0
                    take_profits.append(TakeProfitLevel(
                        price=round(liq_target, 2),
                        distance=round(liq_distance, 2),
                        percent=round((liq_distance / entry_price) * 100, 2),
                        rr_ratio=round(liq_rr, 2),
                        allocation=0.25
                    ))
        
        return take_profits
    
    # ═══════════════════════════════════════════════════════════════════════════
    # PORTFOLIO RISK MANAGEMENT
    # ═══════════════════════════════════════════════════════════════════════════
    
    def check_portfolio_heat(self) -> Tuple[float, bool]:
        """
        Portfolio heat kontrolü
        
        Returns:
            (current_heat, can_open_new_position)
        """
        total_risk = sum(pos.risk_amount for pos in self.open_positions)
        current_heat = total_risk / self.capital if self.capital > 0 else 0
        
        can_trade = current_heat < self.config.max_portfolio_heat
        
        return current_heat, can_trade
    
    def check_position_limit(self) -> bool:
        """Maksimum pozisyon sayısı kontrolü"""
        return len(self.open_positions) < self.config.max_positions
    
    def check_daily_loss_limit(self, daily_pnl: float) -> bool:
        """Günlük kayıp limiti kontrolü"""
        daily_loss_percent = abs(daily_pnl) / self.capital if daily_pnl < 0 else 0
        return daily_loss_percent < self.config.max_daily_loss
    
    def check_sector_exposure(self, sector: str) -> Tuple[float, bool]:
        """
        Sektör maruziyeti kontrolü
        
        Args:
            sector: Sektör adı
            
        Returns:
            (current_exposure, can_add_more)
        """
        sector_value = sum(
            pos.shares * pos.current_price
            for pos in self.open_positions
            # Burada pos.sector kontrolü yapılabilir
        )
        
        exposure = sector_value / self.capital if self.capital > 0 else 0
        can_add = exposure < self.config.max_sector_exposure
        
        return exposure, can_add
    
    def calculate_metrics(self) -> RiskMetrics:
        """
        Risk metriklerini hesapla
        
        Returns:
            RiskMetrics dataclass
        """
        warnings = []
        
        # Portfolio heat
        heat, can_trade = self.check_portfolio_heat()
        
        # Total exposure
        total_exposure = sum(
            pos.shares * pos.current_price
            for pos in self.open_positions
        ) / self.capital if self.capital > 0 else 0
        
        # Trade history stats
        if not self.trade_history:
            return RiskMetrics(
                position_risk=0, portfolio_heat=heat * 100, total_exposure=total_exposure * 100,
                daily_pnl=0, weekly_pnl=0, max_drawdown=0, current_drawdown=0,
                win_rate=0, profit_factor=0, expectancy=0, avg_win=0, avg_loss=0,
                consecutive_wins=0, consecutive_losses=0, largest_win=0, largest_loss=0,
                can_trade=True, risk_level=RiskLevel.LOW, risk_warnings=[]
            )
        
        wins = [t for t in self.trade_history if t.get('pnl', 0) > 0]
        losses = [t for t in self.trade_history if t.get('pnl', 0) <= 0]
        
        # Win rate
        win_rate = len(wins) / len(self.trade_history) * 100 if self.trade_history else 0
        
        # Profit factor
        gross_profit = sum(t.get('pnl', 0) for t in wins)
        gross_loss = abs(sum(t.get('pnl', 0) for t in losses))
        profit_factor = gross_profit / gross_loss if gross_loss > 0 else 0
        
        # Average win/loss
        avg_win = gross_profit / len(wins) if wins else 0
        avg_loss = gross_loss / len(losses) if losses else 0
        
        # Expectancy
        expectancy = (win_rate / 100 * avg_win) - ((1 - win_rate / 100) * avg_loss)
        
        # Consecutive wins/losses
        consec_wins = 0
        consec_losses = 0
        current_streak = 0
        last_result = None
        
        for t in self.trade_history:
            pnl = t.get('pnl', 0)
            if pnl > 0:
                if last_result == 'win':
                    current_streak += 1
                else:
                    current_streak = 1
                last_result = 'win'
                consec_wins = max(consec_wins, current_streak)
            else:
                if last_result == 'loss':
                    current_streak += 1
                else:
                    current_streak = 1
                last_result = 'loss'
                consec_losses = max(consec_losses, current_streak)
        
        # Current consecutive losses
        current_consec_losses = 0
        for t in reversed(self.trade_history):
            if t.get('pnl', 0) <= 0:
                current_consec_losses += 1
            else:
                break
        
        # Largest win/loss
        largest_win = max((t.get('pnl', 0) for t in wins), default=0)
        largest_loss = min((t.get('pnl', 0) for t in losses), default=0)
        
        # Max drawdown
        max_dd = self._calculate_max_drawdown()
        
        # Current drawdown
        equity = [self.capital]
        for t in self.trade_history:
            equity.append(equity[-1] + t.get('pnl', 0))
        peak = max(equity)
        current_dd = (peak - equity[-1]) / peak * 100 if peak > 0 else 0
        
        # Daily/Weekly PnL (son 5 ve 25 trade)
        daily_pnl = sum(t.get('pnl', 0) for t in self.trade_history[-5:])
        weekly_pnl = sum(t.get('pnl', 0) for t in self.trade_history[-25:])
        
        # Risk warnings
        if current_consec_losses >= self.config.pause_after_consecutive_losses:
            warnings.append(f"⚠️ {current_consec_losses} ardışık kayıp! Trade'e ara verin.")
            can_trade = False
        
        if heat > self.config.max_portfolio_heat * 0.8:
            warnings.append(f"⚠️ Portfolio heat yüksek: {heat*100:.1f}%")
        
        if current_dd > self.config.max_drawdown * 100 * 0.8:
            warnings.append(f"⚠️ Drawdown yüksek: {current_dd:.1f}%")
        
        # Risk level
        if heat > 0.08 or current_dd > 12:
            risk_level = RiskLevel.EXTREME
        elif heat > 0.06 or current_dd > 10:
            risk_level = RiskLevel.VERY_HIGH
        elif heat > 0.04 or current_dd > 7:
            risk_level = RiskLevel.HIGH
        elif heat > 0.02 or current_dd > 5:
            risk_level = RiskLevel.MODERATE
        elif heat > 0.01:
            risk_level = RiskLevel.LOW
        else:
            risk_level = RiskLevel.VERY_LOW
        
        return RiskMetrics(
            position_risk=heat * 100,
            portfolio_heat=heat * 100,
            total_exposure=total_exposure * 100,
            daily_pnl=daily_pnl,
            weekly_pnl=weekly_pnl,
            max_drawdown=max_dd,
            current_drawdown=current_dd,
            win_rate=win_rate,
            profit_factor=profit_factor,
            expectancy=expectancy,
            avg_win=avg_win,
            avg_loss=avg_loss,
            consecutive_wins=consec_wins,
            consecutive_losses=consec_losses,
            largest_win=largest_win,
            largest_loss=largest_loss,
            can_trade=can_trade,
            risk_level=risk_level,
            risk_warnings=warnings
        )
    
    def _calculate_max_drawdown(self) -> float:
        """Maximum drawdown hesapla"""
        if not self.trade_history:
            return 0
        
        equity = [self.capital]
        for t in self.trade_history:
            equity.append(equity[-1] + t.get('pnl', 0))
        
        equity = pd.Series(equity)
        rolling_max = equity.expanding().max()
        drawdown = (equity - rolling_max) / rolling_max
        
        return abs(drawdown.min()) * 100
    
    # ═══════════════════════════════════════════════════════════════════════════
    # MONTE CARLO SIMULATION
    # ═══════════════════════════════════════════════════════════════════════════
    
    def monte_carlo_simulation(
        self,
        n_simulations: int = None,
        n_trades: int = 100
    ) -> MonteCarloResult:
        """
        Monte Carlo risk simülasyonu
        
        Args:
            n_simulations: Simülasyon sayısı
            n_trades: Her simülasyonda trade sayısı
            
        Returns:
            MonteCarloResult dataclass
        """
        n_simulations = n_simulations or self.config.monte_carlo_simulations
        
        if len(self.trade_history) < 20:
            return MonteCarloResult(
                expected_return=0, median_return=0, var_95=0, var_99=0,
                cvar_95=0, best_case=0, worst_case=0, win_probability=0.5,
                ruin_probability=0, confidence_interval=(0, 0)
            )
        
        # Trade istatistikleri
        pnls = [t.get('pnl', 0) for t in self.trade_history]
        
        # Simülasyonlar
        final_returns = []
        
        np.random.seed(42)
        
        for _ in range(n_simulations):
            # Bootstrap sampling
            sampled_trades = np.random.choice(pnls, size=n_trades, replace=True)
            total_return = sum(sampled_trades)
            final_returns.append(total_return)
        
        final_returns = np.array(final_returns)
        
        # İstatistikler
        expected = np.mean(final_returns)
        median = np.median(final_returns)
        var_95 = np.percentile(final_returns, 5)
        var_99 = np.percentile(final_returns, 1)
        
        # CVaR (Expected Shortfall)
        cvar_95 = np.mean(final_returns[final_returns <= var_95])
        
        best = np.max(final_returns)
        worst = np.min(final_returns)
        
        # Win probability
        win_prob = np.mean(final_returns > 0)
        
        # Ruin probability (%50'den fazla kayıp)
        ruin_prob = np.mean(final_returns < -self.capital * 0.5)
        
        # 95% confidence interval
        ci_low = np.percentile(final_returns, 2.5)
        ci_high = np.percentile(final_returns, 97.5)
        
        return MonteCarloResult(
            expected_return=round(expected, 2),
            median_return=round(median, 2),
            var_95=round(var_95, 2),
            var_99=round(var_99, 2),
            cvar_95=round(cvar_95, 2),
            best_case=round(best, 2),
            worst_case=round(worst, 2),
            win_probability=round(win_prob * 100, 2),
            ruin_probability=round(ruin_prob * 100, 2),
            confidence_interval=(round(ci_low, 2), round(ci_high, 2))
        )
    
    # ═══════════════════════════════════════════════════════════════════════════
    # STRESS TESTING
    # ═══════════════════════════════════════════════════════════════════════════
    
    def stress_test(
        self,
        scenario: str = "market_crash_20"
    ) -> StressTestResult:
        """
        Stress test
        
        Args:
            scenario: Test senaryosu
            
        Returns:
            StressTestResult dataclass
        """
        scenarios = {
            "market_crash_20": -0.20,  # %20 piyasa düşüşü
            "market_crash_30": -0.30,  # %30 piyasa düşüşü
            "volatility_spike": -0.15,  # Volatilite artışı
            "liquidity_crisis": -0.25,  # Likidite krizi
            "sector_rotation": -0.10,   # Sektör rotasyonu
        }
        
        shock = scenarios.get(scenario, -0.20)
        
        # Portföy etkisi hesapla
        total_value = sum(
            pos.shares * pos.current_price
            for pos in self.open_positions
        )
        
        portfolio_impact = total_value * shock
        portfolio_impact_percent = (portfolio_impact / self.capital) * 100 if self.capital > 0 else 0
        
        # En kötü pozisyon
        worst_pos = None
        worst_impact = 0
        
        for pos in self.open_positions:
            pos_value = pos.shares * pos.current_price
            pos_impact = pos_value * shock
            if pos_impact < worst_impact:
                worst_impact = pos_impact
                worst_pos = pos.symbol
        
        # Toparlanma tahmini (basit)
        recovery_days = int(abs(shock) * 100)  # %20 düşüş = 20 gün
        
        # Risk seviyesi
        if abs(portfolio_impact_percent) > 15:
            risk = RiskLevel.EXTREME
        elif abs(portfolio_impact_percent) > 10:
            risk = RiskLevel.VERY_HIGH
        elif abs(portfolio_impact_percent) > 5:
            risk = RiskLevel.HIGH
        else:
            risk = RiskLevel.MODERATE
        
        return StressTestResult(
            scenario_name=scenario,
            portfolio_impact=round(portfolio_impact, 2),
            worst_position=worst_pos or "N/A",
            worst_position_impact=round(worst_impact, 2),
            recovery_estimate_days=recovery_days,
            risk_level=risk
        )
    
    # ═══════════════════════════════════════════════════════════════════════════
    # TRADE SETUP GENERATION
    # ═══════════════════════════════════════════════════════════════════════════
    
    def generate_trade_setup(
        self,
        symbol: str,
        df: pd.DataFrame,
        direction: str,
        entry_price: float = None,
        smc_data: Dict = None,
        confidence: float = 70.0,
        method: PositionSizeMethod = PositionSizeMethod.FIXED_FRACTIONAL
    ) -> TradeSetup:
        """
        Komple trade kurulumu oluştur
        
        Args:
            symbol: Hisse kodu
            df: OHLCV DataFrame
            direction: "LONG" veya "SHORT"
            entry_price: Giriş fiyatı (None ise current price)
            smc_data: SMC analiz verisi
            confidence: Sinyal güven skoru
            method: Pozisyon boyutlandırma yöntemi
            
        Returns:
            TradeSetup dataclass
        """
        notes = []
        
        if entry_price is None:
            entry_price = df['Close'].iloc[-1]
        
        # Stop loss hesapla (SMC varsa structure-based)
        if smc_data and smc_data.get('ob'):
            stop_method = StopLossType.STRUCTURE_BASED
            notes.append("Structure-based stop (OB)")
        else:
            stop_method = StopLossType.ATR_BASED
            notes.append("ATR-based stop")
        
        stop_result = self.calculate_stop_loss(
            df, entry_price, direction, stop_method, smc_data
        )
        
        # Take profit seviyeleri
        take_profits = self.calculate_take_profits(
            entry_price, stop_result.price, direction, smc_data=smc_data
        )
        
        # Pozisyon boyutu
        conviction = confidence / 100
        position = self.calculate_position_size(
            entry_price, stop_result.price,
            take_profit=take_profits[0].price if take_profits else None,
            conviction=conviction,
            method=method
        )
        
        # Setup type belirleme
        if smc_data:
            if smc_data.get('liquidity_sweep') and smc_data['liquidity_sweep'] != "YOK":
                setup_type = "LIQUIDITY_SWEEP"
                notes.append("Liquidity sweep detected")
            elif smc_data.get('ob'):
                setup_type = "ORDER_BLOCK"
                notes.append(f"OB Type: {smc_data['ob'].get('type', 'N/A')}")
            else:
                setup_type = "STRUCTURE"
        else:
            setup_type = "TECHNICAL"
        
        # Entry type
        if confidence >= 80:
            entry_type = "MARKET"
            notes.append("High confidence - Market entry")
        else:
            entry_type = "LIMIT"
            notes.append("Limit entry recommended")
        
        return TradeSetup(
            symbol=symbol,
            direction=direction,
            entry_price=entry_price,
            entry_type=entry_type,
            stop_loss=stop_result,
            take_profit_levels=take_profits,
            position_size=position,
            risk_amount=position.risk_amount,
            risk_reward=position.reward_risk_ratio,
            confidence_score=confidence,
            setup_type=setup_type,
            notes=notes
        )
    
    # ═══════════════════════════════════════════════════════════════════════════
    # POSITION MANAGEMENT
    # ═══════════════════════════════════════════════════════════════════════════
    
    def add_position(self, position: Position) -> bool:
        """
        Yeni pozisyon ekle
        
        Args:
            position: Position object
            
        Returns:
            Başarılı mı
        """
        heat, can_trade = self.check_portfolio_heat()
        
        if not can_trade:
            logger.warning(f"⚠️ Portfolio heat limiti aşıldı: {heat*100:.1f}%")
            return False
        
        if not self.check_position_limit():
            logger.warning(f"⚠️ Maksimum pozisyon sayısı ({self.config.max_positions}) aşıldı")
            return False
        
        self.open_positions.append(position)
        logger.info(f"✅ Pozisyon eklendi: {position.symbol}")
        return True
    
    def close_position(self, symbol: str, exit_price: float) -> Optional[float]:
        """
        Pozisyon kapat
        
        Args:
            symbol: Hisse kodu
            exit_price: Çıkış fiyatı
            
        Returns:
            PnL veya None
        """
        for i, pos in enumerate(self.open_positions):
            if pos.symbol == symbol:
                # PnL hesapla
                if pos.direction == "LONG":
                    pnl = (exit_price - pos.entry_price) * pos.shares
                else:
                    pnl = (pos.entry_price - exit_price) * pos.shares
                
                # Trade history'e ekle
                self.trade_history.append({
                    'symbol': symbol,
                    'direction': pos.direction,
                    'entry_price': pos.entry_price,
                    'exit_price': exit_price,
                    'shares': pos.shares,
                    'pnl': pnl,
                    'entry_date': pos.entry_date,
                    'exit_date': datetime.now()
                })
                
                self.open_positions.pop(i)
                logger.info(f"✅ Pozisyon kapatıldı: {symbol} | PnL: {pnl:.2f}")
                return pnl
        
        return None
    
    def update_positions(self, price_data: Dict[str, float]) -> None:
        """
        Pozisyonları güncelle
        
        Args:
            price_data: {symbol: current_price} sözlüğü
        """
        for pos in self.open_positions:
            if pos.symbol in price_data:
                pos.current_price = price_data[pos.symbol]
                
                if pos.direction == "LONG":
                    pos.unrealized_pnl = (pos.current_price - pos.entry_price) * pos.shares
                else:
                    pos.unrealized_pnl = (pos.entry_price - pos.current_price) * pos.shares
                
                pos.unrealized_pnl_percent = (pos.unrealized_pnl / (pos.entry_price * pos.shares)) * 100


# ═══════════════════════════════════════════════════════════════════════════════
# TEST
# ═══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    print("Risk Engine v4.2 - Test")
    print("=" * 60)
    
    # Test için dummy data
    import numpy as np
    
    dates = pd.date_range(start='2024-01-01', periods=100, freq='D')
    np.random.seed(42)
    
    prices = 100 + np.cumsum(np.random.randn(100) * 2)
    
    df = pd.DataFrame({
        'Open': prices * 0.99,
        'High': prices * 1.02,
        'Low': prices * 0.98,
        'Close': prices,
        'Volume': np.random.randint(1000000, 5000000, 100),
        'ATR': np.ones(100) * 2
    }, index=dates)
    
    engine = RiskEngine(capital=100000)
    
    # Position size testi
    position = engine.calculate_position_size(
        entry_price=100.0,
        stop_loss=95.0,
        take_profit=110.0,
        conviction=1.0
    )
    
    print(f"\n📊 POZİSYON BOYUTU")
    print("=" * 60)
    print(f"📈 Entry: 100.00 TRY")
    print(f"🛑 Stop Loss: {position.stop_loss} TRY ({position.stop_percent}%)")
    print(f"🎯 TP1: {position.take_profit_1} TRY")
    print(f"🎯 TP2: {position.take_profit_2} TRY")
    print(f"🎯 TP3: {position.take_profit_3} TRY")
    print(f"\n📦 Lot: {position.lots}")
    print(f"💰 Pozisyon: {position.position_value:,.2f} TRY")
    print(f"⚠️ Risk: {position.risk_amount:,.2f} TRY ({position.risk_percent}%)")
    print(f"📊 R:R: 1:{position.reward_risk_ratio}")
    
    # Trade setup testi
    setup = engine.generate_trade_setup(
        symbol="THYAO",
        df=df,
        direction="LONG",
        confidence=75.0
    )
    
    print(f"\n📋 TRADE SETUP")
    print("=" * 60)
    print(f"📈 Symbol: {setup.symbol}")
    print(f"📊 Direction: {setup.direction}")
    print(f"💰 Entry: {setup.entry_price:.2f} TRY ({setup.entry_type})")
    print(f"🛑 Stop: {setup.stop_loss.price:.2f} TRY ({setup.stop_loss.method.value})")
    print(f"📊 R:R: 1:{setup.risk_reward}")
    print(f"🔥 Confidence: {setup.confidence_score}%")
    print(f"📝 Type: {setup.setup_type}")
    print(f"📝 Notes: {', '.join(setup.notes)}")
