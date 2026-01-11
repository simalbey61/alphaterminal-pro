"""
AlphaTerminal Pro - Shadow Mode System v4.2
============================================

Paper Trading / Simülasyon Sistemi

Author: AlphaTerminal Team
Version: 4.2.0
"""

import json
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, field, asdict
from enum import Enum
from pathlib import Path
import numpy as np

from app.core.config import logger, SHADOW_MODE_CONFIG, DATA_DIR


class ShadowTradeStatus(Enum):
    OPEN = "OPEN"
    CLOSED_TP1 = "CLOSED_TP1"
    CLOSED_TP2 = "CLOSED_TP2"
    CLOSED_TP3 = "CLOSED_TP3"
    CLOSED_SL = "CLOSED_SL"
    CLOSED_MANUAL = "CLOSED_MANUAL"
    EXPIRED = "EXPIRED"


class ShadowModeState(Enum):
    ACTIVE = "ACTIVE"
    PAUSED = "PAUSED"
    STOPPED = "STOPPED"


@dataclass
class ShadowTrade:
    trade_id: str
    signal_id: str
    symbol: str
    direction: str
    entry_price: float
    entry_time: datetime
    position_size: int
    position_value: float
    stop_loss: float
    take_profit_1: float
    take_profit_2: float
    take_profit_3: float
    current_price: float = 0.0
    unrealized_pnl: float = 0.0
    status: ShadowTradeStatus = ShadowTradeStatus.OPEN
    closed_at_tp1: bool = False
    closed_at_tp2: bool = False
    tp1_pnl: float = 0.0
    tp2_pnl: float = 0.0
    exit_price: Optional[float] = None
    exit_time: Optional[datetime] = None
    exit_reason: Optional[str] = None
    realized_pnl: float = 0.0
    entry_scores: Dict[str, float] = field(default_factory=dict)

    def update_price(self, price: float) -> None:
        self.current_price = price
        if self.direction == "LONG":
            self.unrealized_pnl = (price - self.entry_price) * self.position_size
        else:
            self.unrealized_pnl = (self.entry_price - price) * self.position_size


@dataclass
class ShadowPortfolio:
    initial_capital: float
    current_capital: float
    available_capital: float
    open_positions: int = 0
    total_exposure: float = 0.0
    realized_pnl: float = 0.0
    unrealized_pnl: float = 0.0
    total_pnl: float = 0.0
    total_pnl_pct: float = 0.0
    max_drawdown: float = 0.0
    peak_equity: float = 0.0
    total_trades: int = 0
    winning_trades: int = 0
    losing_trades: int = 0
    win_rate: float = 0.0
    profit_factor: float = 0.0
    pnl_by_symbol: Dict[str, float] = field(default_factory=dict)


@dataclass
class ShadowSession:
    session_id: str
    start_time: datetime
    end_time: Optional[datetime] = None
    state: ShadowModeState = ShadowModeState.ACTIVE
    initial_capital: float = 100000.0
    max_positions: int = 5
    risk_per_trade: float = 0.02
    signals_received: int = 0
    signals_executed: int = 0
    portfolio: ShadowPortfolio = None


class ShadowModeSystem:
    def __init__(self, config=None, data_dir: Path = None):
        self.config = config or SHADOW_MODE_CONFIG
        self.data_dir = data_dir or DATA_DIR / "shadow"
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self._session: Optional[ShadowSession] = None
        self._trades: Dict[str, ShadowTrade] = {}
        self._trade_counter = 0

    def _generate_trade_id(self) -> str:
        self._trade_counter += 1
        return f"SHD_{datetime.now().strftime('%Y%m%d%H%M%S')}_{self._trade_counter:04d}"

    def start_session(self, initial_capital: float = None) -> ShadowSession:
        session_id = f"SESSION_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        capital = initial_capital or self.config.initial_capital
        
        self._session = ShadowSession(
            session_id=session_id,
            start_time=datetime.now(),
            initial_capital=capital,
            portfolio=ShadowPortfolio(
                initial_capital=capital,
                current_capital=capital,
                available_capital=capital,
                peak_equity=capital
            )
        )
        logger.info(f"Shadow session started: {session_id}")
        return self._session

    @property
    def is_active(self) -> bool:
        return self._session is not None and self._session.state == ShadowModeState.ACTIVE

    def execute_signal(
        self, signal_id: str, symbol: str, direction: str,
        entry_price: float, stop_loss: float,
        take_profit_1: float, take_profit_2: float, take_profit_3: float,
        scores: Dict[str, float] = None
    ) -> Optional[ShadowTrade]:
        if not self.is_active:
            return None

        self._session.signals_received += 1
        
        open_count = len([t for t in self._trades.values() if t.status == ShadowTradeStatus.OPEN])
        if open_count >= self._session.max_positions:
            return None

        risk_amount = abs(entry_price - stop_loss)
        position_size = max(1, int(
            (self._session.portfolio.available_capital * self._session.risk_per_trade) / risk_amount
        ))
        position_value = position_size * entry_price

        if position_value > self._session.portfolio.available_capital:
            return None

        trade_id = self._generate_trade_id()
        trade = ShadowTrade(
            trade_id=trade_id, signal_id=signal_id, symbol=symbol,
            direction=direction, entry_price=entry_price,
            entry_time=datetime.now(), position_size=position_size,
            position_value=position_value, stop_loss=stop_loss,
            take_profit_1=take_profit_1, take_profit_2=take_profit_2,
            take_profit_3=take_profit_3, current_price=entry_price,
            entry_scores=scores or {}
        )

        self._trades[trade_id] = trade
        self._session.signals_executed += 1
        self._session.portfolio.available_capital -= position_value
        self._session.portfolio.open_positions += 1

        logger.info(f"Shadow trade: {symbol} {direction} @ {entry_price}")
        return trade

    def update_prices(self, price_data: Dict[str, float]) -> List[ShadowTrade]:
        if not self.is_active:
            return []

        closed = []
        for trade in list(self._trades.values()):
            if trade.status != ShadowTradeStatus.OPEN:
                continue
            if trade.symbol not in price_data:
                continue

            price = price_data[trade.symbol]
            trade.update_price(price)

            # Check SL/TP
            if trade.direction == "LONG":
                if price <= trade.stop_loss:
                    self._close_trade(trade.trade_id, price, "STOP_LOSS")
                    closed.append(trade)
                elif price >= trade.take_profit_3:
                    self._close_trade(trade.trade_id, price, "TP3")
                    closed.append(trade)
            else:
                if price >= trade.stop_loss:
                    self._close_trade(trade.trade_id, price, "STOP_LOSS")
                    closed.append(trade)
                elif price <= trade.take_profit_3:
                    self._close_trade(trade.trade_id, price, "TP3")
                    closed.append(trade)

        self._update_portfolio()
        return closed

    def _close_trade(self, trade_id: str, exit_price: float, reason: str):
        trade = self._trades.get(trade_id)
        if not trade:
            return

        if trade.direction == "LONG":
            pnl = (exit_price - trade.entry_price) * trade.position_size
        else:
            pnl = (trade.entry_price - exit_price) * trade.position_size

        trade.exit_price = exit_price
        trade.exit_time = datetime.now()
        trade.exit_reason = reason
        trade.realized_pnl = pnl
        trade.status = ShadowTradeStatus.CLOSED_SL if "SL" in reason else ShadowTradeStatus.CLOSED_TP3

        self._session.portfolio.available_capital += trade.position_value + pnl
        self._session.portfolio.realized_pnl += pnl
        self._session.portfolio.open_positions -= 1
        self._session.portfolio.total_trades += 1

        if pnl > 0:
            self._session.portfolio.winning_trades += 1
        else:
            self._session.portfolio.losing_trades += 1

        logger.info(f"Shadow closed: {trade.symbol} @ {exit_price} | PnL: {pnl:+.2f}")

    def _update_portfolio(self):
        if not self._session:
            return
        
        p = self._session.portfolio
        p.unrealized_pnl = sum(t.unrealized_pnl for t in self._trades.values() if t.status == ShadowTradeStatus.OPEN)
        p.total_pnl = p.realized_pnl + p.unrealized_pnl
        p.total_pnl_pct = (p.total_pnl / p.initial_capital) * 100
        p.current_capital = p.initial_capital + p.total_pnl

        if p.current_capital > p.peak_equity:
            p.peak_equity = p.current_capital
        
        if p.peak_equity > 0:
            dd = ((p.peak_equity - p.current_capital) / p.peak_equity) * 100
            p.max_drawdown = max(p.max_drawdown, dd)

        total = p.winning_trades + p.losing_trades
        if total > 0:
            p.win_rate = (p.winning_trades / total) * 100

    def get_portfolio(self) -> Optional[ShadowPortfolio]:
        return self._session.portfolio if self._session else None

    def get_open_trades(self) -> List[ShadowTrade]:
        return [t for t in self._trades.values() if t.status == ShadowTradeStatus.OPEN]

    def stop_session(self) -> Optional[ShadowSession]:
        if not self._session:
            return None
        
        self._session.state = ShadowModeState.STOPPED
        self._session.end_time = datetime.now()
        
        for trade in self.get_open_trades():
            self._close_trade(trade.trade_id, trade.current_price, "SESSION_END")
        
        logger.info(f"Shadow stopped: PnL={self._session.portfolio.total_pnl:+.2f}")
        return self._session


_shadow_system: Optional[ShadowModeSystem] = None

def get_shadow_system() -> ShadowModeSystem:
    global _shadow_system
    if _shadow_system is None:
        _shadow_system = ShadowModeSystem()
    return _shadow_system


if __name__ == "__main__":
    print("Shadow Mode System v4.2 - Test")
    system = ShadowModeSystem()
    session = system.start_session(100000)
    
    trade = system.execute_signal(
        "SIG_001", "THYAO", "LONG", 145.5, 140.0, 155.0, 165.0, 175.0
    )
    print(f"Trade: {trade.trade_id if trade else 'None'}")
    
    system.update_prices({"THYAO": 160.0})
    print(f"Portfolio PnL: {system.get_portfolio().total_pnl:+.2f}")
    
    system.stop_session()
