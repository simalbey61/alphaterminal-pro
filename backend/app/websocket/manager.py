"""
AlphaTerminal Pro - WebSocket Manager v4.2
==========================================

Real-time data streaming ve event broadcasting

Özellikler:
- Fiyat güncellemeleri
- Sinyal bildirimleri
- Portföy değişiklikleri
- Market durumu
- Trade execution events

Author: AlphaTerminal Team
Version: 4.2.0
"""

import asyncio
import json
from datetime import datetime
from typing import Dict, List, Set, Optional, Any, Callable
from dataclasses import dataclass, field
from enum import Enum
import logging

from fastapi import WebSocket, WebSocketDisconnect
from starlette.websockets import WebSocketState

logger = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════════════════════════════════
# ENUMS & TYPES
# ═══════════════════════════════════════════════════════════════════════════════

class ChannelType(Enum):
    """WebSocket kanal türleri"""
    PRICES = "prices"              # Fiyat güncellemeleri
    SIGNALS = "signals"            # Sinyal bildirimleri
    PORTFOLIO = "portfolio"        # Portföy değişiklikleri
    MARKET = "market"              # Market durumu
    TRADES = "trades"              # Trade events
    ALERTS = "alerts"              # Risk/sistem uyarıları
    ANALYSIS = "analysis"          # Analiz güncellemeleri
    ALL = "all"                    # Tüm kanallar


class MessageType(Enum):
    """Mesaj türleri"""
    PRICE_UPDATE = "price_update"
    SIGNAL_NEW = "signal_new"
    SIGNAL_UPDATE = "signal_update"
    SIGNAL_TRIGGERED = "signal_triggered"
    TRADE_OPENED = "trade_opened"
    TRADE_CLOSED = "trade_closed"
    TRADE_UPDATE = "trade_update"
    PORTFOLIO_UPDATE = "portfolio_update"
    MARKET_STATUS = "market_status"
    ALERT = "alert"
    ERROR = "error"
    SUBSCRIBED = "subscribed"
    UNSUBSCRIBED = "unsubscribed"
    HEARTBEAT = "heartbeat"


# ═══════════════════════════════════════════════════════════════════════════════
# DATA CLASSES
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass
class WebSocketMessage:
    """WebSocket mesajı"""
    type: MessageType
    channel: ChannelType
    data: Dict[str, Any]
    timestamp: datetime = field(default_factory=datetime.now)
    
    def to_json(self) -> str:
        return json.dumps({
            "type": self.type.value,
            "channel": self.channel.value,
            "data": self.data,
            "timestamp": self.timestamp.isoformat()
        })


@dataclass
class ClientConnection:
    """Client bağlantısı"""
    websocket: WebSocket
    client_id: str
    user_id: Optional[str] = None
    subscriptions: Set[ChannelType] = field(default_factory=set)
    connected_at: datetime = field(default_factory=datetime.now)
    last_heartbeat: datetime = field(default_factory=datetime.now)
    
    @property
    def is_authenticated(self) -> bool:
        return self.user_id is not None


# ═══════════════════════════════════════════════════════════════════════════════
# CONNECTION MANAGER
# ═══════════════════════════════════════════════════════════════════════════════

class ConnectionManager:
    """
    WebSocket Connection Manager
    
    Tüm client bağlantılarını yönetir, mesaj broadcast eder
    """
    
    def __init__(self):
        # Active connections: client_id -> ClientConnection
        self._connections: Dict[str, ClientConnection] = {}
        
        # Channel subscriptions: channel -> set of client_ids
        self._channel_subscribers: Dict[ChannelType, Set[str]] = {
            channel: set() for channel in ChannelType
        }
        
        # Symbol subscriptions for prices: symbol -> set of client_ids
        self._symbol_subscribers: Dict[str, Set[str]] = {}
        
        # Message handlers
        self._handlers: Dict[str, Callable] = {}
        
        # Stats
        self._total_connections = 0
        self._total_messages_sent = 0
        
        # Heartbeat task
        self._heartbeat_task: Optional[asyncio.Task] = None
    
    # ═══════════════════════════════════════════════════════════════════════════
    # CONNECTION LIFECYCLE
    # ═══════════════════════════════════════════════════════════════════════════
    
    async def connect(
        self,
        websocket: WebSocket,
        client_id: str,
        user_id: Optional[str] = None
    ) -> ClientConnection:
        """
        Yeni client bağlantısı kabul et
        
        Args:
            websocket: WebSocket instance
            client_id: Unique client ID
            user_id: Optional authenticated user ID
            
        Returns:
            ClientConnection
        """
        await websocket.accept()
        
        connection = ClientConnection(
            websocket=websocket,
            client_id=client_id,
            user_id=user_id
        )
        
        self._connections[client_id] = connection
        self._total_connections += 1
        
        logger.info(f"Client connected: {client_id} (user: {user_id})")
        
        # Send welcome message
        await self.send_personal(client_id, WebSocketMessage(
            type=MessageType.SUBSCRIBED,
            channel=ChannelType.ALL,
            data={"message": "Connected to AlphaTerminal Pro", "client_id": client_id}
        ))
        
        return connection
    
    async def disconnect(self, client_id: str) -> None:
        """
        Client bağlantısını kapat
        
        Args:
            client_id: Client ID
        """
        if client_id not in self._connections:
            return
        
        connection = self._connections[client_id]
        
        # Remove from all channel subscriptions
        for channel in connection.subscriptions:
            self._channel_subscribers[channel].discard(client_id)
        
        # Remove from symbol subscriptions
        for subscribers in self._symbol_subscribers.values():
            subscribers.discard(client_id)
        
        # Remove connection
        del self._connections[client_id]
        
        logger.info(f"Client disconnected: {client_id}")
    
    # ═══════════════════════════════════════════════════════════════════════════
    # SUBSCRIPTIONS
    # ═══════════════════════════════════════════════════════════════════════════
    
    async def subscribe(
        self,
        client_id: str,
        channel: ChannelType,
        symbols: List[str] = None
    ) -> bool:
        """
        Client'ı kanala subscribe et
        
        Args:
            client_id: Client ID
            channel: Kanal türü
            symbols: Opsiyonel sembol listesi (prices kanalı için)
            
        Returns:
            Başarılı mı
        """
        if client_id not in self._connections:
            return False
        
        connection = self._connections[client_id]
        connection.subscriptions.add(channel)
        self._channel_subscribers[channel].add(client_id)
        
        # Symbol subscriptions for prices
        if channel == ChannelType.PRICES and symbols:
            for symbol in symbols:
                if symbol not in self._symbol_subscribers:
                    self._symbol_subscribers[symbol] = set()
                self._symbol_subscribers[symbol].add(client_id)
        
        logger.debug(f"Client {client_id} subscribed to {channel.value}")
        
        await self.send_personal(client_id, WebSocketMessage(
            type=MessageType.SUBSCRIBED,
            channel=channel,
            data={"channel": channel.value, "symbols": symbols}
        ))
        
        return True
    
    async def unsubscribe(
        self,
        client_id: str,
        channel: ChannelType,
        symbols: List[str] = None
    ) -> bool:
        """
        Client'ı kanaldan unsubscribe et
        
        Args:
            client_id: Client ID
            channel: Kanal türü
            symbols: Opsiyonel sembol listesi
            
        Returns:
            Başarılı mı
        """
        if client_id not in self._connections:
            return False
        
        connection = self._connections[client_id]
        connection.subscriptions.discard(channel)
        self._channel_subscribers[channel].discard(client_id)
        
        # Remove symbol subscriptions
        if channel == ChannelType.PRICES and symbols:
            for symbol in symbols:
                if symbol in self._symbol_subscribers:
                    self._symbol_subscribers[symbol].discard(client_id)
        
        logger.debug(f"Client {client_id} unsubscribed from {channel.value}")
        
        await self.send_personal(client_id, WebSocketMessage(
            type=MessageType.UNSUBSCRIBED,
            channel=channel,
            data={"channel": channel.value}
        ))
        
        return True
    
    # ═══════════════════════════════════════════════════════════════════════════
    # MESSAGE SENDING
    # ═══════════════════════════════════════════════════════════════════════════
    
    async def send_personal(
        self,
        client_id: str,
        message: WebSocketMessage
    ) -> bool:
        """
        Tek bir client'a mesaj gönder
        
        Args:
            client_id: Hedef client ID
            message: Gönderilecek mesaj
            
        Returns:
            Başarılı mı
        """
        if client_id not in self._connections:
            return False
        
        connection = self._connections[client_id]
        
        try:
            if connection.websocket.client_state == WebSocketState.CONNECTED:
                await connection.websocket.send_text(message.to_json())
                self._total_messages_sent += 1
                return True
        except Exception as e:
            logger.error(f"Failed to send message to {client_id}: {e}")
            await self.disconnect(client_id)
        
        return False
    
    async def broadcast_channel(
        self,
        channel: ChannelType,
        message: WebSocketMessage
    ) -> int:
        """
        Bir kanaldaki tüm subscriber'lara mesaj gönder
        
        Args:
            channel: Hedef kanal
            message: Gönderilecek mesaj
            
        Returns:
            Gönderilen mesaj sayısı
        """
        subscribers = self._channel_subscribers.get(channel, set())
        sent_count = 0
        
        for client_id in list(subscribers):  # Copy to avoid modification during iteration
            if await self.send_personal(client_id, message):
                sent_count += 1
        
        return sent_count
    
    async def broadcast_symbol(
        self,
        symbol: str,
        message: WebSocketMessage
    ) -> int:
        """
        Belirli bir sembole subscribe olan client'lara mesaj gönder
        
        Args:
            symbol: Hedef sembol
            message: Gönderilecek mesaj
            
        Returns:
            Gönderilen mesaj sayısı
        """
        subscribers = self._symbol_subscribers.get(symbol, set())
        sent_count = 0
        
        for client_id in list(subscribers):
            if await self.send_personal(client_id, message):
                sent_count += 1
        
        return sent_count
    
    async def broadcast_all(self, message: WebSocketMessage) -> int:
        """
        Tüm bağlı client'lara mesaj gönder
        
        Args:
            message: Gönderilecek mesaj
            
        Returns:
            Gönderilen mesaj sayısı
        """
        sent_count = 0
        
        for client_id in list(self._connections.keys()):
            if await self.send_personal(client_id, message):
                sent_count += 1
        
        return sent_count
    
    # ═══════════════════════════════════════════════════════════════════════════
    # CONVENIENCE METHODS
    # ═══════════════════════════════════════════════════════════════════════════
    
    async def broadcast_price_update(
        self,
        symbol: str,
        price: float,
        change: float,
        change_pct: float,
        volume: int
    ) -> int:
        """Fiyat güncellemesi broadcast et"""
        message = WebSocketMessage(
            type=MessageType.PRICE_UPDATE,
            channel=ChannelType.PRICES,
            data={
                "symbol": symbol,
                "price": price,
                "change": change,
                "change_pct": change_pct,
                "volume": volume
            }
        )
        return await self.broadcast_symbol(symbol, message)
    
    async def broadcast_new_signal(
        self,
        signal_id: str,
        symbol: str,
        direction: str,
        entry_price: float,
        confidence: float,
        **kwargs
    ) -> int:
        """Yeni sinyal bildirimi broadcast et"""
        message = WebSocketMessage(
            type=MessageType.SIGNAL_NEW,
            channel=ChannelType.SIGNALS,
            data={
                "signal_id": signal_id,
                "symbol": symbol,
                "direction": direction,
                "entry_price": entry_price,
                "confidence": confidence,
                **kwargs
            }
        )
        return await self.broadcast_channel(ChannelType.SIGNALS, message)
    
    async def broadcast_trade_event(
        self,
        event_type: str,
        trade_id: str,
        symbol: str,
        **kwargs
    ) -> int:
        """Trade event broadcast et"""
        msg_type = {
            "opened": MessageType.TRADE_OPENED,
            "closed": MessageType.TRADE_CLOSED,
            "update": MessageType.TRADE_UPDATE
        }.get(event_type, MessageType.TRADE_UPDATE)
        
        message = WebSocketMessage(
            type=msg_type,
            channel=ChannelType.TRADES,
            data={
                "trade_id": trade_id,
                "symbol": symbol,
                "event": event_type,
                **kwargs
            }
        )
        return await self.broadcast_channel(ChannelType.TRADES, message)
    
    async def broadcast_portfolio_update(
        self,
        total_value: float,
        daily_pnl: float,
        daily_pnl_pct: float,
        open_positions: int
    ) -> int:
        """Portföy güncellemesi broadcast et"""
        message = WebSocketMessage(
            type=MessageType.PORTFOLIO_UPDATE,
            channel=ChannelType.PORTFOLIO,
            data={
                "total_value": total_value,
                "daily_pnl": daily_pnl,
                "daily_pnl_pct": daily_pnl_pct,
                "open_positions": open_positions
            }
        )
        return await self.broadcast_channel(ChannelType.PORTFOLIO, message)
    
    async def broadcast_alert(
        self,
        alert_type: str,
        title: str,
        message_text: str,
        severity: str = "info"
    ) -> int:
        """Alert broadcast et"""
        message = WebSocketMessage(
            type=MessageType.ALERT,
            channel=ChannelType.ALERTS,
            data={
                "alert_type": alert_type,
                "title": title,
                "message": message_text,
                "severity": severity
            }
        )
        return await self.broadcast_channel(ChannelType.ALERTS, message)
    
    async def broadcast_market_status(
        self,
        is_open: bool,
        index_value: float,
        index_change: float
    ) -> int:
        """Market durumu broadcast et"""
        message = WebSocketMessage(
            type=MessageType.MARKET_STATUS,
            channel=ChannelType.MARKET,
            data={
                "is_open": is_open,
                "index_value": index_value,
                "index_change": index_change
            }
        )
        return await self.broadcast_channel(ChannelType.MARKET, message)
    
    # ═══════════════════════════════════════════════════════════════════════════
    # HEARTBEAT & MAINTENANCE
    # ═══════════════════════════════════════════════════════════════════════════
    
    async def start_heartbeat(self, interval: int = 30):
        """Heartbeat task başlat"""
        self._heartbeat_task = asyncio.create_task(self._heartbeat_loop(interval))
    
    async def stop_heartbeat(self):
        """Heartbeat task durdur"""
        if self._heartbeat_task:
            self._heartbeat_task.cancel()
            try:
                await self._heartbeat_task
            except asyncio.CancelledError:
                pass
    
    async def _heartbeat_loop(self, interval: int):
        """Heartbeat döngüsü"""
        while True:
            await asyncio.sleep(interval)
            
            message = WebSocketMessage(
                type=MessageType.HEARTBEAT,
                channel=ChannelType.ALL,
                data={"status": "alive", "connections": len(self._connections)}
            )
            
            await self.broadcast_all(message)
            
            # Clean up stale connections
            await self._cleanup_stale_connections()
    
    async def _cleanup_stale_connections(self, timeout: int = 120):
        """Stale bağlantıları temizle"""
        now = datetime.now()
        stale_clients = []
        
        for client_id, connection in self._connections.items():
            if (now - connection.last_heartbeat).seconds > timeout:
                stale_clients.append(client_id)
        
        for client_id in stale_clients:
            logger.warning(f"Removing stale connection: {client_id}")
            await self.disconnect(client_id)
    
    # ═══════════════════════════════════════════════════════════════════════════
    # STATS & INFO
    # ═══════════════════════════════════════════════════════════════════════════
    
    @property
    def active_connections(self) -> int:
        """Aktif bağlantı sayısı"""
        return len(self._connections)
    
    @property
    def total_connections(self) -> int:
        """Toplam bağlantı sayısı (historical)"""
        return self._total_connections
    
    @property
    def total_messages_sent(self) -> int:
        """Toplam gönderilen mesaj sayısı"""
        return self._total_messages_sent
    
    def get_stats(self) -> Dict[str, Any]:
        """İstatistikleri getir"""
        channel_stats = {
            channel.value: len(subscribers)
            for channel, subscribers in self._channel_subscribers.items()
        }
        
        return {
            "active_connections": self.active_connections,
            "total_connections": self.total_connections,
            "total_messages_sent": self.total_messages_sent,
            "channel_subscribers": channel_stats,
            "symbol_subscriptions": len(self._symbol_subscribers)
        }


# ═══════════════════════════════════════════════════════════════════════════════
# SINGLETON INSTANCE
# ═══════════════════════════════════════════════════════════════════════════════

_manager: Optional[ConnectionManager] = None


def get_connection_manager() -> ConnectionManager:
    """Global connection manager instance"""
    global _manager
    if _manager is None:
        _manager = ConnectionManager()
    return _manager


# ═══════════════════════════════════════════════════════════════════════════════
# WEBSOCKET HANDLER
# ═══════════════════════════════════════════════════════════════════════════════

async def websocket_handler(
    websocket: WebSocket,
    client_id: str,
    user_id: Optional[str] = None
):
    """
    WebSocket bağlantı handler
    
    Args:
        websocket: WebSocket instance
        client_id: Client ID
        user_id: Optional user ID
    """
    manager = get_connection_manager()
    
    try:
        connection = await manager.connect(websocket, client_id, user_id)
        
        while True:
            # Receive message
            data = await websocket.receive_text()
            
            try:
                message = json.loads(data)
                action = message.get("action")
                
                if action == "subscribe":
                    channel = ChannelType(message.get("channel", "all"))
                    symbols = message.get("symbols", [])
                    await manager.subscribe(client_id, channel, symbols)
                    
                elif action == "unsubscribe":
                    channel = ChannelType(message.get("channel", "all"))
                    symbols = message.get("symbols", [])
                    await manager.unsubscribe(client_id, channel, symbols)
                    
                elif action == "heartbeat":
                    connection.last_heartbeat = datetime.now()
                    await manager.send_personal(client_id, WebSocketMessage(
                        type=MessageType.HEARTBEAT,
                        channel=ChannelType.ALL,
                        data={"status": "alive"}
                    ))
                    
            except json.JSONDecodeError:
                await manager.send_personal(client_id, WebSocketMessage(
                    type=MessageType.ERROR,
                    channel=ChannelType.ALL,
                    data={"error": "Invalid JSON"}
                ))
                
    except WebSocketDisconnect:
        await manager.disconnect(client_id)
    except Exception as e:
        logger.error(f"WebSocket error for {client_id}: {e}")
        await manager.disconnect(client_id)
