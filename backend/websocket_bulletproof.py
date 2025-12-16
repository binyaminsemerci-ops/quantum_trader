"""
WebSocket Bulletproofing - Never crash from WebSocket failures
Handles disconnections, reconnections, and error scenarios gracefully

Key Features:
- Automatic reconnection with exponential backoff
- Heartbeat/ping-pong for connection health
- Message queue for offline periods
- Error recovery and logging
- Multiple connection management
"""

import asyncio
import json
import logging
from typing import Optional, Dict, Any, Callable, List, Coroutine
from datetime import datetime, timedelta
from enum import Enum
from dataclasses import dataclass, field
import aiohttp

logger = logging.getLogger(__name__)


class WSState(Enum):
    """WebSocket connection states"""
    DISCONNECTED = "disconnected"
    CONNECTING = "connecting"
    CONNECTED = "connected"
    RECONNECTING = "reconnecting"
    CLOSED = "closed"


@dataclass
class WSStats:
    """WebSocket connection statistics"""
    total_connections: int = 0
    successful_connections: int = 0
    failed_connections: int = 0
    total_messages_received: int = 0
    total_messages_sent: int = 0
    total_errors: int = 0
    last_connected: Optional[datetime] = None
    last_disconnected: Optional[datetime] = None
    last_error: Optional[str] = None
    reconnection_attempts: int = 0
    
    def connection_success_rate(self) -> float:
        """Calculate connection success rate"""
        if self.total_connections == 0:
            return 1.0
        return self.successful_connections / self.total_connections


class BulletproofWebSocket:
    """
    Bulletproof WebSocket client with automatic reconnection
    Never crashes, always tries to maintain connection
    """
    
    def __init__(
        self,
        url: str,
        name: str = "WebSocket",
        max_reconnect_attempts: int = -1,  # -1 = infinite
        reconnect_delay: float = 1.0,
        max_reconnect_delay: float = 60.0,
        backoff_factor: float = 2.0,
        ping_interval: float = 30.0,
        ping_timeout: float = 10.0,
        message_queue_size: int = 1000,
        on_message: Optional[Callable[[Dict[str, Any]], Coroutine]] = None,
        on_connect: Optional[Callable[[], Coroutine]] = None,
        on_disconnect: Optional[Callable[[], Coroutine]] = None,
        on_error: Optional[Callable[[Exception], Coroutine]] = None
    ):
        """
        Initialize bulletproof WebSocket client
        
        Args:
            url: WebSocket URL to connect to
            name: Name for logging
            max_reconnect_attempts: Max reconnection attempts (-1 = infinite)
            reconnect_delay: Initial reconnection delay in seconds
            max_reconnect_delay: Maximum reconnection delay
            backoff_factor: Exponential backoff multiplier
            ping_interval: Seconds between ping messages
            ping_timeout: Timeout for ping response
            message_queue_size: Max messages to queue when disconnected
            on_message: Callback for received messages
            on_connect: Callback when connected
            on_disconnect: Callback when disconnected
            on_error: Callback for errors
        """
        self.url = url
        self.name = name
        self.max_reconnect_attempts = max_reconnect_attempts
        self.reconnect_delay = reconnect_delay
        self.max_reconnect_delay = max_reconnect_delay
        self.backoff_factor = backoff_factor
        self.ping_interval = ping_interval
        self.ping_timeout = ping_timeout
        self.message_queue_size = message_queue_size
        
        # Callbacks
        self.on_message = on_message
        self.on_connect = on_connect
        self.on_disconnect = on_disconnect
        self.on_error = on_error
        
        # State
        self.state = WSState.DISCONNECTED
        self.ws: Optional[aiohttp.ClientWebSocketResponse] = None
        self.session: Optional[aiohttp.ClientSession] = None
        self.stats = WSStats()
        
        # Message queue for offline periods
        self.message_queue: List[str] = []
        
        # Tasks
        self._receive_task: Optional[asyncio.Task] = None
        self._ping_task: Optional[asyncio.Task] = None
        self._reconnect_task: Optional[asyncio.Task] = None
        
        # Control
        self._should_reconnect = True
        self._last_pong = datetime.now()
    
    async def connect(self) -> bool:
        """
        Connect to WebSocket
        
        Returns:
            True if connected successfully, False otherwise
        """
        if self.state == WSState.CONNECTED:
            logger.warning(f"{self.name}: Already connected")
            return True
        
        self.state = WSState.CONNECTING
        self.stats.total_connections += 1
        
        try:
            logger.info(f"{self.name}: Connecting to {self.url}")
            
            self.session = aiohttp.ClientSession()
            self.ws = await self.session.ws_connect(
                self.url,
                heartbeat=self.ping_interval,
                timeout=aiohttp.ClientTimeout(total=30)
            )
            
            self.state = WSState.CONNECTED
            self.stats.successful_connections += 1
            self.stats.last_connected = datetime.now()
            self.stats.reconnection_attempts = 0
            self._last_pong = datetime.now()
            
            logger.info(f"{self.name}: Connected successfully")
            
            # Start background tasks
            self._receive_task = asyncio.create_task(self._receive_loop())
            self._ping_task = asyncio.create_task(self._ping_loop())
            
            # Call on_connect callback
            if self.on_connect:
                try:
                    await self.on_connect()
                except Exception as e:
                    logger.error(f"{self.name}: Error in on_connect callback: {e}")
            
            # Send queued messages
            await self._send_queued_messages()
            
            return True
            
        except Exception as e:
            self.state = WSState.DISCONNECTED
            self.stats.failed_connections += 1
            self.stats.total_errors += 1
            self.stats.last_error = str(e)
            
            logger.error(f"{self.name}: Connection failed: {e}")
            
            # Call on_error callback
            if self.on_error:
                try:
                    await self.on_error(e)
                except Exception as callback_error:
                    logger.error(f"{self.name}: Error in on_error callback: {callback_error}")
            
            # Cleanup
            await self._cleanup()
            
            return False
    
    async def disconnect(self):
        """Gracefully disconnect from WebSocket"""
        logger.info(f"{self.name}: Disconnecting")
        
        self._should_reconnect = False
        self.state = WSState.CLOSED
        
        # Cancel tasks
        if self._receive_task:
            self._receive_task.cancel()
        if self._ping_task:
            self._ping_task.cancel()
        if self._reconnect_task:
            self._reconnect_task.cancel()
        
        # Close connection
        await self._cleanup()
        
        # Call on_disconnect callback
        if self.on_disconnect:
            try:
                await self.on_disconnect()
            except Exception as e:
                logger.error(f"{self.name}: Error in on_disconnect callback: {e}")
    
    async def send(self, data: Dict[str, Any]) -> bool:
        """
        Send message to WebSocket
        
        Args:
            data: Data to send (will be JSON encoded)
            
        Returns:
            True if sent successfully, False otherwise
        """
        try:
            if self.state != WSState.CONNECTED or not self.ws:
                # Queue message if disconnected
                if len(self.message_queue) < self.message_queue_size:
                    self.message_queue.append(json.dumps(data))
                    logger.debug(f"{self.name}: Message queued (offline)")
                else:
                    logger.warning(f"{self.name}: Message queue full, dropping message")
                return False
            
            await self.ws.send_json(data)
            self.stats.total_messages_sent += 1
            logger.debug(f"{self.name}: Message sent")
            return True
            
        except Exception as e:
            logger.error(f"{self.name}: Error sending message: {e}")
            self.stats.total_errors += 1
            
            # Queue message for retry
            if len(self.message_queue) < self.message_queue_size:
                self.message_queue.append(json.dumps(data))
            
            return False
    
    async def _receive_loop(self):
        """Background task to receive messages"""
        try:
            logger.info(f"{self.name}: Starting receive loop")
            
            async for msg in self.ws:
                if msg.type == aiohttp.WSMsgType.TEXT:
                    try:
                        data = json.loads(msg.data)
                        self.stats.total_messages_received += 1
                        
                        # Call on_message callback
                        if self.on_message:
                            try:
                                await self.on_message(data)
                            except Exception as e:
                                logger.error(f"{self.name}: Error in on_message callback: {e}")
                                self.stats.total_errors += 1
                    
                    except json.JSONDecodeError as e:
                        logger.error(f"{self.name}: Invalid JSON received: {e}")
                        self.stats.total_errors += 1
                
                elif msg.type == aiohttp.WSMsgType.PONG:
                    self._last_pong = datetime.now()
                    logger.debug(f"{self.name}: Pong received")
                
                elif msg.type == aiohttp.WSMsgType.ERROR:
                    logger.error(f"{self.name}: WebSocket error: {msg.data}")
                    self.stats.total_errors += 1
                    break
                
                elif msg.type == aiohttp.WSMsgType.CLOSED:
                    logger.warning(f"{self.name}: WebSocket closed by server")
                    break
        
        except asyncio.CancelledError:
            logger.info(f"{self.name}: Receive loop cancelled")
        
        except Exception as e:
            logger.error(f"{self.name}: Error in receive loop: {e}")
            self.stats.total_errors += 1
        
        finally:
            logger.info(f"{self.name}: Receive loop ended")
            await self._handle_disconnection()
    
    async def _ping_loop(self):
        """Background task to send ping messages"""
        try:
            logger.info(f"{self.name}: Starting ping loop")
            
            while self.state == WSState.CONNECTED:
                await asyncio.sleep(self.ping_interval)
                
                # Check if last pong is too old
                time_since_pong = (datetime.now() - self._last_pong).total_seconds()
                
                if time_since_pong > self.ping_timeout + self.ping_interval:
                    logger.warning(f"{self.name}: Ping timeout ({time_since_pong:.1f}s since last pong)")
                    self.stats.total_errors += 1
                    break
                
                # Send ping
                if self.ws and not self.ws.closed:
                    try:
                        await self.ws.ping()
                        logger.debug(f"{self.name}: Ping sent")
                    except Exception as e:
                        logger.error(f"{self.name}: Error sending ping: {e}")
                        break
        
        except asyncio.CancelledError:
            logger.info(f"{self.name}: Ping loop cancelled")
        
        except Exception as e:
            logger.error(f"{self.name}: Error in ping loop: {e}")
            self.stats.total_errors += 1
        
        finally:
            logger.info(f"{self.name}: Ping loop ended")
    
    async def _handle_disconnection(self):
        """Handle disconnection and trigger reconnection"""
        if self.state == WSState.CLOSED:
            return
        
        self.state = WSState.DISCONNECTED
        self.stats.last_disconnected = datetime.now()
        
        logger.warning(f"{self.name}: Disconnected")
        
        # Cleanup
        await self._cleanup()
        
        # Call on_disconnect callback
        if self.on_disconnect:
            try:
                await self.on_disconnect()
            except Exception as e:
                logger.error(f"{self.name}: Error in on_disconnect callback: {e}")
        
        # Trigger reconnection
        if self._should_reconnect:
            self._reconnect_task = asyncio.create_task(self._reconnect_loop())
    
    async def _reconnect_loop(self):
        """Background task to handle reconnection"""
        try:
            self.state = WSState.RECONNECTING
            current_delay = self.reconnect_delay
            attempt = 0
            
            while self._should_reconnect:
                if self.max_reconnect_attempts > 0 and attempt >= self.max_reconnect_attempts:
                    logger.error(f"{self.name}: Max reconnection attempts reached")
                    self.state = WSState.CLOSED
                    break
                
                attempt += 1
                self.stats.reconnection_attempts += 1
                
                logger.info(f"{self.name}: Reconnection attempt {attempt} (delay: {current_delay:.1f}s)")
                
                await asyncio.sleep(current_delay)
                
                success = await self.connect()
                
                if success:
                    logger.info(f"{self.name}: Reconnected successfully after {attempt} attempts")
                    break
                
                # Exponential backoff
                current_delay = min(current_delay * self.backoff_factor, self.max_reconnect_delay)
        
        except asyncio.CancelledError:
            logger.info(f"{self.name}: Reconnection loop cancelled")
        
        except Exception as e:
            logger.error(f"{self.name}: Error in reconnection loop: {e}")
            self.stats.total_errors += 1
    
    async def _send_queued_messages(self):
        """Send queued messages after reconnection"""
        if not self.message_queue:
            return
        
        logger.info(f"{self.name}: Sending {len(self.message_queue)} queued messages")
        
        while self.message_queue:
            try:
                msg_str = self.message_queue.pop(0)
                data = json.loads(msg_str)
                await self.send(data)
            except Exception as e:
                logger.error(f"{self.name}: Error sending queued message: {e}")
    
    async def _cleanup(self):
        """Cleanup connection resources"""
        if self.ws and not self.ws.closed:
            try:
                await self.ws.close()
            except Exception as e:
                logger.debug(f"{self.name}: Error closing WebSocket: {e}")
        
        if self.session and not self.session.closed:
            try:
                await self.session.close()
            except Exception as e:
                logger.debug(f"{self.name}: Error closing session: {e}")
        
        self.ws = None
        self.session = None
    
    def get_stats(self) -> Dict[str, Any]:
        """Get connection statistics"""
        return {
            'name': self.name,
            'state': self.state.value,
            'total_connections': self.stats.total_connections,
            'successful_connections': self.stats.successful_connections,
            'failed_connections': self.stats.failed_connections,
            'connection_success_rate': self.stats.connection_success_rate(),
            'total_messages_received': self.stats.total_messages_received,
            'total_messages_sent': self.stats.total_messages_sent,
            'total_errors': self.stats.total_errors,
            'reconnection_attempts': self.stats.reconnection_attempts,
            'last_connected': self.stats.last_connected.isoformat() if self.stats.last_connected else None,
            'last_disconnected': self.stats.last_disconnected.isoformat() if self.stats.last_disconnected else None,
            'last_error': self.stats.last_error,
            'queued_messages': len(self.message_queue)
        }


# Global WebSocket manager
_websocket_connections: Dict[str, BulletproofWebSocket] = {}


def create_bulletproof_websocket(
    url: str,
    name: str,
    **kwargs
) -> BulletproofWebSocket:
    """Create a bulletproof WebSocket connection"""
    ws = BulletproofWebSocket(url=url, name=name, **kwargs)
    _websocket_connections[name] = ws
    return ws


def get_websocket(name: str) -> Optional[BulletproofWebSocket]:
    """Get an existing WebSocket connection by name"""
    return _websocket_connections.get(name)


def get_all_websocket_stats() -> Dict[str, Any]:
    """Get statistics for all WebSocket connections"""
    return {
        name: ws.get_stats()
        for name, ws in _websocket_connections.items()
    }


async def cleanup_all_websockets():
    """Cleanup all WebSocket connections"""
    logger.info("Cleaning up all WebSocket connections")
    
    for name, ws in _websocket_connections.items():
        try:
            await ws.disconnect()
        except Exception as e:
            logger.error(f"Error disconnecting {name}: {e}")
    
    _websocket_connections.clear()
