"""
Dashboard WebSocket Handler
Sprint 4

Real-time event streaming for dashboard updates.
"""

import logging
import asyncio
import json
from fastapi import APIRouter, WebSocket, WebSocketDisconnect
from datetime import datetime, timezone
from typing import Set, Dict, Any

from backend.api.dashboard.models import (
    DashboardEvent,
    create_position_updated_event,
    create_pnl_updated_event,
    create_signal_generated_event,
    create_ess_state_changed_event,
    create_health_alert_event,
    EventType
)


logger = logging.getLogger(__name__)

router = APIRouter(tags=["dashboard"])


# ========== WEBSOCKET CONNECTION MANAGER ==========

class DashboardConnectionManager:
    """
    Manages WebSocket connections for dashboard clients.
    
    Features:
    - Multiple client connections
    - Broadcast events to all clients
    - Auto-cleanup on disconnect
    - [SPRINT 5 - PATCH #6] Event batching for high-load scenarios
    """
    
    def __init__(self):
        self.active_connections: Set[WebSocket] = set()
        self._lock = asyncio.Lock()
        
        # [SPRINT 5 - PATCH #6] Event batching for flood protection
        self._event_batch: list = []
        self._batch_size = 10  # Batch events in groups of 10
        self._batch_interval = 0.1  # Send batch every 100ms
        self._last_batch_send = asyncio.get_event_loop().time()
        self._max_events_per_second = 50  # Rate limit: 50 events/sec per client
        self._event_send_times: Dict[float, int] = {}  # timestamp -> count
        logger.info("[PATCH #6] WebSocket event batching enabled: batch_size=10, max_rate=50/sec")
    
    async def connect(self, websocket: WebSocket):
        """Accept and register new client."""
        await websocket.accept()
        async with self._lock:
            self.active_connections.add(websocket)
        logger.info(f"[DASHBOARD-WS] Client connected. Total: {len(self.active_connections)}")
    
    async def disconnect(self, websocket: WebSocket):
        """Remove client from active connections."""
        async with self._lock:
            self.active_connections.discard(websocket)
        logger.info(f"[DASHBOARD-WS] Client disconnected. Total: {len(self.active_connections)}")
    
    async def broadcast(self, event: DashboardEvent):
        """
        Broadcast event to all connected clients with batching.
        
        Args:
            event: DashboardEvent to broadcast
        """
        if not self.active_connections:
            return
        
        # [SPRINT 5 - PATCH #6] Add event to batch
        self._event_batch.append(event)
        
        # Check if we should send batch (either size or time threshold reached)
        now = asyncio.get_event_loop().time()
        should_send_batch = (
            len(self._event_batch) >= self._batch_size or
            (now - self._last_batch_send) >= self._batch_interval
        )
        
        if not should_send_batch:
            return
        
        # Rate limiting: Clean old timestamps (older than 1 second)
        cutoff_time = now - 1.0
        self._event_send_times = {t: c for t, c in self._event_send_times.items() if t > cutoff_time}
        
        # Count events in last second
        events_last_second = sum(self._event_send_times.values())
        
        # Check rate limit
        if events_last_second >= self._max_events_per_second:
            dropped_count = len(self._event_batch)
            logger.warning(
                f"[THROTTLE] Rate limit reached ({events_last_second}/{self._max_events_per_second}/sec), "
                f"dropping {dropped_count} events"
            )
            self._event_batch.clear()
            return
        
        # Send batch
        if self._event_batch:
            # Create batched message
            batch_message = json.dumps({
                "type": "event_batch",
                "count": len(self._event_batch),
                "events": [e.to_dict() for e in self._event_batch],
                "timestamp": datetime.now(timezone.utc).isoformat()
            })
            
            disconnected = set()
            
            async with self._lock:
                for connection in self.active_connections:
                    try:
                        await connection.send_text(batch_message)
                    except Exception as e:
                        logger.warning(f"[DASHBOARD-WS] Failed to send to client: {e}")
                        disconnected.add(connection)
                
                # Clean up disconnected clients
                for connection in disconnected:
                    self.active_connections.discard(connection)
            
            # Record send time for rate limiting
            self._event_send_times[now] = len(self._event_batch)
            
            # Clear batch
            self._event_batch.clear()
            self._last_batch_send = now
    
    @property
    def connection_count(self) -> int:
        """Get number of active connections."""
        return len(self.active_connections)


# Global connection manager
manager = DashboardConnectionManager()


# ========== EVENT BUS INTEGRATION ==========

async def subscribe_to_events(callback):
    """
    Subscribe to internal EventBus for dashboard-relevant events.
    
    Events to listen for:
    - position.updated (from execution-service)
    - pnl.updated (from portfolio-intelligence)
    - signal.generated (from ai-engine)
    - ess.state_changed (from risk-safety)
    - health.alert (from monitoring-health)
    - trade.executed (from execution-service)
    - order.placed (from execution-service)
    
    Args:
        callback: Async function to call with DashboardEvent
    
    Note: This is a placeholder. Real implementation should use
    the actual EventBus from infrastructure/event_bus.py.
    
    TODO: Integrate with real EventBus subscription.
    """
    logger.info("[DASHBOARD-WS] Subscribing to EventBus")
    
    # Mock implementation for now
    # In production, use:
    #   from infrastructure.event_bus import EventBus
    #   bus = EventBus()
    #   await bus.subscribe("position.updated", callback)
    #   await bus.subscribe("pnl.updated", callback)
    #   etc.
    
    # For testing, simulate events every 10 seconds
    while True:
        await asyncio.sleep(10)
        
        # Mock event: position updated
        mock_event = DashboardEvent(
            type=EventType.POSITION_UPDATED,
            timestamp=datetime.now(timezone.utc).isoformat(),
            payload={
                "symbol": "BTCUSDT",
                "side": "LONG",
                "size": 0.5,
                "unrealized_pnl": 125.50
            }
        )
        
        await callback(mock_event)


async def handle_internal_event(event_type: str, event_data: Dict[str, Any]):
    """
    Handle internal event from EventBus and convert to DashboardEvent.
    
    Args:
        event_type: Event type from EventBus (e.g., "position.updated")
        event_data: Event payload
    """
    logger.debug(f"[DASHBOARD-WS] Received internal event: {event_type}")
    
    dashboard_event = None
    
    # Convert to DashboardEvent based on type
    if event_type == "position.updated":
        # Extract position data from event_data
        # In production, event_data should match Position model
        dashboard_event = create_position_updated_event(event_data)
    
    elif event_type == "pnl.updated":
        # Extract portfolio data from event_data
        dashboard_event = create_pnl_updated_event(event_data)
    
    elif event_type == "signal.generated":
        # Extract signal data from event_data
        dashboard_event = create_signal_generated_event(event_data)
    
    elif event_type == "ess.state_changed":
        # Extract ESS state and reason
        new_state = event_data.get("state", "UNKNOWN")
        reason = event_data.get("reason")
        dashboard_event = create_ess_state_changed_event(new_state, reason)
    
    elif event_type == "health.alert":
        # Extract health alert data
        service = event_data.get("service", "unknown")
        status = event_data.get("status", "UNKNOWN")
        message = event_data.get("message", "")
        dashboard_event = create_health_alert_event(service, status, message)
    
    elif event_type in ["trade.executed", "order.placed"]:
        # Create generic event
        dashboard_event = DashboardEvent(
            type=EventType(event_type.replace(".", "_")),
            timestamp=datetime.now(timezone.utc).isoformat(),
            payload=event_data
        )
    
    # Broadcast to all clients
    if dashboard_event:
        await manager.broadcast(dashboard_event)


# ========== WEBSOCKET ENDPOINT ==========

@router.websocket("/ws/dashboard")
async def dashboard_websocket(websocket: WebSocket):
    """
    WebSocket endpoint for real-time dashboard updates.
    
    Protocol:
    1. Client connects
    2. Server sends heartbeat every 30 seconds
    3. Server broadcasts events as they occur
    4. Client can send ping to keep connection alive
    
    Event types:
    - position_updated: Position size/PnL changed
    - pnl_updated: Portfolio PnL changed
    - signal_generated: New AI signal
    - ess_state_changed: ESS tripped/armed
    - health_alert: Service health degraded
    - trade_executed: Trade filled
    - order_placed: New order submitted
    
    Example message:
    {
        "type": "position_updated",
        "timestamp": "2025-01-27T10:30:00Z",
        "payload": {
            "symbol": "BTCUSDT",
            "side": "LONG",
            "size": 0.5,
            "unrealized_pnl": 125.50
        }
    }
    """
    await manager.connect(websocket)
    
    try:
        # Start event subscription task
        event_task = asyncio.create_task(
            subscribe_to_events(lambda event: manager.broadcast(event))
        )
        
        # Send initial connection confirmation
        await websocket.send_json({
            "type": "connected",
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "message": "Dashboard WebSocket connected"
        })
        
        # Keep connection alive and handle client messages
        while True:
            try:
                # Wait for client message (ping/pong)
                data = await asyncio.wait_for(websocket.receive_text(), timeout=60.0)
                
                # Handle ping
                if data == "ping":
                    await websocket.send_json({
                        "type": "pong",
                        "timestamp": datetime.now(timezone.utc).isoformat()
                    })
            
            except asyncio.TimeoutError:
                # Send heartbeat if no message in 60 seconds
                await websocket.send_json({
                    "type": "heartbeat",
                    "timestamp": datetime.now(timezone.utc).isoformat(),
                    "clients": manager.connection_count
                })
    
    except WebSocketDisconnect:
        logger.info("[DASHBOARD-WS] Client disconnected normally")
    
    except Exception as e:
        logger.error(f"[DASHBOARD-WS] WebSocket error: {e}", exc_info=True)
    
    finally:
        await manager.disconnect(websocket)
        
        # Cancel event subscription task
        if 'event_task' in locals():
            event_task.cancel()


# ========== TESTING HELPERS ==========

async def broadcast_test_event(event_type: str, payload: Dict[str, Any]):
    """
    Broadcast test event to all connected clients.
    
    For testing/debugging purposes only.
    
    Args:
        event_type: Event type (position_updated, pnl_updated, etc.)
        payload: Event payload
    """
    event = DashboardEvent(
        type=EventType(event_type),
        timestamp=datetime.now(timezone.utc).isoformat(),
        payload=payload
    )
    
    await manager.broadcast(event)
    logger.info(f"[DASHBOARD-WS] Broadcasted test event: {event_type}")


async def get_connection_stats() -> Dict[str, Any]:
    """
    Get WebSocket connection statistics.
    
    Returns:
        Dict with connection count and status
    """
    return {
        "active_connections": manager.connection_count,
        "status": "OK" if manager.connection_count > 0 else "IDLE",
        "timestamp": datetime.now(timezone.utc).isoformat()
    }
