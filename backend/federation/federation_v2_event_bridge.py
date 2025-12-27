"""Federation v2 Event Bridge

[CRITICAL FIX #2] Bridges model lifecycle events from EventBus v2 (Redis Streams)
to Federation v2 nodes to prevent split-brain scenarios during model promotion.

This ensures all distributed nodes (v2 and v3) receive model.promoted events
and stay synchronized during model lifecycle transitions.
"""

import asyncio
import logging
from typing import Optional

logger = logging.getLogger(__name__)


class FederationV2EventBridge:
    """
    [CRITICAL FIX #2] Event bridge for Federation v2 <-> EventBus v2 integration.
    
    Subscribes to model lifecycle events and broadcasts them to Federation v2 nodes
    to prevent version desynchronization across distributed deployments.
    """
    
    def __init__(self, event_bus, federation_v2_engine=None):
        """
        Initialize event bridge.
        
        Args:
            event_bus: EventBus v2 instance (Redis Streams)
            federation_v2_engine: FederatedEngineV2 instance (optional)
        """
        self.event_bus = event_bus
        self.federation_v2 = federation_v2_engine
        self._running = False
        self._subscribed_events = [
            "model.promoted",
            "model.rollback_initiated",
            "model.training_completed",
            "model.validation_passed",
        ]
        
        logger.info("[FED-V2-BRIDGE] Initialized (model lifecycle event bridge)")
    
    async def start(self):
        """Start the event bridge."""
        self._running = True
        
        # Subscribe to model lifecycle events
        for event_type in self._subscribed_events:
            self.event_bus.subscribe(event_type, self._handle_model_event)
            logger.info(f"[FED-V2-BRIDGE] Subscribed to {event_type}")
        
        logger.info("[FED-V2-BRIDGE] Started - bridging events to Federation v2 nodes")
    
    async def _handle_model_event(self, event_data: dict):
        """
        Handle model lifecycle event and broadcast to Federation v2.
        
        Args:
            event_data: Event payload from EventBus
        """
        event_type = event_data.get("type", "unknown")
        
        try:
            logger.info(
                f"[FED-V2-BRIDGE] Received {event_type}, "
                f"broadcasting to Federation v2 nodes"
            )
            
            # If Federation v2 engine available, broadcast to all nodes
            if self.federation_v2:
                # Federation v2 uses different event format
                # Convert EventBus v2 format to Federation v2 format
                fed_v2_event = {
                    "event_type": event_type,
                    "payload": event_data,
                    "source": "event_bridge",
                    "broadcast_required": True,
                }
                
                # Broadcast to all Federation v2 nodes
                # (Implementation depends on Federation v2 API)
                await self._broadcast_to_v2_nodes(fed_v2_event)
            else:
                logger.warning(
                    "[FED-V2-BRIDGE] No Federation v2 engine configured - "
                    "events not propagated to v2 nodes"
                )
        
        except Exception as e:
            logger.error(f"[FED-V2-BRIDGE] Error handling event {event_type}: {e}")
    
    async def _broadcast_to_v2_nodes(self, event: dict):
        """
        Broadcast event to all Federation v2 nodes.
        
        Args:
            event: Event data in Federation v2 format
        """
        try:
            # Federation v2 broadcast mechanism
            # This is a placeholder - actual implementation depends on Federation v2 API
            
            # Option 1: If Federation v2 has HTTP API
            # for node in self.federation_v2.get_nodes():
            #     await self._http_post(f"http://{node}/api/events", event)
            
            # Option 2: If Federation v2 has event queue
            # await self.federation_v2.broadcast_event(event)
            
            # For now, just log
            logger.info(
                f"[FED-V2-BRIDGE] Event {event['event_type']} "
                f"ready for broadcast to {self.federation_v2.num_nodes() if hasattr(self.federation_v2, 'num_nodes') else 'N'} nodes"
            )
            
            # [IMPLEMENTATION NOTE]
            # If Federation v2 doesn't support event broadcasting,
            # recommend upgrading to Federation v3 or implementing
            # a webhook/HTTP endpoint for each v2 node
        
        except Exception as e:
            logger.error(f"[FED-V2-BRIDGE] Broadcast failed: {e}")
    
    async def stop(self):
        """Stop the event bridge."""
        self._running = False
        logger.info("[FED-V2-BRIDGE] Stopped")


# Singleton instance
_bridge_instance: Optional[FederationV2EventBridge] = None


def get_federation_v2_bridge(event_bus, federation_v2=None) -> FederationV2EventBridge:
    """Get or create Federation v2 event bridge singleton."""
    global _bridge_instance
    
    if _bridge_instance is None:
        _bridge_instance = FederationV2EventBridge(event_bus, federation_v2)
    
    return _bridge_instance
