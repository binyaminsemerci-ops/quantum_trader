"""
Risk & Safety Service - Core Service Logic

Manages ESS and PolicyStore with event-driven updates.
"""
import asyncio
import logging
from typing import Dict, Any, Optional
from datetime import datetime, timezone

# Import existing modules (will be refactored)
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))

from backend.core.safety.ess import ESS
from backend.core.policy_store import PolicyStore
from backend.core.event_bus import EventBus
from backend.core.disk_buffer import DiskBuffer

logger = logging.getLogger(__name__)


class RiskSafetyService:
    """
    Risk & Safety Service
    
    Responsibilities:
    - Emergency Stop System (ESS) management
    - PolicyStore (Single Source of Truth)
    - Risk limit enforcement
    - Event-driven state synchronization
    """
    
    def __init__(self):
        self.ess: Optional[ESS] = None
        self.policy_store: Optional[PolicyStore] = None
        self.event_bus: Optional[EventBus] = None
        self.disk_buffer: Optional[DiskBuffer] = None
        self._running = False
        self._event_loop_task: Optional[asyncio.Task] = None
        
        logger.info("[RISK-SAFETY] Service initialized")
    
    async def start(self):
        """Start the service."""
        if self._running:
            logger.warning("[RISK-SAFETY] Service already running")
            return
        
        try:
            # Initialize PolicyStore
            logger.info("[RISK-SAFETY] Initializing PolicyStore...")
            self.policy_store = PolicyStore()
            await self.policy_store.initialize()
            logger.info("[RISK-SAFETY] ✅ PolicyStore ready")
            
            # Initialize ESS
            logger.info("[RISK-SAFETY] Initializing ESS...")
            self.ess = ESS()
            logger.info(f"[RISK-SAFETY] ✅ ESS ready: state={self.ess.get_state()}")
            
            # Initialize EventBus
            logger.info("[RISK-SAFETY] Initializing EventBus...")
            self.event_bus = EventBus()
            
            # Subscribe to relevant events
            self.event_bus.subscribe("trade.closed", self._handle_trade_closed)
            self.event_bus.subscribe("order.failed", self._handle_order_failed)
            logger.info("[RISK-SAFETY] ✅ EventBus subscriptions active")
            
            # Initialize DiskBuffer for event persistence
            self.disk_buffer = DiskBuffer(
                buffer_dir="data/event_buffers/risk_safety",
                buffer_name="risk_safety_events"
            )
            logger.info("[RISK-SAFETY] ✅ DiskBuffer ready")
            
            # Start event processing loop
            self._running = True
            self._event_loop_task = asyncio.create_task(self._event_processing_loop())
            
            logger.info("[RISK-SAFETY] ✅ Service started successfully")
            
        except Exception as e:
            logger.error(f"[RISK-SAFETY] ❌ Failed to start service: {e}", exc_info=True)
            raise
    
    async def stop(self):
        """Stop the service."""
        if not self._running:
            return
        
        logger.info("[RISK-SAFETY] Stopping service...")
        self._running = False
        
        # Cancel event loop
        if self._event_loop_task:
            self._event_loop_task.cancel()
            try:
                await self._event_loop_task
            except asyncio.CancelledError:
                pass
        
        # Flush disk buffer
        if self.disk_buffer:
            self.disk_buffer.flush()
        
        logger.info("[RISK-SAFETY] ✅ Service stopped")
    
    async def _event_processing_loop(self):
        """Background loop for processing buffered events."""
        logger.info("[RISK-SAFETY] Event processing loop started")
        
        while self._running:
            try:
                # Process buffered events
                if self.disk_buffer:
                    while True:
                        event = self.disk_buffer.pop()
                        if event is None:
                            break
                        
                        # Re-publish to EventBus
                        event_type = event.get("type")
                        event_data = event.get("data")
                        if event_type and event_data:
                            await self.event_bus.publish(event_type, event_data)
                            logger.debug(f"[RISK-SAFETY] Replayed buffered event: {event_type}")
                
                # Sleep to avoid busy loop
                await asyncio.sleep(1.0)
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"[RISK-SAFETY] Error in event processing loop: {e}", exc_info=True)
                await asyncio.sleep(5.0)  # Backoff on error
        
        logger.info("[RISK-SAFETY] Event processing loop stopped")
    
    async def _handle_trade_closed(self, event_data: Dict[str, Any]):
        """Handle trade.closed events for ESS tracking."""
        try:
            symbol = event_data.get("symbol")
            pnl_usd = event_data.get("pnl_usd", 0.0)
            
            logger.info(f"[RISK-SAFETY] Trade closed: {symbol} PnL=${pnl_usd:.2f}")
            
            # Update ESS
            old_state = self.ess.get_state()
            self.ess.record_trade(pnl_usd=pnl_usd)
            new_state = self.ess.get_state()
            
            # Publish state change event if changed
            if old_state != new_state:
                await self.event_bus.publish("ess.state.changed", {
                    "old_state": old_state,
                    "new_state": new_state,
                    "timestamp": datetime.now(timezone.utc).isoformat()
                })
                logger.warning(f"[RISK-SAFETY] 🚨 ESS state changed: {old_state} → {new_state}")
                
                # If transitioned to CRITICAL, publish ess.tripped
                if new_state == "CRITICAL":
                    await self.event_bus.publish("ess.tripped", {
                        "reason": self.ess.get_status().get("trip_reason", "Unknown"),
                        "timestamp": datetime.now(timezone.utc).isoformat()
                    })
                    logger.error("[RISK-SAFETY] 🛑 ESS TRIPPED - All trading blocked!")
            
            # Buffer event for reliability
            if self.disk_buffer:
                self.disk_buffer.push({
                    "type": "trade.closed",
                    "data": event_data,
                    "processed_at": datetime.now(timezone.utc).isoformat()
                })
        
        except Exception as e:
            logger.error(f"[RISK-SAFETY] Error handling trade.closed: {e}", exc_info=True)
    
    async def _handle_order_failed(self, event_data: Dict[str, Any]):
        """Handle order.failed events for failure rate monitoring."""
        try:
            symbol = event_data.get("symbol")
            reason = event_data.get("reason", "Unknown")
            
            logger.warning(f"[RISK-SAFETY] Order failed: {symbol} - {reason}")
            
            # TODO: Implement failure rate tracking (future enhancement)
            # For now, just log
            
        except Exception as e:
            logger.error(f"[RISK-SAFETY] Error handling order.failed: {e}", exc_info=True)
    
    # ========================================================================
    # PUBLIC API METHODS (called by REST endpoints)
    # ========================================================================
    
    async def get_ess_status(self) -> Dict[str, Any]:
        """Get current ESS status."""
        if not self.ess:
            return {"error": "ESS not initialized", "healthy": False}
        
        return self.ess.get_status()
    
    async def override_ess(self, new_state: str, reason: str) -> Dict[str, Any]:
        """Manually override ESS state."""
        if not self.ess:
            return {"error": "ESS not initialized", "success": False}
        
        old_state = self.ess.get_state()
        
        # Manual override (bypass ESS logic)
        self.ess._state = new_state
        logger.warning(f"[RISK-SAFETY] 🔧 ESS manually overridden: {old_state} → {new_state} (reason: {reason})")
        
        # Publish state change event
        await self.event_bus.publish("ess.state.changed", {
            "old_state": old_state,
            "new_state": new_state,
            "manual_override": True,
            "reason": reason,
            "timestamp": datetime.now(timezone.utc).isoformat()
        })
        
        return {
            "success": True,
            "old_state": old_state,
            "new_state": new_state,
            "reason": reason
        }
    
    async def reset_ess(self) -> Dict[str, Any]:
        """Reset ESS to NORMAL state."""
        if not self.ess:
            return {"error": "ESS not initialized", "success": False}
        
        old_state = self.ess.get_state()
        self.ess.reset()
        new_state = self.ess.get_state()
        
        logger.info(f"[RISK-SAFETY] ESS reset: {old_state} → {new_state}")
        
        # Publish state change event
        await self.event_bus.publish("ess.state.changed", {
            "old_state": old_state,
            "new_state": new_state,
            "reset": True,
            "timestamp": datetime.now(timezone.utc).isoformat()
        })
        
        return {
            "success": True,
            "old_state": old_state,
            "new_state": new_state
        }
    
    async def get_policy(self, key: str) -> Optional[Any]:
        """Get policy value by key."""
        if not self.policy_store:
            return None
        
        return self.policy_store.get_policy(key)
    
    async def update_policy(self, key: str, value: Any) -> Dict[str, Any]:
        """Update policy value."""
        if not self.policy_store:
            return {"error": "PolicyStore not initialized", "success": False}
        
        old_value = self.policy_store.get_policy(key)
        self.policy_store.set_policy(key, value)
        
        logger.info(f"[RISK-SAFETY] Policy updated: {key} = {value} (was: {old_value})")
        
        # Publish policy update event
        await self.event_bus.publish("policy.updated", {
            "key": key,
            "old_value": old_value,
            "new_value": value,
            "timestamp": datetime.now(timezone.utc).isoformat()
        })
        
        return {
            "success": True,
            "key": key,
            "old_value": old_value,
            "new_value": value
        }
    
    async def get_all_policies(self) -> Dict[str, Any]:
        """Get all policy values."""
        if not self.policy_store:
            return {}
        
        # Return snapshot of all policies
        policies = {}
        for key in [
            "execution.max_slippage_pct",
            "execution.max_order_retries",
            "execution.retry_backoff_base_sec",
            "execution.max_retry_backoff_sec",
            "execution.sl_adjustment_buffer_pct",
            "execution.max_sl_distance_pct",
            "execution.max_tp_distance_pct",
            "rl.max_leverage",
            "rl.min_confidence",
            "ess.max_daily_loss_pct",
            "ess.max_consecutive_losses",
            "ess.max_drawdown_pct",
            "ess.min_win_rate",
        ]:
            value = self.policy_store.get_policy(key)
            if value is not None:
                policies[key] = value
        
        return policies
    
    async def get_health(self) -> Dict[str, Any]:
        """Get service health status."""
        healthy = True
        components = {}
        
        # Check ESS
        if self.ess:
            ess_status = self.ess.get_status()
            components["ess"] = {
                "healthy": True,
                "state": ess_status.get("state"),
                "can_execute": ess_status.get("can_execute")
            }
        else:
            components["ess"] = {"healthy": False, "error": "Not initialized"}
            healthy = False
        
        # Check PolicyStore
        if self.policy_store:
            components["policy_store"] = {"healthy": True}
        else:
            components["policy_store"] = {"healthy": False, "error": "Not initialized"}
            healthy = False
        
        # Check EventBus
        if self.event_bus:
            components["event_bus"] = {"healthy": True}
        else:
            components["event_bus"] = {"healthy": False, "error": "Not initialized"}
            healthy = False
        
        return {
            "service": "risk-safety",
            "healthy": healthy,
            "running": self._running,
            "components": components,
            "timestamp": datetime.now(timezone.utc).isoformat()
        }
