"""
Exit Brain Heartbeat Mixin

Add this mixin to Exit Brain to publish heartbeats for watchdog monitoring.

Usage:
    class ExitBrain(HeartbeatMixin, BaseExitBrain):
        def __init__(self):
            super().__init__()
            self.init_heartbeat(redis_client)
            
        async def run(self):
            self.start_heartbeat()
            try:
                await self._main_loop()
            finally:
                await self.stop_heartbeat()
"""

import time
import asyncio
import logging
from typing import Optional, Dict, Any
from abc import ABC

logger = logging.getLogger(__name__)

# Heartbeat configuration
HEARTBEAT_STREAM = "quantum:stream:exit_brain.heartbeat"
HEARTBEAT_INTERVAL = 1.0  # seconds
HEARTBEAT_MAXLEN = 1000   # Keep last 1000 messages


class HeartbeatMixin(ABC):
    """
    Mixin class that adds heartbeat publishing to Exit Brain.
    
    Attributes required in inheriting class:
    - redis: Redis client
    - active_positions: dict or list of current positions
    - last_decision_ts: timestamp of last decision
    """
    
    def __init__(self):
        self._heartbeat_task: Optional[asyncio.Task] = None
        self._heartbeat_running = False
        self._last_cycle_ms = 0
        self._pending_exits = 0
        self._degraded = False
        self._heartbeat_interval = HEARTBEAT_INTERVAL
    
    def init_heartbeat(self, redis_client, interval: float = HEARTBEAT_INTERVAL):
        """Initialize heartbeat with Redis client"""
        self._hb_redis = redis_client
        self._heartbeat_interval = interval
    
    def start_heartbeat(self):
        """Start the heartbeat background task"""
        if self._heartbeat_task is not None:
            logger.warning("Heartbeat already running")
            return
        
        self._heartbeat_running = True
        self._heartbeat_task = asyncio.create_task(self._heartbeat_loop())
        logger.info(f"Heartbeat started (interval={self._heartbeat_interval}s)")
    
    async def stop_heartbeat(self):
        """Stop the heartbeat background task"""
        self._heartbeat_running = False
        if self._heartbeat_task:
            self._heartbeat_task.cancel()
            try:
                await self._heartbeat_task
            except asyncio.CancelledError:
                pass
            self._heartbeat_task = None
        logger.info("Heartbeat stopped")
    
    def set_cycle_time(self, cycle_ms: int):
        """Set the last decision cycle time (call after each decision loop)"""
        self._last_cycle_ms = cycle_ms
    
    def set_pending_exits(self, count: int):
        """Set number of pending exit orders"""
        self._pending_exits = count
    
    def set_degraded(self, degraded: bool):
        """Set degraded status"""
        self._degraded = degraded
    
    async def _heartbeat_loop(self):
        """Background task that publishes heartbeats"""
        while self._heartbeat_running:
            try:
                await self._publish_heartbeat()
            except Exception as e:
                logger.error(f"Heartbeat publish error: {e}")
            
            await asyncio.sleep(self._heartbeat_interval)
    
    async def _publish_heartbeat(self):
        """Publish a single heartbeat"""
        # Get position count (override this property in your class)
        position_count = self._get_position_count()
        
        # Get last decision timestamp (override this in your class)
        last_decision = self._get_last_decision_ts()
        
        # Determine status
        status = "DEGRADED" if self._is_degraded() else "OK"
        
        # Build heartbeat message
        message = {
            "timestamp": str(time.time()),
            "status": status,
            "active_positions_count": str(position_count),
            "last_decision_ts": str(last_decision),
            "loop_cycle_ms": str(self._last_cycle_ms),
            "pending_exits": str(self._pending_exits)
        }
        
        # Publish to stream
        await self._hb_redis.xadd(
            HEARTBEAT_STREAM,
            message,
            maxlen=HEARTBEAT_MAXLEN
        )
    
    def _get_position_count(self) -> int:
        """
        Get current open position count.
        Override this in your Exit Brain class.
        """
        # Try common attribute names
        if hasattr(self, 'active_positions'):
            pos = getattr(self, 'active_positions')
            if isinstance(pos, dict):
                return len(pos)
            elif isinstance(pos, (list, set)):
                return len(pos)
            elif isinstance(pos, int):
                return pos
        
        if hasattr(self, 'positions'):
            pos = getattr(self, 'positions')
            if isinstance(pos, dict):
                return len(pos)
            return len(pos) if pos else 0
        
        return 0
    
    def _get_last_decision_ts(self) -> float:
        """
        Get timestamp of last exit decision.
        Override this in your Exit Brain class.
        """
        if hasattr(self, 'last_decision_ts'):
            return getattr(self, 'last_decision_ts', 0)
        
        if hasattr(self, 'last_decision_time'):
            return getattr(self, 'last_decision_time', 0)
        
        return 0.0
    
    def _is_degraded(self) -> bool:
        """
        Check if Exit Brain is in degraded state.
        Override this for custom degraded detection.
        """
        if self._degraded:
            return True
        
        # Auto-detect degraded conditions
        if self._last_cycle_ms > 500:  # Decision loop too slow
            return True
        
        # Check for stale decisions with active positions
        position_count = self._get_position_count()
        last_decision = self._get_last_decision_ts()
        if position_count > 0 and last_decision > 0:
            age = time.time() - last_decision
            if age > 10:  # No decisions in 10 seconds
                return True
        
        return False


# Example usage
class ExampleExitBrain(HeartbeatMixin):
    """Example Exit Brain with heartbeat integration"""
    
    def __init__(self, redis_client):
        HeartbeatMixin.__init__(self)
        self.redis = redis_client
        self.active_positions = {}
        self.last_decision_ts = 0.0
        self.running = False
        
        # Initialize heartbeat
        self.init_heartbeat(redis_client)
    
    async def run(self):
        """Main run loop"""
        self.running = True
        self.start_heartbeat()
        
        try:
            while self.running:
                start = time.time()
                
                # Your decision logic here
                await self._process_positions()
                
                # Record cycle time
                cycle_ms = int((time.time() - start) * 1000)
                self.set_cycle_time(cycle_ms)
                
                await asyncio.sleep(1)
        finally:
            await self.stop_heartbeat()
    
    async def _process_positions(self):
        """Example position processing"""
        # Update last decision timestamp
        self.last_decision_ts = time.time()
        
        # Your exit logic here
        pass
