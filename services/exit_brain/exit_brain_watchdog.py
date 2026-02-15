"""
Exit Brain Watchdog Service

Monitors Exit Brain health via heartbeats and triggers panic_close
when failure is detected.

CRITICAL RULES:
- No grace periods during volatility
- False positives are acceptable
- False negatives are NOT acceptable

Trigger panic_close if:
- Heartbeat missing > 5 seconds
- Status = DEGRADED > 10 seconds
- Decisions stagnant > 30 seconds with active positions
- Positions unguarded > 3 seconds (heartbeat missing + positions > 0)
"""

import os
import sys
import time
import json
import asyncio
import logging
from datetime import datetime
from typing import Optional, Dict, Any
from dataclasses import dataclass

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

try:
    import redis.asyncio as aioredis
except ImportError:
    print("Missing redis-py: pip install redis")
    sys.exit(1)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | %(levelname)s | [WATCHDOG] %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)

# =============================================================================
# CONSTANTS
# =============================================================================

# Heartbeat stream
HEARTBEAT_STREAM = "exit_brain:heartbeat"

# Panic close stream
PANIC_CLOSE_STREAM = "system:panic_close"

# Thresholds (DO NOT RELAX THESE)
HEARTBEAT_MISSING_THRESHOLD = 5.0    # seconds
DEGRADED_THRESHOLD = 5.0             # seconds (spec says >5s)
DECISION_STAGNANT_THRESHOLD = 30.0   # seconds
UNGUARDED_THRESHOLD = 3.0            # seconds (CRITICAL: positions + no heartbeat)

# Check interval
CHECK_INTERVAL = 1.0  # seconds


@dataclass
class ExitBrainState:
    """Current state of Exit Brain as seen by watchdog"""
    last_heartbeat_ts: float = 0.0
    last_heartbeat_status: str = "UNKNOWN"
    active_positions: int = 0
    last_decision_ts: float = 0.0
    degraded_since: Optional[float] = None
    loop_cycle_ms: int = 0
    
    def heartbeat_age(self) -> float:
        """Seconds since last heartbeat"""
        if self.last_heartbeat_ts == 0:
            return float('inf')
        return time.time() - self.last_heartbeat_ts
    
    def decision_age(self) -> float:
        """Seconds since last decision"""
        if self.last_decision_ts == 0:
            return float('inf')
        return time.time() - self.last_decision_ts
    
    def degraded_duration(self) -> float:
        """Seconds in degraded state"""
        if self.degraded_since is None:
            return 0.0
        return time.time() - self.degraded_since


class ExitBrainWatchdog:
    """
    Monitors Exit Brain and triggers panic_close on failure.
    
    This is a fail-closed system:
    - If uncertain, assume failure
    - If watchdog itself fails, Exit Brain should detect and self-halt
    """
    
    def __init__(self):
        self.redis: Optional[aioredis.Redis] = None
        self.state = ExitBrainState()
        self._running = False
        self._panic_triggered = False
        
        self.redis_url = os.getenv("REDIS_URL", "redis://localhost:6379")
    
    async def start(self):
        """Start the watchdog"""
        logger.info("=" * 60)
        logger.info("EXIT BRAIN WATCHDOG STARTING")
        logger.info("=" * 60)
        logger.info(f"Thresholds:")
        logger.info(f"  Heartbeat missing: {HEARTBEAT_MISSING_THRESHOLD}s")
        logger.info(f"  Degraded duration: {DEGRADED_THRESHOLD}s")
        logger.info(f"  Decision stagnant: {DECISION_STAGNANT_THRESHOLD}s")
        logger.info(f"  Unguarded (positions + no HB): {UNGUARDED_THRESHOLD}s")
        logger.info("=" * 60)
        
        # Connect to Redis
        self.redis = aioredis.from_url(self.redis_url, decode_responses=True)
        
        self._running = True
        self._panic_triggered = False
        
        # Main monitoring loop
        await self._monitor_loop()
    
    async def stop(self):
        """Stop the watchdog"""
        self._running = False
        if self.redis:
            await self.redis.close()
        logger.info("Watchdog stopped")
    
    async def _monitor_loop(self):
        """Main monitoring loop"""
        last_log_time = 0
        
        while self._running:
            try:
                # Read latest heartbeat (non-blocking with short timeout)
                await self._read_heartbeat()
                
                # Check health
                failure_reason = self._check_health()
                
                if failure_reason:
                    logger.error(f"🚨 FAILURE DETECTED: {failure_reason}")
                    await self._trigger_panic_close(failure_reason)
                    break
                
                # Periodic status log (every 30 seconds)
                now = time.time()
                if now - last_log_time > 30:
                    self._log_status()
                    last_log_time = now
                
                await asyncio.sleep(CHECK_INTERVAL)
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error in monitor loop: {e}")
                await asyncio.sleep(1)
    
    async def _read_heartbeat(self):
        """Read latest heartbeat from stream"""
        try:
            # Read newest message
            messages = await self.redis.xrevrange(
                HEARTBEAT_STREAM,
                count=1
            )
            
            if not messages:
                return
            
            msg_id, data = messages[0]
            
            # Update state from schema fields
            # ts is int64 (epoch ms) - convert to seconds
            ts_ms = int(data.get("ts", 0))
            self.state.last_heartbeat_ts = ts_ms / 1000.0 if ts_ms > 0 else 0.0
            self.state.last_heartbeat_status = data.get("status", "UNKNOWN")
            self.state.active_positions = int(data.get("active_positions", 0))
            self.state.last_decision_ts = float(data.get("last_decision_ts", 0))
            self.state.loop_cycle_ms = int(data.get("loop_cycle_ms", 0))
            
            # Track degraded state
            if self.state.last_heartbeat_status == "DEGRADED":
                if self.state.degraded_since is None:
                    self.state.degraded_since = time.time()
                    logger.warning("Exit Brain entered DEGRADED state")
            else:
                if self.state.degraded_since is not None:
                    duration = time.time() - self.state.degraded_since
                    logger.info(f"Exit Brain recovered from DEGRADED (was {duration:.1f}s)")
                self.state.degraded_since = None
                
        except Exception as e:
            logger.error(f"Error reading heartbeat: {e}")
    
    def _check_health(self) -> Optional[str]:
        """
        Check Exit Brain health.
        
        Returns failure reason if panic_close should trigger, None if healthy.
        """
        if self._panic_triggered:
            return None  # Already triggered
        
        hb_age = self.state.heartbeat_age()
        decision_age = self.state.decision_age()
        degraded_duration = self.state.degraded_duration()
        positions = self.state.active_positions
        
        # Rule 1: Heartbeat missing too long
        if hb_age > HEARTBEAT_MISSING_THRESHOLD:
            return f"Heartbeat missing ({hb_age:.1f}s > {HEARTBEAT_MISSING_THRESHOLD}s)"
        
        # Rule 2: Degraded too long
        if degraded_duration > DEGRADED_THRESHOLD:
            return f"DEGRADED too long ({degraded_duration:.1f}s > {DEGRADED_THRESHOLD}s)"
        
        # Rule 3: Decision loop stuck (only if has positions)
        if positions > 0 and decision_age > DECISION_STAGNANT_THRESHOLD:
            return f"Decisions stagnant ({decision_age:.1f}s) with {positions} positions"
        
        # Rule 4: CRITICAL - Positions unguarded
        if positions > 0 and hb_age > UNGUARDED_THRESHOLD:
            return f"Positions UNGUARDED ({positions} pos, heartbeat {hb_age:.1f}s old)"
        
        return None  # Healthy
    
    async def _trigger_panic_close(self, reason: str):
        """Trigger system:panic_close per schema"""
        if self._panic_triggered:
            logger.warning("Panic close already triggered, skipping")
            return
        
        self._panic_triggered = True
        
        logger.error("=" * 60)
        logger.error("🚨 TRIGGERING PANIC CLOSE 🚨")
        logger.error(f"Reason: {reason}")
        logger.error("=" * 60)
        
        import uuid
        event_id = str(uuid.uuid4())
        ts = int(time.time() * 1000)  # Epoch ms
        
        try:
            await self.redis.xadd(
                PANIC_CLOSE_STREAM,
                {
                    "event_id": event_id,
                    "reason": f"EXIT_BRAIN_{reason.replace(' ', '_').upper()}",
                    "severity": "CRITICAL",
                    "issued_by": "watchdog",
                    "ts": str(ts)
                }
            )
            logger.error(f"✅ Panic close published to {PANIC_CLOSE_STREAM}")
            logger.error(f"Event ID: {event_id}")
            
        except Exception as e:
            logger.error(f"❌ FAILED to publish panic close: {e}")
            # This is catastrophic - watchdog failed to protect
            # Log loudly and hope ops notices
            logger.critical("WATCHDOG FAILED TO TRIGGER PANIC CLOSE")
            logger.critical("MANUAL INTERVENTION REQUIRED IMMEDIATELY")
    
    def _log_status(self):
        """Log current status"""
        hb_age = self.state.heartbeat_age()
        decision_age = self.state.decision_age()
        positions = self.state.active_positions
        status = self.state.last_heartbeat_status
        
        if hb_age == float('inf'):
            hb_str = "NEVER"
        else:
            hb_str = f"{hb_age:.1f}s"
        
        if decision_age == float('inf'):
            dec_str = "NEVER"
        else:
            dec_str = f"{decision_age:.1f}s"
        
        logger.info(
            f"Status: heartbeat={hb_str} ago, status={status}, "
            f"positions={positions}, last_decision={dec_str} ago"
        )


async def main():
    """Main entry point"""
    watchdog = ExitBrainWatchdog()
    
    try:
        await watchdog.start()
    except KeyboardInterrupt:
        logger.info("Shutting down...")
    finally:
        await watchdog.stop()


if __name__ == "__main__":
    asyncio.run(main())
