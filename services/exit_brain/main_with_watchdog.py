#!/usr/bin/env python3
"""
Exit Brain with Systemd Watchdog Integration

This wrapper:
1. Calls sd_notify("READY=1") when initialized
2. Calls sd_notify("WATCHDOG=1") every 1 second via heartbeat
3. Integrates with Redis heartbeat stream

Requirements:
- WatchdogSec=3 in systemd unit
- Type=notify in systemd unit
"""

import os
import sys
import time
import asyncio
import logging
import signal
import functools
from typing import Optional

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

try:
    import sdnotify
    SDNOTIFY_AVAILABLE = True
except ImportError:
    SDNOTIFY_AVAILABLE = False
    print("WARNING: sdnotify not installed. Install with: pip install sdnotify")

try:
    import redis.asyncio as aioredis
except ImportError:
    print("ERROR: redis.asyncio required")
    sys.exit(1)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("exit_brain_watchdog_wrapper")

# Redis stream for heartbeat
HEARTBEAT_STREAM = "exit_brain:heartbeat"
REDIS_URL = os.getenv("REDIS_URL", "redis://localhost:6379")

# Watchdog interval (should be < WatchdogSec/2)
WATCHDOG_INTERVAL = 1.0  # 1 second


class ExitBrainWatchdogWrapper:
    """
    Wrapper that manages sd_notify and Redis heartbeat for Exit Brain
    """
    
    def __init__(self):
        self.redis: Optional[aioredis.Redis] = None
        self._running = False
        self._exit_brain_healthy = True
        self._positions_count = 0
        
        # Initialize sdnotify if available
        if SDNOTIFY_AVAILABLE:
            self.notifier = sdnotify.SystemdNotifier()
        else:
            self.notifier = None
            logger.warning("sdnotify not available - systemd watchdog disabled")
    
    def sd_notify(self, state: str):
        """Send sd_notify message"""
        if self.notifier:
            try:
                self.notifier.notify(state)
                logger.debug(f"sd_notify: {state}")
            except Exception as e:
                logger.error(f"sd_notify failed: {e}")
    
    async def start(self):
        """Start the wrapper and Exit Brain"""
        logger.info("=" * 60)
        logger.info("EXIT BRAIN WATCHDOG WRAPPER STARTING")
        logger.info("=" * 60)
        
        # Connect to Redis
        self.redis = await aioredis.from_url(REDIS_URL)
        logger.info(f"Connected to Redis: {REDIS_URL}")
        
        # Import and start the actual Exit Brain
        # This would normally import the Exit Brain module
        # For now, we simulate it
        logger.info("Initializing Exit Brain...")
        
        # Notify systemd we're ready
        self.sd_notify("READY=1")
        logger.info("Sent READY=1 to systemd")
        
        self._running = True
        
        # Start watchdog loop
        await self._watchdog_loop()
    
    async def _watchdog_loop(self):
        """
        Main watchdog loop:
        1. Ping systemd watchdog
        2. Publish heartbeat to Redis
        3. Monitor Exit Brain health
        """
        logger.info("Starting watchdog loop...")
        loop_count = 0
        
        while self._running:
            try:
                loop_count += 1
                
                # 1. Send systemd watchdog notification
                self.sd_notify("WATCHDOG=1")
                
                # 2. Determine current status
                if self._exit_brain_healthy:
                    status = "ALIVE"
                else:
                    status = "DEGRADED"
                
                # 3. Publish heartbeat to Redis stream
                await self._publish_heartbeat(status)
                
                # 4. Log every 30 loops (30 seconds)
                if loop_count % 30 == 0:
                    logger.info(f"Watchdog OK - loop {loop_count}, status={status}")
                
                # Sleep until next interval
                await asyncio.sleep(WATCHDOG_INTERVAL)
                
            except asyncio.CancelledError:
                logger.info("Watchdog loop cancelled")
                break
            except Exception as e:
                logger.error(f"Watchdog loop error: {e}")
                # Still notify systemd we're alive even if there's an error
                self.sd_notify("WATCHDOG=1")
                await asyncio.sleep(WATCHDOG_INTERVAL)
    
    async def _publish_heartbeat(self, status: str):
        """Publish heartbeat to exit_brain:heartbeat stream"""
        try:
            ts = int(time.time() * 1000)  # Epoch ms
            
            await self.redis.xadd(
                HEARTBEAT_STREAM,
                {
                    "status": status,
                    "active_positions": str(self._positions_count),
                    "ts": str(ts)
                },
                maxlen=100  # Keep last 100 entries
            )
        except Exception as e:
            logger.error(f"Failed to publish heartbeat: {e}")
    
    def set_health(self, healthy: bool):
        """Update Exit Brain health status"""
        self._exit_brain_healthy = healthy
    
    def set_positions_count(self, count: int):
        """Update active positions count"""
        self._positions_count = count
    
    async def stop(self):
        """Stop the wrapper"""
        logger.info("Stopping Exit Brain wrapper...")
        self._running = False
        
        # Notify systemd we're stopping
        self.sd_notify("STOPPING=1")
        
        if self.redis:
            await self.redis.close()


async def main():
    """Main entry point"""
    wrapper = ExitBrainWatchdogWrapper()
    
    # Signal handlers
    loop = asyncio.get_event_loop()
    
    def signal_handler(sig):
        logger.info(f"Received signal {sig}")
        asyncio.create_task(wrapper.stop())
    
    for sig in (signal.SIGTERM, signal.SIGINT):
        loop.add_signal_handler(sig, functools.partial(signal_handler, sig))
    
    try:
        await wrapper.start()
    except KeyboardInterrupt:
        logger.info("Keyboard interrupt received")
    finally:
        await wrapper.stop()


if __name__ == "__main__":
    asyncio.run(main())
