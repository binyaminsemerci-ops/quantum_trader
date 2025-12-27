"""
Position Monitor Service - Standalone with ExitBrain v3.5
Monitors open positions and manages TP/SL protection with Exit Brain v3.5
"""
import asyncio
import logging
import sys
import os
from pathlib import Path

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent.parent))
sys.path.insert(0, '/app/microservices')  # For ExitBrain v3.5

from backend.services.monitoring.position_monitor import PositionMonitor
from backend.core.event_bus import EventBus

# Import ExitBrain v3.5
try:
    from exitbrain_v3_5.exit_brain import ExitBrainV35, SignalContext
    EXITBRAIN_V35_AVAILABLE = True
    logging.info("✅ ExitBrain v3.5 imported successfully")
except ImportError as e:
    EXITBRAIN_V35_AVAILABLE = False
    logging.warning(f"⚠️ ExitBrain v3.5 not available: {e}")

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


async def main():
    """Start Position Monitor with EventBus and ExitBrain v3.5 integration"""
    try:
        # Write PID file for healthcheck
        pid_file = Path("/tmp/position_monitor.pid")
        pid_file.write_text(str(os.getpid()))

        # Initialize Redis client and EventBus
        import redis.asyncio as redis_async

        redis_host = os.getenv("REDIS_HOST", "localhost")
        redis_port = int(os.getenv("REDIS_PORT", 6379))

        redis_client = redis_async.Redis(
            host=redis_host,
            port=redis_port,
            decode_responses=True
        )

        event_bus = EventBus(redis_client, service_name="position_monitor")
        await event_bus.initialize()

        logger.info(f"[POSITION MONITOR] Connected to Redis: {redis_host}:{redis_port}")

        # Initialize ExitBrain v3.5 if available
        exit_brain = None
        if EXITBRAIN_V35_AVAILABLE:
            try:
                exit_brain = ExitBrainV35()
                logger.info("✅ [POSITION MONITOR] ExitBrain v3.5 initialized and ACTIVE")
            except Exception as e:
                logger.error(f"❌ [POSITION MONITOR] Failed to initialize ExitBrain v3.5: {e}")
        else:
            logger.warning("⚠️ [POSITION MONITOR] Running without ExitBrain v3.5")

        # Initialize Position Monitor with Exit Brain v3.5
        check_interval = int(os.getenv("POSITION_CHECK_INTERVAL", "10"))

        monitor = PositionMonitor(
            check_interval=check_interval,
            ai_engine=None,
            app_state=None,
            event_bus=event_bus
        )

        # Attach ExitBrain to monitor if available
        if exit_brain:
            monitor.exit_brain_v35 = exit_brain
            logger.info(f"[POSITION MONITOR] ExitBrain v3.5 attached to monitor")

        logger.info(f"[POSITION MONITOR] Starting with {check_interval}s check interval")
        logger.info(f"[POSITION MONITOR] Exit Brain v3.5: {'ACTIVE' if exit_brain else 'DISABLED'}")

        # Start monitoring loop
        await monitor.monitor_loop()

    except KeyboardInterrupt:
        logger.info("[POSITION MONITOR] Shutting down...")
    except Exception as e:
        logger.error(f"[POSITION MONITOR] Fatal error: {e}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    asyncio.run(main())
