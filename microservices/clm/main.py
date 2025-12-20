"""
CLM v3 Microservice - Continuous Learning Manager

Monitors model performance and triggers retraining when needed.
Runs as standalone microservice with EventBus integration.
"""

import asyncio
import logging
import os
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import redis.asyncio as redis_async

from backend.core.event_bus import EventBus
from backend.services.clm_v3.main import ClmV3Service

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - [%(name)s] [%(levelname)s] %(message)s'
)
logger = logging.getLogger(__name__)


async def main():
    """Main entry point for CLM microservice."""
    logger.info("=" * 60)
    logger.info("🧠 CLM v3 Microservice Starting...")
    logger.info("=" * 60)
    
    # Connect to Redis
    redis_url = os.getenv("REDIS_URL", "redis://redis:6379")
    logger.info(f"Connecting to Redis: {redis_url}")
    redis_client = redis_async.from_url(redis_url, decode_responses=False)
    
    try:
        await redis_client.ping()
        logger.info("✅ Redis connected")
    except Exception as e:
        logger.error(f"❌ Failed to connect to Redis: {e}")
        return
    
    # Initialize EventBus
    event_bus = EventBus(redis_client, service_name="clm_service")
    await event_bus.start()
    logger.info("✅ EventBus initialized")
    
    # Load CLM configuration from environment
    config = {
        "models_dir": os.getenv("MODELS_DIR", "/app/models"),
        "metadata_dir": os.getenv("CLM_METADATA_DIR", "/app/data/clm_v3/registry"),
        
        "orchestrator": {
            "auto_promote_to_candidate": True,
            "auto_promote_to_production": False,  # Manual production promotion
            "require_shadow_testing": True,
        },
        
        "scheduler": {
            "enabled": True,
            "retrain_hours": float(os.getenv("QT_CLM_RETRAIN_HOURS", "0.5")),  # 30 min default
            "drift_check_hours": float(os.getenv("QT_CLM_DRIFT_HOURS", "0.25")),  # 15 min
            "performance_check_hours": float(os.getenv("QT_CLM_PERF_HOURS", "0.17")),  # 10 min
        },
        
        "event_subscriptions": {
            "drift_detected": True,
            "performance_degraded": True,
            "manual_training_requested": True,
            "regime_changed": True,
        },
    }
    
    logger.info(f"📊 CLM Configuration:")
    logger.info(f"   Retrain interval: {config['scheduler']['retrain_hours']} hours")
    logger.info(f"   Drift check: {config['scheduler']['drift_check_hours']} hours")
    logger.info(f"   Performance check: {config['scheduler']['performance_check_hours']} hours")
    
    # Initialize CLM v3 Service
    clm_service = ClmV3Service(event_bus=event_bus, config=config)
    
    # Start CLM service
    await clm_service.start()
    logger.info("✅ CLM v3 Service started")
    
    logger.info("=" * 60)
    logger.info("🚀 CLM v3 Microservice Running")
    logger.info("=" * 60)
    
    try:
        # Keep running
        while True:
            await asyncio.sleep(60)
            logger.debug("CLM heartbeat - service running")
    except KeyboardInterrupt:
        logger.info("Shutting down CLM service...")
    finally:
        await clm_service.stop()
        await event_bus.stop()
        await redis_client.close()
        logger.info("✅ CLM service stopped")


if __name__ == "__main__":
    asyncio.run(main())
