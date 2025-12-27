#!/usr/bin/env python3
"""
Runner for TradeIntentSubscriber
Initializes all required dependencies and starts the subscriber
"""
import sys
import os
import logging

# Set up paths
sys.path.insert(0, '/app')
sys.path.insert(0, '/app/backend')
sys.path.insert(0, '/app/microservices')
sys.path.insert(0, '/app/ai_engine')
os.environ['PYTHONPATH'] = '/app/backend:/app/microservices:/app/ai_engine:/app'

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Import required classes
from backend.core.event_bus import EventBus
from backend.services.execution.execution import BinanceFuturesExecutionAdapter
from backend.events.subscribers.trade_intent_subscriber import TradeIntentSubscriber

def main():
    """Initialize and start TradeIntentSubscriber"""
    logger.info("[runner.py] Initializing TradeIntentSubscriber dependencies...")
    
    try:
        # Initialize async Redis client for EventBus
        import redis.asyncio as redis
        redis_host = os.getenv("REDIS_HOST", "localhost")
        redis_port = int(os.getenv("REDIS_PORT", 6379))
        redis_client = redis.Redis(host=redis_host, port=redis_port, decode_responses=True)
        logger.info(f"✅ Redis client initialized (host={redis_host}:{redis_port})")
        
        # Initialize EventBus
        event_bus = EventBus(redis_client=redis_client)
        logger.info("✅ EventBus initialized")
        
        # Initialize BinanceFuturesExecutionAdapter with API keys from environment
        api_key = os.getenv("BINANCE_API_KEY")
        api_secret = os.getenv("BINANCE_API_SECRET")
        
        execution_adapter = BinanceFuturesExecutionAdapter(
            api_key=api_key,
            api_secret=api_secret
        )
        logger.info("✅ BinanceFuturesExecutionAdapter initialized")
        
        # Initialize TradeIntentSubscriber with dependencies
        subscriber = TradeIntentSubscriber(
            event_bus=event_bus,
            execution_adapter=execution_adapter
        )
        logger.info("✅ TradeIntentSubscriber initialized")
        
        # Start the subscriber and event bus (both are async)
        logger.info("[runner.py] Starting TradeIntentSubscriber...")
        import asyncio
        
        async def run_subscriber():
            """Run the subscriber and event bus"""
            await subscriber.start()
            await event_bus.start()
            
            # Keep running forever to let consumer tasks process events
            logger.info("✅ EventBus and subscriber started, waiting for events...")
            try:
                # Wait indefinitely - consumer tasks will run in background
                await asyncio.Event().wait()
            except asyncio.CancelledError:
                logger.info("Shutdown signal received, stopping...")
                await event_bus.stop()
                raise
        
        asyncio.run(run_subscriber())
        
    except Exception as e:
        logger.error(f"❌ Failed to start TradeIntentSubscriber: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == '__main__':
    main()
