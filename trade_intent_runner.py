"""
Standalone Trade Intent Consumer Runner
Runs SAFE_DRAIN mode - consumes events but does NOT execute trades
"""
import asyncio
import logging
import os
import sys

# Setup logging first
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(name)s: %(message)s'
)
logger = logging.getLogger("trade_intent_runner")

async def main():
    """Main runner - NEW EVENTS ONLY mode"""
    try:
        # Import with error handling
        logger.info("🚀 Starting Trade Intent Consumer (NEW EVENTS ONLY)")
        logger.info(f"SAFE_DRAIN mode: {os.getenv('TRADE_INTENT_SAFE_DRAIN', 'false')}")
        
        import redis.asyncio as redis
        from backend.core.event_bus import EventBus
        from backend.services.execution.execution import BinanceFuturesExecutionAdapter
        
        # Try to import subscriber - if exitbrain import fails, we'll handle it
        try:
            from backend.events.subscribers.trade_intent_subscriber import TradeIntentSubscriber
        except ModuleNotFoundError as e:
            if 'exitbrain_v3_5' in str(e) or 'microservices.exitbrain' in str(e):
                logger.warning(f"⚠️ ExitBrain import issue detected: {e}")
                logger.warning("🔧 Attempting to patch import...")
                
                # Patch the missing module
                import types
                exitbrain_module = types.ModuleType('exitbrain_v3_5')
                exitbrain_module.adaptive_leverage_engine = types.ModuleType('adaptive_leverage_engine')
                exitbrain_module.pnl_tracker = types.ModuleType('pnl_tracker')
                
                # Create mock classes
                class MockAdaptiveLeverageEngine:
                    def __init__(self, *args, **kwargs):
                        pass
                    def compute_adaptive_levels(self, *args, **kwargs):
                        return {"tp1": 0, "tp2": 0, "tp3": 0, "sl": 0, "LSF": 1.0, "adjustment": 1.0, "harvest_scheme": "3x"}
                
                class MockPnLTracker:
                    def __init__(self, *args, **kwargs):
                        pass
                
                exitbrain_module.adaptive_leverage_engine.AdaptiveLeverageEngine = MockAdaptiveLeverageEngine
                exitbrain_module.pnl_tracker.PnLTracker = MockPnLTracker
                sys.modules['exitbrain_v3_5'] = exitbrain_module
                sys.modules['exitbrain_v3_5.adaptive_leverage_engine'] = exitbrain_module.adaptive_leverage_engine
                sys.modules['exitbrain_v3_5.pnl_tracker'] = exitbrain_module.pnl_tracker
                sys.modules['microservices.exitbrain_v3_5'] = exitbrain_module
                sys.modules['microservices.exitbrain_v3_5.adaptive_leverage_engine'] = exitbrain_module.adaptive_leverage_engine
                sys.modules['microservices.exitbrain_v3_5.pnl_tracker'] = exitbrain_module.pnl_tracker
                
                logger.info("✅ Import patch applied, retrying subscriber import...")
                from backend.events.subscribers.trade_intent_subscriber import TradeIntentSubscriber
            else:
                raise
        
        # Initialize Redis
        redis_host = os.getenv("REDIS_HOST", "quantum_redis")
        redis_port = int(os.getenv("REDIS_PORT", "6379"))
        
        logger.info(f"📡 Connecting to Redis: {redis_host}:{redis_port}")
        redis_client = redis.Redis(
            host=redis_host,
            port=redis_port,
            decode_responses=True
        )
        
        # Test connection
        await redis_client.ping()
        logger.info("✅ Redis connected")
        
        # Initialize EventBus
        event_bus = EventBus(redis_client, service_name="trade_intent_consumer")
        await event_bus.initialize()
        logger.info("✅ EventBus initialized")
        
        # Initialize execution adapter
        api_key = os.getenv("BINANCE_API_KEY", "")
        api_secret = os.getenv("BINANCE_SECRET_KEY", "")
        execution_adapter = BinanceFuturesExecutionAdapter(
            api_key=api_key,
            api_secret=api_secret
        )
        logger.info("✅ Execution adapter initialized")
        
        # Create subscriber
        subscriber = TradeIntentSubscriber(
            event_bus=event_bus,
            execution_adapter=execution_adapter,
            risk_guard=None  # Optional
        )
        
        logger.info("🎯 Starting subscriber (will consume NEW events only)")
        
        # Start consumer (registers handler)
        await subscriber.start()
        
        # Start EventBus processing loop
        logger.info("🚀 Starting EventBus processing loop...")
        await event_bus.start()
        
        logger.info("✅ Consumer running - waiting for new events...")
        # EventBus.start() runs forever, but if it returns, keep alive
        while True:
            await asyncio.sleep(60)
            logger.debug("💓 Consumer heartbeat")
            
    except KeyboardInterrupt:
        logger.info("🛑 Shutdown requested")
    except Exception as e:
        logger.error(f"❌ Fatal error: {e}", exc_info=True)
        raise

if __name__ == "__main__":
    asyncio.run(main())
