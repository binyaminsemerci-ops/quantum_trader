import asyncio
import json
import redis.asyncio as redis
import logging
import sys
sys.path.append("/app")
from backend.infrastructure.redis_manager import RedisConnectionManager

# [EPIC-OBS-001] Initialize observability (tracing, metrics, structured logging)
try:
    from backend.infra.observability import init_observability, get_logger
    OBSERVABILITY_AVAILABLE = True
except ImportError:
    OBSERVABILITY_AVAILABLE = False
    logging.basicConfig(level=logging.INFO)

# Initialize observability at module level
if OBSERVABILITY_AVAILABLE:
    init_observability(
        service_name="eventbus-bridge",
        log_level="INFO",
        enable_tracing=True,
        enable_metrics=True,
    )
    logger = get_logger(__name__)
else:
    logger = logging.getLogger(__name__)

async def main():
    logger.info('[BRIDGE] Starting EventBus → Redis signal bridge...')
    
    # Use Redis Connection Manager with resilience
    redis_manager = RedisConnectionManager(redis_url='redis://redis:6379')
    await redis_manager.start()
    
    logger.info('[BRIDGE] ✅ Connected to Redis via RedisConnectionManager')
    
    stream_id = '$'  # Start from newest messages only
    signal_buffer = []
    
    while True:
        try:
            # Use the underlying Redis client for xread (not in manager)
            r = redis_manager.redis
            
            # Read from EventBus stream 'quantum:stream:trade.intent'
            result = await r.xread({'quantum:stream:trade.intent': stream_id}, count=50, block=1000)
            
            for stream_name, messages in result:
                for message_id, data in messages:
                    stream_id = message_id
                    
                    # Parse signal from payload field
                    payload = data.get(b'payload', b'{}')
                    signal = json.loads(payload)
                    signal_buffer.append(signal)
                    
                    # Keep latest 50 signals
                    if len(signal_buffer) > 50:
                        signal_buffer = signal_buffer[-50:]
                    
                    # Write to Redis key using manager (with retry)
                    await redis_manager.set('live_signals', json.dumps(signal_buffer))
                    
                    symbol = signal.get('symbol', 'UNKNOWN')
                    side = signal.get('side', 'UNKNOWN')
                    logger.info(f'[BRIDGE] ✅ Forwarded: {symbol} {side} (total buffered: {len(signal_buffer)})')
            
            await asyncio.sleep(0.5)
            
        except Exception as e:
            logger.error(f'[BRIDGE] Error: {e}')
            # Redis Manager handles reconnection automatically
            await asyncio.sleep(5)

if __name__ == '__main__':
    asyncio.run(main())
