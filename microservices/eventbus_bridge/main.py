import asyncio
import json
import redis.asyncio as redis
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

async def main():
    logger.info('[BRIDGE] Starting EventBus → Redis signal bridge...')
    r = await redis.from_url('redis://redis:6379', decode_responses=False)
    
    # Test connection
    await r.ping()
    logger.info('[BRIDGE] ✅ Connected to Redis')
    
    stream_id = '$'  # Start from newest messages only
    signal_buffer = []
    
    while True:
        try:
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
                    
                    # Write to Redis key that auto-executor reads
                    await r.set('live_signals', json.dumps(signal_buffer))
                    
                    symbol = signal.get('symbol', 'UNKNOWN')
                    side = signal.get('side', 'UNKNOWN')
                    logger.info(f'[BRIDGE] ✅ Forwarded: {symbol} {side} (total buffered: {len(signal_buffer)})')
            
            await asyncio.sleep(0.5)
            
        except Exception as e:
            logger.error(f'[BRIDGE] Error: {e}')
            await asyncio.sleep(5)

if __name__ == '__main__':
    asyncio.run(main())
