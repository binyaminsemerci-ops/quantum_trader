#!/usr/bin/env python3
"""
Mock market data publisher - generates fake ticks for testing.
Uses only Redis (no external dependencies).
"""
import asyncio
import json
import random
import logging
from datetime import datetime, timezone

try:
    import redis.asyncio as redis_async
except ImportError:
    print("ERROR: redis package not installed")
    exit(1)

logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s')
logger = logging.getLogger(__name__)

# Configuration
REDIS_HOST = "redis"
REDIS_PORT = 6379
SYMBOLS = ["BTCUSDT", "ETHUSDT", "BNBUSDT"]
BASE_PRICES = {"BTCUSDT": 43000, "ETHUSDT": 2300, "BNBUSDT": 310}


async def publish_tick(redis_client, symbol: str, price: float):
    """Publish market.tick event to Redis"""
    event_data = {
        "symbol": symbol,
        "price": price,
        "volume": random.uniform(0.1, 10.0),
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }
    
    # FIXED: Use correct stream prefix to match EventBus (quantum:stream: not quantum:events:)
    # FIXED: Use 'payload' key to match EventBus expectation (not 'data')
    stream_key = "quantum:stream:market.tick"
    await redis_client.xadd(
        stream_key,
        {"payload": json.dumps(event_data)},
        maxlen=1000,
    )
    logger.info(f"✅ Published: {symbol} @ ${price:.2f}")


async def generate_mock_ticks():
    """Generate mock market ticks continuously"""
    redis_client = await redis_async.Redis(
        host=REDIS_HOST,
        port=REDIS_PORT,
        db=0,
        decode_responses=False,
    )
    logger.info(f"🔌 Connected to Redis at {REDIS_HOST}:{REDIS_PORT}")
    
    # Current prices (will fluctuate)
    prices = BASE_PRICES.copy()
    
    try:
        counter = 0
        while True:
            # Update prices with small random changes (-0.1% to +0.1%)
            for symbol in SYMBOLS:
                change_pct = random.uniform(-0.001, 0.001)
                prices[symbol] *= (1 + change_pct)
                
                # Publish tick
                await publish_tick(redis_client, symbol, prices[symbol])
                counter += 1
            
            # Status every 100 ticks
            if counter % 100 == 0:
                logger.info(f"📊 Total ticks published: {counter}")
            
            # Publish 3 ticks/second (one per symbol)
            await asyncio.sleep(1)
            
    except KeyboardInterrupt:
        logger.info("⏹️  Stopping...")
    finally:
        await redis_client.close()


if __name__ == "__main__":
    logger.info("🚀 Mock Market Data Publisher Starting...")
    asyncio.run(generate_mock_ticks())
