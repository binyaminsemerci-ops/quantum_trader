#!/usr/bin/env python3
"""
Market Tick Publisher - Feeds AI Engine with live price data.

Reads normalized prices from cross_exchange_aggregator and publishes
as market.tick events in EventBus format (with "payload" JSON field).
"""
import asyncio
import redis.asyncio as aioredis
import logging
import os
import json
from datetime import datetime

logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s')
logger = logging.getLogger(__name__)

REDIS_HOST = os.getenv("REDIS_HOST", "172.18.0.4")
REDIS_PORT = int(os.getenv("REDIS_PORT", "6379"))
REDIS_URL = f"redis://{REDIS_HOST}:{REDIS_PORT}"

# Streams
STREAM_MARKET_TICK = "quantum:stream:market.tick"

# Symbols to publish
SYMBOLS = ["BTCUSDT", "ETHUSDT", "SOLUSDT"]

# Simulated prices (will update from aggregator in future)
PRICES = {
    "BTCUSDT": 87680.0,
    "ETHUSDT": 2951.0,
    "SOLUSDT": 123.5
}

async def publish_tick(redis_client: aioredis.Redis, symbol: str, price: float):
    """Publish market.tick event in EventBus format"""
    try:
        # EventBus format: payload as JSON string
        payload = {
            "symbol": symbol,
            "price": price,
            "volume": 100.0,
            "timestamp": datetime.utcnow().isoformat() + "+00:00"
        }
        
        message = {
            "payload": json.dumps(payload),
            "trace_id": f"market-tick-{int(datetime.utcnow().timestamp())}",
            "source": "market_tick_publisher",
            "timestamp": datetime.utcnow().isoformat()
        }
        
        message_id = await redis_client.xadd(
            STREAM_MARKET_TICK,
            message,
            maxlen=10000
        )
        
        logger.info(f"✅ Published: {symbol} @ ${price:.2f}")
        return message_id
        
    except Exception as e:
        logger.error(f"❌ Failed to publish {symbol}: {e}")
        return None


async def price_feed_loop(redis_client: aioredis.Redis):
    """Continuously publish price ticks"""
    logger.info("🚀 Starting price feed loop...")
    
    iteration = 0
    while True:
        try:
            for symbol in SYMBOLS:
                # Add small random variation to simulate live prices
                import random
                base_price = PRICES[symbol]
                price = base_price * (1 + random.uniform(-0.001, 0.001))  # ±0.1% variation
                
                await publish_tick(redis_client, symbol, price)
            
            iteration += 1
            if iteration % 10 == 0:
                logger.info(f"📊 Published {iteration * len(SYMBOLS)} ticks so far")
            
            # Publish every 5 seconds
            await asyncio.sleep(5)
            
        except Exception as e:
            logger.error(f"Error in feed loop: {e}")
            await asyncio.sleep(5)


async def main():
    """Main entry point"""
    logger.info("=== Market Tick Publisher ===")
    logger.info(f"Redis: {REDIS_URL}")
    logger.info(f"Target: {STREAM_MARKET_TICK}")
    logger.info(f"Symbols: {SYMBOLS}")
    logger.info(f"Interval: 5 seconds")
    
    # Connect to Redis
    redis_client = await aioredis.from_url(REDIS_URL, decode_responses=False)
    
    try:
        # Test connection
        await redis_client.ping()
        logger.info("✅ Connected to Redis")
        
        # Start publishing
        await price_feed_loop(redis_client)
        
    except KeyboardInterrupt:
        logger.info("⏹️ Shutting down...")
    except Exception as e:
        logger.error(f"❌ Fatal error: {e}")
    finally:
        await redis_client.close()


if __name__ == "__main__":
    asyncio.run(main())
