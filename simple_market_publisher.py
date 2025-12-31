#!/usr/bin/env python3
"""
Simple standalone market data publisher for VPS deployment.
Publishes market.tick events to Redis for AI Engine consumption.
"""
import asyncio
import json
import logging
import os
from datetime import datetime, timezone

# Redis for event publishing
import redis.asyncio as redis_async

# Binance websocket for market data (use python-binance library)
from binance import AsyncClient, BinanceSocketManager

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Configuration
REDIS_HOST = os.getenv("REDIS_HOST", "redis")
REDIS_PORT = int(os.getenv("REDIS_PORT", "6379"))
REDIS_DB = int(os.getenv("REDIS_DB", "0"))

# Parse symbols from environment or use defaults
symbols_str = os.getenv("MARKET_SYMBOLS", "BTCUSDT,ETHUSDT,BNBUSDT,SOLUSDT,AVAXUSDT")
SYMBOLS = [s.strip() for s in symbols_str.split(",")]

redis_client: redis_async.Redis = None


async def publish_market_tick(symbol: str, price: float, volume: float):
    """Publish market.tick event to Redis"""
    try:
        event_data = {
            "symbol": symbol,
            "price": price,
            "volume": volume,
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }
        
        stream_key = "quantum:events:market.tick"
        await redis_client.xadd(
            stream_key,
            {"data": json.dumps(event_data)},
            maxlen=1000,  # Keep last 1000 events
        )
        logger.debug(f"[MARKET] Published tick: {symbol} @ ${price:.2f}")
    except Exception as e:
        logger.error(f"[ERROR] Failed to publish tick: {e}")


async def handle_trade_message(msg):
    """Handle Binance trade message"""
    try:
        if msg.get("e") == "trade":
            symbol = msg["s"]
            price = float(msg["p"])
            volume = float(msg["q"])
            await publish_market_tick(symbol, price, volume)
    except Exception as e:
        logger.error(f"[ERROR] Error handling trade message: {e}")


async def start_market_streams():
    """Start Binance WebSocket streams for all symbols"""
    global redis_client
    
    # Connect to Redis
    redis_client = redis_async.Redis(
        host=REDIS_HOST,
        port=REDIS_PORT,
        db=REDIS_DB,
        decode_responses=False,
    )
    await redis_client.ping()
    logger.info(f"[REDIS] ✅ Connected to {REDIS_HOST}:{REDIS_PORT}")
    
    # Initialize Binance Async Client
    client = await AsyncClient.create()
    bsm = BinanceSocketManager(client)
    
    # Split symbols into batches of 10 to avoid Binance rate limits
    batch_size = 10
    symbol_batches = [SYMBOLS[i:i + batch_size] for i in range(0, len(SYMBOLS), batch_size)]
    
    logger.info(f"[BINANCE] Starting {len(symbol_batches)} multiplex streams for {len(SYMBOLS)} symbols")
    
    # Handle one batch stream
    async def handle_batch_stream(batch_symbols, batch_num):
        streams = [f"{symbol.lower()}@trade" for symbol in batch_symbols]
        logger.info(f"[BINANCE] Batch {batch_num}: {len(streams)} symbols - {', '.join(batch_symbols[:3])}...")
        
        ms = bsm.multiplex_socket(streams)
        
        async with ms as stream:
            while True:
                try:
                    msg = await stream.recv()
                    
                    # Multiplex format wraps data
                    if "data" in msg:
                        await handle_trade_message(msg["data"])
                    else:
                        await handle_trade_message(msg)
                except Exception as e:
                    logger.error(f"[ERROR] Batch {batch_num} error: {e}")
                    await asyncio.sleep(1)
    
    # Run all batch streams concurrently
    tasks = [handle_batch_stream(batch, i+1) for i, batch in enumerate(symbol_batches)]
    await asyncio.gather(*tasks)


async def main():
    """Main entry point"""
    logger.info("=" * 60)
    logger.info("📈 SIMPLE MARKET DATA PUBLISHER")
    logger.info(f"Redis: {REDIS_HOST}:{REDIS_PORT}")
    logger.info(f"Symbols: {', '.join(SYMBOLS)}")
    logger.info("=" * 60)
    
    try:
        await start_market_streams()
    except KeyboardInterrupt:
        logger.info("[STOP] Shutting down...")
    except Exception as e:
        logger.error(f"[FATAL] {e}", exc_info=True)
    finally:
        if redis_client:
            await redis_client.close()


if __name__ == "__main__":
    asyncio.run(main())
