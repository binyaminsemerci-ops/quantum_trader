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

# Binance websocket for market data
from binance.client import Client
from binance.streams import BinanceSocketManager

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Configuration
REDIS_HOST = os.getenv("REDIS_HOST", "redis")
REDIS_PORT = int(os.getenv("REDIS_PORT", "6379"))
REDIS_DB = int(os.getenv("REDIS_DB", "0"))
SYMBOLS = ["BTCUSDT", "ETHUSDT", "BNBUSDT"]  # Symbols to track

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
        logger.info(f"[MARKET] Published tick: {symbol} @ ${price:.2f}")
    except Exception as e:
        logger.error(f"[ERROR] Failed to publish tick: {e}")


async def handle_socket_message(msg):
    """Handle Binance WebSocket message"""
    if msg["e"] == "trade":
        symbol = msg["s"]
        price = float(msg["p"])
        volume = float(msg["q"])
        await publish_market_tick(symbol, price, volume)


async def start_market_streams():
    """Start Binance WebSocket streams for all symbols"""
    global redis_client
    
    # Connect to Redis
    redis_client = await redis_async.Redis(
        host=REDIS_HOST,
        port=REDIS_PORT,
        db=REDIS_DB,
        decode_responses=False,
    )
    logger.info(f"[REDIS] Connected to {REDIS_HOST}:{REDIS_PORT}")
    
    # Initialize Binance client
    client = Client("", "")  # No API keys needed for public streams
    bsm = BinanceSocketManager(client)
    
    # Start streams for each symbol
    streams = []
    for symbol in SYMBOLS:
        stream = bsm.trade_socket(symbol)
        streams.append(stream)
        logger.info(f"[BINANCE] Starting stream for {symbol}")
    
    # Listen to streams
    async with streams[0] as stream_0, streams[1] as stream_1, streams[2] as stream_2:
        while True:
            # Read from all streams
            msg0 = await stream_0.recv()
            await handle_socket_message(msg0)
            
            msg1 = await stream_1.recv()
            await handle_socket_message(msg1)
            
            msg2 = await stream_2.recv()
            await handle_socket_message(msg2)


async def main():
    """Main entry point"""
    logger.info("[START] Simple Market Data Publisher")
    logger.info(f"[CONFIG] Redis: {REDIS_HOST}:{REDIS_PORT}")
    logger.info(f"[CONFIG] Symbols: {SYMBOLS}")
    
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
