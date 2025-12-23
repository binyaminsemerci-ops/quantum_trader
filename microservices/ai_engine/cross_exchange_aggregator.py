"""
Cross-Exchange Aggregator
Consume raw exchange data, merge, normalize, and compute derived features
Publishes normalized data to Redis for AI Engine
"""
import asyncio
import redis.asyncio as aioredis
import pandas as pd
import numpy as np
from typing import Dict, List, Optional
import logging
import os
from datetime import datetime
from collections import defaultdict
import json
import sys
sys.path.append("/app")
from backend.infrastructure.redis_manager import RedisConnectionManager

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Redis configuration - use localhost if not in Docker
REDIS_URL = os.getenv("REDIS_URL", "redis://localhost:6379")
REDIS_STREAM_RAW = "quantum:stream:exchange.raw"
REDIS_STREAM_NORMALIZED = "quantum:stream:exchange.normalized"

# Symbols to process
SYMBOLS = ["BTCUSDT", "ETHUSDT", "SOLUSDT"]


class CrossExchangeAggregator:
    """Aggregate and normalize multi-exchange data"""
    
    def __init__(self, symbols: List[str] = None, redis_host: str = None, redis_port: int = 6379):
        # Use defaults if not provided
        if redis_host is None:
            redis_host = os.getenv("REDIS_HOST", "localhost")
        
        redis_url = f"redis://{redis_host}:{redis_port}"
        
        self.redis_url = redis_url
        self.redis_manager = RedisConnectionManager(redis_url=redis_url)
        self.redis_client: Optional[aioredis.Redis] = None
        self.running = False
        
        # Use provided symbols or default
        self.symbols = symbols or SYMBOLS
        
        # Buffer for merging data by timestamp
        self.data_buffer: Dict[str, Dict[int, Dict[str, float]]] = {
            symbol: defaultdict(dict) for symbol in self.symbols
        }
        
        # Last processed ID
        self.last_id = "0-0"
        
    async def connect_redis(self):
        """Connect to Redis"""
        try:
            await self.redis_manager.start()
            self.redis_client = self.redis_manager.redis
            logger.info("✅ Connected to Redis via RedisConnectionManager")
        except Exception as e:
            logger.error(f"❌ Failed to connect to Redis: {e}")
            raise
    
    async def close_redis(self):
        """Close Redis connection"""
        await self.redis_manager.stop()
        logger.info("Redis connection closed")
    
    def merge_exchange_data(self, symbol: str, timestamp: int) -> Optional[Dict]:
        """Merge data from all exchanges for a given timestamp"""
        if symbol not in self.data_buffer:
            return None
        
        if timestamp not in self.data_buffer[symbol]:
            return None
        
        exchange_data = self.data_buffer[symbol][timestamp]
        
        # Need data from at least 2 exchanges
        if len(exchange_data) < 2:
            return None
        
        try:
            prices = list(exchange_data.values())
            
            # Compute aggregated metrics
            avg_price = np.mean(prices)
            price_divergence = np.std(prices)
            
            result = {
                "symbol": symbol,
                "timestamp": timestamp,
                "avg_price": float(avg_price),
                "price_divergence": float(price_divergence),
                "num_exchanges": len(prices),
                "prices": {
                    "binance": exchange_data.get("binance"),
                    "bybit": exchange_data.get("bybit"),
                    "coinbase": exchange_data.get("coinbase")
                }
            }
            
            logger.debug(f"{symbol} @ {timestamp}: avg={avg_price:.2f}, div={price_divergence:.4f}")
            return result
        
        except Exception as e:
            logger.error(f"Failed to merge data for {symbol} @ {timestamp}: {e}")
            return None
    
    async def publish_normalized(self, data: Dict):
        """Publish normalized data to Redis stream"""
        try:
            message = {
                "symbol": data["symbol"],
                "timestamp": str(data["timestamp"]),
                "avg_price": str(data["avg_price"]),
                "price_divergence": str(data["price_divergence"]),
                "num_exchanges": str(data["num_exchanges"]),
                "binance_price": str(data["prices"].get("binance", "null")),
                "bybit_price": str(data["prices"].get("bybit", "null")),
                "coinbase_price": str(data["prices"].get("coinbase", "null"))
            }
            
            # Use Redis Manager with retry logic
            success = await self.redis_manager.publish(REDIS_STREAM_NORMALIZED, json.dumps(message))
            if success:
                logger.info(f"Published normalized: {data['symbol']} @ {data['avg_price']:.2f}")
            else:
                logger.warning(f"Failed to publish (circuit breaker open): {data['symbol']}")
        
        except Exception as e:
            logger.error(f"Failed to publish normalized data: {e}")
    
    def add_funding_delta(self, symbol: str, funding_data: Dict):
        """Add funding rate delta to aggregated data"""
        # This would be implemented when funding rate data is available
        # For now, we'll set it to 0
        return 0.0
    
    def get_latest_features(self, symbol: str) -> Optional[Dict]:
        """
        Get latest cross-exchange features for a symbol.
        Returns: Dict with volatility_factor, divergence, lead_lag
        """
        if symbol not in self.data_buffer:
            return None
        
        # Get most recent timestamp
        if not self.data_buffer[symbol]:
            return None
        
        latest_ts = max(self.data_buffer[symbol].keys())
        exchange_data = self.data_buffer[symbol][latest_ts]
        
        if len(exchange_data) < 2:
            return None
        
        try:
            prices = list(exchange_data.values())
            avg_price = np.mean(prices)
            std_price = np.std(prices)
            
            # Compute features
            volatility_factor = std_price / avg_price if avg_price > 0 else 0.0
            divergence = std_price
            
            # Lead/lag: which exchange is highest (leader)
            if "binance" in exchange_data and "bybit" in exchange_data:
                lead_lag = (exchange_data["binance"] - exchange_data["bybit"]) / avg_price
            else:
                lead_lag = 0.0
            
            return {
                "volatility_factor": float(volatility_factor),
                "divergence": float(divergence),
                "lead_lag": float(lead_lag),
                "num_exchanges": len(prices),
                "timestamp": latest_ts
            }
        except Exception as e:
            logger.error(f"Error computing features for {symbol}: {e}")
            return None
    
    async def process_raw_stream(self):
        """Consume raw exchange stream and aggregate"""
        logger.info(f"📡 Starting to consume {REDIS_STREAM_RAW}")
        
        while self.running:
            try:
                # Read new entries from stream
                entries = await self.redis_client.xread(
                    {REDIS_STREAM_RAW: self.last_id},
                    count=10,
                    block=1000  # Block for 1 second
                )
                
                if not entries:
                    continue
                
                for stream_name, messages in entries:
                    for message_id, message_data in messages:
                        try:
                            # Skip init message
                            if "init" in message_data:
                                self.last_id = message_id
                                continue
                            
                            # Parse tick data
                            exchange = message_data.get("exchange")
                            symbol = message_data.get("symbol")
                            timestamp = int(message_data.get("timestamp", 0))
                            close_price = float(message_data.get("close", 0))
                            
                            if not all([exchange, symbol, timestamp, close_price]):
                                logger.warning(f"Incomplete data: {message_data}")
                                self.last_id = message_id
                                continue
                            
                            # Add to buffer
                            if symbol in self.data_buffer:
                                self.data_buffer[symbol][timestamp][exchange] = close_price
                            
                            # Try to merge and publish
                            merged = self.merge_exchange_data(symbol, timestamp)
                            if merged:
                                # Add funding delta (placeholder for now)
                                merged["funding_delta"] = 0.0
                                
                                await self.publish_normalized(merged)
                                
                                # Clean up old timestamps from buffer
                                timestamps_to_remove = [
                                    ts for ts in self.data_buffer[symbol].keys()
                                    if ts < timestamp - 60  # Keep last 60 seconds
                                ]
                                for ts in timestamps_to_remove:
                                    del self.data_buffer[symbol][ts]
                            
                            self.last_id = message_id
                        
                        except Exception as e:
                            logger.error(f"Error processing message {message_id}: {e}")
                            self.last_id = message_id
            
            except Exception as e:
                logger.error(f"Error reading stream: {e}")
                await asyncio.sleep(1)
    
    async def start(self):
        """Start aggregator"""
        self.running = True
        await self.connect_redis()
        
        # Create normalized stream if it doesn't exist
        try:
            await self.redis_client.xadd(REDIS_STREAM_NORMALIZED, {"init": "1"}, maxlen=10000)
            logger.info(f"✅ Normalized stream ready: {REDIS_STREAM_NORMALIZED}")
        except Exception as e:
            logger.warning(f"Stream init warning: {e}")
        
        logger.info("🚀 Cross-Exchange Aggregator started")
        await self.process_raw_stream()
    
    async def stop(self):
        """Stop aggregator"""
        logger.info("⏹️ Stopping aggregator...")
        self.running = False
        await asyncio.sleep(1)
        await self.close_redis()


async def test_aggregator():
    """Test aggregator for 30 seconds"""
    logger.info("=== Testing Cross-Exchange Aggregator ===")
    logger.info("Make sure exchange_stream_bridge is running!")
    logger.info("Testing for 30 seconds...")
    
    aggregator = CrossExchangeAggregator()
    
    # Start aggregator
    start_task = asyncio.create_task(aggregator.start())
    
    # Run for 30 seconds
    await asyncio.sleep(30)
    
    # Stop aggregator
    await aggregator.stop()
    start_task.cancel()
    
    # Check normalized stream
    try:
        redis_client = await aioredis.from_url(REDIS_URL, decode_responses=True)
        
        raw_len = await redis_client.xlen(REDIS_STREAM_RAW)
        norm_len = await redis_client.xlen(REDIS_STREAM_NORMALIZED)
        
        logger.info(f"\n✅ Raw stream length: {raw_len}")
        logger.info(f"✅ Normalized stream length: {norm_len}")
        
        # Get latest normalized entries
        entries = await redis_client.xrevrange(REDIS_STREAM_NORMALIZED, count=3)
        logger.info(f"\n✅ Latest 3 normalized entries:")
        for entry_id, entry_data in entries:
            if "init" not in entry_data:
                logger.info(f"  {entry_id}:")
                logger.info(f"    Symbol: {entry_data.get('symbol')}")
                logger.info(f"    Avg Price: {entry_data.get('avg_price')}")
                logger.info(f"    Price Divergence: {entry_data.get('price_divergence')}")
                logger.info(f"    Exchanges: {entry_data.get('num_exchanges')}")
        
        await redis_client.close()
        
        if norm_len > 0:
            logger.info("\n✅ Aggregator test PASSED")
            return True
        else:
            logger.error("\n❌ Aggregator test FAILED - No normalized data")
            return False
    
    except Exception as e:
        logger.error(f"\n❌ Aggregator test FAILED: {e}")
        return False


if __name__ == "__main__":
    import sys
    
    if "--test" in sys.argv:
        result = asyncio.run(test_aggregator())
        sys.exit(0 if result else 1)
    else:
        # Run continuously
        aggregator = CrossExchangeAggregator()
        try:
            asyncio.run(aggregator.start())
        except KeyboardInterrupt:
            logger.info("Stopped by user")
