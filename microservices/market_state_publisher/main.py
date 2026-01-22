#!/usr/bin/env python3
"""
P0.5 MarketState Metrics Publisher
Computes MarketState from live candles and publishes metrics only (NO trading)

systemd-only service
"""

import os
import sys
import time
import logging
from pathlib import Path
from collections import deque
from datetime import datetime
import asyncio

import redis.asyncio as aioredis
import numpy as np

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from ai_engine.market_state import MarketState


# Configuration from environment
SYMBOLS = os.getenv("MARKETSTATE_SYMBOLS", "BTCUSDT,ETHUSDT").split(",")
PUBLISH_INTERVAL_SEC = int(os.getenv("MARKETSTATE_PUBLISH_INTERVAL", "60"))
REDIS_HOST = os.getenv("MARKETSTATE_REDIS_HOST", "localhost")
REDIS_PORT = int(os.getenv("MARKETSTATE_REDIS_PORT", "6379"))
WINDOW_SIZE = int(os.getenv("MARKETSTATE_WINDOW_SIZE", "300"))  # Safety margin: 300 candles
SOURCE_MODE = os.getenv("MARKETSTATE_SOURCE", "candles")  # candles | synthetic
LOG_LEVEL = os.getenv("MARKETSTATE_LOG_LEVEL", "INFO")

# Logging setup
logging.basicConfig(
    level=getattr(logging, LOG_LEVEL),
    format='%(asctime)s [%(levelname)s] %(name)s: %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger("marketstate_publisher")


class MarketStatePublisher:
    """Publishes MarketState metrics to Redis (calc-only, no trading)"""
    
    def __init__(self):
        self.redis = None
        self.market_state = MarketState()
        
        # Rolling buffers for each symbol (FIFO deques)
        self.price_buffers = {symbol: deque(maxlen=WINDOW_SIZE) for symbol in SYMBOLS}
        
        # Rate limiting
        self.last_publish = {symbol: 0 for symbol in SYMBOLS}
        
        logger.info(f"Initialized MarketStatePublisher: symbols={SYMBOLS}, interval={PUBLISH_INTERVAL_SEC}s, source={SOURCE_MODE}")
    
    async def connect_redis(self):
        """Connect to Redis"""
        try:
            self.redis = await aioredis.from_url(
                f"redis://{REDIS_HOST}:{REDIS_PORT}",
                encoding="utf-8",
                decode_responses=True
            )
            await self.redis.ping()
            logger.info(f"✅ Connected to Redis at {REDIS_HOST}:{REDIS_PORT}")
        except Exception as e:
            logger.error(f"❌ Failed to connect to Redis: {e}")
            raise
    
    async def fetch_candles_from_cache(self, symbol: str, limit: int = WINDOW_SIZE) -> list:
        """
        Fetch recent candles from existing candle cache
        Tries multiple potential Redis key patterns used by the system
        """
        try:
            # Try different key patterns
            patterns = [
                f"candles:{symbol}:1m",  # Standard pattern
                f"quantum:candles:{symbol}",  # Namespaced
                f"klines:{symbol}:1m",  # Alternative
            ]
            
            for key in patterns:
                # Try sorted set (ZREVRANGE)
                candles = await self.redis.zrevrange(key, 0, limit - 1)
                if candles:
                    # Parse closes from JSON or direct values
                    closes = []
                    for candle in reversed(candles):  # Oldest first
                        try:
                            import json
                            data = json.loads(candle)
                            closes.append(float(data.get('close', data.get('c', 0))))
                        except:
                            # If not JSON, try direct float
                            try:
                                closes.append(float(candle))
                            except:
                                pass
                    
                    if closes:
                        logger.debug(f"Loaded {len(closes)} candles from {key}")
                        return closes
                
                # Try list (LRANGE)
                candles = await self.redis.lrange(key, -limit, -1)
                if candles:
                    closes = []
                    for candle in candles:
                        try:
                            import json
                            data = json.loads(candle)
                            closes.append(float(data.get('close', data.get('c', 0))))
                        except:
                            try:
                                closes.append(float(candle))
                            except:
                                pass
                    
                    if closes:
                        logger.debug(f"Loaded {len(closes)} candles from {key}")
                        return closes
            
            logger.warning(f"No candles found for {symbol} in Redis cache")
            return []
        
        except Exception as e:
            logger.error(f"Error fetching candles for {symbol}: {e}")
            return []
    
    def generate_synthetic_prices(self, symbol: str, n: int = WINDOW_SIZE) -> list:
        """Generate synthetic prices for testing (fallback mode)"""
        np.random.seed(hash(symbol) % 2**32)
        
        # Different behavior per symbol
        if "BTC" in symbol:
            # Trending
            returns = np.random.randn(n) * 0.01 + 0.001
            prices = 45000 * np.exp(np.cumsum(returns))
        elif "ETH" in symbol:
            # Mean-reverting
            prices = np.zeros(n)
            prices[0] = 2500.0
            for i in range(1, n):
                reversion = 0.05 * (2500 - prices[i-1])
                shock = np.random.randn() * 15
                prices[i] = prices[i-1] + reversion + shock
        else:
            # Choppy
            returns = np.random.randn(n) * 0.015
            prices = 100 + np.cumsum(returns)
        
        return prices.tolist()
    
    async def update_price_buffer(self, symbol: str):
        """Update price buffer for a symbol"""
        if SOURCE_MODE == "synthetic":
            # Generate full synthetic buffer
            prices = self.generate_synthetic_prices(symbol, WINDOW_SIZE)
            self.price_buffers[symbol] = deque(prices, maxlen=WINDOW_SIZE)
            logger.debug(f"{symbol}: Generated {len(prices)} synthetic prices")
        else:
            # Fetch from candle cache
            closes = await self.fetch_candles_from_cache(symbol)
            if closes:
                # Append to buffer (deque auto-maintains maxlen)
                for close in closes:
                    self.price_buffers[symbol].append(close)
                logger.debug(f"{symbol}: Buffer size = {len(self.price_buffers[symbol])}")
            else:
                # Fallback: generate synthetic if no real data
                logger.warning(f"{symbol}: No candles found, using synthetic fallback")
                prices = self.generate_synthetic_prices(symbol, min(50, WINDOW_SIZE))
                for price in prices:
                    self.price_buffers[symbol].append(price)
    
    async def compute_and_publish(self, symbol: str):
        """Compute MarketState and publish metrics"""
        now = time.time()
        
        # Rate limiting
        if now - self.last_publish[symbol] < PUBLISH_INTERVAL_SEC:
            return
        
        # Check if we have enough data
        buffer = self.price_buffers[symbol]
        if len(buffer) < 257:  # Need window + 1 for returns
            logger.debug(f"{symbol}: Insufficient data ({len(buffer)}/257), skipping")
            return
        
        try:
            # Convert buffer to numpy array
            prices = np.array(list(buffer))
            
            # Compute MarketState
            state = self.market_state.get_state(symbol, prices)
            
            if not state:
                logger.warning(f"{symbol}: MarketState returned None")
                return
            
            # Publish to Redis hash
            redis_key = f"quantum:marketstate:{symbol}"
            
            metrics = {
                'sigma': f"{state['sigma']:.8f}",
                'mu': f"{state['mu']:.8f}",
                'ts': f"{state['ts']:.6f}",
                'p_trend': f"{state['regime_probs']['trend']:.6f}",
                'p_mr': f"{state['regime_probs']['mr']:.6f}",
                'p_chop': f"{state['regime_probs']['chop']:.6f}",
                'dp': f"{state['features']['dp']:.6f}",
                'vr': f"{state['features']['vr']:.6f}",
                'spike_proxy': f"{state['features']['spike_proxy']:.6f}",
                'ts_timestamp': str(int(now)),
                'buffer_size': str(len(buffer))
            }
            
            await self.redis.hset(redis_key, mapping=metrics)
            await self.redis.expire(redis_key, PUBLISH_INTERVAL_SEC * 10)  # TTL
            
            # Optional: publish to stream for time series
            stream_key = "quantum:stream:marketstate"
            stream_data = {
                'symbol': symbol,
                'ts': f"{state['ts']:.6f}",
                'sigma': f"{state['sigma']:.8f}",
                'mu': f"{state['mu']:.8f}",
                'regime': max(state['regime_probs'].items(), key=lambda x: x[1])[0],
                'timestamp': str(int(now))
            }
            await self.redis.xadd(stream_key, stream_data, maxlen=10000)
            
            # Log summary
            dominant = max(state['regime_probs'].items(), key=lambda x: x[1])
            logger.info(
                f"{symbol}: σ={state['sigma']:.6f} μ={state['mu']:.6f} TS={state['ts']:.4f} "
                f"regime={dominant[0]}({dominant[1]:.1%}) buffer={len(buffer)}"
            )
            
            self.last_publish[symbol] = now
        
        except Exception as e:
            logger.error(f"{symbol}: Error computing/publishing: {e}", exc_info=True)
    
    async def run_loop(self):
        """Main event loop"""
        logger.info("Starting MarketState publisher loop...")
        
        while True:
            try:
                for symbol in SYMBOLS:
                    # Update buffer
                    await self.update_price_buffer(symbol)
                    
                    # Compute and publish
                    await self.compute_and_publish(symbol)
                
                # Sleep interval
                await asyncio.sleep(PUBLISH_INTERVAL_SEC)
            
            except KeyboardInterrupt:
                logger.info("Received shutdown signal")
                break
            except Exception as e:
                logger.error(f"Error in main loop: {e}", exc_info=True)
                await asyncio.sleep(5)  # Backoff on errors
    
    async def shutdown(self):
        """Cleanup"""
        if self.redis:
            await self.redis.close()
        logger.info("Shutdown complete")


async def main():
    """Entry point"""
    logger.info("=" * 80)
    logger.info("P0.5 MarketState Metrics Publisher — CALC ONLY (NO TRADING)")
    logger.info("=" * 80)
    logger.info(f"Symbols: {SYMBOLS}")
    logger.info(f"Publish interval: {PUBLISH_INTERVAL_SEC}s")
    logger.info(f"Source mode: {SOURCE_MODE}")
    logger.info(f"Redis: {REDIS_HOST}:{REDIS_PORT}")
    logger.info("=" * 80)
    
    publisher = MarketStatePublisher()
    
    try:
        await publisher.connect_redis()
        await publisher.run_loop()
    except KeyboardInterrupt:
        logger.info("Shutting down...")
    finally:
        await publisher.shutdown()


if __name__ == "__main__":
    asyncio.run(main())
