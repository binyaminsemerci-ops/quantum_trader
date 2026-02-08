#!/usr/bin/env python3
"""
Quantum Price Feed Service
Subscribes to Binance testnet websocket and publishes prices to Redis.
"""
import os
import sys
import json
import time
import redis
import logging
import asyncio
import websockets
from typing import Dict, Any

# Logging setup
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | %(levelname)-8s | %(name)s | %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)


class PriceFeedService:
    """Binance testnet websocket → Redis price feed"""
    
    def __init__(self):
        self.redis = redis.Redis(
            host=os.getenv('REDIS_HOST', 'localhost'),
            port=int(os.getenv('REDIS_PORT', '6379')),
            decode_responses=True
        )
        
        # Binance testnet WebSocket
        self.ws_url = "wss://stream.binancefuture.com/ws/!markPrice@arr@1s"
        
        # Statistics
        self.price_updates = 0
        self.error_count = 0
        self.start_time = time.time()
        self.last_stats_time = time.time()
        
        # Active symbols (loaded from universe)
        self.active_symbols = set()
        self.load_active_symbols()
        
        logger.info(f"✅ Price Feed initialized: {len(self.active_symbols)} symbols tracked")
    
    def load_active_symbols(self):
        """Load active symbols from Redis universe"""
        try:
            universe_key = "quantum:universe:active"
            symbols = self.redis.smembers(universe_key)
            
            if symbols:
                self.active_symbols = set(symbols)
                logger.info(f"📡 Loaded {len(self.active_symbols)} symbols from universe")
            else:
                # Fallback 1: load from AI Engine's active symbols
                ai_symbols_key = "quantum:ai:active_symbols"
                ai_symbols = self.redis.smembers(ai_symbols_key)
                if ai_symbols:
                    self.active_symbols = set(ai_symbols)
                    logger.info(f"📡 Loaded {len(self.active_symbols)} symbols from AI Engine")
                else:
                    # Fallback 2: load from open positions
                    position_keys = self.redis.keys("quantum:position:*")
                    if position_keys:
                        self.active_symbols = {
                            key.replace("quantum:position:", "") 
                            for key in position_keys
                        }
                        logger.info(f"📡 Loaded {len(self.active_symbols)} symbols from open positions")
                    else:
                        # Final fallback: common symbols
                        self.active_symbols = {
                            'BTCUSDT', 'ETHUSDT', 'SOLUSDT', 'XRPUSDT',
                            'AAVEUSDT', 'ZKPUSDT', 'RIVERUSDT', 'XMRUSDT'
                        }
                        logger.warning(f"⚠️  Using fallback symbols: {len(self.active_symbols)} symbols")
        except Exception as e:
            logger.error(f"Failed to load symbols: {e}")
            self.active_symbols = {'BTCUSDT', 'ETHUSDT', 'SOLUSDT'}
    
    async def publish_price(self, symbol: str, price: float, mark_price: float = None):
        """Publish price to Redis in format that Harvest Brain expects"""
        try:
            # Format 1: quantum:ticker:{symbol} (preferred by Harvest Brain)
            ticker_key = f"quantum:ticker:{symbol}"
            ticker_data = {
                'symbol': symbol,
                'price': str(price),
                'markPrice': str(mark_price if mark_price else price),
                'timestamp': str(int(time.time() * 1000))
            }
            self.redis.hset(ticker_key, mapping=ticker_data)
            self.redis.expire(ticker_key, 10)  # 10 second TTL
            
            # Format 2: quantum:market:{symbol} (secondary fallback)
            market_key = f"quantum:market:{symbol}"
            market_data = {
                'symbol': symbol,
                'price': str(price),
                'timestamp': str(int(time.time()))
            }
            self.redis.hset(market_key, mapping=market_data)
            self.redis.expire(market_key, 10)
            
            self.price_updates += 1
            
        except Exception as e:
            logger.error(f"Failed to publish {symbol}: {e}")
            self.error_count += 1
    
    async def process_mark_price_message(self, data: Dict[str, Any]):
        """Process Binance mark price array message"""
        try:
            # Binance sends array of all mark prices
            if isinstance(data, list):
                for item in data:
                    symbol = item.get('s')
                    mark_price = float(item.get('p', 0))
                    
                    # Only publish if symbol is in active universe
                    if symbol in self.active_symbols and mark_price > 0:
                        await self.publish_price(symbol, mark_price, mark_price)
                        
        except Exception as e:
            logger.error(f"Failed to process mark price: {e}")
            self.error_count += 1
    
    async def print_stats(self):
        """Print statistics every 60 seconds"""
        while True:
            await asyncio.sleep(60)
            
            now = time.time()
            elapsed = now - self.start_time
            updates_per_sec = self.price_updates / elapsed if elapsed > 0 else 0
            
            logger.info(
                f"📊 STATS: {self.price_updates} updates, "
                f"{updates_per_sec:.1f} updates/sec, "
                f"{self.error_count} errors, "
                f"{len(self.active_symbols)} symbols"
            )
    
    async def websocket_listener(self):
        """Main websocket listener loop"""
        reconnect_delay = 1
        max_reconnect_delay = 60
        
        while True:
            try:
                logger.info(f"🔌 Connecting to Binance testnet: {self.ws_url}")
                
                async with websockets.connect(self.ws_url) as ws:
                    logger.info("✅ WebSocket connected successfully")
                    reconnect_delay = 1  # Reset on successful connection
                    
                    async for message in ws:
                        try:
                            data = json.loads(message)
                            await self.process_mark_price_message(data)
                            
                        except json.JSONDecodeError as e:
                            logger.warning(f"Invalid JSON: {e}")
                        except Exception as e:
                            logger.error(f"Message processing error: {e}")
                            
            except websockets.exceptions.ConnectionClosed:
                logger.warning(f"⚠️  WebSocket closed, reconnecting in {reconnect_delay}s...")
                await asyncio.sleep(reconnect_delay)
                reconnect_delay = min(reconnect_delay * 2, max_reconnect_delay)
                
            except Exception as e:
                logger.error(f"WebSocket error: {e}", exc_info=True)
                await asyncio.sleep(reconnect_delay)
                reconnect_delay = min(reconnect_delay * 2, max_reconnect_delay)
    
    async def symbol_refresh_loop(self):
        """Refresh active symbols every 5 minutes"""
        while True:
            await asyncio.sleep(300)  # 5 minutes
            self.load_active_symbols()
    
    async def run(self):
        """Run all tasks concurrently"""
        try:
            await asyncio.gather(
                self.websocket_listener(),
                self.print_stats(),
                self.symbol_refresh_loop()
            )
        except KeyboardInterrupt:
            logger.info("🛑 Service interrupted")
        except Exception as e:
            logger.error(f"Fatal error: {e}", exc_info=True)
            sys.exit(1)


if __name__ == "__main__":
    service = PriceFeedService()
    asyncio.run(service.run())
