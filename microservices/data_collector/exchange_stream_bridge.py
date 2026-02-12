"""
Exchange Stream Bridge
Maintain live WebSocket streams for real-time candles and price updates
Publishes to Redis EventBus
"""
import asyncio
import websockets
import json
import redis.asyncio as aioredis
from typing import Optional, Dict, List
import logging
import os
from datetime import datetime
import sys
sys.path.append('/app/backend/infrastructure')
try:
    from redis_manager import RedisConnectionManager
except ImportError:
    # Fallback: Create simple Redis manager if import fails
    class RedisConnectionManager:
        def __init__(self, url):
            self.url = url
            self.client = None
            self.healthy = False
            self.consecutive_failures = 0
        
        async def start(self):
            """Start connection"""
            if not self.client:
                self.client = await aioredis.from_url(self.url)
                self.healthy = True
        
        async def stop(self):
            """Stop connection"""
            if self.client:
                await self.client.close()
                self.client = None
                self.healthy = False
        
        async def get_connection(self):
            if not self.client:
                await self.start()
            return self.client

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# WebSocket Endpoints
WS_BINANCE_BASE = "wss://stream.binance.com:9443/ws"
WS_BYBIT = "wss://stream.bybit.com/v5/public/linear"

# Symbols to stream - DYNAMIC from Universe or config
SYMBOLS_CONFIG = os.getenv("EXCHANGE_BRIDGE_SYMBOLS", "USE_UNIVERSE_SERVICE")
MAX_SYMBOLS = int(os.getenv("EXCHANGE_BRIDGE_MAX_SYMBOLS", "20"))  # WebSocket limits

# Redis configuration - use localhost if not in Docker
REDIS_URL = os.getenv("REDIS_URL", "redis://localhost:6379")
REDIS_STREAM_RAW = "quantum:stream:exchange.raw"


def load_universe_symbols(redis_host: str = "127.0.0.1", redis_port: int = 6379, max_count: int = 20) -> List[str]:
    """
    Load top N symbols from Universe Service (sorted by volume/liquidity).
    
    Returns:
        List[str]: Top N symbols from universe (max_count limit)
    
    Raises:
        RuntimeError: If Universe Service unavailable or config error
    """
    import redis
    
    if SYMBOLS_CONFIG != "USE_UNIVERSE_SERVICE":
        # Parse comma-separated list
        symbols = [s.strip() for s in SYMBOLS_CONFIG.split(",") if s.strip()]
        logger.info(f"✅ Loaded {len(symbols)} symbols from config")
        return symbols[:max_count]  # Respect limit
    
    try:
        r = redis.Redis(host=redis_host, port=redis_port, decode_responses=True)
        
        # Universe Service publishes to quantum:cfg:universe:active
        universe_key = "quantum:cfg:universe:active"
        universe_json = r.get(universe_key)
        
        if not universe_json:
            raise RuntimeError(
                f"❌ Universe Service NOT AVAILABLE! Key '{universe_key}' empty. "
                f"Start quantum-universe-service first or set EXCHANGE_BRIDGE_SYMBOLS!"
            )
        
        universe_data = json.loads(universe_json)
        symbols = universe_data.get("symbols", [])
        
        if not symbols:
            raise RuntimeError(f"❌ Universe data empty! Check quantum-universe-service logs.")
        
        # Take top N symbols (Universe already sorted by volume/liquidity)
        selected = symbols[:max_count]
        
        logger.info(f"🌍 Universe loaded: {len(symbols)} total, using top {len(selected)} for WebSocket streams")
        logger.info(f"📊 Symbols: {', '.join(selected[:10])}{'...' if len(selected) > 10 else ''}")
        
        return selected
        
    except redis.ConnectionError as e:
        raise RuntimeError(f"❌ Cannot connect to Redis at {redis_host}:{redis_port}: {e}")
    except json.JSONDecodeError as e:
        raise RuntimeError(f"❌ Invalid JSON in universe data: {e}")


class ExchangeStreamBridge:
    """WebSocket bridge for multi-exchange live data"""
    
    def __init__(self, redis_url: str = REDIS_URL, symbols: List[str] = None):
        self.redis_url = redis_url
        self.redis_manager = RedisConnectionManager(url=redis_url)
        self.redis_client: Optional[aioredis.Redis] = None
        self.binance_ws: Optional[websockets.WebSocketClientProtocol] = None
        self.bybit_ws: Optional[websockets.WebSocketClientProtocol] = None
        self.running = False
        
        # Load symbols dynamically (NO hardcoding!)
        if symbols:
            self.symbols = symbols
        else:
            # Extract redis host/port from URL for universe loading
            redis_host = os.getenv("REDIS_HOST", "127.0.0.1")
            redis_port = int(os.getenv("REDIS_PORT", "6379"))
            self.symbols = load_universe_symbols(redis_host, redis_port, MAX_SYMBOLS)
            logger.info(f"✅ Loaded {len(self.symbols)} symbols for WebSocket streams")
        
    async def connect_redis(self):
        """Connect to Redis"""
        try:
            await self.redis_manager.start()
            self.redis_client = self.redis_manager.client
            logger.info("βœ… Connected to Redis via RedisConnectionManager")
        except Exception as e:
            logger.error(f"❌ Failed to connect to Redis: {e}")
            raise
    
    async def close_redis(self):
        """Close Redis connection"""
        await self.redis_manager.stop()
        logger.info("Redis connection closed")
    
    async def publish_to_redis(self, data: Dict):
        """Publish tick data to Redis stream"""
        # Use Redis Manager client with resilience
        if not self.redis_manager.healthy or not self.redis_manager.client:
            return  # Silently skip when unhealthy
        
        try:
            # Convert to JSON string for Redis
            message = {
                "exchange": data["exchange"],
                "symbol": data["symbol"],
                "timestamp": str(data["t"]),
                "open": str(data["o"]),
                "high": str(data["h"]),
                "low": str(data["l"]),
                "close": str(data["c"]),
                "volume": str(data["v"])
            }
            
            await self.redis_manager.client.xadd(REDIS_STREAM_RAW, message)
            logger.debug(f"Published {data['exchange']}:{data['symbol']} @ {data['c']}")
            
        except Exception as e:
            # Only log first error to avoid spam
            if self.redis_manager.healthy:
                logger.warning(f"Redis publish failed - marking unhealthy: {e}")
                self.redis_manager.healthy = False
                self.redis_manager.consecutive_failures += 1
                # Close broken connection
                if self.redis_manager.client:
                    try:
                        await self.redis_manager.client.close()
                    except:
                        pass
                    self.redis_manager.client = None
    
    async def handle_binance_stream(self, symbol: str):
        """Handle Binance WebSocket stream for a symbol"""
        stream_name = f"{symbol.lower()}@kline_1m"
        ws_url = f"{WS_BINANCE_BASE}/{stream_name}"
        
        while self.running:
            try:
                async with websockets.connect(ws_url) as websocket:
                    logger.info(f"Γ£à Connected to Binance stream: {symbol}")
                    
                    async for message in websocket:
                        if not self.running:
                            break
                        
                        try:
                            data = json.loads(message)
                            
                            # Binance kline format
                            if 'k' in data:
                                kline = data['k']
                                tick = {
                                    "exchange": "binance",
                                    "symbol": symbol,
                                    "t": int(kline['t']) // 1000,  # Convert to seconds
                                    "o": float(kline['o']),
                                    "h": float(kline['h']),
                                    "l": float(kline['l']),
                                    "c": float(kline['c']),
                                    "v": float(kline['v'])
                                }
                                
                                await self.publish_to_redis(tick)
                        
                        except json.JSONDecodeError as e:
                            logger.error(f"Failed to parse Binance message: {e}")
                        except Exception as e:
                            logger.error(f"Error processing Binance tick: {e}")
            
            except websockets.exceptions.WebSocketException as e:
                logger.error(f"Binance WebSocket error for {symbol}: {e}")
                if self.running:
                    logger.info("Reconnecting in 5 seconds...")
                    await asyncio.sleep(5)
            except Exception as e:
                logger.error(f"Unexpected error in Binance stream {symbol}: {e}")
                if self.running:
                    await asyncio.sleep(5)
    
    async def handle_bybit_stream(self):
        """Handle Bybit WebSocket stream for all symbols"""
        while self.running:
            try:
                async with websockets.connect(WS_BYBIT) as websocket:
                    logger.info("Γ£à Connected to Bybit stream")
                    
                    # Subscribe to kline streams
                    subscribe_msg = {
                        "op": "subscribe",
                        "args": [f"kline.1.{symbol}" for symbol in self.symbols]
                    }
                    await websocket.send(json.dumps(subscribe_msg))
                    
                    async for message in websocket:
                        if not self.running:
                            break
                        
                        try:
                            data = json.loads(message)
                            
                            # Bybit kline format
                            if data.get('topic', '').startswith('kline'):
                                if 'data' in data and len(data['data']) > 0:
                                    kline = data['data'][0]
                                    
                                    # Extract symbol from topic (e.g., "kline.1.BTCUSDT")
                                    symbol = data['topic'].split('.')[-1]
                                    
                                    tick = {
                                        "exchange": "bybit",
                                        "symbol": symbol,
                                        "t": int(kline['start']) // 1000,  # Convert to seconds
                                        "o": float(kline['open']),
                                        "h": float(kline['high']),
                                        "l": float(kline['low']),
                                        "c": float(kline['close']),
                                        "v": float(kline['volume'])
                                    }
                                    
                                    await self.publish_to_redis(tick)
                        
                        except json.JSONDecodeError as e:
                            logger.error(f"Failed to parse Bybit message: {e}")
                        except Exception as e:
                            logger.error(f"Error processing Bybit tick: {e}")
            
            except websockets.exceptions.WebSocketException as e:
                logger.error(f"Bybit WebSocket error: {e}")
                if self.running:
                    logger.info("Reconnecting in 5 seconds...")
                    await asyncio.sleep(5)
            except Exception as e:
                logger.error(f"Unexpected error in Bybit stream: {e}")
                if self.running:
                    await asyncio.sleep(5)
    
    async def start(self):
        """Start all WebSocket streams"""
        self.running = True
        
        # Connect to Redis
        await self.connect_redis()
        
        # Create stream if it doesn't exist
        try:
            await self.redis_client.xadd(REDIS_STREAM_RAW, {"init": "1"}, maxlen=10000)
            logger.info(f"Γ£à Redis stream ready: {REDIS_STREAM_RAW}")
        except Exception as e:
            logger.warning(f"Stream init warning: {e}")
        
        # Start all streams concurrently
        tasks = []
        
        # Binance streams (one per symbol)
        for symbol in self.symbols:
            tasks.append(asyncio.create_task(self.handle_binance_stream(symbol)))
        
        # Bybit stream (all symbols in one connection)
        tasks.append(asyncio.create_task(self.handle_bybit_stream()))
        
        logger.info(f"≡ƒÜÇ Started {len(tasks)} WebSocket streams")
        
        # Wait for all tasks
        await asyncio.gather(*tasks, return_exceptions=True)
    
    async def stop(self):
        """Stop all streams"""
        logger.info("ΓÅ╣∩╕Å Stopping all streams...")
        self.running = False
        await asyncio.sleep(2)  # Give time for graceful shutdown
        await self.close_redis()


async def test_stream_bridge():
    """Test stream bridge for 30 seconds"""
    logger.info("=== Testing Exchange Stream Bridge ===")
    logger.info("Running for 30 seconds...")
    
    bridge = ExchangeStreamBridge()
    
    # Start bridge
    start_task = asyncio.create_task(bridge.start())
    
    # Run for 30 seconds
    await asyncio.sleep(30)
    
    # Stop bridge
    await bridge.stop()
    start_task.cancel()
    
    # Check Redis stream
    try:
        redis_client = await aioredis.from_url(REDIS_URL, decode_responses=True)
        stream_len = await redis_client.xlen(REDIS_STREAM_RAW)
        logger.info(f"\nΓ£à Redis stream length: {stream_len}")
        
        # Get latest entries
        entries = await redis_client.xrevrange(REDIS_STREAM_RAW, count=3)
        logger.info(f"Γ£à Latest 3 entries:")
        for entry_id, entry_data in entries:
            logger.info(f"  {entry_id}: {entry_data}")
        
        await redis_client.close()
        
        if stream_len > 0:
            logger.info("\nΓ£à Stream bridge test PASSED")
            return True
        else:
            logger.error("\nΓ¥î Stream bridge test FAILED - No data in stream")
            return False
    
    except Exception as e:
        logger.error(f"\nΓ¥î Stream bridge test FAILED: {e}")
        return False


if __name__ == "__main__":
    import sys
    
    if "--test" in sys.argv:
        result = asyncio.run(test_stream_bridge())
        sys.exit(0 if result else 1)
    else:
        # Run continuously
        bridge = ExchangeStreamBridge()
        try:
            asyncio.run(bridge.start())
        except KeyboardInterrupt:
            logger.info("Stopped by user")
