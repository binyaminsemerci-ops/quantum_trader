#!/usr/bin/env python3
"""Production Market Data Publisher - v1.1
Fetches real-time market data from Binance and publishes to Redis Streams.
"""
import asyncio, json, logging, os, random, sys, time, uuid
from datetime import datetime
from typing import Dict, List, Optional
import redis
from binance import AsyncClient, BinanceSocketManager

# Configuration
REDIS_HOST = os.getenv("REDIS_HOST", "127.0.0.1")
REDIS_PORT = int(os.getenv("REDIS_PORT", "6379"))
REDIS_DB = int(os.getenv("REDIS_DB", "0"))
LOGLEVEL = os.getenv("LOGLEVEL", "INFO")
LOG_FILE = os.getenv("LOG_FILE", "/var/log/quantum/market-publisher.log")
SYMBOLS_STR = os.getenv("SYMBOLS", "BTCUSDT,ETHUSDT,BNBUSDT,SOLUSDT,XRPUSDT,ADAUSDT,AVAXUSDT,DOTUSDT,MATICUSDT,LINKUSDT,UNIUSDT,ATOMUSDT,ETCUSDT,LTCUSDT,DOGEUSDT,SHIBUSDT,TRXUSDT,AAVEUSDT,INJUSDT,ARBUSDT,OPUSDT,ICPUSDT,APTUSDT,SUIUSDT,STXUSDT,FILUSDT,RENDERUSDT,IMXUSDT,LDOUSDT,TIAUSDT")
SYMBOLS: List[str] = [s.strip() for s in SYMBOLS_STR.split(",") if s.strip()]

STREAM_TICK = "quantum:stream:market.tick"
STREAM_KLINES = "quantum:stream:market.klines"
RECONNECT_BASE, RECONNECT_MAX, RECONNECT_JITTER = 1.0, 60.0, 0.2
HEALTH_INTERVAL = 30.0

logging.basicConfig(level=getattr(logging, LOGLEVEL.upper(), logging.INFO),
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers=[logging.FileHandler(LOG_FILE), logging.StreamHandler(sys.stdout)])
logger = logging.getLogger(__name__)

redis_client: Optional[redis.Redis] = None
health_counters = {"ticks_published": 0, "klines_published": 0, "reconnects": 0, "last_tick_ts": 0.0, "last_kline_ts": 0.0}

def create_event_envelope(event_type: str, payload: Dict, source: str = "market-publisher") -> Dict:
    return {"event_type": event_type, "payload": json.dumps(payload), "correlation_id": str(uuid.uuid4()), 
            "timestamp": datetime.utcnow().isoformat(), "source": source, "trace_id": ""}

def publish_market_tick(symbol: str, price: float, qty: float, is_buyer_maker: bool):
    try:
        payload = {"symbol": symbol, "price": price, "qty": qty, "is_buyer_maker": is_buyer_maker, "timestamp": time.time()}
        redis_client.xadd(STREAM_TICK, create_event_envelope("market.tick", payload), maxlen=10000, approximate=True)
        health_counters["ticks_published"] += 1
        health_counters["last_tick_ts"] = time.time()
    except Exception as e:
        logger.error(f"Failed to publish market.tick for {symbol}: {e}")

def publish_market_klines(symbol: str, interval: str, kline_data: Dict):
    try:
        k = kline_data.get("k", {})
        payload = {"symbol": symbol, "interval": interval, "open": float(k.get("o", 0)), "high": float(k.get("h", 0)),
                   "low": float(k.get("l", 0)), "close": float(k.get("c", 0)), "volume": float(k.get("v", 0)),
                   "open_time": int(k.get("t", 0)), "close_time": int(k.get("T", 0)), 
                   "is_closed": k.get("x", False), "timestamp": time.time()}
        if not payload["is_closed"]:
            return
        redis_client.xadd(STREAM_KLINES, create_event_envelope("market.klines", payload), maxlen=10000, approximate=True)
        health_counters["klines_published"] += 1
        health_counters["last_kline_ts"] = time.time()
        logger.debug(f"[KLINE] {symbol} @ {payload['close']}")
    except Exception as e:
        logger.error(f"Failed to publish market.klines for {symbol}: {e}")

async def handle_symbol_trade_stream(symbol: str, bsm: BinanceSocketManager):
    reconnect_delay = RECONNECT_BASE
    while True:
        try:
            logger.info(f"[TRADE] Starting trade stream: {symbol}")
            ts = bsm.trade_socket(symbol)
            async with ts as tscm:
                while True:
                    res = await tscm.recv()
                    if res and "e" in res and res["e"] == "trade":
                        publish_market_tick(symbol, float(res.get("p", 0)), float(res.get("q", 0)), res.get("m", False))
            logger.warning(f"[TRADE] {symbol} stream closed, reconnecting...")
            reconnect_delay = RECONNECT_BASE
        except Exception as e:
            logger.error(f"[ERROR] {symbol} trade stream error: {e}")
            health_counters["reconnects"] += 1
            wait_time = min(reconnect_delay * random.uniform(1-RECONNECT_JITTER, 1+RECONNECT_JITTER), RECONNECT_MAX)
            logger.info(f"[RECONNECT] {symbol} trade stream in {wait_time:.1f}s...")
            await asyncio.sleep(wait_time)
            reconnect_delay = min(reconnect_delay * 2, RECONNECT_MAX)

async def handle_symbol_kline_stream(symbol: str, bsm: BinanceSocketManager, interval: str = "1m"):
    reconnect_delay = RECONNECT_BASE
    while True:
        try:
            logger.info(f"[KLINE] Starting kline stream: {symbol} ({interval})")
            ks = bsm.kline_socket(symbol, interval=interval)
            async with ks as kscm:
                while True:
                    res = await kscm.recv()
                    if res and "e" in res and res["e"] == "kline":
                        publish_market_klines(symbol, interval, res)
            logger.warning(f"[KLINE] {symbol} stream closed, reconnecting...")
            reconnect_delay = RECONNECT_BASE
        except Exception as e:
            logger.error(f"[ERROR] {symbol} kline stream error: {e}")
            health_counters["reconnects"] += 1
            wait_time = min(reconnect_delay * random.uniform(1-RECONNECT_JITTER, 1+RECONNECT_JITTER), RECONNECT_MAX)
            logger.info(f"[RECONNECT] {symbol} kline stream in {wait_time:.1f}s...")
            await asyncio.sleep(wait_time)
            reconnect_delay = min(reconnect_delay * 2, RECONNECT_MAX)

async def health_monitor():
    while True:
        await asyncio.sleep(HEALTH_INTERVAL)
        logger.info(f"[HEALTH] ticks={health_counters['ticks_published']}, klines={health_counters['klines_published']}, "
                   f"reconnects={health_counters['reconnects']}, last_tick={time.time()-health_counters['last_tick_ts']:.1f}s ago, "
                   f"last_kline={time.time()-health_counters['last_kline_ts']:.1f}s ago")

async def start_market_streams():
    global redis_client
    logger.info("=== MARKET PUBLISHER STARTING ===")
    logger.info(f"Redis: {REDIS_HOST}:{REDIS_PORT}/{REDIS_DB}")
    logger.info(f"Symbols: {len(SYMBOLS)} pairs")
    logger.info(f"Streams: {STREAM_TICK}, {STREAM_KLINES}")
    
    redis_client = redis.Redis(host=REDIS_HOST, port=REDIS_PORT, db=REDIS_DB, decode_responses=False)
    try:
        redis_client.ping()
        logger.info("Redis connection OK")
    except Exception as e:
        logger.error(f"Redis connection FAILED: {e}")
        sys.exit(1)
    
    client = await AsyncClient.create()
    bsm = BinanceSocketManager(client)
    asyncio.create_task(health_monitor())
    
    tasks = []
    for symbol in SYMBOLS:
        tasks.append(asyncio.create_task(handle_symbol_trade_stream(symbol, bsm)))
        tasks.append(asyncio.create_task(handle_symbol_kline_stream(symbol, bsm)))
    
    logger.info(f"[READY] {len(tasks)} streams started ({len(SYMBOLS)} symbols × 2 streams)")
    await asyncio.gather(*tasks)

if __name__ == "__main__":
    try:
        asyncio.run(start_market_streams())
    except KeyboardInterrupt:
        logger.info("Market publisher stopped by user")
    except Exception as e:
        logger.error(f"Fatal error: {e}", exc_info=True)
        sys.exit(1)
