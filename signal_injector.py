#!/usr/bin/env python3
"""
QuantumTrader Signal Injector
Generates price-momentum signals and publishes to Redis stream every 60s.
This bypasses the AI Engine HTTP layer and feeds scanner directly.

Stream: quantum:stream:ai.signal_generated
Scanner reads: symbol, action (BUY/SELL), confidence >= 0.65, age < 300s
"""
import asyncio
import json
import os
import uuid
import logging
from datetime import datetime, timezone

import redis.asyncio as aioredis
import httpx

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [INJECTOR] %(levelname)s %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler("/tmp/qt_signal_injector.log", mode="a"),
    ]
)
logger = logging.getLogger("signal_injector")

# Config
SYMBOLS = os.getenv(
    "QT_SIGNAL_SYMBOLS",
    "ETHUSDT,BTCUSDT,SOLUSDT,XRPUSDT,BNBUSDT,ADAUSDT,SUIUSDT,LINKUSDT,AVAXUSDT,LTCUSDT,DOTUSDT,NEARUSDT"
).split(",")
SYMBOLS = [s.strip() for s in SYMBOLS if s.strip()]

STREAM_KEY = "quantum:stream:ai.signal_generated"
INTERVAL_SEC = int(os.getenv("QT_INJECT_INTERVAL", "60"))
MIN_MOVE_PCT = float(os.getenv("QT_INJECT_MIN_MOVE", "0.003"))   # 0.3% move
CONFIDENCE = float(os.getenv("QT_INJECT_CONFIDENCE", "0.68"))
KLINE_INTERVAL = os.getenv("QT_INJECT_KLINE", "15m")             # 15-min candles
KLINE_LIMIT = 3                                                    # compare open[0] vs close[-1]
BINANCE_FAPI = "https://fapi.binance.com/fapi/v1"
REDIS_URL = os.getenv("REDIS_URL", "redis://127.0.0.1:6379")
MAX_STREAM_LEN = 500                                               # trim old entries


async def fetch_klines(client: httpx.AsyncClient, symbol: str):
    """Return (open_price, close_price) from last KLINE_LIMIT candles, or None."""
    try:
        resp = await client.get(
            f"{BINANCE_FAPI}/klines",
            params={"symbol": symbol, "interval": KLINE_INTERVAL, "limit": KLINE_LIMIT},
            timeout=5.0,
        )
        resp.raise_for_status()
        data = resp.json()
        if len(data) < 2:
            return None
        open_price = float(data[0][1])
        close_price = float(data[-1][4])
        return open_price, close_price
    except Exception as e:
        logger.warning(f"Kline fetch failed for {symbol}: {e}")
        return None


async def publish_signal(r: aioredis.Redis, symbol: str, action: str, price: float, change_pct: float):
    """Write a signal entry to the Redis stream."""
    now_iso = datetime.now(timezone.utc).isoformat()
    payload = json.dumps({
        "symbol": symbol,
        "action": action,                         # "buy" or "sell"
        "confidence": CONFIDENCE,
        "ensemble_confidence": CONFIDENCE,
        "model_votes": {action.upper(): "momentum"},
        "consensus": 1,
        "price": price,
        "regime": "TRENDING",
        "timestamp": now_iso,
    })
    # outer timestamp is what scanner uses for age check
    entry = {
        "event_type": "ai.signal_generated",
        "payload": payload,
        "trace_id": "",
        "correlation_id": str(uuid.uuid4()),
        "timestamp": now_iso,
        "source": "signal-injector",
    }
    await r.xadd(STREAM_KEY, entry, maxlen=MAX_STREAM_LEN, approximate=True)
    logger.info(
        f"  PUBLISHED {symbol:20s} {action.upper():4s}  "
        f"conf={CONFIDENCE:.2f}  move={change_pct:+.2%}  price={price:.4f}"
    )


async def run_cycle(r: aioredis.Redis):
    """One scan cycle across all symbols."""
    logger.info(f"--- Cycle start: {len(SYMBOLS)} symbols ---")
    published = 0
    async with httpx.AsyncClient() as client:
        for symbol in SYMBOLS:
            result = await fetch_klines(client, symbol)
            if result is None:
                continue
            open_price, close_price = result
            if open_price <= 0:
                continue
            change = (close_price - open_price) / open_price

            if change >= MIN_MOVE_PCT:
                await publish_signal(r, symbol, "buy", close_price, change)
                published += 1
            elif change <= -MIN_MOVE_PCT:
                await publish_signal(r, symbol, "sell", close_price, change)
                published += 1
            else:
                logger.debug(f"  SKIP {symbol:20s}  move={change:+.2%} (below {MIN_MOVE_PCT:.1%})")

    logger.info(f"--- Cycle done: {published}/{len(SYMBOLS)} signals published ---")


async def main():
    logger.info(
        f"Signal Injector starting — symbols={len(SYMBOLS)}, "
        f"interval={INTERVAL_SEC}s, min_move={MIN_MOVE_PCT:.1%}, "
        f"confidence={CONFIDENCE}"
    )
    r = aioredis.from_url(REDIS_URL, decode_responses=True)

    # Run immediately, then every INTERVAL_SEC
    while True:
        try:
            await run_cycle(r)
        except Exception as e:
            logger.error(f"Cycle error: {e}", exc_info=True)
        await asyncio.sleep(INTERVAL_SEC)


if __name__ == "__main__":
    asyncio.run(main())
