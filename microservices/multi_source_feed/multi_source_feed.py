#!/usr/bin/env python3
"""
multi_source_feed.py — Multi-Exchange Data Aggregator

Parallelle datakilder (alle gratis, ingen API-nøkkel nødvendig for public data):

  WS-1  Bybit Linear WebSocket
        wss://stream.bybit.com/v5/public/linear
        → Reelle priser, volum, bid/ask spread (Binance testnet-volum er syntetisk)
        → quantum:market:<SYM>:bybit  {price, volume, bid, ask, ts}

  WS-2  OKX Public WebSocket
        wss://ws.okx.com:8443/ws/v5/public
        → Priser og funding rate direkte i tickeren
        → quantum:market:<SYM>:okx  {price, volume, funding_rate, ts}

  REST-1  CoinGecko Global Market (poll hvert 60s)
        https://api.coingecko.com/api/v3/global
        → Total market cap, BTC/ETH dominance, 24h volume global
        → quantum:market:GLOBAL  {total_market_cap_usd, btc_dominance, eth_dominance,
                                   total_volume_24h, market_cap_change_24h_pct, ts}

  REST-2  Alternative.me Fear & Greed Index (poll hvert 15min)
        https://api.alternative.me/fng/
        → Sentiment 0–100 (0=Extreme Fear, 100=Extreme Greed)
        → quantum:sentiment:fear_greed  {value, classification, ts}

  REST-3  Binance Futures Funding Rates (poll hvert 30min)
        https://fapi.binance.com/fapi/v1/premiumIndex  (public, no auth)
        → Reell funding rate, mark price, index price for alle symbols
        → quantum:funding:<SYM>  {funding_rate, mark_price, index_price, ts}

  REST-4  Bybit Funding Rates (poll hvert 30min)
        https://api.bybit.com/v5/market/tickers?category=linear
        → Cross-exchange funding rate sammenligning
        → quantum:funding:<SYM>:bybit  {funding_rate, open_interest, ts}

Aggregert cross-exchange signal:
        → quantum:market:<SYM>:spread  {binance_price, bybit_price, okx_price,
                                         spread_pct, funding_avg, ts}
        → quantum:stream:market_events  (stream for AI-konsumenter)

Aktiverte symboler hentes fra quantum:universe:active (samme som eksisterende feed).
For WS: starter med TOP-20 etter volum (begrenser WS-abonnementer).
"""

import asyncio
import json
import logging
import os
import signal
import sys
import time
from typing import Optional

import aiohttp
import redis as redis_lib
import websockets

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s msf %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger("msf")

REDIS_HOST = os.getenv("REDIS_HOST", "localhost")
REDIS_PORT = int(os.getenv("REDIS_PORT", "6379"))

# Top symbols for WS feeds (Bybit + OKX har abonnementsgrense per kobling)
TOP_SYMBOLS = [
    "BTCUSDT", "ETHUSDT", "SOLUSDT", "BNBUSDT", "XRPUSDT",
    "DOGEUSDT", "ADAUSDT", "AVAXUSDT", "LINKUSDT", "DOTUSDT",
    "MATICUSDT", "UNIUSDT", "LTCUSDT", "ATOMUSDT", "NEARUSDT",
    "AAVEUSDT", "OPUSDT", "ARBUSDT", "INJUSDT", "SUIUSDT",
]

COINGECKO_GLOBAL     = "https://api.coingecko.com/api/v3/global"
FEAR_GREED_URL       = "https://api.alternative.me/fng/?limit=1&format=json"
BINANCE_PREMIUM      = "https://fapi.binance.com/fapi/v1/premiumIndex"
BYBIT_TICKERS        = "https://api.bybit.com/v5/market/tickers?category=linear"
BYBIT_WS_URL         = "wss://stream.bybit.com/v5/public/linear"
OKX_WS_URL           = "wss://ws.okx.com:8443/ws/v5/public"

GLOBAL_POLL_INTERVAL   = 60
FEAR_GREED_INTERVAL    = 900   # 15 min
FUNDING_POLL_INTERVAL  = 1800  # 30 min

STREAM_KEY = "quantum:stream:market_events"

_RUNNING = True


def _handle_signal(sig, _):
    global _RUNNING
    logger.info("Signal %s — shutting down", sig)
    _RUNNING = False


signal.signal(signal.SIGTERM, _handle_signal)
signal.signal(signal.SIGINT, _handle_signal)


# ── Redis helpers ─────────────────────────────────────────────────────────

def _r() -> redis_lib.Redis:
    return redis_lib.Redis(host=REDIS_HOST, port=REDIS_PORT, db=0)


def _set_market(r: redis_lib.Redis, symbol: str, source: str, data: dict, ttl: int = 120):
    key = f"quantum:market:{symbol}:{source}"
    r.hset(key, mapping={str(k): str(v) for k, v in data.items()})
    r.expire(key, ttl)


def _publish_event(r: redis_lib.Redis, event_type: str, symbol: str, data: dict):
    payload = {"type": event_type, "symbol": symbol, "ts": int(time.time()), **data}
    r.xadd(STREAM_KEY, {k: str(v) for k, v in payload.items()}, maxlen=50000)


# ── Source 1: Bybit WebSocket ─────────────────────────────────────────────

async def bybit_ws_feed(r: redis_lib.Redis):
    """
    Subscribe to Bybit tickers for TOP_SYMBOLS.
    Publishes: quantum:market:<SYM>:bybit  {price, volume, bid, ask, ts}
    """
    topics = [f"tickers.{sym}" for sym in TOP_SYMBOLS]

    while _RUNNING:
        try:
            logger.info("[BYBIT-WS] Connecting ...")
            async with websockets.connect(BYBIT_WS_URL, ping_interval=20,
                                          close_timeout=10) as ws:
                # Subscribe in batches of 10
                for i in range(0, len(topics), 10):
                    batch = topics[i:i+10]
                    await ws.send(json.dumps({"op": "subscribe", "args": batch}))
                    await asyncio.sleep(0.1)
                logger.info("[BYBIT-WS] Subscribed to %d topics", len(topics))

                while _RUNNING:
                    try:
                        raw = await asyncio.wait_for(ws.recv(), timeout=30)
                        msg = json.loads(raw)
                        if msg.get("topic", "").startswith("tickers."):
                            d = msg.get("data", {})
                            symbol = d.get("symbol", "")
                            if not symbol:
                                continue
                            price   = d.get("lastPrice", "")
                            bid     = d.get("bid1Price", "")
                            ask     = d.get("ask1Price", "")
                            volume  = d.get("volume24h", "")
                            ts = int(time.time())
                            if price:
                                payload = {"price": price, "bid": bid, "ask": ask,
                                           "volume_24h": volume, "ts": ts,
                                           "source": "bybit"}
                                _set_market(r, symbol, "bybit", payload)

                                # Update spread aggregate
                                _update_spread(r, symbol, "bybit", float(price))
                    except asyncio.TimeoutError:
                        await ws.send(json.dumps({"op": "ping"}))
        except Exception as e:
            logger.warning("[BYBIT-WS] Disconnected: %s — reconnecting in 10s", e)
            await asyncio.sleep(10)


# ── Source 2: OKX WebSocket ───────────────────────────────────────────────

async def okx_ws_feed(r: redis_lib.Redis):
    """
    Subscribe to OKX tickers.
    OKX perp symbols: BTC-USDT-SWAP etc.
    Publishes: quantum:market:<SYM>:okx  {price, volume, funding_rate, ts}
    """
    # Convert BTCUSDT → BTC-USDT-SWAP
    def to_okx(sym: str) -> Optional[str]:
        if sym.endswith("USDT"):
            base = sym[:-4]
            return f"{base}-USDT-SWAP"
        return None

    okx_symbols = [(sym, to_okx(sym)) for sym in TOP_SYMBOLS]
    okx_symbols = [(s, o) for s, o in okx_symbols if o]
    channels = [{"channel": "tickers", "instId": okx} for _, okx in okx_symbols]

    while _RUNNING:
        try:
            logger.info("[OKX-WS] Connecting ...")
            async with websockets.connect(OKX_WS_URL, ping_interval=20,
                                          close_timeout=10) as ws:
                await ws.send(json.dumps({"op": "subscribe", "args": channels}))
                logger.info("[OKX-WS] Subscribed to %d channels", len(channels))

                while _RUNNING:
                    try:
                        raw = await asyncio.wait_for(ws.recv(), timeout=30)
                        msg = json.loads(raw)
                        if msg.get("event") == "subscribe":
                            continue
                        data_list = msg.get("data", [])
                        for d in data_list:
                            inst_id = d.get("instId", "")
                            # Convert back: BTC-USDT-SWAP → BTCUSDT
                            if "-USDT-SWAP" in inst_id:
                                symbol = inst_id.replace("-USDT-SWAP", "") + "USDT"
                            else:
                                continue
                            price         = d.get("last", "")
                            volume        = d.get("vol24h", "")
                            funding_rate  = d.get("fundingRate", "")
                            ts = int(time.time())
                            if price:
                                payload = {"price": price, "volume_24h": volume,
                                           "funding_rate": funding_rate,
                                           "ts": ts, "source": "okx"}
                                _set_market(r, symbol, "okx", payload)
                                _update_spread(r, symbol, "okx", float(price))
                    except asyncio.TimeoutError:
                        await ws.send(json.dumps({"op": "ping"}))
        except Exception as e:
            logger.warning("[OKX-WS] Disconnected: %s — reconnecting in 10s", e)
            await asyncio.sleep(10)


# ── Cross-exchange spread ─────────────────────────────────────────────────

_price_cache: dict[str, dict] = {}


def _update_spread(r: redis_lib.Redis, symbol: str, source: str, price: float):
    """Maintain cross-exchange price spread. Publish if all 3 sources available."""
    if symbol not in _price_cache:
        _price_cache[symbol] = {}
    _price_cache[symbol][source] = price
    _price_cache[symbol][f"{source}_ts"] = int(time.time())

    cache = _price_cache[symbol]
    sources_available = [s for s in ("binance", "bybit", "okx") if s in cache]
    if len(sources_available) < 2:
        return

    prices = [cache[s] for s in sources_available]
    min_p, max_p = min(prices), max(prices)
    spread_pct = (max_p - min_p) / min_p * 100 if min_p > 0 else 0.0

    spread_data = {
        "binance_price": cache.get("binance", ""),
        "bybit_price":   cache.get("bybit", ""),
        "okx_price":     cache.get("okx", ""),
        "spread_pct":    f"{spread_pct:.4f}",
        "sources":       ",".join(sources_available),
        "ts":            int(time.time()),
    }
    key = f"quantum:market:{symbol}:spread"
    r.hset(key, mapping={str(k): str(v) for k, v in spread_data.items()})
    r.expire(key, 120)

    if spread_pct > 0.5:
        logger.warning("[SPREAD] %s spread=%.3f%% binance=%s bybit=%s okx=%s",
                       symbol, spread_pct,
                       cache.get("binance", "?"), cache.get("bybit", "?"),
                       cache.get("okx", "?"))
        _publish_event(r, "SPREAD_ALERT", symbol, {"spread_pct": f"{spread_pct:.4f}"})


# ── Source 3: CoinGecko Global Market ────────────────────────────────────

async def coingecko_global_poller(r: redis_lib.Redis, session: aiohttp.ClientSession):
    """
    Poll CoinGecko /global every 60s (free tier: 30 calls/min).
    Publishes: quantum:market:GLOBAL
    """
    while _RUNNING:
        try:
            async with session.get(COINGECKO_GLOBAL, timeout=aiohttp.ClientTimeout(total=15)) as resp:
                if resp.status == 200:
                    body = await resp.json()
                    d = body.get("data", {})
                    mcd = d.get("market_cap_change_percentage_24h_usd", 0)
                    total_mc = d.get("total_market_cap", {}).get("usd", 0)
                    total_vol = d.get("total_volume", {}).get("usd", 0)
                    btc_dom = d.get("market_cap_percentage", {}).get("btc", 0)
                    eth_dom = d.get("market_cap_percentage", {}).get("eth", 0)
                    active_cryptos = d.get("active_cryptocurrencies", 0)

                    payload = {
                        "total_market_cap_usd":       f"{total_mc:.0f}",
                        "total_volume_24h_usd":       f"{total_vol:.0f}",
                        "btc_dominance_pct":          f"{btc_dom:.2f}",
                        "eth_dominance_pct":          f"{eth_dom:.2f}",
                        "market_cap_change_24h_pct":  f"{mcd:.4f}",
                        "active_cryptos":             str(active_cryptos),
                        "ts":                         int(time.time()),
                        "source":                     "coingecko",
                    }
                    r.hset("quantum:market:GLOBAL", mapping={str(k): str(v) for k, v in payload.items()})
                    r.expire("quantum:market:GLOBAL", 180)

                    logger.info("[COINGECKO] mc=%.1fT btc_dom=%.1f%% eth_dom=%.1f%% change_24h=%.2f%%",
                                total_mc / 1e12, btc_dom, eth_dom, mcd)
                elif resp.status == 429:
                    logger.warning("[COINGECKO] Rate limited — sleeping 120s")
                    await asyncio.sleep(120)
                    continue
                else:
                    logger.warning("[COINGECKO] HTTP %d", resp.status)
        except Exception as e:
            logger.warning("[COINGECKO] Error: %s", e)
        await asyncio.sleep(GLOBAL_POLL_INTERVAL)


# ── Source 4: Fear & Greed Index ──────────────────────────────────────────

async def fear_greed_poller(r: redis_lib.Redis, session: aiohttp.ClientSession):
    """
    Poll Alternative.me F&G index every 15 min (completely free).
    Publishes: quantum:sentiment:fear_greed
    """
    while _RUNNING:
        try:
            async with session.get(FEAR_GREED_URL, timeout=aiohttp.ClientTimeout(total=10)) as resp:
                if resp.status == 200:
                    body = await resp.json(content_type=None)
                    d = body.get("data", [{}])[0]
                    value          = int(d.get("value", 50))
                    classification = d.get("value_classification", "Neutral")

                    # Map to numeric regime for AI models: 0=extreme_fear .. 4=extreme_greed
                    regime = 0
                    if value >= 75:    regime = 4  # Extreme Greed
                    elif value >= 55:  regime = 3  # Greed
                    elif value >= 45:  regime = 2  # Neutral
                    elif value >= 25:  regime = 1  # Fear
                    else:              regime = 0  # Extreme Fear

                    payload = {
                        "value":          str(value),
                        "classification": classification,
                        "regime":         str(regime),
                        "regime_label":   ["extreme_fear", "fear", "neutral", "greed", "extreme_greed"][regime],
                        "ts":             int(time.time()),
                        "source":         "alternative.me",
                    }
                    r.hset("quantum:sentiment:fear_greed", mapping={str(k): str(v) for k, v in payload.items()})
                    r.expire("quantum:sentiment:fear_greed", 1800)

                    logger.info("[FEAR&GREED] value=%d classification=%s regime=%s",
                                value, classification, payload["regime_label"])
                else:
                    logger.warning("[FEAR&GREED] HTTP %d", resp.status)
        except Exception as e:
            logger.warning("[FEAR&GREED] Error: %s", e)
        await asyncio.sleep(FEAR_GREED_INTERVAL)


# ── Source 5: Binance Futures Funding Rates (real, public endpoint) ───────

async def binance_funding_poller(r: redis_lib.Redis, session: aiohttp.ClientSession):
    """
    Poll Binance /fapi/v1/premiumIndex — public, no auth.
    Returns real funding rates, mark price, index price.
    Publishes: quantum:funding:<SYM>  (+ updates quantum:market:<SYM>:binance with mark/index)
    """
    while _RUNNING:
        try:
            async with session.get(BINANCE_PREMIUM, timeout=aiohttp.ClientTimeout(total=15)) as resp:
                if resp.status == 200:
                    data = await resp.json()
                    updated = 0
                    for item in data:
                        symbol       = item.get("symbol", "")
                        funding_rate = item.get("lastFundingRate", "0")
                        mark_price   = item.get("markPrice", "0")
                        index_price  = item.get("indexPrice", "0")
                        next_funding = item.get("nextFundingTime", 0)

                        if not symbol:
                            continue

                        payload = {
                            "funding_rate":      funding_rate,
                            "mark_price":        mark_price,
                            "index_price":       index_price,
                            "next_funding_ts":   str(int(next_funding / 1000)),
                            "ts":                int(time.time()),
                            "source":            "binance_mainnet",
                        }
                        r.hset(f"quantum:funding:{symbol}", mapping={str(k): str(v) for k, v in payload.items()})
                        r.expire(f"quantum:funding:{symbol}", 3600)

                        # Also write binance mainnet price (separate from testnet feed)
                        r.hset(f"quantum:market:{symbol}:binance_main", mapping={
                            "price":  mark_price,
                            "index":  index_price,
                            "funding_rate": funding_rate,
                            "ts":     int(time.time()),
                            "source": "binance_mainnet",
                        })
                        r.expire(f"quantum:market:{symbol}:binance_main", 3600)

                        # Track price for spread
                        if symbol in TOP_SYMBOLS and mark_price:
                            _update_spread(r, symbol, "binance", float(mark_price))

                        updated += 1

                    logger.info("[BINANCE-FUNDING] Updated %d symbols", updated)
                else:
                    logger.warning("[BINANCE-FUNDING] HTTP %d", resp.status)
        except Exception as e:
            logger.warning("[BINANCE-FUNDING] Error: %s", e)
        await asyncio.sleep(FUNDING_POLL_INTERVAL)


# ── Source 6: Bybit Funding Rates ────────────────────────────────────────

async def bybit_funding_poller(r: redis_lib.Redis, session: aiohttp.ClientSession):
    """
    Poll Bybit /v5/market/tickers — public endpoint.
    Gives funding rate + open interest per symbol.
    Publishes: quantum:funding:<SYM>:bybit
    """
    while _RUNNING:
        try:
            async with session.get(BYBIT_TICKERS, timeout=aiohttp.ClientTimeout(total=15)) as resp:
                if resp.status == 200:
                    body = await resp.json()
                    items = body.get("result", {}).get("list", [])
                    updated = 0
                    for item in items:
                        symbol         = item.get("symbol", "")
                        funding_rate   = item.get("fundingRate", "0")
                        open_interest  = item.get("openInterest", "0")
                        oi_value       = item.get("openInterestValue", "0")
                        last_price     = item.get("lastPrice", "0")

                        if not symbol or not symbol.endswith("USDT"):
                            continue

                        payload = {
                            "funding_rate":    funding_rate,
                            "open_interest":   open_interest,
                            "oi_value_usd":    oi_value,
                            "ts":              int(time.time()),
                            "source":          "bybit",
                        }
                        r.hset(f"quantum:funding:{symbol}:bybit", mapping={str(k): str(v) for k, v in payload.items()})
                        r.expire(f"quantum:funding:{symbol}:bybit", 3600)
                        updated += 1

                    logger.info("[BYBIT-FUNDING] Updated %d symbols OI+funding", updated)

                    # Funding spread alarm: compare Binance vs Bybit
                    _check_funding_spread(r, items)
                else:
                    logger.warning("[BYBIT-FUNDING] HTTP %d", resp.status)
        except Exception as e:
            logger.warning("[BYBIT-FUNDING] Error: %s", e)
        await asyncio.sleep(FUNDING_POLL_INTERVAL)


def _check_funding_spread(r: redis_lib.Redis, bybit_items: list):
    """
    Compare Binance vs Bybit funding rates.
    Large spread = carry trade opportunity / market stress indicator.
    """
    for item in bybit_items:
        symbol = item.get("symbol", "")
        if symbol not in TOP_SYMBOLS:
            continue
        bybit_rate = float(item.get("fundingRate", 0) or 0)
        binance_raw = r.hget(f"quantum:funding:{symbol}", "funding_rate")
        if not binance_raw:
            continue
        binance_rate = float(binance_raw)
        spread_bps = abs(bybit_rate - binance_rate) * 10000  # basis points

        if spread_bps > 5:  # >5 bps spread = significant
            r.hset(f"quantum:funding:{symbol}:spread", mapping={
                "binance_rate":   str(binance_rate),
                "bybit_rate":     str(bybit_rate),
                "spread_bps":     f"{spread_bps:.2f}",
                "ts":             str(int(time.time())),
            })
            r.expire(f"quantum:funding:{symbol}:spread", 3600)
            logger.info("[FUNDING-SPREAD] %s binance=%.4f%% bybit=%.4f%% spread=%.1fbps",
                        symbol, binance_rate * 100, bybit_rate * 100, spread_bps)


# ── Status publisher ──────────────────────────────────────────────────────

async def status_publisher(r: redis_lib.Redis):
    """Publish health state every 30s."""
    while _RUNNING:
        await asyncio.sleep(30)
        try:
            # Count populated keys
            bybit_count   = len(r.keys("quantum:market:*:bybit"))
            okx_count     = len(r.keys("quantum:market:*:okx"))
            funding_count = len(r.keys("quantum:funding:*"))
            spread_count  = len(r.keys("quantum:market:*:spread"))
            has_global    = r.exists("quantum:market:GLOBAL")
            has_fg        = r.exists("quantum:sentiment:fear_greed")

            fg = {k.decode() if isinstance(k, bytes) else k: v.decode() if isinstance(v, bytes) else v
                  for k, v in r.hgetall("quantum:sentiment:fear_greed").items()}

            state = {
                "bybit_symbols":   str(bybit_count),
                "okx_symbols":     str(okx_count),
                "funding_symbols": str(funding_count),
                "spread_symbols":  str(spread_count),
                "global_ok":       str(bool(has_global)),
                "fear_greed_ok":   str(bool(has_fg)),
                "fear_greed_val":  fg.get("value", "?"),
                "fear_greed_cls":  fg.get("classification", "?"),
                "ts":              int(time.time()),
            }
            r.hset("quantum:multi_source_feed:status", mapping={str(k): str(v) for k, v in state.items()})
            r.expire("quantum:multi_source_feed:status", 120)

            logger.info("[STATUS] bybit=%d okx=%d funding=%d spread=%d fg=%s(%s)",
                        bybit_count, okx_count, funding_count, spread_count,
                        fg.get("value", "?"), fg.get("classification", "?"))
        except Exception as e:
            logger.warning("[STATUS] Error: %s", e)


# ── Main ──────────────────────────────────────────────────────────────────

async def main_async():
    r = redis_lib.Redis(host=REDIS_HOST, port=REDIS_PORT, db=0)
    try:
        r.ping()
        logger.info("[MSF] Redis OK")
    except redis_lib.ConnectionError as e:
        logger.error("[MSF] Redis FAILED: %s", e)
        sys.exit(1)

    logger.info("[MSF] Multi-Source Feed starting")
    logger.info("[MSF] Sources: Bybit WS + OKX WS + CoinGecko + Fear&Greed + "
                "Binance Funding (real) + Bybit Funding")
    logger.info("[MSF] Top symbols for WS: %s", TOP_SYMBOLS)

    connector = aiohttp.TCPConnector(limit=20)
    async with aiohttp.ClientSession(connector=connector) as session:
        tasks = [
            asyncio.create_task(bybit_ws_feed(r),           name="bybit_ws"),
            asyncio.create_task(okx_ws_feed(r),             name="okx_ws"),
            asyncio.create_task(coingecko_global_poller(r, session), name="coingecko"),
            asyncio.create_task(fear_greed_poller(r, session),       name="fear_greed"),
            asyncio.create_task(binance_funding_poller(r, session),  name="binance_funding"),
            asyncio.create_task(bybit_funding_poller(r, session),    name="bybit_funding"),
            asyncio.create_task(status_publisher(r),                 name="status"),
        ]

        # Run until shutdown
        while _RUNNING:
            await asyncio.sleep(1)

        logger.info("[MSF] Shutting down ...")
        for t in tasks:
            t.cancel()
        await asyncio.gather(*tasks, return_exceptions=True)

    logger.info("[MSF] Stopped")


def main():
    asyncio.run(main_async())


if __name__ == "__main__":
    main()
