#!/usr/bin/env python3
"""
QuantumTrader PnL Reconciler
Watches quantum:stream:trade.closed, fetches REAL realized PnL from Binance
userTrades for each closed order, and writes correct data to all RL streams.

This is the authoritative bridge between Binance fills and the RL reward pipeline.
"""
import asyncio
import hashlib
import hmac
import json
import logging
import os
import time
import urllib.parse
from datetime import datetime, timezone

import aiohttp
import redis.asyncio as aioredis

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [PNL-RECON] %(levelname)s %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler("/tmp/qt_pnl_reconciler.log", mode="a"),
    ],
)
logger = logging.getLogger("pnl_reconciler")

# ---------- Config ----------
REDIS_URL         = os.getenv("REDIS_URL", "redis://127.0.0.1:6379")
BINANCE_BASE_URL  = os.getenv("BINANCE_BASE_URL", "https://testnet.binancefuture.com")
API_KEY           = os.getenv("BINANCE_TESTNET_API_KEY", "")
API_SECRET        = os.getenv("BINANCE_TESTNET_API_SECRET", "")

TRADE_CLOSED_STREAM = "quantum:stream:trade.closed"
RL_REWARDS_STREAM   = "quantum:stream:rl_rewards"
EXITBRAIN_PNL_STREAM = "quantum:stream:exitbrain.pnl"
EXIT_LOG_PREFIX     = "quantum:exit_log"
CONSUMER_GROUP      = "pnl-reconciler"
CONSUMER_NAME       = "recon-1"
STREAM_MAXLEN       = 500

FETCH_DELAY_SEC = 3.0   # wait a bit for fills to settle on exchange
MAX_RETRIES     = 3
# ----------------------------


def _sign(params: dict) -> str:
    params["timestamp"] = int(time.time() * 1000)
    params["recvWindow"] = 10000
    qs = urllib.parse.urlencode(params)
    sig = hmac.new(API_SECRET.encode(), qs.encode(), hashlib.sha256).hexdigest()
    return qs + "&signature=" + sig


async def fetch_user_trades(session: aiohttp.ClientSession, symbol: str, order_id: str) -> list:
    """Fetch fills for a specific order from Binance."""
    params = {"symbol": symbol}
    if order_id and order_id not in ("?", "None", ""):
        params["orderId"] = int(order_id)
    else:
        params["limit"] = 20  # fallback: last 20 fills

    for attempt in range(MAX_RETRIES):
        try:
            qs = _sign(params.copy())
            url = f"{BINANCE_BASE_URL}/fapi/v1/userTrades?{qs}"
            async with session.get(url, headers={"X-MBX-APIKEY": API_KEY}, timeout=aiohttp.ClientTimeout(total=8)) as resp:
                if resp.status == 200:
                    return await resp.json()
                text = await resp.text()
                logger.warning(f"userTrades {symbol} status={resp.status}: {text[:100]}")
        except Exception as e:
            logger.warning(f"userTrades attempt {attempt+1}/{MAX_RETRIES} failed: {e}")
        await asyncio.sleep(1.5)
    return []


async def reconcile_trade(r: aioredis.Redis, session: aiohttp.ClientSession, msg_id: str, fields: dict):
    """
    Fetch real PnL for a closed trade and publish to RL streams.
    fields come from quantum:stream:trade.closed
    """
    symbol   = fields.get("symbol", "?")
    side     = fields.get("side", "?")
    order_id = fields.get("order_id", "")
    est_pnl  = float(fields.get("pnl_usd", 0))
    confidence = float(fields.get("confidence", 0.7))
    entry_price = float(fields.get("entry_price", 0))
    exit_price  = float(fields.get("exit_price", 0))
    reason      = fields.get("reason", "unknown")
    r_net       = float(fields.get("R_net", 0))

    if symbol == "?":
        return

    # Small delay to let exchange settle
    await asyncio.sleep(FETCH_DELAY_SEC)

    # Fetch real fills from Binance
    fills = await fetch_user_trades(session, symbol, order_id)

    # Sum realizedPnl from fills for this order
    real_pnl = 0.0
    real_commission = 0.0
    if fills:
        for fill in fills:
            if not order_id or str(fill.get("orderId")) == str(order_id):
                real_pnl       += float(fill.get("realizedPnl", 0))
                real_commission += float(fill.get("commission", 0))

    # If no fills matched, fall back to estimated PnL
    data_source = "binance"
    if real_pnl == 0.0 and fills:
        # fills exist but none matched orderId — sum all recent
        real_pnl = sum(float(f.get("realizedPnl", 0)) for f in fills[-5:])
        real_commission = sum(float(f.get("commission", 0)) for f in fills[-5:])
        data_source = "binance_recent"

    if real_pnl == 0.0:
        # Truly no data — fall back to estimated
        real_pnl = est_pnl
        data_source = "estimated"

    net_pnl = real_pnl - real_commission
    pnl_pct = (net_pnl / (entry_price * 1)) * 100 if entry_price > 0 else 0.0

    logger.info(
        f"RECONCILE {symbol:20s} order={order_id}  "
        f"estimated={est_pnl:+.4f}  real={real_pnl:+.4f}  "
        f"commission={real_commission:.4f}  net={net_pnl:+.4f}  "
        f"src={data_source}"
    )

    now_iso = datetime.now(timezone.utc).isoformat()

    # 1. Reward signal for RL sizing agent
    reward = net_pnl / 10.0  # normalize: $10 of profit = reward 1.0
    reward = max(-1.0, min(1.0, reward))
    await r.xadd(RL_REWARDS_STREAM, {
        "symbol":     symbol,
        "reward":     str(round(reward, 6)),
        "pnl":        str(round(real_pnl, 6)),
        "pnl_net":    str(round(net_pnl, 6)),
        "commission": str(round(real_commission, 6)),
        "pnl_pct":    str(round(pnl_pct, 4)),
        "confidence": str(confidence),
        "side":       side,
        "order_id":   order_id,
        "source":     data_source,
        "timestamp":  now_iso,
    }, maxlen=STREAM_MAXLEN, approximate=True)

    # 2. ExitBrain PnL stream (used by rl_feedback_bridge_v2 via quantum:signal:strategy)
    await r.xadd(EXITBRAIN_PNL_STREAM, {
        "symbol":     symbol,
        "pnl":        str(round(net_pnl, 6)),
        "confidence": str(confidence),
        "volatility": "0.02",
        "side":       side,
        "source":     data_source,
        "timestamp":  now_iso,
    }, maxlen=STREAM_MAXLEN, approximate=True)

    # Also push to quantum:signal:strategy (what rl_feedback_bridge_v2 actually reads)
    await r.xadd("quantum:signal:strategy", {
        "symbol":     symbol,
        "reward":     str(round(net_pnl, 6)),   # bridge uses this as reward
        "confidence": str(confidence),
        "pnl":        str(round(net_pnl, 6)),
        "source":     data_source,
        "timestamp":  now_iso,
    }, maxlen=STREAM_MAXLEN, approximate=True)

    # 3. Exit log for rl_calibrator (reads quantum:exit_log:* keys)
    log_key = f"{EXIT_LOG_PREFIX}:{symbol}:{int(time.time())}"
    await r.setex(log_key, 86400, json.dumps({
        "symbol":       symbol,
        "side":         side,
        "realized_pnl": round(net_pnl, 6),
        "raw_pnl":      round(real_pnl, 6),
        "commission":   round(real_commission, 6),
        "confidence":   confidence,
        "entry_price":  entry_price,
        "exit_price":   exit_price,
        "reason":       reason,
        "R_net":        r_net,
        "data_source":  data_source,
        "timestamp":    now_iso,
    }))

    # 4. Update rl:reward hash for RL agent realtime access
    await r.setex(f"quantum:rl:reward:{symbol}", 3600, json.dumps({
        "symbol":           symbol,
        "realized_pnl":     round(net_pnl, 6),
        "reward":           round(reward, 6),
        "confidence":       confidence,
        "last_close_time":  now_iso,
        "data_source":      data_source,
    }))

    logger.info(
        f"  → Published: rl_rewards reward={reward:+.4f}  exit_log={log_key.split(':')[-1]}  src={data_source}"
    )


async def main():
    if not API_KEY or not API_SECRET:
        logger.error("BINANCE_TESTNET_API_KEY/SECRET not set — cannot reconcile")
        return

    logger.info(f"PnL Reconciler starting — watching {TRADE_CLOSED_STREAM}")
    logger.info(f"Binance: {BINANCE_BASE_URL}")

    r = aioredis.from_url(REDIS_URL, decode_responses=True)

    # Create consumer group (ignore if exists)
    try:
        await r.xgroup_create(TRADE_CLOSED_STREAM, CONSUMER_GROUP, id="$", mkstream=True)
        logger.info(f"Consumer group '{CONSUMER_GROUP}' created")
    except Exception as e:
        if "BUSYGROUP" in str(e):
            logger.info(f"Consumer group '{CONSUMER_GROUP}' already exists")
        else:
            # Stream may be empty — start from beginning once it gets data
            logger.warning(f"Group create: {e}")

    connector = aiohttp.TCPConnector(limit=5)
    async with aiohttp.ClientSession(connector=connector) as session:
        while True:
            try:
                # Read new messages from trade.closed
                msgs = await r.xreadgroup(
                    CONSUMER_GROUP, CONSUMER_NAME,
                    {TRADE_CLOSED_STREAM: ">"},
                    count=5, block=5000
                )
                if not msgs:
                    continue

                for stream_name, entries in msgs:
                    for msg_id, fields in entries:
                        logger.info(f"Processing close: {fields.get('symbol','?')} id={msg_id}")
                        try:
                            await reconcile_trade(r, session, msg_id, fields)
                            await r.xack(TRADE_CLOSED_STREAM, CONSUMER_GROUP, msg_id)
                        except Exception as e:
                            logger.error(f"Failed to reconcile {msg_id}: {e}", exc_info=True)

            except Exception as e:
                logger.error(f"Main loop error: {e}", exc_info=True)
                await asyncio.sleep(5)


if __name__ == "__main__":
    asyncio.run(main())
