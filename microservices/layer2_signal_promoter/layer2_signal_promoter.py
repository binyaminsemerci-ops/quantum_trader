#!/usr/bin/env python3
"""
Layer 2 Signal Auto-Promoter
==============================
Monitors quantum:sandbox:gate:latest every 60 seconds.
When gate transitions CLOSED → OPEN:
  1. Reads top-performing strategies from Layer 3 backtest leaderboard (db=1)
  2. Reads Layer 4 Kelly sizing recommendations
  3. Builds a "promoted strategy config" for each symbol with positive Kelly
  4. Writes to quantum:strategy:active_config (JSON) — picked up by strategy router
  5. Records promotion event to quantum:layer2:promotion:events (LPUSH, capped 50)
  6. Updates quantum:sandbox:promotion:status

Only promotes once per gate-open event (tracks gate_open_ts to avoid duplicates).
Reverts config when gate closes (gate goes back to CLOSED).

Reads (never writes to live control keys):
  quantum:sandbox:gate:latest              — Layer 2 gate state
  quantum:sandbox:accuracy:latest          — accuracy metrics
  quantum:backtest:leaderboard:<SYM>       — (Redis db=1) backtest rankings
  quantum:backtest:results:<job_id>        — (Redis db=1) backtest details
  quantum:layer4:sizing:<SYM>             — Kelly sizing per symbol

Writes:
  quantum:strategy:active_config           — promoted strategy config (JSON)
  quantum:layer2:promotion:events          — event log (LPUSH, cap 50)
  quantum:sandbox:promotion:status         — current promotion status

Operator:
  redis-cli hgetall quantum:sandbox:promotion:status
  redis-cli lrange quantum:layer2:promotion:events 0 4
  redis-cli get quantum:strategy:active_config
"""

import asyncio
import json
import logging
import os
import time
from datetime import datetime, timezone
from typing import Dict, List, Optional

import redis.asyncio as aioredis

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s promoter %(message)s",
)
log = logging.getLogger("promoter")

# ── Config ────────────────────────────────────────────────────────────────
REDIS_HOST        = os.getenv("REDIS_HOST", "localhost")
REDIS_LIVE_DB     = 0
REDIS_BT_DB       = 1

CHECK_INTERVAL    = int(os.getenv("CHECK_INTERVAL",   "60"))    # seconds
TOP_N_SYMBOLS     = int(os.getenv("TOP_N_SYMBOLS",    "10"))    # max symbols in promoted config
MIN_SHARPE        = float(os.getenv("MIN_SHARPE",     "2.0"))   # min backtest sharpe to include
MIN_KELLY_EDGE    = float(os.getenv("MIN_KELLY_EDGE",  "0.01"))  # min Kelly fraction > 0

KEY_GATE          = "quantum:sandbox:gate:latest"
KEY_ACCURACY      = "quantum:sandbox:accuracy:latest"
KEY_ACTIVE_CONFIG = "quantum:strategy:active_config"
KEY_PROMO_STATUS  = "quantum:sandbox:promotion:status"
KEY_PROMO_EVENTS  = "quantum:layer2:promotion:events"

TOP_SYMBOLS = [
    "BTCUSDT", "ETHUSDT", "SOLUSDT", "BNBUSDT", "XRPUSDT",
    "DOGEUSDT", "AVAXUSDT", "ADAUSDT", "LINKUSDT", "MATICUSDT",
    "DOTUSDT", "UNIUSDT", "LTCUSDT", "NEARUSDT", "INJUSDT",
]


# ── Strategy Config Builder ───────────────────────────────────────────────
async def build_promoted_config(
    r:   aioredis.Redis,
    rbt: aioredis.Redis,
    accuracy_data: dict,
) -> Optional[dict]:
    """Build promoted strategy config based on backtest + Kelly data."""
    candidates = []

    for sym in TOP_SYMBOLS:
        # Layer 4 Kelly sizing
        l4_data = await r.hgetall(f"quantum:layer4:sizing:{sym}")
        if not l4_data:
            continue
        l4_ts  = int(l4_data.get("ts", 0))
        l4_rec = l4_data.get("recommendation", "SKIP")
        l4_size = float(l4_data.get("size_usdt", 0))
        l4_kelly = float(l4_data.get("kelly_adj", 0))
        l4_lev   = int(l4_data.get("max_leverage", 1))
        l4_sharpe = float(l4_data.get("metrics_sharpe", 0))
        l4_pf     = float(l4_data.get("metrics_pf", 0))
        l4_wr     = float(l4_data.get("metrics_wr", 0))

        # Skip if no edge or stale data
        if l4_rec == "SKIP":
            continue
        if int(time.time()) - l4_ts > 600:  # stale > 10min
            continue
        if l4_sharpe < MIN_SHARPE:
            continue

        # Find best backtest strategy
        best_strategy = None
        best_sharpe   = -999.0
        async for key in rbt.scan_iter("quantum:backtest:results:*"):
            bt = await rbt.hgetall(key)
            if bt.get("symbol") != sym:
                continue
            sh = float(bt.get("metrics_sharpe", 0))
            if sh > best_sharpe:
                best_sharpe    = sh
                best_strategy  = bt.get("strategy", "ema_cross")

        candidates.append({
            "symbol":       sym,
            "strategy":     best_strategy or "ema_cross",
            "kelly_adj":    round(l4_kelly, 4),
            "size_usdt":    round(l4_size, 2),
            "max_leverage": l4_lev,
            "metrics": {
                "sharpe":        round(l4_sharpe, 3),
                "profit_factor": round(l4_pf, 3),
                "win_rate_pct":  round(l4_wr, 1),
            },
            "promoted_at": datetime.now(timezone.utc).isoformat(),
        })

    if not candidates:
        log.warning("[PROMOTER] No symbols with positive Kelly + good sharpe — no promotion")
        return None

    # Sort by sharpe, take top N
    candidates.sort(key=lambda x: x["metrics"]["sharpe"], reverse=True)
    top_candidates = candidates[:TOP_N_SYMBOLS]

    return {
        "version":      int(time.time()),
        "promoted_at":  datetime.now(timezone.utc).isoformat(),
        "gate_metrics": {
            "accuracy_pct":  float(accuracy_data.get("accuracy_pct", 0)),
            "n_trades":      int(accuracy_data.get("n_trades", 0)),
            "profit_factor": float(accuracy_data.get("profit_factor", 0)),
        },
        "symbols": top_candidates,
        "n_symbols": len(top_candidates),
        "status": "PROMOTED",
    }


async def do_promotion(
    r:            aioredis.Redis,
    rbt:          aioredis.Redis,
    gate_data:    dict,
    accuracy_data: dict,
) -> bool:
    """Execute a gate-open promotion event."""
    log.info("[PROMOTER] Gate OPEN — building promoted strategy config...")
    config = await build_promoted_config(r, rbt, accuracy_data)

    if config is None:
        await r.hset(KEY_PROMO_STATUS, mapping={
            "status":  "NO_PROMOTION",
            "reason":  "No symbols with positive Kelly passed filters",
            "ts":      str(int(time.time())),
        })
        return False

    # Write active config
    await r.set(KEY_ACTIVE_CONFIG, json.dumps(config))
    await r.expire(KEY_ACTIVE_CONFIG, 86400 * 7)  # 7 day TTL

    # Log promotion event
    event = {
        "event":      "GATE_OPEN_PROMOTION",
        "ts":         int(time.time()),
        "ts_human":   datetime.now(timezone.utc).isoformat(),
        "n_symbols":  config["n_symbols"],
        "symbols":    [c["symbol"] for c in config["symbols"]],
        "accuracy":   accuracy_data.get("accuracy_pct", "0"),
    }
    await r.lpush(KEY_PROMO_EVENTS, json.dumps(event))
    await r.ltrim(KEY_PROMO_EVENTS, 0, 49)  # cap 50

    # Update promotion status
    await r.hset(KEY_PROMO_STATUS, mapping={
        "status":   "PROMOTED",
        "n_symbols": str(config["n_symbols"]),
        "symbols":   ",".join(c["symbol"] for c in config["symbols"]),
        "version":   str(config["version"]),
        "ts":        str(int(time.time())),
        "accuracy":  str(accuracy_data.get("accuracy_pct", "0")),
    })

    log.info(
        f"[PROMOTER] Promoted {config['n_symbols']} symbols: "
        f"{[c['symbol'] for c in config['symbols']]} | "
        f"acc={accuracy_data.get('accuracy_pct', '?')}%"
    )
    return True


async def do_demotion(r: aioredis.Redis):
    """Gate closed — mark config as inactive."""
    config_raw = await r.get(KEY_ACTIVE_CONFIG)
    if config_raw:
        config = json.loads(config_raw)
        config["status"] = "GATE_RECLOSED"
        config["revoked_at"] = datetime.now(timezone.utc).isoformat()
        await r.set(KEY_ACTIVE_CONFIG, json.dumps(config))

    event = {
        "event":    "GATE_CLOSED_DEMOTION",
        "ts":       int(time.time()),
        "ts_human": datetime.now(timezone.utc).isoformat(),
        "reason":   "Layer 2 gate closed — accuracy/trades criteria no longer met",
    }
    await r.lpush(KEY_PROMO_EVENTS, json.dumps(event))
    await r.ltrim(KEY_PROMO_EVENTS, 0, 49)

    await r.hset(KEY_PROMO_STATUS, mapping={
        "status": "GATE_RECLOSED",
        "reason": "Gate closed — accuracy fell below threshold",
        "ts":     str(int(time.time())),
    })
    log.warning("[PROMOTER] Gate CLOSED — strategy config marked inactive")


# ── Main Loop ─────────────────────────────────────────────────────────────
async def main():
    log.info("[PROMOTER] Layer 2 Signal Auto-Promoter starting")
    r   = aioredis.Redis(host=REDIS_HOST, port=6379, db=REDIS_LIVE_DB, decode_responses=True)
    rbt = aioredis.Redis(host=REDIS_HOST, port=6379, db=REDIS_BT_DB,   decode_responses=True)
    await r.ping()
    log.info("[PROMOTER] Redis OK")

    last_gate     = "CLOSED"
    last_promoted_version = 0

    while True:
        try:
            gate_data    = await r.hgetall(KEY_GATE)
            accuracy_data = await r.hgetall(KEY_ACCURACY)

            gate_status = gate_data.get("gate", "CLOSED") if gate_data else "CLOSED"
            n_trades    = int(accuracy_data.get("n_trades",   0)) if accuracy_data else 0
            accuracy    = float(accuracy_data.get("accuracy_pct", 0)) if accuracy_data else 0.0

            if gate_status == "OPEN" and last_gate != "OPEN":
                log.info(
                    f"[PROMOTER] Gate opened! n={n_trades} "
                    f"acc={accuracy:.1f}% → promoting..."
                )
                await do_promotion(r, rbt, gate_data or {}, accuracy_data or {})
                last_gate = "OPEN"

            elif gate_status == "CLOSED" and last_gate == "OPEN":
                await do_demotion(r)
                last_gate = "CLOSED"

            else:
                last_gate = gate_status
                log.debug(
                    f"[PROMOTER] gate={gate_status} n={n_trades}/{30} "
                    f"acc={accuracy:.1f}%"
                )

        except Exception as e:
            log.error(f"[PROMOTER] Loop error: {e}", exc_info=True)

        await asyncio.sleep(CHECK_INTERVAL)


if __name__ == "__main__":
    asyncio.run(main())
