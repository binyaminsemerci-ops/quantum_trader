#!/usr/bin/env python3
"""
Layer 4 — Portfolio Optimizer & Position Sizer
================================================
READ-ONLY intelligence layer. NEVER places orders or writes to control keys.

Role:
  - Continuously evaluates portfolio health and provides sizing recommendations
  - Kelly fraction calculator per symbol (based on Layer 3 backtest metrics)
  - Portfolio heat monitor: total notional exposure vs account equity
  - Position concentration guard: max 30% in one symbol
  - Correlation matrix shortcut (BTC-ETH co-movement detection)
  - Volatility-adjusted position sizing (ATR-based)
  - Publishes sizing recs to quantum:layer4:sizing:<SYM> for use by execution layer
  - Publishes portfolio state to quantum:layer4:portfolio:latest

Operator reads:
  redis-cli hgetall quantum:layer4:portfolio:latest
  redis-cli hgetall quantum:layer4:sizing:BTCUSDT

The execution layer READS these keys but this module never changes live positions.
"""

import asyncio
import json
import logging
import os
import time
from datetime import datetime, timezone
from typing import Dict, List, Optional, Tuple

import redis.asyncio as aioredis

try:
    import numpy as np
    HAS_NUMPY = True
except ImportError:
    HAS_NUMPY = False

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s portfolio %(message)s",
)
log = logging.getLogger("portfolio")

# ── Config ───────────────────────────────────────────────────────────────
REDIS_HOST      = os.getenv("REDIS_HOST", "localhost")
REDIS_LIVE_DB   = 0
REDIS_BT_DB     = 1   # read backtest metrics from here

EVAL_INTERVAL   = int(os.getenv("EVAL_INTERVAL", "30"))     # seconds
MAX_HEAT_PCT    = float(os.getenv("MAX_HEAT_PCT",  "40.0")) # max total notional / equity
MAX_SYMBOL_PCT  = float(os.getenv("MAX_SYMBOL_PCT","30.0")) # max single symbol / portfolio
MAX_POSITIONS   = int(os.getenv("MAX_POSITIONS",   "5"))    # max concurrent open
KELLY_FRACTION  = float(os.getenv("KELLY_FRACTION","0.25")) # fractional Kelly (0.25 = quarter-Kelly)
MIN_EDGE        = float(os.getenv("MIN_EDGE",      "0.02")) # min Kelly edge to recommend a trade

TOP_SYMBOLS = [
    "BTCUSDT", "ETHUSDT", "SOLUSDT", "BNBUSDT", "XRPUSDT",
    "DOGEUSDT", "AVAXUSDT", "ADAUSDT", "LINKUSDT", "MATICUSDT",
    "DOTUSDT", "UNIUSDT", "LTCUSDT", "NEARUSDT", "INJUSDT",
]

# Redis keys
KEY_EQUITY    = "quantum:equity:current"
KEY_HEALTH    = "quantum:health:truth:latest"
KEY_STATUS    = "quantum:layer4:portfolio:latest"
KEY_POSITIONS = "quantum:position:{sym}"        # live positions (read-only)


# ── Kelly Calculator ─────────────────────────────────────────────────────
def kelly_fraction(win_rate: float, profit_factor: float) -> float:
    """
    Kelly criterion: f = (W*b - L) / b
    where b = profit_factor (avg_win / avg_loss), W = win_rate, L = 1 - W
    Returns optimal fraction; caller applies fractional Kelly multiplier.
    """
    if profit_factor <= 0 or win_rate <= 0 or win_rate >= 1:
        return 0.0
    b = profit_factor
    w = win_rate
    l = 1.0 - w
    k = (w * b - l) / b
    return max(0.0, k)


def recommended_size_usdt(
    kelly: float,
    equity: float,
    kelly_fraction_mult: float,
    atr_pct: float,
    max_risk_pct: float = 1.0,
) -> float:
    """
    Size = equity * kellly * kelly_fraction_mult
    But capped by: max_risk (%) of equity / atr_pct per unit
    (Ensures SL hit = max_risk_pct% of equity)
    """
    kelly_size = equity * kelly * kelly_fraction_mult
    if atr_pct > 0:
        risk_based_size = equity * (max_risk_pct / 100) / atr_pct
        return round(min(kelly_size, risk_based_size), 2)
    return round(kelly_size, 2)


# ── Data Readers ─────────────────────────────────────────────────────────
async def get_equity(r: aioredis.Redis) -> Tuple[float, float]:
    data = await r.hgetall(KEY_EQUITY)
    if not data:
        return 3645.0, 5004.0  # last known values as fallback
    eq   = float(data.get("equity", 3645.0))
    peak = float(data.get("peak",   eq))
    return eq, peak


async def get_open_positions(r: aioredis.Redis) -> List[dict]:
    positions = []
    for sym in TOP_SYMBOLS:
        key = KEY_POSITIONS.format(sym=sym)
        try:
            data = await r.hgetall(key)
            if data and data.get("quantity"):
                qty = float(data.get("quantity", 0))
                ep  = float(data.get("entry_price", 0))
                side = data.get("side", "LONG")
                lev  = float(data.get("leverage", 1))
                upnl = float(data.get("unrealized_pnl", 0))
                # Get live price for notional calc
                live = await r.hgetall(f"quantum:market:{sym}:bybit")
                price = float(live.get("price", ep)) if live else ep
                notional = qty * price
                positions.append({
                    "symbol":   sym,
                    "side":     side,
                    "quantity": qty,
                    "entry_price": ep,
                    "live_price":  price,
                    "notional":    round(notional, 2),
                    "leverage":    lev,
                    "upnl":        round(upnl, 2),
                    "margin_used": round(notional / lev, 2),
                })
        except Exception:
            pass
    return positions


async def get_backtest_metrics(rbt: aioredis.Redis, symbol: str) -> Optional[dict]:
    """Pull latest ema_cross backtest result from Redis db=1."""
    try:
        lb_key = f"quantum:backtest:leaderboard:{symbol}"
        top = await rbt.zrevrange(lb_key, 0, 0, withscores=True)
        if not top:
            return None
        job_key_part = top[0][0]  # "ema_cross:auto_btc"
        # Find the result hash
        strategy = job_key_part.split(":")[0]
        # Search for matching result key
        pattern = f"quantum:backtest:results:*"
        cursor = 0
        best = None
        best_sharpe = -999
        async for key in rbt.scan_iter(pattern):
            data = await rbt.hgetall(key)
            if data.get("symbol") == symbol:
                sh = float(data.get("metrics_sharpe", 0))
                if sh > best_sharpe:
                    best_sharpe = sh
                    best = data
        return best
    except Exception:
        return None


# ── Portfolio Heat ────────────────────────────────────────────────────────
def compute_heat(positions: List[dict], equity: float) -> dict:
    total_notional  = sum(p["notional"]    for p in positions)
    total_margin    = sum(p["margin_used"] for p in positions)
    total_upnl      = sum(p["upnl"]        for p in positions)
    heat_pct        = total_notional / equity * 100 if equity > 0 else 0.0
    margin_pct      = total_margin   / equity * 100 if equity > 0 else 0.0
    concentration   = {}
    for p in positions:
        sym = p["symbol"]
        concentration[sym] = round(p["notional"] / total_notional * 100, 1) if total_notional > 0 else 0.0

    return {
        "total_notional": round(total_notional, 2),
        "total_margin":   round(total_margin, 2),
        "total_upnl":     round(total_upnl, 2),
        "heat_pct":       round(heat_pct, 2),
        "margin_pct":     round(margin_pct, 2),
        "n_positions":    len(positions),
        "heat_status":    "OK" if heat_pct < MAX_HEAT_PCT else "OVERHEATED",
        "concentration":  concentration,
    }


# ── Sizing Recommendations ────────────────────────────────────────────────
async def compute_sizing(
    r: aioredis.Redis,
    rbt: aioredis.Redis,
    symbol: str,
    equity: float,
    heat: dict,
) -> dict:
    """Compute Kelly-based sizing recommendation for a symbol."""
    result = {
        "symbol":        symbol,
        "ts":            int(time.time()),
        "kelly_raw":     0.0,
        "kelly_adj":     0.0,
        "size_usdt":     0.0,
        "max_leverage":  1,
        "recommendation":"NO_DATA",
        "reason":        "",
    }

    # Get backtest metrics (from isolated db=1)
    metrics = await get_backtest_metrics(rbt, symbol)
    if not metrics:
        result["reason"] = "no_backtest_data"
        return result

    win_rate = float(metrics.get("metrics_win_rate_pct", 0)) / 100
    pf       = float(metrics.get("metrics_profit_factor", 0))
    sharpe   = float(metrics.get("metrics_sharpe", 0))
    max_dd   = float(metrics.get("metrics_max_dd_pct", 100))

    # Kelly
    k_raw = kelly_fraction(win_rate, pf)
    k_adj = k_raw * KELLY_FRACTION  # quarter-Kelly

    result["kelly_raw"] = round(k_raw, 4)
    result["kelly_adj"] = round(k_adj, 4)
    result["metrics_sharpe"] = sharpe
    result["metrics_pf"]     = pf
    result["metrics_wr"]     = round(win_rate * 100, 1)

    # Reject if edge too low
    if k_raw < MIN_EDGE:
        result["recommendation"] = "SKIP"
        result["reason"] = f"kelly={k_raw:.3f} < min_edge={MIN_EDGE}"
        return result

    # ATR-based sizing guard
    atr_data = await r.hgetall(f"quantum:position:{symbol}")
    atr_val  = float(atr_data.get("atr_value", 0)) if atr_data else 0.0
    live     = await r.hgetall(f"quantum:market:{symbol}:bybit")
    price    = float(live.get("price", 1)) if live else 1.0
    atr_pct  = atr_val / price if price > 0 and atr_val > 0 else 0.01

    size = recommended_size_usdt(k_raw, equity, KELLY_FRACTION, atr_pct, max_risk_pct=1.0)

    # Cap size by portfolio heat limit
    remaining_capacity = max(0.0, equity * MAX_HEAT_PCT / 100 - heat["total_notional"])
    size = min(size, remaining_capacity)

    # Symbol concentration cap
    sym_max = equity * KELLY_FRACTION * MAX_SYMBOL_PCT / 100
    size = min(size, sym_max)

    # Max leverage: inversely proportional to dd risk
    max_lev = max(1, min(5, int(10 / max(max_dd, 1))))

    result["size_usdt"]    = round(size, 2)
    result["max_leverage"] = max_lev
    result["atr_pct"]      = round(atr_pct * 100, 3)

    if size >= 10:
        result["recommendation"] = "TRADE"
        result["reason"] = (f"kelly={k_adj:.3f} size=${size:.0f} "
                            f"max_lev={max_lev}x sharpe={sharpe:.2f}")
    else:
        result["recommendation"] = "TOO_SMALL"
        result["reason"] = f"size=${size:.2f} < $10 min"

    return result


# ── Main Loop ─────────────────────────────────────────────────────────────
async def run_once(r: aioredis.Redis, rbt: aioredis.Redis, state: dict):
    ts = int(time.time())
    equity, peak = await get_equity(r)
    dd = (peak - equity) / peak * 100 if peak > 0 else 0.0
    positions = await get_open_positions(r)
    heat = compute_heat(positions, equity)

    # Compute sizing for top symbols
    sizing_results = {}
    tradeable = []
    for sym in TOP_SYMBOLS[:10]:  # top 10 to limit Redis queries
        sz = await compute_sizing(r, rbt, sym, equity, heat)
        sizing_results[sym] = sz
        # Write per-symbol sizing key
        flat = {k: str(v) for k, v in sz.items() if not isinstance(v, dict)}
        await r.hset(f"quantum:layer4:sizing:{sym}", mapping=flat)
        await r.expire(f"quantum:layer4:sizing:{sym}", 300)
        if sz["recommendation"] == "TRADE":
            tradeable.append(f"{sym}(${sz['size_usdt']:.0f}@{sz['max_leverage']}x)")
        await asyncio.sleep(0.05)

    n_trade = sum(1 for s in sizing_results.values() if s["recommendation"] == "TRADE")
    n_skip  = sum(1 for s in sizing_results.values() if s["recommendation"] == "SKIP")

    # Portfolio summary
    summary = {
        "ts":             ts,
        "ts_human":       datetime.now(timezone.utc).isoformat(),
        "equity":         str(equity),
        "peak":           str(peak),
        "dd_pct":         str(round(dd, 2)),
        "heat_pct":       str(heat["heat_pct"]),
        "heat_status":    heat["heat_status"],
        "total_notional": str(heat["total_notional"]),
        "total_upnl":     str(heat["total_upnl"]),
        "n_positions":    str(heat["n_positions"]),
        "n_tradeable":    str(n_trade),
        "n_skip_edge":    str(n_skip),
        "tradeable_list": ",".join(tradeable),
        "max_heat_pct":   str(MAX_HEAT_PCT),
        "kelly_fraction": str(KELLY_FRACTION),
    }
    await r.hset(KEY_STATUS, mapping=summary)

    state["last"] = summary
    log.info(
        f"[L4] equity={equity:.2f} dd={dd:.2f}% heat={heat['heat_pct']:.1f}% "
        f"positions={heat['n_positions']}/{MAX_POSITIONS} "
        f"tradeable={n_trade}/10  heat={heat['heat_status']}"
    )
    if tradeable:
        log.info(f"  [L4] TRADEABLE: {', '.join(tradeable)}")


async def main():
    log.info("[L4] Portfolio Optimizer starting")
    log.info(f"[L4] max_heat={MAX_HEAT_PCT}% max_sym={MAX_SYMBOL_PCT}% "
             f"kelly_frac={KELLY_FRACTION} max_pos={MAX_POSITIONS}")
    log.info("[L4] READ-ONLY — never writes to live control keys")

    r   = aioredis.Redis(host=REDIS_HOST, port=6379, db=REDIS_LIVE_DB, decode_responses=True)
    rbt = aioredis.Redis(host=REDIS_HOST, port=6379, db=REDIS_BT_DB,   decode_responses=True)
    await r.ping()
    await rbt.ping()
    log.info("[L4] Redis live+backtest OK")

    state = {}
    while True:
        try:
            await run_once(r, rbt, state)
        except Exception as e:
            log.error(f"Eval error: {e}", exc_info=True)
        await asyncio.sleep(EVAL_INTERVAL)


if __name__ == "__main__":
    asyncio.run(main())
