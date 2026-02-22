#!/usr/bin/env python3
"""
Layer 5 — Execution Quality Monitor
======================================
READ-ONLY post-trade analyzer. Consumes from quantum:stream:trade.closed
and quantum:stream:apply.plan to compute execution quality metrics.

Tracks:
  A) Slippage: (actual_fill_price - signal_price) / signal_price * 10000  (basis points)
  B) Fill rate: % of signals that resulted in fills
  C) P&L attribution: alpha (strategy edge) vs beta (market move)
  D) Execution latency: signal_ts → fill_ts in milliseconds
  E) Per-symbol quality scores (rolling 30-trade window)
  F) Churning detection: too many round-trips per symbol per day

All results read-only from streams. NEVER writes to apply.plan or position keys.

Operator reads:
  redis-cli hgetall quantum:layer5:execution:latest
  redis-cli hgetall quantum:layer5:slippage:BTCUSDT
  redis-cli hgetall quantum:layer5:quality:BTCUSDT
"""

import asyncio
import json
import logging
import os
import time
from collections import defaultdict, deque
from datetime import datetime, timezone
from typing import Dict, List, Optional

import redis.asyncio as aioredis

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s exec_qual %(message)s",
)
log = logging.getLogger("exec_qual")

# ── Config ───────────────────────────────────────────────────────────────
REDIS_HOST         = os.getenv("REDIS_HOST", "localhost")
REDIS_DB           = 0
STREAM_TRADES      = "quantum:stream:trade.closed"
STREAM_PLAN        = "quantum:stream:apply.plan"
HEARTBEAT_INTERVAL = 30   # seconds
ROLLING_WINDOW     = 30   # trades per symbol for rolling metrics
MAX_SLIPPAGE_BPS   = float(os.getenv("MAX_SLIPPAGE_BPS", "10.0"))  # alert threshold
CHURN_THRESHOLD    = int(os.getenv("CHURN_THRESHOLD",     "6"))     # round-trips/day/symbol = churn

KEY_STATUS   = "quantum:layer5:execution:latest"
KEY_HISTORY  = "quantum:layer5:execution:history"
CG_TRADES    = "layer5_exec_quality_trades"
CG_PLAN      = "layer5_exec_quality_plan"


# ── In-memory State ───────────────────────────────────────────────────────
# Per-symbol rolling windows (deque of dicts)
_symbol_trades: Dict[str, deque] = defaultdict(lambda: deque(maxlen=ROLLING_WINDOW))
_symbol_plans:  Dict[str, list]  = defaultdict(list)   # unmatched plans awaiting fill
_daily_churn:   Dict[str, int]   = defaultdict(int)     # symbol → round-trips today
_daily_date:    str = ""

_totals = {
    "trades_processed": 0,
    "plans_processed":  0,
    "alerts_fired":     0,
    "symbols_seen":     set(),
}


# ── Metrics Calculators ───────────────────────────────────────────────────
def compute_slippage_bps(signal_price: float, fill_price: float, side: str) -> float:
    """
    Positive = good slippage (filled better than signal).
    Negative = adverse slippage (filled worse than signal).
    For LONG:  positive if fill_price < signal_price
    For SHORT: positive if fill_price > signal_price
    """
    if signal_price <= 0:
        return 0.0
    raw_bps = (signal_price - fill_price) / signal_price * 10_000
    return round(raw_bps if side.upper() == "LONG" else -raw_bps, 3)


def compute_pnl_attribution(entry: float, exit_p: float, market_ret: float,
                             side: str) -> dict:
    """
    Total return = alpha (strategy) + beta (market)
    Market return (beta) = abs(market_ret) during holding period
    Alpha = total - beta
    """
    if entry <= 0:
        return {"total_pct": 0.0, "alpha_pct": 0.0, "beta_pct": 0.0}
    total = (exit_p - entry) / entry * 100
    if side.upper() == "SHORT":
        total = -total
    beta  = round(market_ret, 4)
    alpha = round(total - beta, 4)
    return {
        "total_pct": round(total, 4),
        "alpha_pct": alpha,
        "beta_pct":  beta,
    }


def rolling_metrics(trades: deque) -> dict:
    if not trades:
        return {"n": 0, "avg_pnl_pct": 0.0, "avg_slippage_bps": 0.0,
                "win_rate": 0.0, "avg_latency_ms": 0.0, "quality_score": 0.0}
    n    = len(trades)
    pnls = [t.get("pnl_pct", 0.0) for t in trades]
    slips = [t.get("slippage_bps", 0.0) for t in trades]
    lats  = [t.get("latency_ms", 0.0) for t in trades if t.get("latency_ms", 0) > 0]
    wins  = sum(1 for p in pnls if p > 0)

    avg_pnl   = round(sum(pnls) / n, 4)
    avg_slip  = round(sum(slips) / n, 3)
    win_rate  = round(wins / n * 100, 1)
    avg_lat   = round(sum(lats) / len(lats), 0) if lats else 0.0

    # Quality score (0-100):
    #   40 pts: positive avg_pnl (scaled)
    #   30 pts: win_rate >= 50%
    #   20 pts: slippage > -5 bps
    #   10 pts: avg_latency < 500ms
    q = 0.0
    q += min(40.0, max(0.0, avg_pnl * 20))   # 2% avg pnl = full 40 pts
    q += 30.0 if win_rate >= 50 else win_rate / 50 * 30
    q += 20.0 if avg_slip > -5 else max(0, (avg_slip + 10) / 5 * 20)
    q += 10.0 if 0 < avg_lat < 500 else (10.0 if avg_lat == 0 else 0.0)

    return {
        "n":              n,
        "avg_pnl_pct":    avg_pnl,
        "avg_slippage_bps": avg_slip,
        "win_rate_pct":   win_rate,
        "avg_latency_ms": avg_lat,
        "quality_score":  round(q, 1),
    }


# ── Stream Processors ─────────────────────────────────────────────────────
async def process_trade_event(r: aioredis.Redis, msg_id: str, fields: dict):
    """Process a quantum:stream:trade.closed event."""
    try:
        sym     = fields.get("symbol", fields.get("Symbol", "UNKNOWN"))
        side    = fields.get("side", "LONG")
        entry   = float(fields.get("entry_price", fields.get("entryPrice", 0)))
        exit_p  = float(fields.get("exit_price",  fields.get("exitPrice",  0)))
        pnl     = float(fields.get("pnl",         fields.get("realizedPnl", 0)))
        pnl_pct = float(fields.get("pnl_pct",     0.0))
        fill_price   = float(fields.get("fill_price",   exit_p))
        signal_price = float(fields.get("signal_price", exit_p))
        signal_ts    = int(fields.get("signal_ts",   0))
        fill_ts      = int(fields.get("fill_ts",     int(time.time() * 1000)))
        market_ret   = float(fields.get("market_return_pct", 0.0))

        if pnl_pct == 0.0 and entry > 0 and exit_p > 0:
            pnl_pct = (exit_p - entry) / entry * 100
            if side.upper() == "SHORT":
                pnl_pct = -pnl_pct

        slippage_bps = compute_slippage_bps(signal_price, fill_price, side)
        latency_ms   = (fill_ts - signal_ts) if signal_ts > 0 else 0
        attribution  = compute_pnl_attribution(entry, exit_p, market_ret, side)

        trade_rec = {
            "symbol":        sym,
            "side":          side,
            "entry":         entry,
            "exit":          exit_p,
            "pnl_pct":       pnl_pct,
            "pnl_usdt":      pnl,
            "slippage_bps":  slippage_bps,
            "latency_ms":    latency_ms,
            "alpha_pct":     attribution["alpha_pct"],
            "msg_id":        msg_id,
        }
        _symbol_trades[sym].append(trade_rec)
        _totals["trades_processed"] += 1
        _totals["symbols_seen"].add(sym)

        # Churn detection
        today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
        global _daily_date
        if today != _daily_date:
            _daily_date = today
            _daily_churn.clear()
        _daily_churn[sym] += 1

        # Compute + publish per-symbol metrics
        metrics = rolling_metrics(_symbol_trades[sym])
        sym_key = f"quantum:layer5:quality:{sym}"
        flat = {k: str(v) for k, v in metrics.items()}
        flat["symbol"]   = sym
        flat["ts"]       = str(int(time.time()))
        flat["churn_today"] = str(_daily_churn[sym])
        await r.hset(sym_key, mapping=flat)
        await r.expire(sym_key, 3600)

        # Slippage key
        slip_key = f"quantum:layer5:slippage:{sym}"
        await r.hset(slip_key, mapping={
            "last_bps":   str(slippage_bps),
            "latency_ms": str(latency_ms),
            "alpha_pct":  str(attribution["alpha_pct"]),
            "ts":         str(int(time.time())),
        })

        # Alerts
        issues = []
        if slippage_bps < -MAX_SLIPPAGE_BPS:
            issues.append(f"HIGH_SLIPPAGE {slippage_bps:.1f}bps")
        if _daily_churn[sym] >= CHURN_THRESHOLD:
            issues.append(f"CHURN {_daily_churn[sym]} round-trips today")
        if metrics["quality_score"] < 30:
            issues.append(f"LOW_QUALITY score={metrics['quality_score']}")

        level = log.warning if issues else log.info
        level(
            f"[L5] {sym} {side} pnl={pnl_pct:+.3f}% "
            f"slip={slippage_bps:+.1f}bps lat={latency_ms}ms "
            f"alpha={attribution['alpha_pct']:+.3f}% "
            f"quality={metrics['quality_score']:.0f}/100"
            + (f" ⚠️ {', '.join(issues)}" if issues else "")
        )
        if issues:
            _totals["alerts_fired"] += 1
            await r.lpush("quantum:layer5:alerts", json.dumps({
                "ts": int(time.time()), "symbol": sym, "issues": issues,
            }))
            await r.ltrim("quantum:layer5:alerts", 0, 99)

    except Exception as e:
        log.error(f"trade event error: {e}", exc_info=True)


async def process_plan_event(r: aioredis.Redis, msg_id: str, fields: dict):
    """Track apply.plan signals for fill-rate computation."""
    try:
        sym = fields.get("symbol", "UNKNOWN")
        _symbol_plans[sym].append({
            "msg_id": msg_id,
            "ts":     int(time.time()),
            "action": fields.get("action", ""),
        })
        # Keep only last 100 per symbol
        if len(_symbol_plans[sym]) > 100:
            _symbol_plans[sym] = _symbol_plans[sym][-100:]
        _totals["plans_processed"] += 1
    except Exception:
        pass


# ── Status Publisher ──────────────────────────────────────────────────────
async def publish_status(r: aioredis.Redis):
    """Publish aggregate execution quality summary."""
    while True:
        try:
            # Aggregate across all symbols
            all_metrics = [rolling_metrics(_symbol_trades[sym])
                           for sym in _symbol_trades if _symbol_trades[sym]]

            if all_metrics:
                avg_quality  = sum(m["quality_score"]    for m in all_metrics) / len(all_metrics)
                avg_slip     = sum(m["avg_slippage_bps"] for m in all_metrics) / len(all_metrics)
                avg_wr       = sum(m["win_rate_pct"]     for m in all_metrics) / len(all_metrics)
                total_trades = sum(m["n"]                for m in all_metrics)
            else:
                avg_quality = avg_slip = avg_wr = total_trades = 0.0

            # Fill rate
            total_plans  = sum(len(v) for v in _symbol_plans.values())
            fill_rate    = (total_trades / total_plans * 100) if total_plans > 0 else 0.0

            # Churn alert
            churning = [sym for sym, cnt in _daily_churn.items() if cnt >= CHURN_THRESHOLD]

            status = {
                "ts":               str(int(time.time())),
                "ts_human":         datetime.now(timezone.utc).isoformat(),
                "trades_processed": str(_totals["trades_processed"]),
                "plans_processed":  str(_totals["plans_processed"]),
                "alerts_fired":     str(_totals["alerts_fired"]),
                "symbols_tracked":  str(len(_totals["symbols_seen"])),
                "avg_quality_score":str(round(avg_quality, 1)),
                "avg_slippage_bps": str(round(avg_slip, 2)),
                "avg_win_rate_pct": str(round(avg_wr, 1)),
                "fill_rate_pct":    str(round(fill_rate, 1)),
                "churning_symbols": ",".join(churning) if churning else "none",
                "rolling_window":   str(ROLLING_WINDOW),
            }
            await r.hset(KEY_STATUS, mapping=status)

            if _totals["trades_processed"] > 0:
                log.info(
                    f"[L5] trades={_totals['trades_processed']} "
                    f"quality={round(avg_quality,1)}/100 "
                    f"slip={round(avg_slip,2)}bps "
                    f"fill_rate={round(fill_rate,1)}% "
                    f"alerts={_totals['alerts_fired']}"
                )
            else:
                log.info("[L5] monitoring — 0 trades processed yet (system in FREEZE)")

        except Exception as e:
            log.warning(f"status publish error: {e}")
        await asyncio.sleep(HEARTBEAT_INTERVAL)


# ── Stream Consumer ───────────────────────────────────────────────────────
async def consume_stream(r: aioredis.Redis, stream: str, cg: str, handler):
    """XREADGROUP consumer loop for a stream."""
    # Create consumer group
    try:
        await r.xgroup_create(stream, cg, id="0", mkstream=True)
        log.info(f"[L5] Created consumer group {cg} on {stream}")
    except Exception:
        log.info(f"[L5] Consumer group {cg} already exists on {stream}")

    consumer_name = f"exec_quality_{os.getpid()}"

    # First: drain pending messages (catch up)
    while True:
        msgs = await r.xreadgroup(cg, consumer_name, {stream: "0"}, count=100, block=0)
        if not msgs or not msgs[0][1]:
            break
        for msg_id, fields in msgs[0][1]:
            await handler(r, msg_id, fields)
            await r.xack(stream, cg, msg_id)

    # Then: new messages
    while True:
        try:
            msgs = await r.xreadgroup(cg, consumer_name, {stream: ">"}, count=10, block=5000)
            if not msgs:
                continue
            for msg_id, fields in msgs[0][1]:
                await handler(r, msg_id, fields)
                await r.xack(stream, cg, msg_id)
        except Exception as e:
            log.error(f"Stream {stream} error: {e}")
            await asyncio.sleep(2)


# ── Main ──────────────────────────────────────────────────────────────────
async def main():
    log.info("[L5] Execution Quality Monitor starting")
    log.info(f"[L5] max_slippage_alert={MAX_SLIPPAGE_BPS}bps churn_threshold={CHURN_THRESHOLD}/day")
    log.info("[L5] READ-ONLY — consumes trade.closed + apply.plan streams")

    r = aioredis.Redis(host=REDIS_HOST, port=6379, db=REDIS_DB, decode_responses=True)
    await r.ping()
    log.info("[L5] Redis OK")

    asyncio.create_task(publish_status(r))

    await asyncio.gather(
        consume_stream(r, STREAM_TRADES, CG_TRADES, process_trade_event),
        consume_stream(r, STREAM_PLAN,   CG_PLAN,   process_plan_event),
    )


if __name__ == "__main__":
    asyncio.run(main())
