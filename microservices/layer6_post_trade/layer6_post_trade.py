#!/usr/bin/env python3
"""
Layer 6 — Post-Trade Analytics Reporter
=========================================
Generates daily/session reports with full P&L decomposition.
Runs in two modes:
  A) Continuous: publishes rolling 24h summary every 5 minutes to Redis
  B) Daily report: triggered at 00:05 UTC, writes JSON to disk + Redis

Reports include:
  - P&L attribution: realized + unrealized, by symbol, by strategy
  - Drawdown decomposition: which trades/symbols caused the DD
  - Execution quality summary (from Layer 5 data)
  - Churn analysis (from anti-churn guard data)
  - Model performance: signal accuracy, profit factor per model
  - Portfolio heat history
  - Comparison: current session vs previous session

READ-ONLY: reads streams + Layer 5 + Layer 4 + Layer 2 metadata.
NEVER writes to live control keys.

Operator reads:
  redis-cli hgetall quantum:layer6:daily:latest
  redis-cli hgetall quantum:layer6:session:latest
  cat /opt/quantum/data/reports/YYYY-MM-DD.json
"""

import asyncio
import json
import logging
import os
import time
from collections import defaultdict
from datetime import datetime, timezone, timedelta
from pathlib import Path
from typing import Dict, List

import redis.asyncio as aioredis

try:
    import pandas as pd
    HAS_PANDAS = True
except ImportError:
    HAS_PANDAS = False

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s layer6 %(message)s",
)
log = logging.getLogger("layer6")

# ── Config ───────────────────────────────────────────────────────────────
REDIS_HOST     = os.getenv("REDIS_HOST", "localhost")
REDIS_DB       = 0
REDIS_BT_DB    = 1
DATA_ROOT      = Path(os.getenv("DATA_ROOT", "/opt/quantum/data"))
REPORTS_DIR    = DATA_ROOT / "reports"
ROLLING_WINDOW = int(os.getenv("ROLLING_WINDOW", "300"))  # seconds for rolling summary
PUBLISH_EVERY  = int(os.getenv("PUBLISH_EVERY",  "300"))  # 5 minutes

STREAM_TRADES   = "quantum:stream:trade.closed"
CG_NAME         = "layer6_reporter"
KEY_DAILY       = "quantum:layer6:daily:latest"
KEY_SESSION     = "quantum:layer6:session:latest"
KEY_EQUITY      = "quantum:equity:current"
KEY_L5_STATUS   = "quantum:layer5:execution:latest"
KEY_L4_STATUS   = "quantum:layer4:portfolio:latest"
KEY_L2_ACCURACY = "quantum:sandbox:accuracy:latest"
KEY_L2_GATE     = "quantum:sandbox:gate:latest"
KEY_HEALTH      = "quantum:health:truth:latest"
KEY_DAG8        = "quantum:dag8:freeze_exit:status"
KEY_FNG         = "quantum:sentiment:fear_greed"


# ── Session State ─────────────────────────────────────────────────────────
_session_start = int(time.time())
_session_trades: List[dict]          = []
_by_symbol:      Dict[str, list]     = defaultdict(list)
_by_hour:        Dict[int, list]     = defaultdict(list)   # hour (0-23) → trades


# ── Stream Reader ─────────────────────────────────────────────────────────
async def backfill_session(r: aioredis.Redis):
    """Read last 24h from trade.closed stream for session baseline."""
    try:
        cutoff_ms = (int(time.time()) - 86400) * 1000
        msgs = await r.xrange(STREAM_TRADES, min=f"{cutoff_ms}-0", max="+")
        for msg_id, fields in msgs:
            ingest_trade(msg_id, fields)
        log.info(f"[L6] Backfilled {len(_session_trades)} trades from last 24h")
    except Exception as e:
        log.warning(f"[L6] Backfill error: {e}")


def ingest_trade(msg_id: str, fields: dict):
    sym      = fields.get("symbol", fields.get("Symbol", "UNKNOWN"))
    side     = fields.get("side", "LONG")
    pnl_pct  = float(fields.get("pnl_pct",          0.0))
    pnl_usdt = float(fields.get("pnl", fields.get("realizedPnl", 0.0)))
    entry    = float(fields.get("entry_price", fields.get("entryPrice", 0.0)))
    exit_p   = float(fields.get("exit_price",  fields.get("exitPrice",  0.0)))
    ts       = int(str(msg_id).split("-")[0]) // 1000

    if pnl_pct == 0.0 and entry > 0 and exit_p > 0:
        pnl_pct = (exit_p - entry) / entry * 100
        if side.upper() == "SHORT":
            pnl_pct = -pnl_pct

    rec = {
        "msg_id":   msg_id,
        "symbol":   sym,
        "side":     side,
        "pnl_pct":  pnl_pct,
        "pnl_usdt": pnl_usdt,
        "entry":    entry,
        "exit":     exit_p,
        "ts":       ts,
    }
    _session_trades.append(rec)
    _by_symbol[sym].append(rec)
    hour = datetime.fromtimestamp(ts, tz=timezone.utc).hour
    _by_hour[hour].append(rec)


# ── Analytics ─────────────────────────────────────────────────────────────
def compute_trade_stats(trades: List[dict]) -> dict:
    if not trades:
        return {
            "n_trades": 0, "realized_pnl_usdt": 0.0, "win_rate_pct": 0.0,
            "profit_factor": 0.0, "avg_pnl_pct": 0.0, "max_win_pct": 0.0,
            "max_loss_pct": 0.0, "avg_win_pct": 0.0, "avg_loss_pct": 0.0,
            "expectancy_usdt": 0.0,
        }
    n     = len(trades)
    wins  = [t for t in trades if t["pnl_pct"] > 0]
    losses = [t for t in trades if t["pnl_pct"] <= 0]

    total_pnl = sum(t["pnl_usdt"] for t in trades)
    gross_win  = sum(t["pnl_usdt"] for t in wins)
    gross_loss = abs(sum(t["pnl_usdt"] for t in losses)) or 1e-10

    return {
        "n_trades":          n,
        "realized_pnl_usdt": round(total_pnl, 2),
        "win_rate_pct":      round(len(wins) / n * 100, 1) if n else 0.0,
        "profit_factor":     round(gross_win / gross_loss, 3),
        "avg_pnl_pct":       round(sum(t["pnl_pct"] for t in trades) / n, 4),
        "max_win_pct":       round(max((t["pnl_pct"] for t in wins),    default=0.0), 4),
        "max_loss_pct":      round(min((t["pnl_pct"] for t in losses),  default=0.0), 4),
        "avg_win_pct":       round(sum(t["pnl_pct"] for t in wins) / len(wins), 4) if wins else 0.0,
        "avg_loss_pct":      round(sum(t["pnl_pct"] for t in losses) / len(losses), 4) if losses else 0.0,
        "expectancy_usdt":   round(total_pnl / n, 3),
    }


def compute_drawdown_decomposition(trades: List[dict], equity_start: float) -> dict:
    """Which symbols/time-periods caused the most drawdown."""
    by_sym = defaultdict(float)
    by_hour_pnl = defaultdict(float)

    for t in trades:
        by_sym[t["symbol"]] += t["pnl_usdt"]
        h = datetime.fromtimestamp(t["ts"], tz=timezone.utc).hour
        by_hour_pnl[h] += t["pnl_usdt"]

    worst_symbols = sorted(by_sym.items(), key=lambda x: x[1])[:5]
    best_symbols  = sorted(by_sym.items(), key=lambda x: x[1], reverse=True)[:5]
    worst_hours   = sorted(by_hour_pnl.items(), key=lambda x: x[1])[:3]

    return {
        "worst_symbols": [{"symbol": s, "pnl_usdt": round(p, 2)} for s, p in worst_symbols],
        "best_symbols":  [{"symbol": s, "pnl_usdt": round(p, 2)} for s, p in best_symbols],
        "worst_hours":   [{"hour_utc": h, "pnl_usdt": round(p, 2)} for h, p in worst_hours],
    }


def compute_churn_analysis(trades: List[dict]) -> dict:
    """Identify churning patterns in this session's trades."""
    daily_rt: Dict[str, int] = defaultdict(int)
    daily_date: Dict[str, str] = {}

    for t in trades:
        sym  = t["symbol"]
        date = datetime.fromtimestamp(t["ts"], tz=timezone.utc).strftime("%Y-%m-%d")
        key  = f"{sym}:{date}"
        daily_rt[key] += 1

    churned = []
    for key, count in daily_rt.items():
        sym, date = key.rsplit(":", 1)
        rt = count // 2
        if rt >= 4:
            churned.append({"symbol": sym, "date": date, "round_trips": rt})

    churned.sort(key=lambda x: x["round_trips"], reverse=True)
    total_churn_trades = sum(c["round_trips"] * 2 for c in churned)
    churn_rate = total_churn_trades / len(trades) * 100 if trades else 0.0

    return {
        "churned_instances": churned[:10],
        "total_churn_trades": total_churn_trades,
        "churn_rate_pct": round(churn_rate, 1),
        "n_sessions_churned": len(churned),
    }


# ── Report Builder ─────────────────────────────────────────────────────────
async def build_report(r: aioredis.Redis, rbt: aioredis.Redis) -> dict:
    ts      = int(time.time())
    ts_human = datetime.now(timezone.utc).isoformat()

    # Get current system state
    equity_data = await r.hgetall(KEY_EQUITY)
    equity = float(equity_data.get("equity", 0.0)) if equity_data else 0.0
    peak   = float(equity_data.get("peak",   0.0)) if equity_data else 0.0
    dd_pct = (peak - equity) / peak * 100 if peak > 0 else 0.0

    l5 = await r.hgetall(KEY_L5_STATUS)
    l4 = await r.hgetall(KEY_L4_STATUS)
    l2a = await r.hgetall(KEY_L2_ACCURACY)
    l2g = await r.hgetall(KEY_L2_GATE)
    health = await r.hgetall(KEY_HEALTH)
    dag8 = await r.hgetall(KEY_DAG8)
    fng = await r.hgetall(KEY_FNG)

    # Compute trade stats
    stats     = compute_trade_stats(_session_trades)
    dd_decomp = compute_drawdown_decomposition(_session_trades, equity)
    churn     = compute_churn_analysis(_session_trades)

    # Per-symbol breakdown (top 10 by absolute PnL)
    sym_stats = {}
    for sym, trades in _by_symbol.items():
        sym_stats[sym] = compute_trade_stats(trades)
    top_losers  = sorted(sym_stats.items(), key=lambda x: x[1]["realized_pnl_usdt"])[:5]
    top_winners = sorted(sym_stats.items(), key=lambda x: x[1]["realized_pnl_usdt"], reverse=True)[:5]

    # Hourly distribution
    hourly = {}
    for h, trades in _by_hour.items():
        hourly[f"hour_{h:02d}"] = round(sum(t["pnl_usdt"] for t in trades), 2)

    report = {
        "report_ts":        ts,
        "report_ts_human":  ts_human,
        "session_start":    _session_start,
        "session_duration_h": round((ts - _session_start) / 3600, 1),

        # Portfolio state
        "equity":           equity,
        "peak":             peak,
        "dd_pct":           round(dd_pct, 2),
        "heat_pct":         float(l4.get("heat_pct", 0.0)) if l4 else 0.0,
        "system_mode":      health.get("system_mode", "UNKNOWN") if health else "UNKNOWN",
        "overall_health":   health.get("overall_health", "UNKNOWN") if health else "UNKNOWN",

        # Trade summary
        "trade_stats":       stats,
        "drawdown_decomp":   dd_decomp,
        "churn_analysis":    churn,

        # Per-symbol
        "top_losers":   [{s: d} for s, d in top_losers],
        "top_winners":  [{s: d} for s, d in top_winners],
        "hourly_pnl":   hourly,

        # Layer health
        "layer5_quality":    float(l5.get("avg_quality_score", 0.0)) if l5 else 0.0,
        "layer5_fill_rate":  float(l5.get("fill_rate_pct", 0.0)) if l5 else 0.0,
        "layer5_slip_bps":   float(l5.get("avg_slippage_bps", 0.0)) if l5 else 0.0,
        "layer5_churn_syms": l5.get("churning_symbols", "none") if l5 else "none",
        "layer2_gate":       l2g.get("gate", "UNKNOWN") if l2g else "UNKNOWN",
        "layer2_accuracy":   float(l2a.get("accuracy_pct", 0.0)) if l2a else 0.0,
        "dag8_criteria":     dag8.get("criteria_green", "0/5") if dag8 else "0/5",
        "dag8_recommend":    dag8.get("recommendation", "UNKNOWN") if dag8 else "UNKNOWN",
        "fng_value":         int(fng.get("value", 0)) if fng else 0,
        "fng_regime":        fng.get("regime", "unknown") if fng else "unknown",
    }
    return report


# ── Continuous Publisher ──────────────────────────────────────────────────
async def publish_loop(r: aioredis.Redis, rbt: aioredis.Redis):
    while True:
        try:
            report = await build_report(r, rbt)

            # Flatten top-level scalars to Redis hash
            flat = {}
            for k, v in report.items():
                if isinstance(v, (str, int, float)):
                    flat[k] = str(v)
                elif isinstance(v, dict):
                    flat[k] = json.dumps(v)

            await r.hset(KEY_SESSION, mapping=flat)

            log.info(
                f"[L6] session={report['session_duration_h']}h "
                f"trades={report['trade_stats']['n_trades']} "
                f"pnl={report['trade_stats']['realized_pnl_usdt']:+.2f}USDT "
                f"wr={report['trade_stats']['win_rate_pct']:.1f}% "
                f"churn={report['churn_analysis']['churn_rate_pct']:.1f}% "
                f"dd={report['dd_pct']:.2f}%"
            )

            # Daily report at ~00:05 UTC
            now = datetime.now(timezone.utc)
            if now.hour == 0 and 5 <= now.minute < 10:
                await write_daily_report(r, report)

        except Exception as e:
            log.error(f"Publish error: {e}", exc_info=True)
        await asyncio.sleep(PUBLISH_EVERY)


async def write_daily_report(r: aioredis.Redis, report: dict):
    """Write daily JSON report to disk and update KEY_DAILY."""
    try:
        REPORTS_DIR.mkdir(parents=True, exist_ok=True)
        date_str = datetime.now(timezone.utc).strftime("%Y-%m-%d")
        out_file = REPORTS_DIR / f"{date_str}.json"
        out_file.write_text(json.dumps(report, indent=2))

        flat = {k: str(v) for k, v in report.items() if not isinstance(v, (dict, list))}
        flat["report_date"] = date_str
        await r.hset(KEY_DAILY, mapping=flat)
        log.info(f"[L6] Daily report written: {out_file} ({out_file.stat().st_size} bytes)")
    except Exception as e:
        log.error(f"Daily report write error: {e}")


# ── Stream Consumer ───────────────────────────────────────────────────────
async def consume_new_trades(r: aioredis.Redis):
    try:
        await r.xgroup_create(STREAM_TRADES, CG_NAME, id="$", mkstream=True)
    except Exception:
        pass
    consumer_name = f"layer6_{os.getpid()}"
    while True:
        try:
            msgs = await r.xreadgroup(
                CG_NAME, consumer_name, {STREAM_TRADES: ">"}, count=20, block=10000
            )
            if msgs:
                for msg_id, fields in msgs[0][1]:
                    ingest_trade(msg_id, fields)
                    await r.xack(STREAM_TRADES, CG_NAME, msg_id)
        except Exception as e:
            log.error(f"Stream error: {e}")
            await asyncio.sleep(2)


# ── Main ──────────────────────────────────────────────────────────────────
async def main():
    log.info("[L6] Post-Trade Analytics Reporter starting")
    r   = aioredis.Redis(host=REDIS_HOST, port=6379, db=REDIS_DB,    decode_responses=True)
    rbt = aioredis.Redis(host=REDIS_HOST, port=6379, db=REDIS_BT_DB, decode_responses=True)
    await r.ping()
    log.info("[L6] Redis OK")

    await backfill_session(r)

    await asyncio.gather(
        publish_loop(r, rbt),
        consume_new_trades(r),
    )


if __name__ == "__main__":
    asyncio.run(main())
