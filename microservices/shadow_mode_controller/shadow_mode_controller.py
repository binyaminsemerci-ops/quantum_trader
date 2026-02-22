#!/usr/bin/env python3
"""
Shadow Mode Controller — Phase 1 Execution Logic
==================================================
Activates ONLY when quantum:dag8:current_phase = 1 (SHADOW_ONLY).

Reads shadow harvest signals from quantum:stream:harvest.v2.shadow
and simulates portfolio entries/exits with ZERO real exposure.
Produces a paper-equity curve in Redis for DAG 8 C2 gate.

Gate requirements fed to Layer 2:
  - Minimum 30 closed shadow trades
  - Accuracy > 55% over rolling 30 trades
  - Profit factor > 1.1

This controller NEVER writes to:
  - quantum:stream:apply.plan
  - quantum:position:*
  - quantum:equity:current
  - Any live order/position key

It DOES write to:
  - quantum:shadow:portfolio:latest      — portfolio-level stats
  - quantum:shadow:position:<SYM>        — simulated open positions
  - quantum:shadow:equity:series         — ZSET of (ts, equity) for chart
  - quantum:sandbox:accuracy:latest      — picked up by Layer 2 gate + DAG 8 C2
  - quantum:shadow:trades:closed         — LPUSH capped 500 — for Layer 6 reports
  - quantum:shadow:status                — human-readable status

When phase ≠ 1, the loop idles and logs "idle, phase=X".

Operator commands:
  # Check status
  redis-cli hgetall quantum:shadow:status
  redis-cli hgetall quantum:shadow:portfolio:latest
  # Reset paper equity (e.g. new test run)
  redis-cli del quantum:shadow:portfolio:latest
  redis-cli del quantum:shadow:equity:series
  # Force phase transition test
  redis-cli set quantum:dag8:current_phase 1
"""

import asyncio
import json
import logging
import os
import time
from collections import defaultdict, deque
from datetime import datetime, timezone
from typing import Dict, Optional

import redis.asyncio as aioredis

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s shadow %(message)s",
)
log = logging.getLogger("shadow_mode")

# ── Config ────────────────────────────────────────────────────────────────
REDIS_HOST       = os.getenv("REDIS_HOST", "localhost")
REDIS_DB         = 0

PAPER_EQUITY_START = float(os.getenv("PAPER_EQUITY_START", "10000.0"))
MAX_POSITION_PCT   = float(os.getenv("MAX_POSITION_PCT",   "0.05"))   # 5% per symbol
MAX_HEAT_PCT       = float(os.getenv("MAX_HEAT_PCT",       "0.40"))   # 40% total heat

# Gate thresholds (fed to Layer 2 + DAG 8 C2)
MIN_SHADOW_TRADES  = int(os.getenv("MIN_SHADOW_TRADES",    "30"))
MIN_ACCURACY       = float(os.getenv("MIN_ACCURACY",       "55.0"))   # %
MIN_PROFIT_FACTOR  = float(os.getenv("MIN_PROFIT_FACTOR",  "1.1"))

# Redis keyspace
KEY_PHASE         = "quantum:dag8:current_phase"
KEY_SHADOW_PORT   = "quantum:shadow:portfolio:latest"
KEY_SHADOW_STATUS = "quantum:shadow:status"
KEY_SHADOW_EQUITY = "quantum:shadow:equity:series"    # ZSET (score=ts, val=equity)
KEY_SHADOW_TRADES = "quantum:shadow:trades:closed"    # LPUSH list
KEY_L2_ACCURACY   = "quantum:sandbox:accuracy:latest" # read by DAG 8 C2
KEY_L2_GATE       = "quantum:sandbox:gate:latest"

STREAM_SHADOW     = "quantum:stream:harvest.v2.shadow"
CG_NAME           = "shadow_controller"

IDLE_SLEEP        = 30     # seconds between phase checks when idle
PUBLISH_EVERY     = 60     # seconds between portfolio snapshots


# ── Shadow Portfolio ──────────────────────────────────────────────────────
class ShadowPortfolio:
    def __init__(self, equity_start: float):
        self.equity          = equity_start
        self.peak            = equity_start
        self.positions:       Dict[str, dict] = {}
        self.closed_trades:   list            = []
        self._win_window:     deque           = deque(maxlen=50)

    @property
    def n_trades(self) -> int:
        return len(self.closed_trades)

    @property
    def rolling_n(self) -> int:
        return len(self._win_window)

    @property
    def rolling_accuracy(self) -> float:
        if not self._win_window:
            return 0.0
        return sum(self._win_window) / len(self._win_window) * 100

    @property
    def profit_factor(self) -> float:
        wins   = sum(t["pnl_usdt"] for t in self.closed_trades if t["pnl_usdt"] > 0)
        losses = sum(abs(t["pnl_usdt"]) for t in self.closed_trades if t["pnl_usdt"] <= 0)
        return round(wins / losses, 3) if losses > 0 else float(wins > 0)

    @property
    def heat_pct(self) -> float:
        total_notional = sum(p["notional"] for p in self.positions.values())
        return total_notional / self.equity if self.equity > 0 else 0.0

    @property
    def dd_pct(self) -> float:
        return (self.peak - self.equity) / self.peak * 100 if self.peak > 0 else 0.0

    @property
    def gate_open(self) -> bool:
        return (
            self.n_trades >= MIN_SHADOW_TRADES and
            self.rolling_accuracy >= MIN_ACCURACY and
            self.profit_factor >= MIN_PROFIT_FACTOR
        )

    def gate_status(self) -> dict:
        return {
            "gate":           "OPEN" if self.gate_open else "CLOSED",
            "n_trades":        self.n_trades,
            "rolling_accuracy": round(self.rolling_accuracy, 1),
            "profit_factor":   self.profit_factor,
            "min_trades_req":  MIN_SHADOW_TRADES,
            "min_accuracy_req": MIN_ACCURACY,
            "min_pf_req":      MIN_PROFIT_FACTOR,
            "trades_needed":   max(0, MIN_SHADOW_TRADES - self.n_trades),
        }

    def open_position(self, sym: str, side: str, entry: float, size: float):
        notional = entry * size
        budget   = self.equity * MAX_POSITION_PCT
        if notional > budget:
            size     = budget / entry
            notional = budget

        if self.heat_pct + (notional / self.equity) > MAX_HEAT_PCT:
            log.debug(f"[SHADOW] skip {sym} — heat cap {self.heat_pct:.1%}")
            return False

        if sym in self.positions:
            log.debug(f"[SHADOW] skip {sym} — already open")
            return False

        self.positions[sym] = {
            "symbol":  sym,
            "side":    side,
            "entry":   entry,
            "size":    size,
            "notional": notional,
            "open_ts": int(time.time()),
        }
        log.info(f"[SHADOW] OPEN {sym} {side} entry={entry:.4f} notional={notional:.2f} USDT")
        return True

    def close_position(self, sym: str, exit_price: float, reason: str = "") -> Optional[dict]:
        if sym not in self.positions:
            return None
        pos  = self.positions.pop(sym)
        side = pos["side"]
        pnl_pct  = (exit_price - pos["entry"]) / pos["entry"] * 100
        if side.upper() == "SHORT":
            pnl_pct = -pnl_pct
        pnl_usdt = pnl_pct / 100 * pos["notional"]

        self.equity += pnl_usdt
        if self.equity > self.peak:
            self.peak = self.equity

        is_win = pnl_usdt > 0
        self._win_window.append(1 if is_win else 0)

        trade = {
            "symbol":   sym,
            "side":     side,
            "entry":    pos["entry"],
            "exit":     exit_price,
            "size":     pos["size"],
            "notional": pos["notional"],
            "pnl_pct":  round(pnl_pct, 4),
            "pnl_usdt": round(pnl_usdt, 4),
            "reason":   reason,
            "open_ts":  pos["open_ts"],
            "close_ts": int(time.time()),
            "duration_s": int(time.time()) - pos["open_ts"],
        }
        self.closed_trades.append(trade)
        log.info(
            f"[SHADOW] CLOSE {sym} exit={exit_price:.4f} "
            f"pnl={pnl_usdt:+.2f}USDT ({pnl_pct:+.3f}%) "
            f"acc={self.rolling_accuracy:.1f}%"
        )
        return trade


# ── Signal Processor ──────────────────────────────────────────────────────
async def process_signal(
    portfolio: ShadowPortfolio,
    fields: dict,
    r: aioredis.Redis,
):
    """Process one shadow harvest signal."""
    signal_type = fields.get("type", fields.get("signal_type", ""))
    sym         = fields.get("symbol", fields.get("Symbol", ""))
    price       = float(fields.get("price", fields.get("close", 0.0)))
    side        = fields.get("side", "LONG")
    size        = float(fields.get("size",     0.01))
    reason      = fields.get("reason", fields.get("signal", ""))

    if not sym or price <= 0:
        return

    if signal_type.upper() in ("OPEN", "ENTRY", "BUY", "LONG", "SHORT"):
        portfolio.open_position(sym, side, price, size)

    elif signal_type.upper() in ("CLOSE", "EXIT", "SELL"):
        trade = portfolio.close_position(sym, price, reason)
        if trade:
            # Publish to shadow trades list (capped 500)
            await r.lpush(KEY_SHADOW_TRADES, json.dumps(trade))
            await r.ltrim(KEY_SHADOW_TRADES, 0, 499)

    # Update equity series
    await r.zadd(KEY_SHADOW_EQUITY, {str(portfolio.equity): int(time.time())})
    await r.zremrangebyrank(KEY_SHADOW_EQUITY, 0, -2001)  # keep 2000 points


# ── Portfolio Publisher ───────────────────────────────────────────────────
async def publish_portfolio(portfolio: ShadowPortfolio, r: aioredis.Redis):
    gate = portfolio.gate_status()
    ts   = int(time.time())

    port_data = {
        "ts":               ts,
        "equity":           round(portfolio.equity, 4),
        "peak":             round(portfolio.peak, 4),
        "dd_pct":           round(portfolio.dd_pct, 2),
        "heat_pct":         round(portfolio.heat_pct * 100, 1),
        "n_open":           len(portfolio.positions),
        "n_closed":         portfolio.n_trades,
        "rolling_accuracy": round(portfolio.rolling_accuracy, 1),
        "profit_factor":    portfolio.profit_factor,
        "gate_status":      gate["gate"],
        "return_pct":       round((portfolio.equity - PAPER_EQUITY_START) / PAPER_EQUITY_START * 100, 2),
        "open_symbols":     ",".join(portfolio.positions.keys()) or "none",
    }

    await r.hset(KEY_SHADOW_PORT, mapping={k: str(v) for k, v in port_data.items()})

    # Also write to Layer 2 accuracy key (read by DAG 8 C2!)
    accuracy_data = {
        "accuracy_pct":      gate["rolling_accuracy"],
        "n_trades":           gate["n_trades"],
        "profit_factor":      gate["profit_factor"],
        "gate":               gate["gate"],
        "min_trades_needed":  gate["trades_needed"],
        "ts":                 ts,
    }
    await r.hset(KEY_L2_ACCURACY, mapping={k: str(v) for k, v in accuracy_data.items()})

    # Gate key
    await r.hset(KEY_L2_GATE, mapping={
        "gate":    gate["gate"],
        "reason":  (
            f"n={gate['n_trades']}/{MIN_SHADOW_TRADES} "
            f"acc={gate['rolling_accuracy']:.1f}% "
            f"pf={gate['profit_factor']:.2f}"
        ),
        "ts": str(ts),
    })


# ── Main Loop ─────────────────────────────────────────────────────────────
async def run_shadow_loop(r: aioredis.Redis):
    portfolio     = ShadowPortfolio(PAPER_EQUITY_START)
    last_publish  = 0.0
    consumer_name = f"shadow_{os.getpid()}"

    # Create consumer group if needed
    try:
        await r.xgroup_create(STREAM_SHADOW, CG_NAME, id="$", mkstream=True)
        log.info(f"[SHADOW] Consumer group '{CG_NAME}' created (id=$)")
    except Exception:
        pass  # already exists

    log.info(f"[SHADOW] Paper equity start: {PAPER_EQUITY_START:.2f} USDT")

    while True:
        # Check phase gate
        phase_raw = await r.get(KEY_PHASE)
        phase_val = int(phase_raw) if phase_raw is not None else 0

        if phase_val != 1:
            await r.hset(KEY_SHADOW_STATUS, mapping={
                "status": "IDLE",
                "phase":  str(phase_val),
                "reason": f"waiting for phase=1, currently phase={phase_val}",
                "ts":     str(int(time.time())),
            })
            log.debug(f"[SHADOW] idle, phase={phase_val}")
            await asyncio.sleep(IDLE_SLEEP)
            continue

        # Phase 1 — active shadow trading
        await r.hset(KEY_SHADOW_STATUS, mapping={
            "status": "ACTIVE",
            "phase":  "1",
            "reason": "SHADOW_ONLY mode — simulating signals",
            "ts":     str(int(time.time())),
        })

        # Consume shadow signals
        try:
            msgs = await r.xreadgroup(
                CG_NAME, consumer_name,
                {STREAM_SHADOW: ">"}, count=20, block=5000
            )
            if msgs:
                for msg_id, fields in msgs[0][1]:
                    await process_signal(portfolio, fields, r)
                    await r.xack(STREAM_SHADOW, CG_NAME, msg_id)
        except Exception as e:
            if "NOGROUP" in str(e):
                try:
                    await r.xgroup_create(STREAM_SHADOW, CG_NAME, id="$", mkstream=True)
                except Exception:
                    pass
            else:
                log.error(f"Stream read error: {e}")
            await asyncio.sleep(2)
            continue

        # Periodic port publish
        now = time.time()
        if now - last_publish >= PUBLISH_EVERY:
            await publish_portfolio(portfolio, r)
            gate = portfolio.gate_status()
            log.info(
                f"[SHADOW] phase={phase_val} "
                f"equity={portfolio.equity:.2f} "
                f"dd={portfolio.dd_pct:.2f}% "
                f"trades={portfolio.n_trades} "
                f"acc={portfolio.rolling_accuracy:.1f}% "
                f"pf={portfolio.profit_factor:.2f} "
                f"gate={gate['gate']} "
                f"({gate['trades_needed']} trades to open)"
            )
            last_publish = now


async def main():
    log.info("[SHADOW] Phase 1 controller starting")
    r = aioredis.Redis(host=REDIS_HOST, port=6379, db=REDIS_DB, decode_responses=True)
    await r.ping()
    log.info("[SHADOW] Redis OK")
    await run_shadow_loop(r)


if __name__ == "__main__":
    asyncio.run(main())
