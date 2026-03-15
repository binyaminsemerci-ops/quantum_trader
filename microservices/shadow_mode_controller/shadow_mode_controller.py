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
  - quantum:state:positions:*
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

# Layer 3 backtest C2 fallback — used when live shadow trades < MIN_SHADOW_TRADES
REDIS_DB_BT      = 1
BT_MIN_WR        = float(os.getenv("BT_MIN_WR",    "35.0"))   # % win rate
BT_MIN_PF        = float(os.getenv("BT_MIN_PF",    "1.1"))
BT_MIN_SHARPE    = float(os.getenv("BT_MIN_SHARPE", "2.0"))
BT_MIN_TRADES    = int(os.getenv("BT_MIN_TRADES",  "50"))     # min backtest n
BT_MIN_QUALIFY   = int(os.getenv("BT_MIN_QUALIFY", "2"))      # symbols needed

BT_SYMBOL_MAP: dict = {
    "SOLUSDT":  "auto_sol",
    "LINKUSDT": "auto_lin",
    "BTCUSDT":  "auto_btc",
    "ETHUSDT":  "auto_eth",
    "BNBUSDT":  "auto_bnb",
    "ADAUSDT":  "auto_ada",
    "AVAXUSDT": "auto_ava",
    "DOGEUSDT": "auto_dog",
    "DOTUSDT":  "auto_dot",
    "INJUSDT":  "auto_inj",
    "LTCUSDT":  "auto_ltc",
    "NEARUSDT": "auto_nea",
    "UNIUSDT":  "auto_uni",
    "XRPUSDT":  "auto_xrp",
}

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
_HARVEST_V2_CLOSES = {"FULL_CLOSE", "PARTIAL_75", "PARTIAL_50", "PARTIAL_25"}


async def process_signal(
    portfolio: ShadowPortfolio,
    fields: dict,
    r: aioredis.Redis,
):
    """Process one shadow harvest signal.

    Supports two formats:
    1. Harvest v2 (current): decision=FULL_CLOSE/HOLD, unrealized_pnl, R_net, initial_risk
    2. Legacy: signal_type=OPEN/CLOSE/SELL, price, size
    """
    decision    = fields.get("decision", "").upper()
    signal_type = fields.get("type", fields.get("signal_type", "")).upper()
    sym         = fields.get("symbol", fields.get("Symbol", ""))
    side        = fields.get("side", "LONG").upper()
    price       = float(fields.get("price", fields.get("close", 0.0)))

    if not sym:
        return

    # ── Harvest v2: decision-based signals ───────────────────────────────
    if decision in _HARVEST_V2_CLOSES:
        unrealized_pnl = float(fields.get("unrealized_pnl", 0.0))
        initial_risk   = float(fields.get("initial_risk",   10.0))
        r_net          = float(fields.get("R_net",           0.0))

        if sym in portfolio.positions and price > 0:
            # Shadow portfolio tracked this position — use normal close
            trade = portfolio.close_position(sym, price, decision)
        else:
            # Pre-existing or untracked position: record directly from P&L fields.
            # unrealized_pnl is the actual USDT result of the live position.
            is_win = unrealized_pnl > 0.0
            portfolio._win_window.append(1 if is_win else 0)
            portfolio.equity += unrealized_pnl
            if portfolio.equity > portfolio.peak:
                portfolio.peak = portfolio.equity
            trade = {
                "symbol":    sym,
                "side":      side,
                "entry":     0.0,
                "exit":      price,
                "size":      round(initial_risk / 10.0, 4),  # size approx (10 USDT/unit)
                "notional":  initial_risk,
                "pnl_usdt":  round(unrealized_pnl, 4),
                "pnl_pct":   round(r_net * 100.0, 3),   # R_net expressed as pct
                "reason":    decision,
                "source":    "harvest_v2_direct",
                "open_ts":   int(time.time()),
                "close_ts":  int(time.time()),
                "duration_s": 0,
            }
            portfolio.closed_trades.append(trade)
            log.info(
                f"[SHADOW] V2_CLOSE {sym} pnl={unrealized_pnl:+.2f}USDT "
                f"R={r_net:+.2f} win={is_win} "
                f"n={portfolio.n_trades} acc={portfolio.rolling_accuracy:.1f}%"
            )

        if trade:
            await r.lpush(KEY_SHADOW_TRADES, json.dumps(trade))
            await r.ltrim(KEY_SHADOW_TRADES, 0, 499)

        await r.zadd(KEY_SHADOW_EQUITY, {str(portfolio.equity): int(time.time())})
        await r.zremrangebyrank(KEY_SHADOW_EQUITY, 0, -2001)
        return  # handled

    # ── Legacy: signal_type-based signals ────────────────────────────────
    if not price or price <= 0:
        return

    size   = float(fields.get("size", 0.01))
    reason = fields.get("reason", fields.get("signal", ""))

    if signal_type in ("OPEN", "ENTRY", "BUY", "LONG", "SHORT"):
        portfolio.open_position(sym, side, price, size)

    elif signal_type in ("CLOSE", "EXIT", "SELL"):
        trade = portfolio.close_position(sym, price, reason)
        if trade:
            await r.lpush(KEY_SHADOW_TRADES, json.dumps(trade))
            await r.ltrim(KEY_SHADOW_TRADES, 0, 499)

    # Update equity series
    await r.zadd(KEY_SHADOW_EQUITY, {str(portfolio.equity): int(time.time())})
    await r.zremrangebyrank(KEY_SHADOW_EQUITY, 0, -2001)  # keep 2000 points



# ── Layer 3 Backtest C2 Evaluator ─────────────────────────────────────────
async def evaluate_c2_from_backtest(r_bt: aioredis.Redis) -> dict:
    """Read Layer 3 backtest results (db=1) and compute C2 gate status.

    A symbol qualifies when:
      - metrics_n_trades  >= BT_MIN_TRADES  (sufficient sample)
      - metrics_win_rate_pct >= BT_MIN_WR
      - metrics_profit_factor >= BT_MIN_PF
      - metrics_sharpe       >= BT_MIN_SHARPE

    Returns a dict compatible with the accuracy key format expected by DAG 8.
    """
    qualifying = []
    all_results = []

    for sym, code in BT_SYMBOL_MAP.items():
        key = f"quantum:backtest:results:{code}"
        try:
            d = await r_bt.hgetall(key)
        except Exception:
            continue
        if not d:
            continue

        n   = int(float(d.get("metrics_n_trades",  "0")))
        wr  = float(d.get("metrics_win_rate_pct",  "0"))
        pf  = float(d.get("metrics_profit_factor", "0"))
        sh  = float(d.get("metrics_sharpe",        "0"))

        passes = (n >= BT_MIN_TRADES and wr >= BT_MIN_WR and
                  pf >= BT_MIN_PF and sh >= BT_MIN_SHARPE)
        all_results.append({"sym": sym, "n": n, "wr": wr, "pf": pf, "sh": sh, "passes": passes})
        if passes:
            qualifying.append(sym)

    gate_open = len(qualifying) >= BT_MIN_QUALIFY
    top_wr    = max((r["wr"] for r in all_results), default=0.0)
    top_pf    = max((r["pf"] for r in all_results), default=0.0)
    top_sh    = max((r["sh"] for r in all_results), default=0.0)

    log.info(
        f"[C2/BT] {len(qualifying)}/{BT_MIN_QUALIFY} qualify "
        f"gate={'OPEN' if gate_open else 'CLOSED'} "
        f"symbols={qualifying}"
    )

    return {
        "gate":             "OPEN" if gate_open else "CLOSED",
        "source":           "layer3_backtest",
        "qualifying_syms":  ",".join(qualifying) or "none",
        "n_qualifying":     len(qualifying),
        "n_evaluated":      len(all_results),
        "best_wr":          round(top_wr, 2),
        "best_pf":          round(top_pf, 3),
        "best_sharpe":      round(top_sh, 3),
        "bt_min_wr":        BT_MIN_WR,
        "bt_min_pf":        BT_MIN_PF,
        "bt_min_sharpe":    BT_MIN_SHARPE,
        "accuracy_pct":     round(top_wr, 2),     # alias for DAG 8 reader compat
        "n_trades":         sum(r["n"] for r in all_results if r["passes"]),
        "profit_factor":    round(top_pf, 3),
        "ts":               int(time.time()),
    }


# ── Portfolio Publisher ───────────────────────────────────────────────────
async def publish_portfolio(portfolio: ShadowPortfolio, r: aioredis.Redis,
                             r_bt: Optional[aioredis.Redis] = None):
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

    # ── C2 gate: prefer Layer 3 backtest stats when live trades insufficient ──
    if portfolio.n_trades < MIN_SHADOW_TRADES and r_bt is not None:
        # Fallback to Layer 3 backtest quality evaluation
        bt = await evaluate_c2_from_backtest(r_bt)
        accuracy_data = {
            "accuracy_pct":     bt["accuracy_pct"],
            "n_trades":          bt["n_trades"],
            "profit_factor":     bt["profit_factor"],
            "gate":              bt["gate"],
            "source":            bt["source"],
            "qualifying_syms":   bt["qualifying_syms"],
            "n_qualifying":      bt["n_qualifying"],
            "best_sharpe":       bt["best_sharpe"],
            "ts":                ts,
        }
        gate_reason = (
            f"backtest: {bt['n_qualifying']}/{BT_MIN_QUALIFY} syms qualify "
            f"(wr>{BT_MIN_WR}% pf>{BT_MIN_PF} sh>{BT_MIN_SHARPE}) "
            f"syms={bt['qualifying_syms']}"
        )
        log.info(f"[C2] Using backtest fallback: gate={bt['gate']} {gate_reason}")
    else:
        # Enough live trades — use shadow portfolio accuracy
        accuracy_data = {
            "accuracy_pct":     gate["rolling_accuracy"],
            "n_trades":          gate["n_trades"],
            "profit_factor":     gate["profit_factor"],
            "gate":              gate["gate"],
            "source":            "live_shadow",
            "ts":                ts,
        }
        gate_reason = (
            f"live: n={gate['n_trades']}/{MIN_SHADOW_TRADES} "
            f"acc={gate['rolling_accuracy']:.1f}% "
            f"pf={gate['profit_factor']:.2f}"
        )

    await r.hset(KEY_L2_ACCURACY, mapping={k: str(v) for k, v in accuracy_data.items()})

    # Gate key
    await r.hset(KEY_L2_GATE, mapping={
        "gate":   accuracy_data["gate"],
        "reason": gate_reason,
        "ts":     str(ts),
    })


# ── Main Loop ─────────────────────────────────────────────────────────────
async def run_shadow_loop(r: aioredis.Redis, r_bt: Optional[aioredis.Redis] = None):
    portfolio     = ShadowPortfolio(PAPER_EQUITY_START)
    last_publish  = 0.0
    consumer_name = f"shadow_{os.getpid()}"

    # Create consumer group if needed
    try:
        await r.xgroup_create(STREAM_SHADOW, CG_NAME, id="0", mkstream=True)
        log.info(f"[SHADOW] Consumer group '{CG_NAME}' created (id=0) — will replay full history")
    except Exception:
        pass  # already exists

    log.info(f"[SHADOW] Paper equity start: {PAPER_EQUITY_START:.2f} USDT")

    while True:
        # Check phase gate
        phase_raw = await r.get(KEY_PHASE)
        phase_val = int(phase_raw) if phase_raw is not None else 0

        # Phase 0 = FREEZE: simulate to build C2 gate (no real execution anyway)
        # Phase 1 = SHADOW_ONLY: simulate as actual execution layer
        # Phase 2+ = PAPER/LIVE: real execution takes over — shadow idles
        if phase_val >= 2:
            await r.hset(KEY_SHADOW_STATUS, mapping={
                "status": "IDLE",
                "phase":  str(phase_val),
                "reason": f"real execution active at phase={phase_val} — shadow idle",
                "ts":     str(int(time.time())),
            })
            log.debug(f"[SHADOW] idle, phase={phase_val} (real execution active)")
            await asyncio.sleep(IDLE_SLEEP)
            continue

        # Phase 0 or 1 — process shadow signals
        status_label = "BUILDING_GATE" if phase_val == 0 else "ACTIVE_SHADOW"
        reason_label = (
            "FREEZE mode — accumulating shadow trades for C2 gate"
            if phase_val == 0 else
            "SHADOW_ONLY mode — shadow simulation is live execution"
        )
        await r.hset(KEY_SHADOW_STATUS, mapping={
            "status": status_label,
            "phase":  str(phase_val),
            "reason": reason_label,
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
                    await r.xgroup_create(STREAM_SHADOW, CG_NAME, id="0", mkstream=True)
                except Exception:
                    pass
            else:
                log.error(f"Stream read error: {e}")
            await asyncio.sleep(2)
            continue

        # Periodic port publish
        now = time.time()
        if now - last_publish >= PUBLISH_EVERY:
            await publish_portfolio(portfolio, r, r_bt)
            gate = portfolio.gate_status()
            log.info(
                f"[SHADOW] phase={phase_val}({'FREEZE→building' if phase_val==0 else 'SHADOW_ONLY'}) "
                f"equity={portfolio.equity:.2f} "
                f"dd={portfolio.dd_pct:.2f}% "
                f"trades={portfolio.n_trades}/{MIN_SHADOW_TRADES} "
                f"acc={portfolio.rolling_accuracy:.1f}% "
                f"pf={portfolio.profit_factor:.2f} "
                f"gate={gate['gate']} "
                f"(need {gate['trades_needed']} more trades)"
            )
            last_publish = now


async def main():
    log.info("[SHADOW] Phase 1 controller starting")
    r    = aioredis.Redis(host=REDIS_HOST, port=6379, db=REDIS_DB,    decode_responses=True)
    r_bt = aioredis.Redis(host=REDIS_HOST, port=6379, db=REDIS_DB_BT, decode_responses=True)
    await r.ping()
    await r_bt.ping()
    log.info("[SHADOW] Redis OK (db=0 live, db=1 backtest)")
    await run_shadow_loop(r, r_bt)


if __name__ == "__main__":
    asyncio.run(main())
