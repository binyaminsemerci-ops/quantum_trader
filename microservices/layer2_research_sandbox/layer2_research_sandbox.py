#!/usr/bin/env python3
"""
layer2_research_sandbox.py — Research Isolation & Signal Validation Gate

Layer 2: Research & Modelling — Isolated from live trading

This service does THREE things independently of the live execution path:

A) SHADOW SIGNAL ARCHIVE
   Reads quantum:stream:harvest.v2.shadow continuously.
   Persists all shadow signals to /opt/quantum/data/research/shadow_signals.parquet
   Never written by any live service — research-only namespace.

B) SIGNAL ACCURACY TRACKER
   Matches shadow EXIT/PARTIAL signals to actual trade outcomes
   (quantum:stream:trade.closed). Computes per-model accuracy:
     - Directional accuracy (correct exit timing %)
     - Average R captured vs shadow R at signal
     - Profit factor (gross wins / gross losses if executed)
   Publishes live metrics to quantum:sandbox:accuracy:latest

C) RESEARCH GATE
   Any model transition (shadow → live) is blocked unless:
     - N_MIN_SIGNALS shadow EXIT signals have been observed (default: 30)
     - Directional accuracy > ACCURACY_THRESHOLD (default: 55%)
     - Paper portfolio PnL > 0 (paper-trades all shadow EXIT signals)
   Gate status published to quantum:sandbox:gate:latest
   Manual override: redis-cli SET quantum:sandbox:gate:override APPROVED

KEY ISOLATION CONTRACT:
  ✅ Reads:  quantum:stream:harvest.v2.shadow, quantum:stream:trade.closed
  ✅ Reads:  quantum:layer1:data_sink:latest (health check only)
  ❌ NEVER writes to: quantum:stream:apply.plan (live execution)
  ❌ NEVER writes to: quantum:system:mode or any live control key
  ✅ Writes only to: quantum:sandbox:* and /opt/quantum/data/research/

Promotion workflow:
  1. View gate status: redis-cli HGETALL quantum:sandbox:gate:latest
  2. View accuracy:    redis-cli HGETALL quantum:sandbox:accuracy:latest
  3. Approve manually: redis-cli SET quantum:sandbox:gate:override APPROVED
  4. Activate live:    redis-cli HSET quantum:config:harvest_v2 stream_live quantum:stream:apply.plan
"""

import json
import logging
import os
import signal
import sys
import time
from collections import defaultdict
from datetime import datetime, timezone

import pandas as pd
import redis as redis_lib

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s sandbox %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger("layer2")

REDIS_HOST  = os.getenv("REDIS_HOST", "localhost")
REDIS_PORT  = int(os.getenv("REDIS_PORT", "6379"))
DATA_ROOT   = os.getenv("QUANTUM_DATA_ROOT", "/opt/quantum/data")
RESEARCH_DIR = os.path.join(DATA_ROOT, "research")

# Gate thresholds — override via env
N_MIN_SIGNALS       = int(os.getenv("GATE_MIN_SIGNALS", "30"))
ACCURACY_THRESHOLD  = float(os.getenv("GATE_ACCURACY_THRESHOLD", "0.55"))
MATCH_WINDOW_SEC    = int(os.getenv("SIGNAL_MATCH_WINDOW_SEC", "3600"))  # 1h

# Redis
SHADOW_STREAM     = "quantum:stream:harvest.v2.shadow"
TRADE_CLOSE_STREAM = "quantum:stream:trade.closed"
CONSUMER_GROUP    = "layer2_research_sandbox"
CONSUMER_ID       = "sandbox_worker_1"

ACCURACY_KEY      = "quantum:sandbox:accuracy:latest"
GATE_KEY          = "quantum:sandbox:gate:latest"
PAPER_PNL_KEY     = "quantum:sandbox:paper_pnl:latest"
OVERRIDE_KEY      = "quantum:sandbox:gate:override"

# ── Shutdown ──────────────────────────────────────────────────────────────
_RUNNING = True

def _handle_signal(sig, frame):
    global _RUNNING
    _RUNNING = False
    logger.info("Signal %s — stopping", sig)

signal.signal(signal.SIGTERM, _handle_signal)
signal.signal(signal.SIGINT, _handle_signal)

# ── Helpers ───────────────────────────────────────────────────────────────

def _decode(v) -> str:
    if isinstance(v, bytes):
        return v.decode()
    return str(v) if v is not None else ""


def _safe_float(v, d=0.0) -> float:
    try:
        return float(v)
    except (TypeError, ValueError):
        return d


def _ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)


def _append_parquet(path: str, df_new: pd.DataFrame, dedup_col: str = "signal_id"):
    if os.path.isfile(path):
        try:
            df_existing = pd.read_parquet(path)
            df_combined = pd.concat([df_existing, df_new], ignore_index=True)
            if dedup_col in df_combined.columns:
                df_combined = df_combined.drop_duplicates(subset=[dedup_col], keep="last")
            df_combined.to_parquet(path, index=False, compression="snappy")
        except Exception as e:
            logger.warning("Parquet append failed, overwriting: %s", e)
            df_new.to_parquet(path, index=False, compression="snappy")
    else:
        df_new.to_parquet(path, index=False, compression="snappy")


# ── Paper Portfolio ───────────────────────────────────────────────────────

class PaperPortfolio:
    """
    Simulates what P&L would have been if we executed all shadow EXIT signals.
    Does NOT send any orders. Pure accounting.
    """

    def __init__(self):
        # symbol → {signal_ts, entry_price, side, initial_risk, R_net_at_signal}
        self._pending: dict[str, dict] = {}
        self.closed_trades: list[dict] = []
        self.total_pnl_usd  = 0.0
        self.win_count  = 0
        self.loss_count = 0

    def on_exit_signal(self, symbol: str, side: str, R_net: float,
                       unrealized_pnl: float, initial_risk: float, ts: float):
        """Record a shadow EXIT signal as a pending paper trade."""
        self._pending[symbol] = {
            "symbol":       symbol,
            "side":         side,
            "R_net_signal": R_net,
            "upnl_at_sig":  unrealized_pnl,
            "initial_risk": initial_risk,
            "signal_ts":    ts,
        }

    def on_trade_closed(self, symbol: str, actual_R: float,
                        actual_pnl_usd: float, close_ts: float):
        """Match actual close to pending paper trade. Compute P&L delta."""
        pending = self._pending.pop(symbol, None)
        if not pending:
            return

        # Paper trade: we exited at shadow signal time
        # Actual trade: exited later at actual_R
        # If signal R_net > actual_R → our paper exit was better (avoided further loss)
        # If signal R_net < actual_R → our paper exit was too early (left gains on table)
        pnl_diff = (pending["R_net_signal"] - actual_R) * pending["initial_risk"]

        # Paper P&L = what we captured from unrealized at signal time
        paper_pnl = pending["upnl_at_sig"]
        self.total_pnl_usd += paper_pnl

        correct = pending["R_net_signal"] >= actual_R  # we exited before it got worse

        trade_record = {
            "symbol":           symbol,
            "side":             pending["side"],
            "R_net_signal":     pending["R_net_signal"],
            "R_net_actual":     actual_R,
            "paper_pnl_usd":    paper_pnl,
            "pnl_improvement":  pnl_diff,
            "correct_timing":   correct,
            "signal_ts":        pending["signal_ts"],
            "close_ts":         close_ts,
        }
        self.closed_trades.append(trade_record)

        if paper_pnl >= 0:
            self.win_count += 1
        else:
            self.loss_count += 1

        logger.info("[L2_PAPER] %s closed: signal_R=%.3f actual_R=%.3f paper_pnl=%.2f correct=%s",
                    symbol, pending["R_net_signal"], actual_R, paper_pnl, correct)

    def expire_stale(self, max_age_sec: int = 86400):
        """Remove pending signals older than max_age_sec (trade never closed)."""
        now = time.time()
        stale = [sym for sym, p in self._pending.items()
                 if now - p["signal_ts"] > max_age_sec]
        for sym in stale:
            self._pending.pop(sym, None)

    def profit_factor(self) -> float:
        gross_win  = sum(t["paper_pnl_usd"] for t in self.closed_trades if t["paper_pnl_usd"] > 0)
        gross_loss = abs(sum(t["paper_pnl_usd"] for t in self.closed_trades if t["paper_pnl_usd"] < 0))
        return gross_win / gross_loss if gross_loss > 0 else float("inf") if gross_win > 0 else 0.0

    def accuracy(self) -> float:
        if not self.closed_trades:
            return 0.0
        correct = sum(1 for t in self.closed_trades if t["correct_timing"])
        return correct / len(self.closed_trades)

    def save_to_parquet(self):
        if not self.closed_trades:
            return
        path = os.path.join(RESEARCH_DIR, "paper_trades.parquet")
        df = pd.DataFrame(self.closed_trades)
        _append_parquet(path, df, dedup_col=None)


# ── Signal Archive ────────────────────────────────────────────────────────

class SignalArchive:
    """Accumulates shadow signals, flushes to Parquet every N signals."""

    def __init__(self):
        self._buf: list[dict] = []
        self._total = 0
        self._by_decision: dict[str, int] = defaultdict(int)

    def add(self, fields: dict):
        self._buf.append(fields)
        self._total += 1
        self._by_decision[fields.get("decision", "?")] += 1
        if len(self._buf) >= 100:
            self.flush()

    def flush(self):
        if not self._buf:
            return
        path = os.path.join(RESEARCH_DIR, "shadow_signals.parquet")
        df = pd.DataFrame(self._buf)
        if "timestamp" not in df.columns:
            df["timestamp"] = time.time()
        _append_parquet(path, df, dedup_col=None)
        self._buf.clear()
        logger.debug("[L2] Flushed shadow signals to disk")

    def total(self) -> int:
        return self._total

    def exit_count(self) -> int:
        return self._by_decision.get("FULL_CLOSE", 0) + self._by_decision.get("EXIT", 0)

    def by_decision(self) -> dict:
        return dict(self._by_decision)


# ── Gate Evaluation ───────────────────────────────────────────────────────

def evaluate_gate(r: redis_lib.Redis, paper: PaperPortfolio,
                  archive: SignalArchive) -> dict:
    """
    Compute gate status. Returns dict with gate decision.
    Gate OPEN = model can be promoted to live.
    Gate CLOSED = needs more signals / higher accuracy.
    """
    n_trades     = len(paper.closed_trades)
    accuracy     = paper.accuracy()
    profit_factor = paper.profit_factor()
    total_pnl    = paper.total_pnl_usd
    exit_signals = archive.exit_count()

    # Check manual override
    override = _decode(r.get(OVERRIDE_KEY) or b"")

    checks = {
        "n_signals_ok":    exit_signals >= N_MIN_SIGNALS,
        "n_trades_ok":     n_trades >= max(10, N_MIN_SIGNALS // 3),
        "accuracy_ok":     accuracy >= ACCURACY_THRESHOLD,
        "profit_factor_ok": profit_factor >= 1.0,
        "pnl_ok":          total_pnl >= 0,
    }
    gate_auto = all(checks.values())

    if override == "APPROVED":
        gate_status = "OPEN_MANUAL_OVERRIDE"
    elif gate_auto:
        gate_status = "OPEN"
    else:
        fails = [k for k, v in checks.items() if not v]
        gate_status = f"CLOSED ({', '.join(fails)})"

    return {
        "gate_status":      gate_status,
        "exit_signals":     str(exit_signals),
        "n_matched_trades": str(n_trades),
        "accuracy_pct":     f"{accuracy * 100:.1f}",
        "profit_factor":    f"{profit_factor:.3f}" if profit_factor != float("inf") else "inf",
        "paper_pnl_usd":    f"{total_pnl:.2f}",
        "n_min_required":   str(N_MIN_SIGNALS),
        "accuracy_required": f"{ACCURACY_THRESHOLD * 100:.0f}%",
        "manual_override":  override or "none",
        "checks":           json.dumps(checks),
        "ts":               time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }


# ── Consumer setup ────────────────────────────────────────────────────────

def _ensure_consumer_groups(r: redis_lib.Redis):
    for stream in [SHADOW_STREAM, TRADE_CLOSE_STREAM]:
        try:
            r.xgroup_create(stream, CONSUMER_GROUP, id="$", mkstream=False)
        except redis_lib.ResponseError as e:
            if "BUSYGROUP" not in str(e):
                logger.warning("Group create error on %s: %s", stream, e)


# ── Main ──────────────────────────────────────────────────────────────────

def main():
    logger.info("[L2] Research Sandbox starting")
    logger.info("[L2] Gate thresholds: N_min=%d accuracy=%.0f%% profit_factor>=1.0",
                N_MIN_SIGNALS, ACCURACY_THRESHOLD * 100)

    _ensure_dir(RESEARCH_DIR)

    r = redis_lib.Redis(host=REDIS_HOST, port=REDIS_PORT, db=0)
    try:
        r.ping()
        logger.info("[L2] Redis OK")
    except redis_lib.ConnectionError as e:
        logger.error("[L2] Redis FAILED: %s", e)
        sys.exit(1)

    _ensure_consumer_groups(r)

    archive = SignalArchive()
    paper   = PaperPortfolio()

    last_publish_ts  = time.monotonic()
    last_save_ts     = time.monotonic()

    while _RUNNING:
        try:
            results = r.xreadgroup(
                groupname=CONSUMER_GROUP,
                consumername=CONSUMER_ID,
                streams={SHADOW_STREAM: ">", TRADE_CLOSE_STREAM: ">"},
                count=50,
                block=3000,
            )
        except Exception as e:
            logger.error("[L2] XREADGROUP error: %s", e)
            time.sleep(5)
            continue

        if results:
            ids_to_ack: dict[str, list] = {}

            for stream_bytes, messages in results:
                stream = _decode(stream_bytes)
                ids_to_ack[stream] = []

                for msg_id, fields in messages:
                    ids_to_ack[stream].append(msg_id)
                    flds = {_decode(k): _decode(v) for k, v in fields.items()}

                    if stream == SHADOW_STREAM:
                        # Archive all shadow signals
                        archive.add(flds)

                        # If it's an exit decision, record for paper portfolio
                        decision = flds.get("decision", "")
                        if decision in ("FULL_CLOSE", "EXIT", "PARTIAL_25", "PARTIAL_50"):
                            symbol  = flds.get("symbol", "")
                            side    = flds.get("side", "LONG")
                            R_net   = _safe_float(flds.get("R_net", 0))
                            upnl    = _safe_float(flds.get("unrealized_pnl", 0))
                            risk    = _safe_float(flds.get("initial_risk", 0))
                            ts      = _safe_float(flds.get("timestamp", time.time()))
                            if symbol:
                                paper.on_exit_signal(symbol, side, R_net, upnl, risk, ts)
                                logger.debug("[L2] Shadow EXIT recorded: %s R=%.3f decision=%s",
                                             symbol, R_net, decision)

                    elif stream == TRADE_CLOSE_STREAM:
                        symbol     = flds.get("symbol", "")
                        actual_R   = _safe_float(flds.get("R_net", 0))
                        actual_pnl = _safe_float(flds.get("pnl_usd", 0))
                        ts         = _safe_float(flds.get("timestamp", time.time()))
                        if symbol:
                            paper.on_trade_closed(symbol, actual_R, actual_pnl, ts)

            for stream, id_list in ids_to_ack.items():
                if id_list:
                    r.xack(stream, CONSUMER_GROUP, *id_list)

        # Expire stale pending paper trades (> 24h)
        paper.expire_stale(86400)

        # Publish metrics every 30s
        if time.monotonic() - last_publish_ts > 30:
            # Accuracy state
            accuracy_state = {
                "total_shadow_signals":  str(archive.total()),
                "exit_signals":          str(archive.exit_count()),
                "matched_trades":        str(len(paper.closed_trades)),
                "accuracy_pct":          f"{paper.accuracy() * 100:.1f}",
                "profit_factor":         f"{paper.profit_factor():.3f}" if paper.profit_factor() != float("inf") else "inf",
                "paper_pnl_usd":         f"{paper.total_pnl_usd:.2f}",
                "paper_wins":            str(paper.win_count),
                "paper_losses":          str(paper.loss_count),
                "by_decision":           json.dumps(archive.by_decision()),
                "ts":                    time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            }
            r.hset(ACCURACY_KEY, mapping=accuracy_state)

            # Gate state
            gate_state = evaluate_gate(r, paper, archive)
            r.hset(GATE_KEY, mapping=gate_state)

            logger.info("[L2_METRICS] signals=%d exit=%d matched=%d accuracy=%.1f%% paper_pnl=%.2f gate=%s",
                        archive.total(), archive.exit_count(), len(paper.closed_trades),
                        paper.accuracy() * 100, paper.total_pnl_usd,
                        gate_state["gate_status"])

            last_publish_ts = time.monotonic()

        # Save paper trades to Parquet every 5 min
        if time.monotonic() - last_save_ts > 300:
            archive.flush()
            paper.save_to_parquet()
            last_save_ts = time.monotonic()

    # Final save
    archive.flush()
    paper.save_to_parquet()
    logger.info("[L2] Research Sandbox stopped — signals=%d matched=%d",
                archive.total(), len(paper.closed_trades))


if __name__ == "__main__":
    main()
