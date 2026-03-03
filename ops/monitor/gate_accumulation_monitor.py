#!/usr/bin/env python3
"""
ops/monitor/gate_accumulation_monitor.py
=========================================
Real-time monitor for the Layer 2 Research Sandbox gate accumulation progress.

Gate opens when ALL three conditions are met:
  - n_matched_trades >= 30   (shadow EXIT signals matched with real trade.closed events)
  - accuracy          >= 55% (correct exit timing = profitable trades)
  - profit_factor     >= 1.0

Usage:
    python ops/monitor/gate_accumulation_monitor.py [--watch] [--interval 30]

    --watch      : Continuous polling (refreshes every --interval seconds)
    --interval N : Polling interval in seconds (default: 30)
    --once       : Print once and exit (default mode)
"""

import argparse
import json
import os
import sys
import time
from datetime import datetime, timezone

import redis

REDIS_HOST = os.getenv("REDIS_HOST", "localhost")
REDIS_PORT = int(os.getenv("REDIS_PORT", "6379"))

# Redis keys
GATE_KEY         = "quantum:sandbox:gate:latest"
ACCURACY_KEY     = "quantum:sandbox:accuracy:latest"
SHADOW_STATUS    = "quantum:shadow:status"
SHADOW_PORTFOLIO = "quantum:shadow:portfolio:latest"
OVERRIDE_KEY     = "quantum:sandbox:gate:override"

# Streams
SHADOW_STREAM    = "quantum:stream:harvest.v2.shadow"
TRADE_CLOSE_STR  = "quantum:stream:trade.closed"
TRADE_INTENT_STR = "quantum:stream:trade.intent"

# Thresholds
N_MIN_TRADES = 30
MIN_ACCURACY = 55.0
MIN_PF       = 1.0


def _r(v) -> str:
    if isinstance(v, bytes):
        return v.decode()
    return str(v) if v is not None else ""


def _pct_bar(pct: float, width: int = 20) -> str:
    """ASCII progress bar scaled 0-100."""
    filled = int(min(pct, 100) / 100 * width)
    return f"[{'█' * filled}{'░' * (width - filled)}] {pct:.1f}%"


def _count_bar(n: int, target: int, width: int = 20) -> str:
    pct = min(n / target, 1.0) * 100
    filled = int(pct / 100 * width)
    return f"[{'█' * filled}{'░' * (width - filled)}] {n}/{target}"


def fetch_gate_state(r: redis.Redis) -> dict:
    raw = r.hgetall(GATE_KEY)
    return {_r(k): _r(v) for k, v in raw.items()}


def fetch_shadow_portfolio(r: redis.Redis) -> dict:
    raw = r.hgetall(SHADOW_PORTFOLIO)
    return {_r(k): _r(v) for k, v in raw.items()}


def fetch_shadow_status(r: redis.Redis) -> dict:
    raw = r.hgetall(SHADOW_STATUS)
    return {_r(k): _r(v) for k, v in raw.items()}


def fetch_stream_stats(r: redis.Redis) -> dict:
    stats = {}
    for key, label in [
        (SHADOW_STREAM,    "harvest.v2.shadow"),
        (TRADE_CLOSE_STR,  "trade.closed"),
        (TRADE_INTENT_STR, "trade.intent"),
    ]:
        try:
            stats[label] = r.xlen(key)
        except Exception:
            stats[label] = -1
    return stats


def fetch_recent_closed_trades(r: redis.Redis, n: int = 5) -> list:
    """Get last N messages from trade.closed stream."""
    try:
        msgs = r.xrevrange(TRADE_CLOSE_STR, "+", "-", count=n)
        trades = []
        for msg_id, fields in msgs:
            d = {_r(k): _r(v) for k, v in fields.items()}
            trades.append(d)
        return trades
    except Exception:
        return []


def fetch_intent_bridge_lag(r: redis.Redis) -> dict:
    """Check intent_bridge consumer group lag on trade.intent."""
    try:
        groups = r.xinfo_groups(TRADE_INTENT_STR)
        for g in groups:
            name = _r(g.get(b"name", g.get("name", "")))
            if "intent_bridge" in name:
                return {
                    "consumers": int(g.get(b"consumers", g.get("consumers", 0))),
                    "pending": int(g.get(b"pending", g.get("pending", 0))),
                    "lag": int(g.get(b"lag", g.get("lag", 0)) or 0),
                }
    except Exception:
        pass
    return {}


def print_report(r: redis.Redis, verbose: bool = True):
    gate     = fetch_gate_state(r)
    shadow   = fetch_shadow_status(r)
    portfolio = fetch_shadow_portfolio(r)
    streams  = fetch_stream_stats(r)
    ib_lag   = fetch_intent_bridge_lag(r)
    override = _r(r.get(OVERRIDE_KEY) or b"")

    n_trades  = int(gate.get("n_matched_trades", 0) or 0)
    accuracy  = float(gate.get("accuracy_pct", 0) or 0)
    pf        = gate.get("profit_factor", "0")
    pf_val    = float(pf) if pf not in ("inf", "nan", "") else float("inf")
    paper_pnl = float(gate.get("paper_pnl_usd", 0) or 0)
    gate_status = gate.get("gate_status", "UNKNOWN")
    exit_signals = int(gate.get("exit_signals", 0) or 0)

    now = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")
    print(f"\n{'='*62}")
    print(f"  GATE ACCUMULATION MONITOR — {now}")
    print(f"{'='*62}")

    # ── Gate status badge ────────────────────────────────────────
    badge = "✅ OPEN" if "OPEN" in gate_status else "🔒 CLOSED"
    if override == "APPROVED":
        badge = "⚡ OPEN (MANUAL OVERRIDE)"
    print(f"\n  Gate:    {badge}")
    print(f"  Status:  {gate_status}")
    if override:
        print(f"  Override:{override}")
    print()

    # ── Condition checks ─────────────────────────────────────────
    trades_ok   = n_trades >= N_MIN_TRADES
    accuracy_ok = accuracy >= MIN_ACCURACY
    pf_ok       = pf_val >= MIN_PF or pf == "inf"

    print("  GATE CONDITIONS")
    print(f"  {'✅' if trades_ok else '❌'}  Matched trades:  {_count_bar(n_trades, N_MIN_TRADES)}")
    print(f"  {'✅' if accuracy_ok else '❌'}  Exit accuracy:   {_pct_bar(accuracy)} (need ≥{MIN_ACCURACY:.0f}%)")
    print(f"  {'✅' if pf_ok else '❌'}  Profit factor:   {pf_val:.3f}  (need ≥{MIN_PF:.1f})")
    print(f"  📄  Paper PnL:      {paper_pnl:+.2f} USDT")
    print(f"  📊  Exit signals:   {exit_signals} seen (signals from harvest.v2.shadow)")
    print()

    # ── Shadow portfolio (shadow_mode_controller) ─────────────────
    sh_n       = portfolio.get("n_trades", "?")
    sh_acc     = portfolio.get("rolling_accuracy", "?")
    sh_pf      = portfolio.get("profit_factor", "?")
    sh_eq      = portfolio.get("equity", "?")
    sh_phase   = shadow.get("phase", "?")
    sh_status  = shadow.get("status", "?")

    print("  SHADOW PORTFOLIO (shadow_mode_controller)")
    print(f"  Phase:   {sh_phase} | Status: {sh_status}")
    print(f"  Trades:  {sh_n} closed | Accuracy: {sh_acc}% | PF: {sh_pf}")
    print(f"  Equity:  {sh_eq} USDT")
    print()

    # ── Stream lengths ────────────────────────────────────────────
    print("  STREAM ACTIVITY")
    for label, count in streams.items():
        print(f"  {label:<25} {count:>8} msgs")

    ib = ib_lag
    if ib:
        print(f"\n  intent_bridge lag:       {ib.get('lag', '?')} msgs pending   ({ib.get('consumers')} consumers)")
    print()

    # ── Recent trade.closed events ───────────────────────────────
    if verbose:
        recent = fetch_recent_closed_trades(r, n=5)
        if recent:
            print("  LAST 5 TRADE.CLOSED EVENTS")
            for t in recent:
                sym  = t.get("symbol", t.get("Symbol", "?"))
                side = t.get("side", "?")
                pnl  = t.get("pnl_usd", t.get("pnl_usdt", t.get("realized_pnl", t.get("pnl", "?"))))
                r    = t.get("R_net", "")
                ts   = t.get("timestamp", t.get("close_ts", ""))
                print(f"  {sym:<12} {side:<6} pnl={pnl} R={r}  ts={ts[:19] if ts else '?'}")
        else:
            print("  LAST 5 TRADE.CLOSED EVENTS: (none yet)")
        print()

    # ── Projection ───────────────────────────────────────────────
    remaining = max(0, N_MIN_TRADES - n_trades)
    if remaining > 0:
        print(f"  ⏳ Remaining: {remaining} more matched trades needed")
        print(f"     Rates depend on position turnover. Each testnet position")
        print(f"     close generates one trade.closed event → one possible match.")
    else:
        if accuracy_ok and pf_ok:
            print("  🚀 ALL CONDITIONS MET — gate should be OPEN!")
        else:
            fails = []
            if not accuracy_ok:
                fails.append(f"accuracy {accuracy:.1f}% < {MIN_ACCURACY:.0f}%")
            if not pf_ok:
                fails.append(f"profit_factor {pf_val:.3f} < {MIN_PF:.1f}")
            print(f"  ⚠️  Trades enough but: {', '.join(fails)}")
            print(f"     To manually override: redis-cli SET {OVERRIDE_KEY} APPROVED")
    print(f"{'='*62}\n")


def main():
    parser = argparse.ArgumentParser(description="Gate accumulation monitor")
    parser.add_argument("--watch", action="store_true", help="Continuous polling")
    parser.add_argument("--interval", type=int, default=30, help="Seconds between refreshes")
    parser.add_argument("--quiet", action="store_true", help="Skip recent trade details")
    args = parser.parse_args()

    r = redis.Redis(host=REDIS_HOST, port=REDIS_PORT, db=0)
    try:
        r.ping()
    except redis.ConnectionError as e:
        print(f"❌ Redis connection failed: {e}", file=sys.stderr)
        sys.exit(1)

    if args.watch:
        print(f"Watching gate every {args.interval}s — Ctrl-C to stop")
        while True:
            print_report(r, verbose=not args.quiet)
            try:
                time.sleep(args.interval)
            except KeyboardInterrupt:
                print("\nStopped.")
                break
    else:
        print_report(r, verbose=not args.quiet)


if __name__ == "__main__":
    main()
