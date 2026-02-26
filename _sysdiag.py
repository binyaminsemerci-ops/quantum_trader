#!/usr/bin/env python3
"""
Systemdiagnose — finn root cause bak:
1. Feb 15-16: 759 trades, -$207 (katastrofal win rate 32-44%)
2. Feb 22-24: 0 trades (dead period)
3. Churning pattern (347 ADAUSDT trades)
4. Nåværende systemhelse
"""
import redis, json, subprocess
from datetime import datetime, timezone, timedelta
from collections import defaultdict

r = redis.Redis(host='localhost', port=6379, decode_responses=True)

print("=" * 65)
print("SYSTEMDIAGNOSE — ROOT CAUSE ANALYSE")
print("=" * 65)

# ─── 1. CHURN ANALYSE ────────────────────────────────────────
print("\n── 1. CHURNING ANALYSE ──")
all_entries = r.xrange("quantum:stream:trade.closed", min="-", max="+")
trades = []
for eid, fields in all_entries:
    ts = datetime.fromtimestamp(int(eid.split('-')[0])/1000, tz=timezone.utc)
    trades.append({
        "ts": ts,
        "symbol": fields.get("symbol", "?"),
        "side": fields.get("side", "?"),
        "pnl_usd": float(fields.get("pnl_usd", 0) or 0),
        "pnl_pct": float(fields.get("pnl_percent", 0) or 0),
        "r_net": float(fields.get("R_net", 0) or 0),
        "reason": fields.get("reason", ""),
        "hold_sec": 0,
    })

# Hold-tid per exit-reason
print("\n  Exit reason breakdown (alle tider):")
by_reason = defaultdict(lambda: {"count": 0, "pnl": 0.0, "wins": 0})
for t in trades:
    reason_key = t["reason"].split("(")[0].strip()[:50] if t["reason"] else "unknown"
    by_reason[reason_key]["count"] += 1
    by_reason[reason_key]["pnl"] += t["pnl_usd"]
    if t["pnl_usd"] > 0:
        by_reason[reason_key]["wins"] += 1

for reason, data in sorted(by_reason.items(), key=lambda x: x[1]["pnl"]):
    wr = data["wins"]/data["count"]*100 if data["count"] else 0
    print(f"  {data['count']:4}x  ${data['pnl']:+8.2f}  W={wr:4.0f}%  {reason}")

# ─── 2. FEB 15-16 KATASTROFE ─────────────────────────────────
print("\n── 2. FEB 15-16 KATASTROFE ANALYSE ──")
bad_days = [t for t in trades if t["ts"].date().strftime("%Y-%m-%d") in ["2026-02-15","2026-02-16"]]
print(f"\n  Feb 15-16 totalt: {len(bad_days)} trades")

# Tidsfordeling — når skjedde det?
by_hour = defaultdict(lambda: {"count": 0, "pnl": 0.0})
for t in bad_days:
    h = t["ts"].strftime("%Y-%m-%d %H:00")
    by_hour[h]["count"] += 1
    by_hour[h]["pnl"] += t["pnl_usd"]

print("  Tidsfordeling (per time):")
for h in sorted(by_hour.keys()):
    d = by_hour[h]
    bar = "█" * min(int(d["count"]/5), 30)
    print(f"  {h}  {d['count']:3}x  ${d['pnl']:+7.2f}  {bar}")

# Symbol churning feb 15-16
by_sym_bad = defaultdict(lambda: {"count": 0, "pnl": 0.0})
for t in bad_days:
    by_sym_bad[t["symbol"]]["count"] += 1
    by_sym_bad[t["symbol"]]["pnl"] += t["pnl_usd"]

print("\n  Symbol breakdown feb 15-16:")
for sym, d in sorted(by_sym_bad.items(), key=lambda x: x[1]["count"], reverse=True)[:15]:
    print(f"  {sym:15}  {d['count']:4}x  ${d['pnl']:+8.2f}")

# ─── 3. HOLD TID ANALYSE ─────────────────────────────────────
print("\n── 3. HOLD-TID — KJERNE PROBLEM ──")
# Beregn hold-tid fra consecutive trades på samme symbol
# Sorter per symbol
by_sym_trades = defaultdict(list)
for t in trades:
    by_sym_trades[t["symbol"]].append(t)

short_holds = []
for sym, sym_trades in by_sym_trades.items():
    sym_trades.sort(key=lambda x: x["ts"])
    for i in range(1, len(sym_trades)):
        diff = (sym_trades[i]["ts"] - sym_trades[i-1]["ts"]).total_seconds()
        if diff < 300:  # under 5 min mellom trades
            short_holds.append({
                "sym": sym,
                "sec": diff,
                "pnl": sym_trades[i]["pnl_usd"],
                "ts": sym_trades[i]["ts"]
            })

print(f"\n  Trades med < 5 min mellom exits: {len(short_holds)}")
print(f"  PnL fra disse: ${sum(t['pnl'] for t in short_holds):+.2f}")
print(f"  (Dette er churning — åpne+lukk på sekunder)")

# ─── 4. KONFIG VED TID FOR KATASTROFEN ───────────────────────
print("\n── 4. SYSTEMKONFIG — NÅTILSTAND ANALYSE ──")

# Sjekk intent-executor konfig
print("\n  intent-executor.env kjerneparametere:")
with open("/etc/quantum/intent-executor.env") as f:
    for line in f:
        line = line.strip()
        if line and not line.startswith("#") and any(k in line for k in [
            "MIN_NOTIONAL", "ALLOWLIST", "ALLOW_UPSIZE", "UPDATE_LEDGER",
            "SOURCE_ALLOWLIST", "MAX_POSITIONS", "COOLDOWN"
        ]):
            # Forkorte lange linjer
            if len(line) > 80:
                print(f"  {line[:80]}...")
            else:
                print(f"  {line}")

# Sjekk apply-layer konfig
print("\n  apply-layer.env kjerneparametere:")
with open("/etc/quantum/apply-layer.env") as f:
    for line in f:
        line = line.strip()
        if line and not line.startswith("#") and any(k in line for k in [
            "DEDUPE_TTL", "POLL_SEC", "MAX_POSITION", "KILL_SWITCH",
            "K_BLOCK", "MODE", "COOLDOWN"
        ]):
            print(f"  {line}")

# ─── 5. NÅVÆRENDE PROBLEMER ──────────────────────────────────
print("\n── 5. AKTIVE SYSTEMPROBLEMER ──")

# Sjekk om INTENT_EXECUTOR_UPDATE_LEDGER_AFTER_EXEC=false er et problem
print("\n  Sjekker INTENT_EXECUTOR_UPDATE_LEDGER_AFTER_EXEC...")
with open("/etc/quantum/intent-executor.env") as f:
    env = f.read()
    val = "false" if "UPDATE_LEDGER_AFTER_EXEC=false" in env else "true"
    print(f"  INTENT_EXECUTOR_UPDATE_LEDGER_AFTER_EXEC = {val}")
    if val == "false":
        print("  ⚠️  POTENSIELT PROBLEM: Ledger ikke oppdatert etter exec")
        print("  → Systemet vet ikke at positions er lukket")
        print("  → Kan forklare ghost position hashes!")

# Sjekk APPLY_DEDUPE_TTL_SEC
print("\n  Sjekker deduplisering...")
with open("/etc/quantum/apply-layer.env") as f:
    for line in f:
        if "DEDUPE_TTL" in line:
            print(f"  {line.strip()}")

# Sjekk om apply-layer sender til testnet
print("\n  apply-layer APPLY_MODE:")
with open("/etc/quantum/apply-layer.env") as f:
    for line in f:
        if "APPLY_MODE" in line and not line.strip().startswith("#"):
            print(f"  {line.strip()}")

# ─── 6. METRICS RATIO ANALYSE ────────────────────────────────
print("\n── 6. INTENT-EXECUTOR METRICS RATIO ANALYSE ──")
out = subprocess.run(['journalctl','-u','quantum-intent-executor','-n','5','--no-pager'],
    capture_output=True, text=True)
for line in reversed(out.stdout.splitlines()):
    if 'harvest_executed=' in line:
        print(f"\n  Kumulativ:")
        parts = line.split()
        metrics = {}
        for p in parts:
            if '=' in p:
                k, v = p.split('=', 1)
                metrics[k] = v
        
        processed = int(metrics.get('processed', 0))
        exec_true = int(metrics.get('executed_true', 0))
        exec_false = int(metrics.get('executed_false', 0))
        blocked = int(metrics.get('p35_guard_blocked', 0))
        h_exec = int(metrics.get('harvest_executed', 0))
        h_fail = int(metrics.get('harvest_failed', 0))
        
        print(f"  Prosessert totalt:     {processed:,}")
        print(f"  Executed TRUE:         {exec_true:,}  ({exec_true/processed*100:.1f}%)")
        print(f"  Executed FALSE:        {exec_false:,}  ({exec_false/processed*100:.1f}%)")
        print(f"  P3.5 Guard blocked:    {blocked:,}")
        print(f"  Harvest executed:      {h_exec:,}")
        print(f"  Harvest failed (401):  {h_fail:,}  ({h_fail/(h_exec+h_fail)*100:.1f}% failure rate)")
        
        print(f"\n  ⚠️  {exec_false/processed*100:.0f}% av alle intents BLOKKERT/AVVIST")
        print(f"  ⚠️  {blocked:,} blokkert av P3.5 guard alene")
        print(f"  ⚠️  {h_fail:,} harvest failures = tapt profit")
        break
