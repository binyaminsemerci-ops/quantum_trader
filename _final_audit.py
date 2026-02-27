#!/usr/bin/env python3
"""Final audit checks: drawdown, testnet mode, P3.5 config, lane status"""
import redis, json, subprocess
from datetime import datetime, timezone

r = redis.Redis(decode_responses=True)

print("=== MANUAL LANE STATE ===")
key = "quantum:manual_lane:enabled"
val = r.get(key)
ttl = r.ttl(key)
print(f"  {key} = {val!r}")
print(f"  TTL = {ttl}  ({'MISSING/EXPIRED — lane OFF' if ttl < 0 else f'{ttl//60}m remaining — lane ON'})")
print(f"\n  To ENABLE: redis-cli SET quantum:manual_lane:enabled 1 EX <TTL_SECONDS>")
print(f"  Example 1h: redis-cli SET quantum:manual_lane:enabled 1 EX 3600")
print(f"  Example 4h: redis-cli SET quantum:manual_lane:enabled 1 EX 14400")

print("\n=== TESTNET vs REAL MODE ===")
# Check env files
for envfile in ["/etc/quantum/intent-executor.env", "/etc/quantum/execution.env"]:
    try:
        result = subprocess.run(["grep", "-i", "binance\\|testnet\\|real\\|api_key", envfile],
                                capture_output=True, text=True)
        print(f"\n  {envfile}:")
        for line in result.stdout.splitlines():
            if "secret" not in line.lower():  # skip secrets
                print(f"    {line.strip()[:100]}")
    except:
        print(f"  {envfile}: not found")

print("\n=== DRAWDOWN STATE ===")
ddkey = "quantum:dag5:lockdown_guard:latest"
dd = r.hgetall(ddkey)
if dd:
    equity = float(dd.get("equity", 0))
    peak   = float(dd.get("peak", 0))
    pct    = float(dd.get("drawdown_pct", 0))
    unreal = float(dd.get("unrealized_pnl", 0)) 
    mode   = dd.get("mode", "?")
    lock   = dd.get("lockdown_active", "?")
    warns  = dd.get("warnings", "none")
    print(f"  Mode              : {mode or '(normal)'}")
    print(f"  Lockdown active   : {lock}")
    print(f"  Equity            : ${equity:.2f}")
    print(f"  Peak equity       : ${peak:.2f}")
    print(f"  Drawdown          : {pct:.2f}%")
    print(f"  Unrealized PnL    : ${unreal:.2f}")
    print(f"  Warnings          : {warns}")
else:
    print("  (key not found)")

print("\n=== KILL SWITCH EVENTS ===")
ks_events = r.get("quantum:metrics:kill_switch_events")
print(f"  Total kill_switch events: {ks_events}")

print("\n=== P3.5 / LAYER4 EDGE THRESHOLD ===")
# Find the kelly min-edge threshold from running service
result2 = subprocess.run(
    ["journalctl", "-u", "quantum-intent-executor", "-n", "200", "--no-pager"],
    capture_output=True, text=True
)
edge_lines = [l for l in result2.stdout.splitlines() if "kelly" in l.lower() or "min_edge" in l.lower() or "P3.5" in l]
for l in edge_lines[-10:]:
    print(f"  {l.strip()[-120:]}")

print("\n=== BLOCKED REASONS summary (last 200 logs) ===")
blocked_lines = [l for l in result2.stdout.splitlines() if "BLOCKED" in l]
reasons = {}
for l in blocked_lines:
    # Extract reason= part
    if "reason=" in l:
        rp = l.split("reason=")[-1].split()[0]
        reasons[rp] = reasons.get(rp, 0) + 1
for rr, cnt in sorted(reasons.items(), key=lambda x: -x[1]):
    print(f"  {cnt:4d}x  {rr}")
