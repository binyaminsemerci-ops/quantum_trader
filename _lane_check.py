#!/usr/bin/env python3
"""Find execution lane controls and MANUAL_LANE key"""
import redis, json, subprocess

r = redis.Redis(decode_responses=True)

# Specifically look for lane/trading controls (NOT permit:p33)
keywords = ["manual_lane", "trade_enabled", "trading:enabled", "entry:enabled",
            "execution:enabled", "lane:open", "autonomous:enabled", "entry_gate",
            "trading_gate", "kill_switch", "halt_trading", "pause_trading",
            "lockdown", "circuit_breaker", "entry_block", "manual_block",
            "open_trading", "execution_gate", "is_live", "is_paused",
            "p35_edge_threshold", "kelly_min", "min_edge"]

print("=== TARGETED CONTROL KEYS ===")
all_keys = r.keys("*")

for k in sorted(all_keys):
    kl = k.lower()
    if any(kw in kl for kw in keywords) and "permit:p33:" not in k:
        t = r.type(k)
        if t == "string":
            v = r.get(k)
            print(f"  {k} = {str(v)[:100]}")
        elif t == "hash":
            v = r.hgetall(k)
            print(f"  {k} [hash]: {str(v)[:200]}")
        else:
            print(f"  {k} [{t}]")

# Check for keys having "manual" or "lane" anywhere in name
print("\n=== ALL KEYS with 'manual' or 'lane' (excluding permit) ===")
for k in sorted(all_keys):
    if ("manual" in k.lower() or "lane" in k.lower()) and "permit:p33:" not in k:
        t = r.type(k)
        v = r.get(k) if t == "string" else "..."
        print(f"  {k} [{t}] = {str(v)[:100]}")

# Find the intent_executor source code
print("\n=== INTENT_EXECUTOR source ===")
result = subprocess.run(
    ["find", "/", "-name", "intent_executor*", "-o", "-name", "*intent*executor*"],
    capture_output=True, text=True, timeout=10
)
for f in result.stdout.strip().splitlines()[:10]:
    if f.endswith(".py"):
        print(f"  {f}")

# Grep for MANUAL_LANE_OFF in all Python files
print("\n=== MANUAL_LANE grep ===")
result2 = subprocess.run(
    ["grep", "-r", "MANUAL_LANE", "/home/qt/quantum_trader", "/opt/quantum",
     "--include=*.py", "-l"],
    capture_output=True, text=True, timeout=10
)
if result2.stdout.strip():
    for f in result2.stdout.strip().splitlines():
        print(f"  {f}")
        result3 = subprocess.run(
            ["grep", "-n", "MANUAL_LANE\|manual_lane\|lane_enabled\|entry_gate", f],
            capture_output=True, text=True
        )
        for line in result3.stdout.strip().splitlines()[:10]:
            print(f"    {line}")
