#!/usr/bin/env python3
"""Find execution lane switch keys in Redis"""
import redis, json

r = redis.Redis(decode_responses=True)
all_keys = r.keys("*")

keywords = ["lane", "manual", "c3", "gate", "enabled", "execution_on", "trading_on",
            "kill", "switch", "mode", "autonomous", "entry_enabled", "trade_on",
            "blocked", "lockdown", "halt", "pause"]

hits = []
for k in sorted(all_keys):
    kl = k.lower()
    if any(kw in kl for kw in keywords):
        hits.append(k)

print(f"=== CONTROL KEYS ({len(hits)} found) ===")
for k in hits:
    t = r.type(k)
    if t == "string":
        v = r.get(k)
        print(f"  {k} = {str(v)[:100]}")
    elif t == "hash":
        v = r.hgetall(k)
        print(f"  {k} [hash/{len(v)}f]: {str(v)[:200]}")
    elif t == "set":
        v = r.smembers(k)
        print(f"  {k} [set]: {v}")
    else:
        print(f"  {k} [{t}]")

# Also check environment config
import subprocess, os
result = subprocess.run(
    ["find", "/opt/quantum", "/home/qt/quantum_trader", "/etc/quantum",
     "-name", "*.env", "-o", "-name", "*.conf", "-o", "-name", "*.yaml",
     "-o", "-name", "*.yml"],
    capture_output=True, text=True
)
conf_files = [f for f in result.stdout.strip().splitlines() if f]
print(f"\n=== CONFIG FILES ===")
for f in conf_files[:20]:
    print(f"  {f}")

# Look for C3 in systemd service files
result2 = subprocess.run(
    ["grep", "-r", "C3", "/etc/systemd/system/", "--include=*.service", "-l"],
    capture_output=True, text=True
)
if result2.stdout.strip():
    print(f"\n=== Services mentioning C3 ===")
    print(result2.stdout.strip())

# Check MANUAL_LANE keys specifically
print(f"\n=== MANUAL/LANE keys ===")
for k in sorted(all_keys):
    if "manual" in k.lower() or "lane" in k.lower():
        t = r.type(k)
        v = r.get(k) if t == "string" else r.hgetall(k) if t == "hash" else "..."
        print(f"  {k} [{t}] = {str(v)[:100]}")
