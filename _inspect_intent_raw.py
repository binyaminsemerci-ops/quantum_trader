#!/usr/bin/env python3
"""
1. Read raw trade.intent stream entries — show ALL field names
2. Find where confidence is actually stored
3. Apply correct patch to execution_service.py
"""
import subprocess, shutil, time, sys

SRC = "/home/qt/quantum_trader/services/execution_service.py"

# ─── 1. Read raw trade.intent entries ─────────────────────────────────────
import redis, json

r = redis.Redis()

print("=== RAW trade.intent latest 3 entries (all fields) ===")
entries = r.xrevrange("quantum:stream:trade.intent", "+", "-", count=3)
for msg_id, data in entries:
    print(f"\n--- Entry: {msg_id.decode()} ---")
    for k, v in data.items():
        key = k.decode()
        val = v.decode()
        # Try to parse JSON if it looks like payload
        if val.startswith('{'):
            try:
                parsed = json.loads(val)
                print(f"  {key} = JSON:")
                for pk, pv in parsed.items():
                    print(f"    .{pk} = {repr(pv)[:80]}")
                continue
            except:
                pass
        print(f"  {key} = {repr(val)[:80]}")

# ─── 2. Show exact lines 1168-1210 of execution_service ───────────────────
print("\n=== execution_service.py lines 1168-1210 (exact repr) ===")
with open(SRC) as f:
    lines = f.readlines()

for i, line in enumerate(lines[1167:1210], 1168):
    print(f"  {i}: {repr(line.rstrip())}")

# ─── 3. Find exact SKIP block that precedes allowed_fields ────────────────
print("\n=== Lines 1140-1170 (exact context for patch anchor) ===")
for i, line in enumerate(lines[1139:1170], 1140):
    print(f"  {i}: {repr(line.rstrip())}")

# ─── 4. Check how signal_data is built (search for signal_data = ) ────────
print("\n=== signal_data construction (first 3 occurrences) ===")
count = 0
for i, line in enumerate(lines, 1):
    if "signal_data" in line and ("=" in line or "json" in line.lower() or "decode" in line.lower()):
        print(f"  {i}: {line.rstrip()}")
        count += 1
        if count >= 10:
            break

# ─── 5. Read raw stream using redis-cli to see actual byte structure ──────
print("\n=== redis-cli xrevrange trade.intent (latest 1) ===")
result = subprocess.run(
    ["redis-cli", "XREVRANGE", "quantum:stream:trade.intent", "+", "-", "COUNT", "1"],
    capture_output=True, text=True
)
for line in result.stdout.splitlines()[:50]:
    print(f"  {line}")
