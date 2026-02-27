#!/usr/bin/env python3
"""
Inspect apply.result executed=True entries — what decision field do they have?
Also read execution_service lines 1040-1160 for full PATH1B logic.
"""
import redis, json

r = redis.Redis(decode_responses=True)

# ─── 1. Show executed=True entries in apply.result ────────────────────────
print("=== apply.result executed=True sample ===")
results = r.xrange("quantum:stream:apply.result", count=10000)
executed_true = []
for sid, data in results:
    raw = data.get("payload", "")
    if raw:
        try:
            p = json.loads(raw)
        except:
            p = {}
    else:
        p = data
    if str(p.get("executed", "")).lower() == "true" or p.get("executed") == True:
        executed_true.append((sid, p))
        
print(f"Total executed=True in apply.result: {len(executed_true)}")
print("\nLast 10 executed=True entries:")
for sid, p in executed_true[-10:]:
    keys = ["plan_id", "symbol", "decision", "executed", "action", "source", "timestamp"]
    row = {k: p.get(k, "?") for k in keys}
    print(f"  {sid}: {row}")

# ─── 2. Show SKIP entries with their timestamp format ─────────────────────
print("\n=== SKIP entry (timestamp field format) ===")
for sid, data in reversed(results[:100]):
    raw = data.get("payload", "")
    if raw:
        try:
            p = json.loads(raw)
        except:
            p = {}
    else:
        p = data
    if p.get("decision") == "SKIP" or str(p.get("executed", "")).lower() == "false":
        ts = p.get("timestamp", "NOT_PRESENT")
        print(f"  timestamp type={type(ts).__name__} value={repr(ts)[:80]}")
        break

# ─── 3. Read execution_service.py lines 1040-1160 ─────────────────────────
print("\n=== execution_service.py lines 1040-1160 ===")
with open("/home/qt/quantum_trader/services/execution_service.py") as f:
    lines = f.readlines()
for i, line in enumerate(lines[1039:1160], 1040):
    print(f"  {i:4}: {line.rstrip()}")

# ─── 4. What does INTENT_MAX_AGE_SEC default to? ──────────────────────────  
print("\n=== INTENT_MAX_AGE_SEC config ===")
for i, line in enumerate(lines[980:1005], 981):
    print(f"  {i:4}: {line.rstrip()}")
