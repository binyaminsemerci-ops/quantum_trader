#!/usr/bin/env python3
import redis, json, time

r = redis.Redis(decode_responses=True)

# Check apply.result for AGLDUSDT
print("--- apply.result: AGLDUSDT entries ---")
results = r.xrange("quantum:stream:apply.result", count=500)
found = 0
for sid, data in reversed(results):
    p = json.loads(data.get("payload", "{}")) if "payload" in data else data
    sym = p.get("symbol", data.get("symbol", ""))
    pid = p.get("plan_id", data.get("plan_id", ""))
    if sym == "AGLDUSDT" or pid == "eaec32a27d9d4f46":
        ex = p.get("executed", data.get("executed", "?"))
        reason = p.get("reason", p.get("decision", ""))
        print(f"  {sid}: plan={pid} sym={sym} executed={ex} reason={reason[:60] if reason else ''}")
        found += 1
        if found >= 5:
            break
if found == 0:
    print("  (none found)")

# Check apply.plan.manual stream
print("\n--- apply.plan.manual stream ---")
manual = r.xrange("quantum:stream:apply.plan.manual", count=50)
print(f"  Total entries: {len(manual)}")
for sid, data in manual[-5:]:
    p = json.loads(data.get("payload", "{}")) if "payload" in data else data
    print(f"  {sid}: sym={p.get('symbol')} action={p.get('action')} plan_id={p.get('plan_id')}")

# Check manual lane
lane_val = r.get("quantum:manual_lane:enabled")
lane_ttl = r.ttl("quantum:manual_lane:enabled")
print(f"\n--- Manual lane: value={lane_val}  TTL={lane_ttl}s ({lane_ttl//60}m)")

# Check recent intent-executor logs for manual stream processing
print("\n--- journalctl intent-executor last 30 lines ---")
import subprocess
result = subprocess.run(
    ["journalctl", "-u", "quantum-intent-executor", "-n", "30", "--no-pager"],
    capture_output=True, text=True
)
for line in result.stdout.splitlines():
    if any(k in line for k in ["manual", "AGLD", "FULL_CLOSE", "eaec32", "ERROR", "error", "exception"]):
        print(f"  {line.strip()}")
