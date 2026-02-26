#!/usr/bin/env python3
"""
Check trade.intent stream + what apply.layer/intent-executor writes
to understand why execution.result is silent
"""
import redis, json, subprocess

r = redis.Redis(decode_responses=True)

# 1. Count trade.intent stream
print("=== quantum:stream:trade.intent ===")
stream_name = "quantum:stream:trade.intent"
if r.exists(stream_name):
    length = r.xlen(stream_name)
    print(f"  Stream length: {length}")
    if length > 0:
        last = r.xrevrange(stream_name, count=3)
        print(f"  Last 3 entries:")
        for sid, data in last:
            raw = data.get("payload", "")
            p = json.loads(raw) if raw else data
            print(f"    {sid}: sym={p.get('symbol')} action={p.get('action')} decision={p.get('decision')} source={p.get('source')}")
    # Check consumer groups
    try:
        groups = r.xinfo_groups(stream_name)
        for g in groups:
            print(f"  Consumer group: {g['name']} pending={g['pending']}")
    except Exception as e:
        print(f"  Groups error: {e}")
else:
    print("  (stream does not exist)")

# 2. Check apply.plan executed=True entries — what fields do they actually have?
print("\n=== apply.result executed=True entry — FULL payload ===")
results = r.xrange("quantum:stream:apply.result", count=10000)
for sid, data in reversed(results):
    raw = data.get("payload", "")
    if raw:
        try:
            p = json.loads(raw)
        except:
            p = {}
    else:
        p = data
    if str(p.get("executed", "")).lower() == "true":
        print(f"  Stream ID: {sid}")
        for k, v in sorted(p.items()):
            val = str(v)
            if len(val) > 100:
                val = val[:100] + "..."
            print(f"    {k}: {val}")
        break

# 3. Check what apply.layer writes to trade.intent  
print("\n=== Check apply-layer service source ===")
import subprocess as sp
# Find apply-layer source
result = sp.run(
    ["find", "/home/qt/quantum_trader/microservices", "-name", "main.py", "-path", "*apply*"],
    capture_output=True, text=True
)
apply_paths = result.stdout.strip().split("\n")
print(f"  apply-layer sources: {apply_paths}")

for path in apply_paths:
    if path:
        try:
            with open(path) as f:
                content = f.read()
            # Look for trade.intent writing
            lines = content.splitlines()
            for i, line in enumerate(lines, 1):
                if "trade.intent" in line or "execution.result" in line or "xadd" in line.lower():
                    print(f"  {path}:{i}: {line.strip()}")
        except Exception as e:
            print(f"  {path}: {e}")

# 4. Check execution_service.py for trade.intent reading
print("\n=== execution_service.py trade.intent consumer ===")
with open("/home/qt/quantum_trader/services/execution_service.py") as f:
    exec_lines = f.readlines()

for i, line in enumerate(exec_lines, 1):
    if "trade.intent" in line or "xreadgroup" in line.lower() or "xread" in line.lower():
        print(f"  {i:4}: {line.rstrip()}")

# 5. Check execution.log for STALE_INTENT_DROP or PATH1B EXEC
print("\n=== execution.log: PATH1B EXEC or STALE ===")
logs = sp.run(
    ["grep", "-a", "PATH1B EXEC\|STALE_INTENT\|trade.intent\|FILLED\|REJECT\|Error placing",
     "/var/log/quantum/execution.log"],
    capture_output=True, text=True
)
output = logs.stdout.strip()
if output:
    for line in output.splitlines()[-20:]:
        print(f"  {line.strip()[-200:]}")
else:
    print("  (no matches)")

# 6. Quick check: INTENT_MAX_AGE_SEC value
print("\n=== INTENT_MAX_AGE_SEC value ===")
for i, line in enumerate(exec_lines, 1):
    if "INTENT_MAX_AGE" in line:
        print(f"  {i:4}: {line.rstrip()}")
