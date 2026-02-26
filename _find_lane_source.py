#!/usr/bin/env python3
import subprocess, redis

r = redis.Redis(decode_responses=True)

# Find MANUAL_LANE in Python files
result = subprocess.run(
    "find /home/qt/quantum_trader -name '*.py' 2>/dev/null | xargs grep -l MANUAL_LANE 2>/dev/null",
    shell=True, capture_output=True, text=True, timeout=30
)
files = result.stdout.strip().splitlines()
print(f"Files with MANUAL_LANE: {files}")

for f in files[:3]:
    result2 = subprocess.run(
        f"grep -n 'MANUAL_LANE\\|manual_lane\\|lane_enabled\\|lane_off' {f}",
        shell=True, capture_output=True, text=True
    )
    print(f"\n{f}:")
    print(result2.stdout[:3000])

# Also look for the Redis key the executor checks
# Common patterns: quantum:config:*, quantum:flag:*, quantum:mode:*
print("\n=== quantum:config:* keys ===")
for k in sorted(r.keys("quantum:config:*"))[:20]:
    t = r.type(k)
    v = r.get(k) if t == "string" else r.hgetall(k) if t == "hash" else "..."
    print(f"  {k} = {str(v)[:100]}")

print("\n=== quantum:flag:* / quantum:mode:* keys ===")
for k in sorted(r.keys("quantum:flag:*") + r.keys("quantum:mode:*"))[:20]:
    t = r.type(k)
    v = r.get(k) if t == "string" else r.hgetall(k) if t == "hash" else "..."
    print(f"  {k} = {str(v)[:100]}")

print("\n=== quantum:intent-executor:* keys ===")
for k in sorted(r.keys("quantum:intent-executor:*"))[:20]:
    t = r.type(k)
    v = r.get(k) if t == "string" else r.hgetall(k) if t == "hash" else "..."
    print(f"  {k} = {str(v)[:100]}")

# Also check systemd service file for intent-executor
result3 = subprocess.run(
    "cat /etc/systemd/system/quantum-intent-executor.service",
    shell=True, capture_output=True, text=True
)
print(f"\n=== quantum-intent-executor.service ===")
print(result3.stdout[:3000])

# Actual service source
result4 = subprocess.run(
    "cat /etc/systemd/system/quantum-intent-executor.service | grep ExecStart",
    shell=True, capture_output=True, text=True
)
print(f"\nExecStart: {result4.stdout.strip()}")
