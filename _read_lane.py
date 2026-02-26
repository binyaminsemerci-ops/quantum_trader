#!/usr/bin/env python3
import subprocess, redis

r = redis.Redis(decode_responses=True)

# Read key section of intent_executor/main.py
with open("/home/qt/quantum_trader/microservices/intent_executor/main.py", "r") as f:
    lines = f.readlines()

# Print lines 50-350 (manual lane logic)
print("=== intent_executor/main.py lines 50-350 ===")
for i, line in enumerate(lines[49:349], start=50):
    print(f"{i:4d}: {line}", end="")

# Check current state of the manual lane key
print("\n=== Current manual_lane state ===")
key = "quantum:manual_lane:enabled"
val = r.get(key)
ttl = r.ttl(key)
print(f"  {key} = {val!r}")
print(f"  TTL = {ttl}s  ({'expired/missing' if ttl < 0 else f'{ttl//60}m {ttl%60}s remaining'})")

# Also check P3.5 edge threshold
print("\n=== P3.5 guard config ===")
for k in sorted(r.keys("quantum:p35:*") + r.keys("quantum:layer4:*") + r.keys("quantum:edge:*"))[:20]:
    t = r.type(k)
    v = r.get(k) if t == "string" else r.hgetall(k) if t == "hash" else "..."
    print(f"  {k} = {str(v)[:100]}")
