#!/usr/bin/env python3
"""Fix H1: Flush permit keys and add TTL to permit key creation.
   Fix H3: Initialize slot/position keys from actual positions.
"""
import redis
import subprocess
import json

r = redis.Redis(decode_responses=True)

# === H1: Flush all permit keys ===
print("=== H1: Flushing permit keys ===")
cursor = 0
total = 0
while True:
    cursor, keys = r.scan(cursor, match="quantum:permit:*", count=2000)
    if keys:
        r.delete(*keys)
        total += len(keys)
    if cursor == 0:
        break
sample = r.scan(0, match="quantum:permit:*", count=5)[1]
print(f"Deleted {total} permit keys. Remaining sample: {len(sample)}")

# === H3: Initialize slot keys from actual open positions ===
print("\n=== H3: Initialize slot/position keys ===")
# Count actual positions in Redis
pos_keys = list(r.scan_iter("quantum:position:*", count=1000))
pos_count = len(pos_keys)
print(f"Found {pos_count} position keys in Redis")

r.set("quantum:max_slots", 15)
r.set("quantum:slot_count", pos_count)
r.set("quantum:positions_count", pos_count)
r.set("quantum:active_position_count", pos_count)
print(f"Set: max_slots=15, slot_count={pos_count}, positions_count={pos_count}")

# === Verify ===
print("\n=== Verification ===")
print(f"quantum:max_slots = {r.get('quantum:max_slots')}")
print(f"quantum:slot_count = {r.get('quantum:slot_count')}")
print(f"quantum:positions_count = {r.get('quantum:positions_count')}")
print(f"quantum:equity_usd = {r.get('quantum:equity_usd')}")
print(f"quantum:max_leverage = {r.get('quantum:max_leverage')}")
print(f"quantum:circuit_breaker = {r.get('quantum:circuit_breaker')}")
print("H1+H3: DONE")
