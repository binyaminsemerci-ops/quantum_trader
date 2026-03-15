#!/usr/bin/env python3
"""
Full Redis position reset after manual testnet close.
Deletes all position keys, stale proposals, HOLD keys, cooldowns.
"""
import sys
sys.path.insert(0, '/home/qt/quantum_trader')
import redis

r = redis.Redis()

def count_and_delete(pattern, label):
    keys = r.keys(pattern)
    if keys:
        r.delete(*keys)
    print(f"  DELETED {len(keys):3d}  {label}  ({pattern})")
    return len(keys)

print("=== REDIS POSITION RESET ===\n")

total = 0
total += count_and_delete("quantum:state:positions:*",       "canonical position keys")
total += count_and_delete("quantum:position:*",             "legacy position keys")
total += count_and_delete("quantum:harvest:proposal:*",     "stale proposals")
total += count_and_delete("quantum:hold:*",                 "HOLD locks")
total += count_and_delete("quantum:cooldown:*",             "cooldown keys")
total += count_and_delete("quantum:apply:done:*",           "apply dedupe (done) keys")
total += count_and_delete("quantum:apply:stream_published:*","stream-published dedupe")
total += count_and_delete("quantum:position:snapshot:*",    "position snapshots")
total += count_and_delete("quantum:position:ledger:*",      "position ledger keys")
total += count_and_delete("quantum:position:dedupe:*",      "position dedupe")

print(f"\n  TOTAL DELETED: {total} keys")
print("\n=== VERIFY ===")
remaining = [k.decode() for k in r.keys("quantum:state:positions:*")] + \
            [k.decode() for k in r.keys("quantum:position:*")]
print(f"  Remaining position/* keys: {len(remaining)}")
for k in remaining:
    print(f"    {k}")

proposals_left = r.keys("quantum:harvest:proposal:*")
print(f"  Remaining proposals: {len(proposals_left)}")

print("\n  ✅ Redis is clean. Harvest brain will generate fresh ENTRY proposals on next tick.")
