#!/usr/bin/env python3
"""Clean CLOSED_GHOST position hashes and verify current live positions"""
import redis, json

r = redis.Redis(host='127.0.0.1', port=6379, decode_responses=True)

# 1. Delete all CLOSED_GHOST hashes
ghost_keys = []
pos_keys = sorted(k for k in r.keys('quantum:position:*') 
                  if not any(x in k for x in ['ledger', 'snapshot', 'phantom', 'tracker']))
for k in pos_keys:
    status = r.hget(k, 'status') or ''
    if 'GHOST' in status.upper() or 'CLOSED' in status.upper():
        ghost_keys.append(k)

print(f"CLOSED_GHOST hashes to delete: {len(ghost_keys)}")
for k in ghost_keys:
    r.delete(k)
    print(f"  DEL {k}")

# 2. Check position_tracker's live data
print("\n=== Live positions from position_tracker ===")
tracker_keys = r.keys('quantum:tracker:*')
print(f"tracker keys: {sorted(tracker_keys)[:10]}")

# Check for any live non-closed position hashes remaining
remaining = sorted(k for k in r.keys('quantum:position:*') 
                   if not any(x in k for x in ['ledger', 'snapshot', 'phantom', 'tracker']))
print(f"\nRemaining position hashes: {len(remaining)}")
for k in remaining:
    data = r.hgetall(k)
    print(f"  {k}: status={data.get('status','?')} side={data.get('side','?')}")

# 3. Check slots:available
print(f"\nSlots available: {r.get('quantum:slots:available')}")
print(f"Slots desired: {r.get('quantum:slots:desired')}")

print("\nDone - CLOSED_GHOST cleanup complete")
