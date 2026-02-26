#!/usr/bin/env python3
"""Find apply.result entries that have 'action' field and show their structure"""
import redis

r = redis.Redis()
STREAM = "quantum:stream:apply.result"

# Scan 2000 entries looking for ones with 'action' field
entries = r.xrevrange(STREAM, count=2000)
with_action = [(mid, d) for mid, d in entries if b'action' in d]

print(f"Scanned: {len(entries)}, with action field: {len(with_action)}")

for mid, d in with_action[:5]:
    print(f"\n--- {mid.decode()} ---")
    for k, v in d.items():
        print(f"  {k.decode()}: {v.decode()[:100]}")

# Also scan for 'confidence' field  
with_confidence = [(mid, d) for mid, d in entries if b'confidence' in d]
print(f"\nWith confidence field: {len(with_confidence)}")

# Check what fields appear in ALL entries (distribution)
from collections import Counter
all_keys = Counter()
for _, d in entries:
    for k in d.keys():
        all_keys[k.decode()] += 1

print("\nField frequency in last 2000 apply.result entries:")
for k, cnt in sorted(all_keys.items(), key=lambda x: -x[1]):
    print(f"  {k}: {cnt}")

# Check if apply.result entries with decision=EXECUTE exist
with_execute = [(mid, d) for mid, d in entries if d.get(b'decision') == b'EXECUTE']
print(f"\ndecision=EXECUTE entries: {len(with_execute)}")
for mid, d in with_execute[:3]:
    print(f"\n--- {mid.decode()} ---")
    for k, v in d.items():
        print(f"  {k.decode()}: {v.decode()[:100]}")
