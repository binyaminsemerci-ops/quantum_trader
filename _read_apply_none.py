#!/usr/bin/env python3
"""Read decision=None apply.result entries to understand structure"""
import redis

r = redis.Redis()
STREAM = "quantum:stream:apply.result"

# scan for decision=None entries
entries = r.xrange(STREAM, count=5000)
none_entries = [(mid, d) for mid, d in entries
                if d.get(b'decision') in (b'None', b'none', None)]

print(f"Total entries scanned: {len(entries)}")
print(f"decision=None entries: {len(none_entries)}")

for mid, d in none_entries[:3]:
    print(f"\n--- {mid.decode()} ---")
    for k, v in d.items():
        print(f"  {k.decode()!r}: {v.decode()[:120]!r}")

# Also check most recent entries for decision=None
print("\n=== Most recent 20 entries with decision != SKIP ===")
recent = r.xrevrange(STREAM, count=1000)
non_skip = [(mid, d) for mid, d in recent
            if d.get(b'decision') not in (b'SKIP', b'skip')]
print(f"Non-SKIP recent: {len(non_skip)}")
for mid, d in non_skip[:3]:
    print(f"\n--- {mid.decode()} ---")
    for k, v in d.items():
        print(f"  {k.decode()}: {v.decode()[:120]}")

# Check ALL unique decision values
from collections import Counter
decisions = Counter(d.get(b'decision', b'MISSING').decode() for _, d in r.xrevrange(STREAM, count=2000))
print(f"\n=== Decision value distribution (last 2000 entries) ===")
for dec, cnt in sorted(decisions.items(), key=lambda x: -x[1]):
    print(f"  {dec!r}: {cnt}")
