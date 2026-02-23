#!/usr/bin/env python3
"""Check the apply.plan stream for pending/recent entries."""
import sys
sys.path.insert(0, '/home/qt/quantum_trader')
import redis
import json

r = redis.Redis()

stream_key = "apply.plan"

# Recent entries
entries = r.xrange(stream_key, count=30)
print(f"apply.plan stream: {len(entries)} recent entries (last 30)")
for entry_id, fields in entries[-10:]:
    data = {}
    for k, v in fields.items():
        try:
            data[k.decode()] = json.loads(v.decode())
        except Exception:
            data[k.decode()] = v.decode()
    symbol = data.get("symbol", "?")
    plan_type = data.get("plan_type", data.get("action", "-"))
    plan_id = str(data.get("plan_id", "-"))[:16]
    ts = entry_id.decode().split("-")[0]
    print(f"  {ts}  symbol={symbol}, type={plan_type}, plan_id={plan_id}")

# Stream info
try:
    info = r.xinfo_stream(stream_key)
    print(f"\nStream length: {info['length']}")
    print(f"First entry: {info.get('first-entry', ['N/A'])[0] if info.get('first-entry') else 'N/A'}")
    print(f"Last entry:  {info.get('last-entry', ['N/A'])[0] if info.get('last-entry') else 'N/A'}")
except Exception as e:
    print(f"xinfo_stream error: {e}")

# Consumer groups
try:
    groups = r.xinfo_groups(stream_key)
    print(f"\nConsumer groups ({len(groups)}):")
    for g in groups:
        name = g.get("name", b"?")
        if isinstance(name, bytes):
            name = name.decode()
        pending = g.get("pending", 0)
        lag = g.get("lag", "?")
        last_delivered = g.get("last-delivered-id", b"?")
        if isinstance(last_delivered, bytes):
            last_delivered = last_delivered.decode()
        print(f"  Group={name}, pending={pending}, lag={lag}, last_delivered={last_delivered}")
except Exception as e:
    print(f"xinfo_groups error: {e}")

# Check for NEW entries (after apply layer started ~05:08)
print("\n\nEntries with plan_type=ENTRY_PROPOSED:")
all_entries = r.xrange(stream_key, count=1000)
entry_count = 0
for entry_id, fields in all_entries:
    data = {}
    for k, v in fields.items():
        try:
            data[k.decode()] = json.loads(v.decode())
        except Exception:
            data[k.decode()] = v.decode()
    plan_type = data.get("plan_type", data.get("action", "-"))
    if "ENTRY" in str(plan_type).upper():
        entry_count += 1
        symbol = data.get("symbol", "?")
        plan_id = str(data.get("plan_id", "-"))[:16]
        print(f"  {entry_id.decode()[:15]}  symbol={symbol}, type={plan_type}, plan_id={plan_id}")

print(f"Total ENTRY plans found: {entry_count}")
