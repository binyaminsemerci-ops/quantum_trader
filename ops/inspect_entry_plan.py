#!/usr/bin/env python3
"""Inspect ENTRY_PROPOSED plan fields and intent_bridge side field"""
import redis
import re
import subprocess

r = redis.Redis(host='localhost', port=6379, decode_responses=True)

# Get ENTRY_PROPOSED plan from apply.plan stream
messages = r.xrevrange('quantum:stream:apply.plan', '+', '-', count=500)
entry_plan = None
for msg_id, fields in messages:
    if fields.get('action') == 'ENTRY_PROPOSED':
        entry_plan = (msg_id, fields)
        break

if entry_plan:
    print(f"=== ENTRY_PROPOSED plan found: {entry_plan[0]} ===")
    for k, v in sorted(entry_plan[1].items()):
        print(f"  {k}: {v}")
else:
    print("No ENTRY_PROPOSED plan found in last 500 messages")

# Check position limit count
all_pos = r.keys("quantum:position:*")
pos_active = [k for k in all_pos if 'snapshot' not in k and 'ledger' not in k]
print(f"\n=== Position counts ===")
print(f"  All quantum:position:* keys: {len(all_pos)}")
print(f"  Active (no snapshot/ledger): {len(pos_active)}")
print(f"  Active keys: {sorted(pos_active)}")

# Check recent apply.action intent from trade.intent
print("\n=== Recent trade.intent (last 3) ===")
intents = r.xrevrange('quantum:stream:trade.intent', '+', '-', count=3)
for mid, fields in intents:
    print(f"  {mid}: action={fields.get('action')} side={fields.get('side')} symbol={fields.get('symbol')} direction={fields.get('direction')}")
