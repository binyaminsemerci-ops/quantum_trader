#!/usr/bin/env python3
"""Full slot/position state check"""
import redis, json

r = redis.Redis(host='127.0.0.1', port=6379, decode_responses=True)

# 1. Slots
print("=== Slot keys ===")
for k in ['quantum:slots:available', 'quantum:slots:desired', 'quantum:slots:regime']:
    t = r.type(k)
    if t == 'string':
        print(f"  {k}: {r.get(k)}")
    elif t == 'hash':
        print(f"  {k}: {r.hgetall(k)}")
    elif t == 'set':
        print(f"  {k}: {r.smembers(k)}")
    else:
        print(f"  {k}: ({t})")

# 2. Active individual position hashes
print("\n=== Active quantum:position:SYMBOL hashes ===")
pos_keys = sorted(k for k in r.keys('quantum:position:*') 
                  if not any(x in k for x in ['ledger', 'snapshot', 'phantom', 'tracker']))
print(f"  Found {len(pos_keys)} active positions")
for k in pos_keys:
    data = r.hgetall(k)
    sym = k.split(':')[-1]
    status = data.get('status', '?')
    side = data.get('side', '?')
    entry = data.get('entry_price', '?')
    size = data.get('size_usd', '?')
    print(f"  {sym:20s} {side:5s} status={status} entry={entry} size_usd={size}")

# 3. Ledger (authoritative)
ledger = r.hgetall('quantum:positions:ledger')
print(f"\n=== quantum:positions:ledger === ({len(ledger)} entries)")
print(f"  {ledger}")

# 4. Check apply-layer.env for UPDATE_LEDGER_AFTER_EXEC
try:
    with open('/etc/quantum/apply-layer.env') as f:
        for line in f:
            if 'LEDGER' in line.upper() or 'UPDATE' in line.upper():
                print(f"\napply-layer.env: {line.strip()}")
except Exception as e:
    print(f"\napply-layer.env: {e}")
