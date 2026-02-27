#!/usr/bin/env python3
"""Check why authoritative_count=0 and ledger stays empty"""
import redis, json, subprocess

r = redis.Redis(host='127.0.0.1', port=6379, decode_responses=True)

# 1. Ledger state
ledger = r.hgetall('quantum:positions:ledger')
print(f"Ledger ({len(ledger)} entries): {ledger}")

# 2. Check actual open positions from Binance via autonomous-trader
positions_key_variants = [
    'quantum:positions:active',
    'quantum:active_positions',
    'qt:positions',
    'quantum:state:positions',
]
for k in positions_key_variants:
    t = r.type(k)
    if t != 'none':
        print(f"\nRedis {k} ({t}):")
        if t == 'hash':
            print(r.hgetall(k))
        elif t == 'set':
            print(r.smembers(k))
        elif t == 'string':
            print(r.get(k))
        elif t == 'list':
            print(r.lrange(k, 0, -1))

# 3. Scan all quantum:position* keys
keys = r.keys('quantum:position*')
print(f"\nAll quantum:position* keys: {sorted(keys)}")
for k in sorted(keys)[:15]:
    t = r.type(k)
    print(f"  {k} ({t})")

# 4. Check current UPDATE_LEDGER setting
try:
    with open('/etc/quantum/apply-layer.env') as f:
        content = f.read()
    for line in content.splitlines():
        if 'LEDGER' in line or 'ledger' in line.lower():
            print(f"\napply-layer.env: {line}")
except Exception as e:
    print(f"\napply-layer.env error: {e}")

# 5. Check slots in autonomous-trader
slots_key = r.keys('*slot*')
print(f"\nSlot keys: {slots_key}")
