#!/usr/bin/env python3
"""Get symbol universe from Redis/env"""
import redis, os, json

r = redis.Redis(host='127.0.0.1', port=6379, decode_responses=True)

# 1. Check common Redis keys for universe
for key in ['quantum:universe', 'quantum:symbols', 'quantum:approved_symbols', 
            'qt:universe', 'qt:symbols', 'ai_engine:universe',
            'quantum:watch:symbols']:
    val = r.get(key)
    if val:
        print(f"KEY {key}: {val[:200]}")
    members = r.smembers(key)
    if members:
        print(f"SET {key}: {sorted(members)[:20]}")

# 2. Check autonomous-trader env
try:
    with open('/etc/quantum/autonomous-trader.env') as f:
        print("\n=== autonomous-trader.env ===")
        print(f.read())
except: pass

# 3. Check position slots
slots = r.hgetall('quantum:positions:ledger')
print(f"\nActual ledger: {slots}")

# 4. Check last published signals to get canonical symbol list
msgs = r.xrevrange('quantum:stream:ai.signal_generated', count=100)
syms = set()
for mid, data in msgs:
    try:
        p = json.loads(data.get('payload', '{}'))
        syms.add(p.get('symbol',''))
    except: pass
print(f"\nSymbols seen in signal stream: {sorted(syms)}")
