#!/usr/bin/env python3
"""Check harvest proposals and current apply layer status."""
import sys
sys.path.insert(0, '/home/qt/quantum_trader')
import redis
import json
import time

r = redis.Redis()

# Check harvest proposals
print("=== HARVEST PROPOSALS ===")
symbols = ['STXUSDT', 'BNBUSDT', 'ETHUSDT', 'OPUSDT', 'DOTUSDT', 'INJUSDT',
           'BTCUSDT', 'SOLUSDT', 'XRPUSDT', 'ARBUSDT']

for sym in symbols:
    key = f"quantum:harvest:proposal:{sym}"
    data = r.hgetall(key)
    if not data:
        print(f"  {sym}: NO PROPOSAL")
        continue
    action = (data.get(b'harvest_action') or data.get(b'action') or b'?').decode()
    kill_score = (data.get(b'kill_score') or b'0').decode()
    calibrated = (data.get(b'calibrated') or b'0').decode()
    ts = (data.get(b'last_update_epoch') or b'0').decode()
    age = int(time.time()) - int(float(ts)) if ts != '0' else -1
    print(f"  {sym}: action={action}, kill_score={kill_score}, calibrated={calibrated}, age={age}s")

# Check quantum:stream:apply.plan
print("\n=== APPLY PLAN STREAM (quantum:stream:apply.plan) ===")
try:
    stream_info = r.xinfo_stream("quantum:stream:apply.plan")
    print(f"  Length: {stream_info['length']}")
    groups = r.xinfo_groups("quantum:stream:apply.plan")
    for g in groups:
        name = g.get(b'name', b'?').decode() if isinstance(g.get(b'name', b'?'), bytes) else str(g.get('name', '?'))
        pending = g.get(b'pending', 0) if isinstance(g.get(b'pending', 0), int) else int(g.get('pending', 0))
        lag = g.get(b'lag', '?')
        print(f"  Group: {name}, pending={pending}, lag={lag}")
    
    # Last 5 entries
    entries = r.xrevrange("quantum:stream:apply.plan", count=5)
    print(f"  Last 5 entries:")
    for entry_id, fields in entries:
        sym = (fields.get(b'symbol') or b'?').decode()
        action = (fields.get(b'action') or fields.get(b'plan_type') or b'?').decode()
        plan_id = (fields.get(b'plan_id') or b'?').decode()[:12]
        print(f"    {entry_id.decode()[:15]} sym={sym} action={action} plan_id={plan_id}")
except Exception as e:
    print(f"  Error: {e}")

# Position count
print("\n=== POSITION COUNT ===")
positions = [k.decode() for k in r.keys("quantum:state:positions:*")]
print(f"  Active: {len(positions)}/10")
for p in positions:
    sym = p.replace("quantum:state:positions:", "")
    data = r.hgetall(p)
    side = (data.get(b'side') or b'?').decode()
    qty = (data.get(b'quantity') or data.get(b'qty') or b'0').decode()
    src = (data.get(b'source') or b'?').decode()
    print(f"    {sym}: {side} qty={qty} src={src}")
