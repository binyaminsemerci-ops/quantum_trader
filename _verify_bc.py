#!/usr/bin/env python3
"""Check harvest proposals for phantom symbols + confirm -2022 errors stopping"""
import redis, json, time

r = redis.Redis()
PHANTOMS = ['SOLUSDT', 'BTCUSDT', 'BNBUSDT', 'XRPUSDT', 'INJUSDT']

# ─── Check harvest proposals ──────────────────────────────────────────────
print("=== Harvest proposals for phantom symbols ===")
for sym in PHANTOMS:
    for key_pattern in [f"quantum:harvest:proposal:{sym}", f"quantum:harvest_v2:state:{sym}"]:
        if r.exists(key_pattern):
            data = {k.decode(): v.decode() for k, v in r.hgetall(key_pattern).items()}
            print(f"\n  {key_pattern}:")
            for k, v in data.items():
                print(f"    {k}: {v[:80]}")

# ─── Check apply.result stream for NEW -2022/-1111 errors (last 5 min) ───
print("\n=== apply.result: new Binance errors in last 5 min ===")
cutoff_ms = (int(time.time()) - 300) * 1000
entries = r.xrange("quantum:stream:apply.result", min=str(cutoff_ms))
recent_errors = []
for mid, d in entries:
    err = d.get(b'error', b'').decode()
    if 'Binance API error' in err or 'execution_failed' in err:
        sym = d.get(b'symbol', b'?').decode()
        action = d.get(b'action', b'?').decode()
        import re
        code = re.search(r'"code"\s*:\s*(-?\d+)', err)
        code = code.group(1) if code else '?'
        age_s = (int(time.time() * 1000) - int(mid.decode().split('-')[0])) // 1000
        recent_errors.append((age_s, sym, action, code))

if recent_errors:
    print(f"  {len(recent_errors)} Binance errors in last 5 min:")
    for age, sym, action, code in sorted(recent_errors):
        print(f"    [{age}s ago] {sym} action={action} code={code}")
else:
    print("  ✅ No Binance errors in last 5 min")

# ─── Summary ──────────────────────────────────────────────────────────────
print("\n=== Final state ===")
print(f"  Manual lane: {r.get('quantum:manual_lane:enabled').decode()} TTL={r.ttl('quantum:manual_lane:enabled')}s")

active_pos = [k.decode() for k in r.keys("quantum:position:*")
              if 'snapshot' not in k.decode() and 'ledger' not in k.decode() and 'backup' not in k.decode()]
print(f"  Active positions (Redis): {len(active_pos)}")

for sym in PHANTOMS:
    key = f"quantum:position:{sym}"
    exists = r.exists(key)
    print(f"  {sym} position key: {'EXISTS' if exists else 'DELETED ✅'}")
