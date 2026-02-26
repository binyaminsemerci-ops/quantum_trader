#!/usr/bin/env python3
"""Verify: phantom proposals gone, Binance errors stopped after harvest fix."""
import redis, time, re

r = redis.Redis()
PHANTOMS = ['SOLUSDT', 'BTCUSDT', 'BNBUSDT', 'XRPUSDT', 'INJUSDT']

print("=== Waiting 90s for regeneration check ===")
time.sleep(90)

print("\n=== Phantom harvest proposal keys (should all be GONE) ===")
all_gone = True
for sym in PHANTOMS:
    key = f"quantum:harvest:proposal:{sym}"
    if r.exists(key):
        data = {k.decode(): v.decode() for k, v in r.hgetall(key).items()}
        print(f"  STILL THERE: {sym}  action={data.get('harvest_action')}  computed={data.get('computed_at_utc', '?')[:19]}")
        all_gone = False
    else:
        print(f"  ✅ GONE: {sym}")

print(f"\nAll phantom proposals gone: {all_gone}")

print("\n=== Binance errors in last 2 min ===")
cutoff_ms = (int(time.time()) - 120) * 1000
entries = r.xrange("quantum:stream:apply.result", min=str(cutoff_ms))
errors = []
for eid, d in entries:
    err = d.get(b'error', b'').decode()
    if 'Binance API error' in err or 'execution_failed' in err or '-2022' in err or '-1111' in err:
        sym = d.get(b'symbol', b'?').decode()
        code = re.search(r'"code"\s*:\s*(-?\d+)', err)
        errors.append((sym, code.group(1) if code else '?'))

if errors:
    from collections import Counter
    for (sym, code), cnt in Counter(errors).most_common():
        print(f"  ❌ {sym} code={code} × {cnt}")
else:
    print("  ✅ ZERO Binance errors in last 2 min")

print(f"\n  Total apply.result entries last 2 min: {len(entries)}")

print("\n=== Remaining apply.result entries last 2 min ===")
for eid, d in entries[:5]:
    sym = d.get(b'symbol', b'?').decode()
    dec = d.get(b'decision', b'?').decode()
    src = d.get(b'source', b'?').decode()
    err = d.get(b'error', b'')[:60].decode() if d.get(b'error') else 'ok'
    print(f"  {sym} decision={dec} src={src} err={err}")
