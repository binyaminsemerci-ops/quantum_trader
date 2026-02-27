#!/usr/bin/env python3
"""
Delete stale harvest proposals driving the phantom close loop.
All 5 symbols have identical fake defaults (entry=50.0, price=52.5, pnl=2.5).
Find and kill whatever is regenerating them.
"""
import redis, json, subprocess, time

r = redis.Redis()
PHANTOMS = ['SOLUSDT', 'BTCUSDT', 'BNBUSDT', 'XRPUSDT', 'INJUSDT']

print("=== Delete stale harvest proposals ===")
for sym in PHANTOMS:
    for key_tpl in [
        f"quantum:harvest:proposal:{sym}",
        f"quantum:harvest_v2:state:{sym}",
        f"quantum:harvest:heat:{sym}",
    ]:
        if r.exists(key_tpl):
            r.delete(key_tpl)
            print(f"  DELETED: {key_tpl}")

# ─── Find the harvest engine / service that regenerates these ─────────────
print("\n=== Find harvest engine source ===")
result = subprocess.run(
    ["find", "/home/qt/quantum_trader", "-name", "*.py", "-not", "-path", "*/.*",
     "-not", "-path", "*/venv*", "-not", "-path", "*/__pycache__*"],
    capture_output=True, text=True
)
for f in result.stdout.splitlines():
    if 'harvest' in f.lower():
        print(f"  {f}")

# Search for which service writes harvest:proposal 
result2 = subprocess.run(
    ["grep", "-rl", "harvest:proposal", "/home/qt/quantum_trader/services/"],
    capture_output=True, text=True
)
print(f"\nServices writing harvest:proposal:")
for f in result2.stdout.splitlines():
    print(f"  {f}")
    # Show the main write lines
    result3 = subprocess.run(
        ["grep", "-n", "harvest:proposal.*hset\|hset.*harvest:proposal\|harvest_proposal", f],
        capture_output=True, text=True
    )
    for line in result3.stdout.splitlines()[:5]:
        print(f"    {line.strip()[:120]}")

# ─── Check if harvest proposals are based on dead snapshot data ───────────
print("\n=== Snapshot keys for phantom symbols ===")
for sym in PHANTOMS:
    snap_key = f"quantum:position:snapshot:{sym}"
    if r.exists(snap_key):
        data = {k.decode(): v.decode() for k, v in r.hgetall(snap_key).items()}
        entry = data.get('entry_price', data.get('open_price', '?'))
        qty = data.get('quantity', data.get('qty', '?'))
        side = data.get('side', '?')
        print(f"  SNAPSHOT {sym}: side={side} qty={qty} entry={entry}")
        # Delete the snapshot too since it's a phantom
        r.delete(snap_key)
        print(f"  → DELETED snapshot {snap_key}")
    else:
        print(f"  {sym}: no snapshot key")

# ─── If harvest engine reads from a different position key, find it ────────
print("\n=== Alternative position storage keys ===")
result4 = subprocess.run(
    ["grep", "-rn", "redis.*get\|hget\|hgetall", "/home/qt/quantum_trader/services/"],
    capture_output=True, text=True
)
# Look for lines that read position-like keys
for line in result4.stdout.splitlines():
    if ('position' in line.lower() or 'pos' in line.lower()) and 'harvest' in line.lower():
        print(f"  {line.strip()[:120]}")

# ─── Wait and verify ─────────────────────────────────────────────────────
print("\n  Waiting 90s to see if proposals regenerate...")
time.sleep(90)

print("\n=== Verify: harvest proposals after 90s ===")
still_exists = []
for sym in PHANTOMS:
    if r.exists(f"quantum:harvest:proposal:{sym}"):
        data = {k.decode(): v.decode() for k, v in r.hgetall(f"quantum:harvest:proposal:{sym}").items()}
        comp_at = data.get('computed_at_utc', '?')
        print(f"  REGENERATED: {sym} computed_at={comp_at}")
        still_exists.append(sym)
    else:
        print(f"  ✅ {sym}: no harvest proposal")

print(f"\n=== Binance errors in last 2 min ===")
cutoff_ms = (int(time.time()) - 120) * 1000
entries = r.xrange("quantum:stream:apply.result", min=str(cutoff_ms))
errors = [(m.decode(), d) for m, d in entries
          if b'Binance API error' in d.get(b'error', b'') or b'execution_failed' in d.get(b'error', b'')]
print(f"  {len(errors)} Binance errors in last 2 min")
if errors:
    for mid, d in errors[:5]:
        sym = d.get(b'symbol', b'?').decode()
        err = d.get(b'error', b'').decode()
        import re
        code = re.search(r'"code"\s*:\s*(-?\d+)', err)
        print(f"  {sym} code={code.group(1) if code else '?'}")
