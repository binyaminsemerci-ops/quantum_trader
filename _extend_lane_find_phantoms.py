#!/usr/bin/env python3
"""
Extend manual lane TTL + investigate remaining phantom close-loop symbols
"""
import redis, json, time, subprocess

r = redis.Redis()
TTL_4H = 4 * 3600

# ─── C: Extend manual lane to 4h from now ─────────────────────────────────
print("=== C: Manual lane TTL extension ===")
LANE_KEY = "quantum:manual_lane:enabled"
old_ttl = r.ttl(LANE_KEY)
r.set(LANE_KEY, "1", ex=TTL_4H)
new_ttl = r.ttl(LANE_KEY)
expire_utc = time.strftime('%H:%M UTC', time.gmtime(time.time() + new_ttl))
print(f"  {LANE_KEY}: TTL {old_ttl}s → {new_ttl}s (expires ~{expire_utc})")

# Also check any hash/list type manual keys
manual_keys = r.keys("*manual*")
for k in manual_keys:
    ks = k.decode()
    typ = r.type(k).decode()
    ttl = r.ttl(k)
    if typ == 'string':
        val = r.get(k).decode()
        print(f"  {ks} [{typ}] = {val!r} TTL={ttl}s")
    elif typ == 'hash':
        val = {kk.decode(): vv.decode() for kk, vv in r.hgetall(k).items()}
        print(f"  {ks} [hash] = {val} TTL={ttl}s")
    else:
        print(f"  {ks} [{typ}] TTL={ttl}s")

# ─── B2: Find where apply-layer tracks SOLUSDT/BTCUSDT/BNBUSDT/XRPUSDT ──
print("\n=== B: Find close-loop sources for remaining phantom symbols ===")
REMAINING = ['SOLUSDT', 'BTCUSDT', 'BNBUSDT', 'XRPUSDT']

for sym in REMAINING:
    print(f"\n  {sym}:")
    # Look for any key containing the symbol
    for k in r.keys(f"*{sym}*"):
        ks = k.decode()
        if 'snapshot' in ks or 'ledger' in ks:
            continue
        typ = r.type(k).decode()
        ttl = r.ttl(k)
        if typ == 'hash':
            data = {kk.decode(): vv.decode() for kk, vv in r.hgetall(k).items()}
            qty = data.get('quantity', data.get('qty', data.get('size', '?')))
            side = data.get('side', data.get('direction', '?'))
            src = data.get('source', data.get('src', '?'))
            print(f"    [{typ}] {ks} | side={side} qty={qty} src={src} TTL={ttl}s")
        elif typ == 'string':
            val = r.get(k)
            print(f"    [{typ}] {ks} = {val.decode()[:60] if val else None} TTL={ttl}s")
        else:
            print(f"    [{typ}] {ks} TTL={ttl}s")

# ─── Also check apply-layer recent errors (look for the stream they read) ─
print("\n=== Apply-layer: recent close attempt error streams ===")
result = subprocess.run(
    ["grep", "-n", "FULL_CLOSE_PROPOSED\|ReduceOnly\|execute_close\|close_position",
     "/home/qt/quantum_trader/services/apply_layer.py"],
    capture_output=True, text=True
)
for line in result.stdout.splitlines()[:8]:
    print(f"  {line.strip()[:130]}")

# Check apply_layer log for close patterns
result2 = subprocess.run(
    ["tail", "-n", "20", "/var/log/quantum/apply_layer.log"],
    capture_output=True, text=True
)
if result2.returncode == 0:
    for line in result2.stdout.splitlines():
        if any(sym in line for sym in REMAINING + ['FULL_CLOSE', 'close', 'ReduceOnly']):
            print(f"  LOG: {line.strip()[-150:]}")
else:
    result3 = subprocess.run(
        ["journalctl", "-u", "quantum-apply-layer", "-n", "30", "--no-pager"],
        capture_output=True, text=True
    )
    for line in result3.stdout.splitlines():
        if any(sym in line for sym in REMAINING + ['FULL_CLOSE', 'close', 'error']):
            print(f"  LOG: {line.strip()[-150:]}")
