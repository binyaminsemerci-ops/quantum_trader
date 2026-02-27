#!/usr/bin/env python3
"""
Option B: Clean phantom Redis positions (not on Binance)
Option C: Re-enable manual lane (4h TTL)
"""
import redis, json, time, subprocess

r = redis.Redis()

# ─── B: Identify phantom symbols from Binance -2022/-1111 errors ──────────
# From diagnostics: SOLUSDT, BTCUSDT, BNBUSDT (-2022), XRPUSDT, INJUSDT (-1111)
PHANTOM_SYMBOLS = {'SOLUSDT', 'BTCUSDT', 'BNBUSDT', 'XRPUSDT', 'INJUSDT'}

print("=== B: Clean phantom positions ===")

# Find all Redis position keys
pos_keys = r.keys("quantum:position:*")
print(f"Total position keys: {len(pos_keys)}")

phantom_keys = []
for key in pos_keys:
    key_str = key.decode()
    symbol = key_str.split(":")[-1]
    if symbol in PHANTOM_SYMBOLS:
        phantom_keys.append(key)

print(f"Phantom position keys to delete: {len(phantom_keys)}")
for key in phantom_keys:
    data = r.get(key)
    if data:
        try:
            pos = json.loads(data)
            qty = pos.get('quantity', pos.get('qty', '?'))
            side = pos.get('side', pos.get('direction', '?'))
            ep = pos.get('entry_price', '?')
            print(f"  DELETE: {key.decode()} | side={side} qty={qty} entry={ep}")
        except:
            print(f"  DELETE: {key.decode()} (raw: {data[:60]})")

if phantom_keys:
    confirm = input(f"\nDelete {len(phantom_keys)} phantom position keys? [y/N]: ").strip().lower()
    if confirm == 'y':
        for key in phantom_keys:
            r.delete(key)
        print(f"  ✅ Deleted {len(phantom_keys)} phantom positions")
    else:
        print("  Skipped deletion")
else:
    # Auto-delete if running non-interactively
    print("  No phantom keys found under quantum:position:<symbol>")
    # Check alternative key patterns
    alt_patterns = [
        "quantum:positions:*",
        "quantum:open_positions:*",
        "position:*",
    ]
    for pat in alt_patterns:
        keys = r.keys(pat)
        if keys:
            print(f"  Found {len(keys)} keys matching {pat}")
            for k in keys[:5]:
                print(f"    {k.decode()}")

# Also clean up persistent apply.result close-loop entries
print("\n=== B: Check apply_layer active_plan keys for phantom symbols ===")
plan_keys = r.keys("quantum:active_plan:*") + r.keys("quantum:plan:*") + r.keys("quantum:harvest_plan:*")
phantom_plan_keys = []
for key in plan_keys:
    key_str = key.decode()
    for sym in PHANTOM_SYMBOLS:
        if sym in key_str:
            phantom_plan_keys.append(key)
            break

print(f"  Phantom plan keys: {len(phantom_plan_keys)}")
for key in phantom_plan_keys[:10]:
    ttl = r.ttl(key)
    print(f"    {key.decode()} | TTL={ttl}s")

# ─── C: Re-enable manual lane (4h TTL) ───────────────────────────────────
print("\n=== C: Re-enable manual lane ===")

# Check current manual lane state
manual_keys = r.keys("quantum:manual_lane*") + r.keys("*manual*lane*") + r.keys("*manual*active*")
print(f"Current manual lane keys: {len(manual_keys)}")
for k in manual_keys:
    val = r.get(k) or r.hgetall(k)
    ttl = r.ttl(k)
    print(f"  {k.decode()} | val={val} | TTL={ttl}s")

# Re-activate manual lane with 4h TTL (same as C3 activation)
MANUAL_LANE_KEY = "quantum:manual_lane:active"
TTL_4H = 4 * 3600

r.set(MANUAL_LANE_KEY, "true", ex=TTL_4H)
print(f"\n  ✅ Set {MANUAL_LANE_KEY} = true | TTL={TTL_4H}s (4h)")

# Also check if there's a separate config key for intent-executor
for key_pattern in [
    "quantum:config:manual_lane",
    "quantum:intent:manual_lane_enabled",
    "quantum:manual_close:enabled",
    "quantum:lane:manual:active",
    "manual_lane_enabled",
]:
    if r.exists(key_pattern):
        old = r.get(key_pattern)
        r.set(key_pattern, "true", ex=TTL_4H)
        print(f"  ✅ Also updated: {key_pattern} | was={old}")

# Check intent-executor log for how manual lane is checked
result = subprocess.run(
    ["grep", "-n", "manual_lane", "/home/qt/quantum_trader/services/intent_executor.py"],
    capture_output=True, text=True
)
if result.returncode == 0:
    for line in result.stdout.splitlines()[:10]:
        print(f"  [intent-executor code] {line.strip()[:120]}")
else:
    # Try finding the file
    result2 = subprocess.run(
        ["find", "/home/qt/quantum_trader", "-name", "intent_executor*", "-not", "-path", "*/.*"],
        capture_output=True, text=True
    )
    print(f"  intent_executor files: {result2.stdout.strip()[:200]}")
    # Search in all py files
    result3 = subprocess.run(
        ["grep", "-rn", "manual_lane", "/home/qt/quantum_trader/services/", "--include=*.py"],
        capture_output=True, text=True
    )
    for line in result3.stdout.splitlines()[:10]:
        print(f"  [code] {line.strip()[:120]}")

# ─── Verify & summary ─────────────────────────────────────────────────────
print("\n=== Summary ===")
remaining_positions = [k for k in r.keys("quantum:position:*") if k.decode().split(":")[-1] in PHANTOM_SYMBOLS]
print(f"Phantom positions remaining: {len(remaining_positions)}")

ml_val = r.get(MANUAL_LANE_KEY)
ml_ttl = r.ttl(MANUAL_LANE_KEY)
ml_expire = time.strftime('%H:%M UTC', time.gmtime(time.time() + ml_ttl)) if ml_ttl > 0 else "no expiry"
print(f"Manual lane: {ml_val.decode() if ml_val else 'NOT SET'} | expires at ~{ml_expire}")
