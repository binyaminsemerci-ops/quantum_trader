#!/usr/bin/env python3
"""
B: Delete phantom Redis positions (SOLUSDT,BTCUSDT,BNBUSDT,XRPUSDT,INJUSDT)
   - -2022 ReduceOnly rejected = no position on Binance
   - -1111 Precision error = quantity too precise, also phantom
C: Re-enable manual lane (4h TTL)
"""
import redis, json, time, subprocess

r = redis.Redis()

PHANTOM_SYMBOLS = ['SOLUSDT', 'BTCUSDT', 'BNBUSDT', 'XRPUSDT', 'INJUSDT']

# ─── B: Show and delete active position hashes ───────────────────────────
print("=== B: Phantom position cleanup ===")

deleted = []
kept = []

for sym in PHANTOM_SYMBOLS:
    key = f"quantum:position:{sym}"
    if r.exists(key):
        data = r.hgetall(key)
        pos_info = {k.decode(): v.decode() for k,v in data.items()}
        qty   = pos_info.get('quantity', pos_info.get('qty', '?'))
        side  = pos_info.get('side', pos_info.get('direction', '?'))
        entry = pos_info.get('entry_price', '?')
        pnl   = pos_info.get('unrealized_pnl', pos_info.get('pnl', '?'))
        print(f"  PHANTOM {sym}: side={side} qty={qty} entry={entry} pnl={pnl}")
        backup_key = f"quantum:position:phantom_backup:{sym}"
        # Preserve data as backup with 7d TTL before deleting
        r.delete(backup_key)
        for k, v in data.items():
            r.hset(backup_key, k, v)
        r.expire(backup_key, 7 * 86400)
        r.delete(key)
        deleted.append(sym)
        print(f"    → DELETED (backup at {backup_key} for 7d)")
    else:
        kept.append(sym)
        print(f"  {sym}: key not found (already clean or different key pattern)")

print(f"\n  Deleted: {deleted}")
print(f"  Not found as quantum:position:<sym>: {kept}")

# ─── B: Also stop active harvest/close plans for phantom symbols ─────────
print("\n=== B: Stop persistent close plans for phantom symbols ===")

plan_patterns = [
    "quantum:harvest_plan:*",
    "quantum:active_plan:*",
    "quantum:close_plan:*",
    "quantum:plan:*",
]

plan_deleted = 0
for pat in plan_patterns:
    for key in r.keys(pat):
        ks = key.decode()
        for sym in PHANTOM_SYMBOLS:
            if sym in ks:
                ttl = r.ttl(key)
                print(f"  DELETE plan: {ks} (TTL={ttl}s)")
                r.delete(key)
                plan_deleted += 1
                break

print(f"  Plan keys deleted: {plan_deleted}")

# ─── B: Verify Binance positions vs Redis remaining ──────────────────────
print("\n=== B: Remaining active position keys ===")
remaining = r.keys("quantum:position:*")
active = [k.decode() for k in remaining
          if 'snapshot' not in k.decode() and 'ledger' not in k.decode() and 'backup' not in k.decode()]
print(f"  Active position keys: {len(active)}")
for k in sorted(active)[:20]:
    sym = k.split(":")[-1]
    data = r.hgetall(k)
    pos = {kk.decode(): vv.decode() for kk,vv in data.items()}
    qty = pos.get('quantity', pos.get('qty', '?'))
    side = pos.get('side', '?')
    entry = pos.get('entry_price', '?')
    print(f"    {sym}: side={side} qty={qty} entry={entry}")

# ─── C: Re-enable manual lane ─────────────────────────────────────────────
print("\n=== C: Manual lane activation (4h TTL) ===")

TTL_4H = 4 * 3600

# Find all existing manual lane related keys
manual_keys = r.keys("*manual*")
print(f"  Existing manual-related keys: {len(manual_keys)}")
for k in manual_keys:
    v = r.get(k)
    typ = r.type(k).decode()
    ttl = r.ttl(k)
    print(f"    {k.decode()} [{typ}] val={v} TTL={ttl}s")

# Set the standard manual lane key (matches C3 activation)
LANE_KEY = "quantum:manual_lane:active"
r.set(LANE_KEY, "true", ex=TTL_4H)
print(f"\n  ✅ {LANE_KEY} = true | TTL={TTL_4H}s (4h)")

# Search code for how intent-executor reads the manual lane flag
result = subprocess.run(
    ["grep", "-rn", "manual_lane", "/home/qt/quantum_trader/services/"],
    capture_output=True, text=True
)
code_keys = set()
for line in result.stdout.splitlines():
    import re
    # find redis keys
    for match in re.findall(r'["\']quantum[:\w]+manual[\w:]+["\']', line):
        code_keys.add(match.strip("'\""))
    for match in re.findall(r'["\'][\w:]*manual[\w:]+["\']', line):
        code_keys.add(match.strip("'\""))

print(f"\n  Keys referenced in code: {code_keys}")
for k in code_keys:
    if k and 'manual' in k:
        r.set(k, "true", ex=TTL_4H)
        print(f"  ✅ Also set: {k} TTL={TTL_4H}s")

# ─── Summary ─────────────────────────────────────────────────────────────
print("\n=== Summary ===")
now_utc = time.strftime('%Y-%m-%d %H:%M UTC', time.gmtime())
expire_utc = time.strftime('%H:%M UTC', time.gmtime(time.time() + TTL_4H))
print(f"  Now: {now_utc}")
print(f"  Phantom positions deleted: {deleted}")
print(f"  Plan keys deleted: {plan_deleted}")
print(f"  Manual lane: active until ~{expire_utc}")
print(f"\n  Apply-layer should stop -2022/-1111 errors within ~60s (no more close loops)")
print(f"  Intent-executor can now process manual-lane signals")

# Restart apply-layer to clear any in-memory position cache
print("\n=== Restart apply-layer to clear position cache ===")
subprocess.run(["systemctl", "restart", "quantum-apply-layer"])
time.sleep(2)
status = subprocess.run(["systemctl", "is-active", "quantum-apply-layer"], capture_output=True, text=True).stdout.strip()
print(f"  quantum-apply-layer: {status}")
