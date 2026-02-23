#!/usr/bin/env python3
"""
Clean up phantom/stale Redis position data.
Removes position and ledger keys for symbols that have no position on exchange (snapshot = 0).
"""
import redis
import sys

r = redis.Redis(host='localhost', port=6379, decode_responses=True)

print("=== Phantom Position Cleanup ===\n")

# All symbols to check
all_syms = set()
for k in r.keys("quantum:position:*"):
    if "snapshot" in k or "cooldown" in k:
        continue
    # Extract symbol from key like quantum:position:SOLUSDT or quantum:position:ledger:SOLUSDT
    parts = k.split(":")
    sym = parts[-1]
    all_syms.add(sym)

print(f"Symbols with position data in Redis: {sorted(all_syms)}\n")

deleted = []
kept = []

for sym in sorted(all_syms):
    snap_amt = r.hget(f"quantum:position:snapshot:{sym}", "position_amt")
    snap_side = r.hget(f"quantum:position:snapshot:{sym}", "side")
    pos_qty_str = r.hget(f"quantum:position:{sym}", "quantity")
    pos_src = r.hget(f"quantum:position:{sym}", "source")
    ledger_qty = r.hget(f"quantum:position:ledger:{sym}", "qty")

    print(f"{sym}:")
    print(f"  snapshot: amt={snap_amt} side={snap_side}")
    print(f"  position: qty={pos_qty_str} source={pos_src}")
    print(f"  ledger:   qty={ledger_qty}")

    # Determine if phantom: snapshot shows 0 or no snapshot
    snap_is_zero = snap_amt is None or float(snap_amt or 0) == 0.0

    if snap_is_zero:
        # Delete stale ledger key (reconcile-engine phantom)
        ldel = r.delete(f"quantum:position:ledger:{sym}")
        # Delete stale position key (harvest_brain_startup_sync phantom)
        pdel = r.delete(f"quantum:position:{sym}")
        if ldel or pdel:
            print(f"  -> DELETED (phantom: ledger={ldel}, position={pdel})")
            deleted.append(sym)
        else:
            print(f"  -> Already clean (no keys to delete)")
    else:
        print(f"  -> KEPT (snapshot shows real position: amt={snap_amt} side={snap_side})")
        kept.append(sym)
    print()

print(f"\n=== Summary ===")
print(f"Deleted phantom data for: {deleted}")
print(f"Kept real positions for:  {kept}")

# Also clear any dedupe keys that may be blocking re-execution
dedupe_count = 0
for k in r.keys("quantum:apply:done:*"):
    r.delete(k)
    dedupe_count += 1
print(f"\nCleared {dedupe_count} dedupe/idempotency keys (allow fresh re-execution)")

# Clear cooldown keys
cooldown_count = 0
for k in r.keys("quantum:cooldown:open:*"):
    r.delete(k)
    cooldown_count += 1
print(f"Cleared {cooldown_count} open-cooldown keys")

print("\nDone.")
