#!/usr/bin/env python3
"""
Clean up phantom/stale Redis position data.
Removes position and ledger keys for symbols that have no position on exchange (snapshot = 0).
"""
import redis
import sys

r = redis.Redis(host='localhost', port=6379, decode_responses=True)

print("=== Phantom Position Cleanup ===\n")

# All symbols to check — scan canonical position keys
all_syms = set()
for k in r.keys("quantum:state:positions:*"):
    sym = k.replace("quantum:state:positions:", "")
    all_syms.add(sym)

print(f"Symbols with position data in Redis: {sorted(all_syms)}\n")

deleted = []
kept = []

for sym in sorted(all_syms):
    can_amt = r.hget(f"quantum:state:positions:{sym}", "position_amt")
    can_side = r.hget(f"quantum:state:positions:{sym}", "side")
    can_src = r.hget(f"quantum:state:positions:{sym}", "source")
    can_qty = r.hget(f"quantum:state:positions:{sym}", "quantity")
    ledger_qty = r.hget(f"quantum:position:ledger:{sym}", "qty")

    print(f"{sym}:")
    print(f"  canonical: amt={can_amt} side={can_side} qty={can_qty} source={can_src}")
    print(f"  ledger:    qty={ledger_qty}")

    # Determine if phantom: canonical shows 0 or FLAT
    is_flat = can_amt is None or float(can_amt or 0) == 0.0

    if is_flat:
        # Delete stale canonical + ledger + legacy keys
        cdel = r.delete(f"quantum:state:positions:{sym}")
        ldel = r.delete(f"quantum:position:ledger:{sym}")
        pdel = r.delete(f"quantum:position:{sym}")
        sdel = r.delete(f"quantum:position:snapshot:{sym}")
        if cdel or ldel or pdel or sdel:
            print(f"  -> DELETED (phantom: canonical={cdel} ledger={ldel} position={pdel} snapshot={sdel})")
            deleted.append(sym)
        else:
            print(f"  -> Already clean (no keys to delete)")
    else:
        print(f"  -> KEPT (canonical shows real position: amt={can_amt} side={can_side})")
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
