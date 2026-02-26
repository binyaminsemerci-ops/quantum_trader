#!/usr/bin/env python3
"""
Fix ghost position hashes: zero out quantity on all quantum:position:SYMBOL
hashes that have no status field (legacy format, never cleaned up on close).
"""
import redis

r = redis.Redis(host='localhost', port=6379, decode_responses=True)

print("=== ZEROING GHOST POSITION QUANTITIES ===\n")

fixed = []
cursor = 0
while True:
    cursor, keys = r.scan(cursor, match="quantum:position:*", count=200)
    for key in keys:
        if ":snapshot:" in key or ":ledger:" in key or ":phantom_backup" in key:
            continue
        
        status = r.hget(key, "status")
        qty_raw = r.hget(key, "quantity")
        
        if status is None and qty_raw is not None:
            try:
                qty = float(qty_raw)
                if abs(qty) > 0:
                    r.hset(key, "quantity", "0")
                    r.hset(key, "status", "CLOSED_GHOST")
                    print(f"  FIXED: {key}  qty {qty} → 0  status=CLOSED_GHOST")
                    fixed.append(key)
            except:
                pass
    
    if cursor == 0:
        break

print(f"\n=== SUMMARY ===")
print(f"Fixed {len(fixed)} ghost position hashes")
print(f"\nNow verifying authoritative count would be...")

# Re-verify
count = 0
cursor = 0
while True:
    cursor, keys = r.scan(cursor, match="quantum:position:*", count=200)
    for key in keys:
        if ":snapshot:" in key or ":ledger:" in key or ":phantom_backup" in key:
            continue
        qty_raw = r.hget(key, "quantity")
        if qty_raw is not None:
            try:
                if abs(float(qty_raw)) > 0:
                    count += 1
            except:
                pass
    if cursor == 0:
        break

print(f"Post-fix authoritative_count = {count}  (was 22, should be 0)")
