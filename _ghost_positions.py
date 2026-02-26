#!/usr/bin/env python3
"""
Diagnose + optionally fix ghost position hashes.
Checks each quantum:position:SYMBOL (no :snapshot: or :ledger:) for quantity > 0.
Shows what would be zeroed.
"""
import redis
import asyncio

r = redis.Redis(host='localhost', port=6379, decode_responses=True)

print("=== DIRECT POSITION HASHES (quantum:position:SYMBOL) ===")
print("Skipping :snapshot: and :ledger: namespaces\n")

ghost_keys = []
open_keys = []

cursor = 0
while True:
    cursor, keys = r.scan(cursor, match="quantum:position:*", count=200)
    for key in keys:
        if ":snapshot:" in key or ":ledger:" in key or ":phantom_backup" in key:
            continue
        qty_raw = r.hget(key, "quantity")
        status = r.hget(key, "status") or "(no status)"
        symbol = key.replace("quantum:position:", "")
        
        if qty_raw is not None:
            try:
                qty = float(qty_raw)
                if abs(qty) > 0:
                    print(f"  OPEN qty={qty:+.4f}  status={status}  key={key}")
                    open_keys.append(key)
                else:
                    print(f"  zero qty={qty}  status={status}  key={key}")
            except:
                print(f"  UNPARSEABLE qty={qty_raw}  key={key}")
        else:
            print(f"  NO-QTY  status={status}  key={key}")
            ghost_keys.append(key)
    
    if cursor == 0:
        break

print(f"\n=== SUMMARY ===")
print(f"Total with qty > 0: {len(open_keys)}")
print(f"Total with no qty: {len(ghost_keys)}")
print(f"\nOpen position keys: {open_keys}")
