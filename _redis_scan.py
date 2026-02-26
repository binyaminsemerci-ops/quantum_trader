#!/usr/bin/env python3
"""Scan Redis for execution-relevant keys"""
import redis
import json

r = redis.Redis(decode_responses=True)

patterns = [
    "quantum:stream:*",
    "quantum:position:*",
    "quantum:execution:*",
    "quantum:order:*",
    "quantum:trade:*",
    "quantum:fill:*",
    "quantum:ledger:*",
    "quantum:journal:*",
]

print("=== REDIS KEY SCAN ===")
for pat in patterns:
    keys = sorted(r.keys(pat))
    if keys:
        print(f"\n[{pat}] ({len(keys)} keys)")
        for k in keys[:20]:
            t = r.type(k)
            if t == "hash":
                val = r.hgetall(k)
                print(f"  {k} [hash/{len(val)} fields]")
                # show first few fields
                for fk, fv in list(val.items())[:5]:
                    print(f"    {fk}: {str(fv)[:80]}")
            elif t == "string":
                val = r.get(k)
                print(f"  {k} = {str(val)[:100]}")
            elif t == "stream":
                length = r.xlen(k)
                print(f"  {k} [stream, len={length}]")
                # peek last few messages
                msgs = r.xrevrange(k, count=3)
                for mid, fields in msgs:
                    print(f"    [{mid}] {list(fields.keys())}")
            elif t == "list":
                length = r.llen(k)
                print(f"  {k} [list, len={length}]")
            else:
                print(f"  {k} [{t}]")
    else:
        print(f"\n[{pat}] (no keys)")

# also check for execution service journal
print("\n=== EXECUTION SERVICE STREAMS ===")
all_streams = [k for k in r.keys("*") if r.type(k) == "stream"]
for s in sorted(all_streams):
    length = r.xlen(s)
    print(f"  {s} [stream, len={length}]")
