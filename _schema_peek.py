#!/usr/bin/env python3
"""Peek at execution.result and apply.result schemas on VPS"""
import redis, json

r = redis.Redis(decode_responses=True)

print("=== execution.result (last 3) ===")
msgs = r.xrevrange("quantum:stream:execution.result", count=3)
for mid, fields in msgs:
    print(f"\n[{mid}]")
    for k, v in fields.items():
        if k == "payload":
            try:
                p = json.loads(v)
                print(f"  payload (decoded):")
                for pk, pv in p.items():
                    print(f"    {pk}: {str(pv)[:100]}")
            except:
                print(f"  payload: {str(v)[:200]}")
        else:
            print(f"  {k}: {str(v)[:100]}")

print("\n=== apply.result (last 3) ===")
msgs = r.xrevrange("quantum:stream:apply.result", count=3)
for mid, fields in msgs:
    print(f"\n[{mid}]")
    for k, v in fields.items():
        if k in ("steps_results", "error"):
            try:
                p = json.loads(v) if v else {}
                print(f"  {k} (decoded): {json.dumps(p, indent=2)[:300]}")
            except:
                print(f"  {k}: {str(v)[:200]}")
        else:
            print(f"  {k}: {str(v)[:100]}")

print("\n=== apply.plan ENTRY_PROPOSED (last 3) ===")
msgs = r.xrevrange("quantum:stream:apply.plan", count=200)
found = 0
for mid, fields in msgs:
    if fields.get("action") == "ENTRY_PROPOSED":
        print(f"\n[{mid}]")
        for k, v in fields.items():
            print(f"  {k}: {str(v)[:100]}")
        found += 1
        if found >= 3:
            break

print("\n=== quantum:position:* sample (3 real positions) ===")
all_pos = r.keys("quantum:position:*")
real = [k for k in all_pos if not any(x in k for x in ["snapshot", "ledger", "claim", "cooldown"])]
for k in real[:3]:
    t = r.type(k)
    print(f"\n{k} [{t}]")
    if t == "hash":
        v = r.hgetall(k)
        for fk, fv in v.items():
            print(f"  {fk}: {str(fv)[:100]}")
    elif t == "string":
        v = r.get(k)
        try:
            p = json.loads(v)
            for fk, fv in (p.items() if isinstance(p, dict) else []):
                print(f"  {fk}: {str(fv)[:100]}")
        except:
            print(f"  {str(v)[:300]}")
