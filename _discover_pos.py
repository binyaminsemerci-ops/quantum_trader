#!/usr/bin/env python3
import redis,json
r=redis.Redis()
patterns=["quantum:position:*","quantum:positions:*","quantum:open_position*","qt:position*","*position*"]
for p in patterns:
    keys=r.keys(p)
    if keys:
        print(f"\n{p}: {len(keys)} keys")
        for k in keys[:10]:
            ks=k.decode()
            typ=r.type(k).decode()
            print(f"  {ks} [{typ}]")

hash_key="quantum:positions"
if r.exists(hash_key):
    fields=r.hkeys(hash_key)
    print(f"\nHASH {hash_key}: {len(fields)} fields: {[f.decode() for f in fields]}")
    for f in fields:
        val=r.hget(hash_key,f)
        try: d=json.loads(val); print(f"  {f.decode()}: qty={d.get('quantity','?')} side={d.get('side','?')} entry={d.get('entry_price','?')}")
        except: print(f"  {f.decode()}: {val[:80] if val else None}")
