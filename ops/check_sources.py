import redis
r = redis.Redis(decode_responses=True)
msgs = r.xrevrange('quantum:stream:apply.plan', count=200)
seen = set()
for mid, f in msgs:
    sym = f.get('symbol', '')
    if sym in ('BTCUSDT', 'LINKUSDT') and sym not in seen:
        src  = f.get('source', '?')
        act  = f.get('action', '?')
        dec  = f.get('decision', '?')
        qty  = f.get('close_qty', '?')
        lane = f.get('lane', '?')
        print(f"{sym} source={src!r} action={act} decision={dec} close_qty={qty} lane={lane}")
        seen.add(sym)
    if len(seen) >= 2:
        break

# Also: sample 5 different sources
print()
srcs = {}
for mid, f in msgs:
    src = f.get('source', 'none')
    if src not in srcs:
        srcs[src] = f.get('symbol', '?') + '/' + f.get('action', '?')
print("Sources seen in apply.plan:", srcs)
