"""Check apply.plan stream for BTC/LINK close signals and close_qty."""
import redis

r = redis.Redis(decode_responses=True)
msgs = r.xrevrange("quantum:stream:apply.plan", count=500)

found = {}
for mid, f in msgs:
    sym = f.get("symbol", "")
    if sym in ("BTCUSDT", "LINKUSDT") and sym not in found:
        found[sym] = f

for sym, f in found.items():
    print(f"{sym}: action={f.get('action')} decision={f.get('decision')} "
          f"close_qty={f.get('close_qty','?')} price={f.get('price','?')} "
          f"R_net={f.get('R_net','?')}")

# Also check execution engine last orders
print()
for sym in ("BTCUSDT", "LINKUSDT"):
    last = r.hgetall(f"quantum:execution:last_order:{sym}")
    if last:
        print(f"last_order {sym}: {last}")
    pos = r.hgetall(f"quantum:state:positions:{sym}")
    if pos:
        print(f"position {sym}: qty={pos.get('quantity')} entry={pos.get('entry_price')} upnl={pos.get('unrealized_pnl')}")
