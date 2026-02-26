import redis, time
r = redis.Redis(decode_responses=True)

DEPLOY_TS = 1771972375000  # 00:32 UTC after restart

msgs = r.xrevrange("quantum:stream:apply.plan", count=3000)

# All action types
from collections import Counter
ctr = Counter(f.get("action","?") for _, f in msgs)
print("Action distribution (last 3000):")
for k, v in ctr.most_common():
    print(f"  {k:<35} {v}")

# Check ENTRY_PROPOSED
entry_msgs = [(mid, f) for mid, f in msgs if "ENTRY" in f.get("action", "")]
print(f"\nENTRY_PROPOSED found: {len(entry_msgs)}")
for mid, f in entry_msgs[:5]:
    ts   = time.strftime("%H:%M:%S", time.gmtime(int(mid.split("-")[0])/1000))
    post = "POST" if int(mid.split("-")[0]) > DEPLOY_TS else "PRE"
    ep   = f.get("entry_price",       "MISSING!")
    atr  = f.get("atr_value",         "MISSING!")
    vol  = f.get("volatility_factor", "MISSING!")
    brk  = f.get("breakeven_price",   "MISSING!")
    sym  = f.get("symbol", "?")
    side = f.get("side", "?")
    print(f"\n  [{post}-deploy @ {ts} UTC] {sym} {side}")
    print(f"    entry_price       = {ep!r:<22} {'OK' if ep != 'MISSING!' else '!!! BROKEN'}")
    print(f"    atr_value         = {atr!r:<22} {'OK' if atr != 'MISSING!' else '!!! BROKEN'}")
    print(f"    volatility_factor = {vol!r:<22} {'OK' if vol != 'MISSING!' else '!!! BROKEN'}")
    print(f"    breakeven_price   = {brk!r:<22} {'OK' if brk != 'MISSING!' else '!!! BROKEN'}")

# Also look at intent_bridge trade.intent side to see raw signals
intents = r.xrevrange("quantum:stream:trade.intent", count=10)
print(f"\ntrade.intent recent messages: {len(intents)}")
for mid, f in intents[:2]:
    ts = time.strftime("%H:%M:%S", time.gmtime(int(mid.split("-")[0]) / 1000))
    print(f"  [{ts}] {f.get('symbol')} {f.get('side')} entry_price={f.get('entry_price','?')!r}  atr_value={f.get('atr_value','?')!r}")
