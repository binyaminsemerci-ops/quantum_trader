#!/usr/bin/env python3
import redis, time, re
from collections import Counter
r = redis.Redis()
print("Waiting 120s for clean window...")
time.sleep(120)
cutoff_ms = (int(time.time()) - 120) * 1000
entries = r.xrange("quantum:stream:apply.result", min=str(cutoff_ms))
errors = []
for _, d in entries:
    err = d.get(b"error", b"").decode()
    if err:
        sym = d.get(b"symbol", b"?").decode()
        m = re.search(r"-[0-9]{4}", err)
        if m:
            errors.append((sym, m.group(0)))
print(f"Total apply.result entries: {len(entries)}")
print("Binance errors last 2 min:")
if errors:
    for (s, c), n in Counter(errors).most_common():
        print(f"  {s} {c} x{n}")
else:
    print("  ✅ ZERO Binance errors")
# Check phantom proposals
print("\nPhantom proposal keys:")
for sym in ["SOLUSDT","BTCUSDT","BNBUSDT","XRPUSDT","INJUSDT"]:
    key = f"quantum:harvest:proposal:{sym}"
    exists = r.exists(key)
    print(f"  {sym}: {'EXISTS ❌' if exists else 'GONE ✅'}")
