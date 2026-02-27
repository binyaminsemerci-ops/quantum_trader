import redis, json, time
r = redis.Redis(host='127.0.0.1', port=6379, decode_responses=True)
msgs = r.xrevrange('quantum:stream:ai.signal_generated', count=8)
now = time.time()
print(f"Last {len(msgs)} signals:")
for mid, d in msgs:
    p = json.loads(d.get('payload', '{}'))
    age = int(now - int(mid.split('-')[0]) / 1000)
    src = d.get('source', '?')
    print(f"  {mid[:20]}  {p.get('symbol','?'):18s}  {p.get('action','?'):4s}  conf={p.get('confidence',0):.2f}  age={age}s  src={src}")
    if age > 300:
        print("    *** TOO OLD - scanner will skip ***")
