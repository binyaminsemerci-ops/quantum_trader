import redis
r = redis.Redis(decode_responses=True)
msgs = r.xrevrange('quantum:stream:apply.plan', count=200)
for mid, f in msgs:
    if f.get('symbol') == 'LINKUSDT' and f.get('source') == 'risk_brake_v1':
        print(f"ID: {mid}")
        print("All fields:")
        for k, v in f.items():
            print(f"  {k!r}: {v!r}")
        break
