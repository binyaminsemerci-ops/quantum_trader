import redis
r = redis.Redis(decode_responses=True)
keys = r.keys('quantum:features:*')
key = keys[0] if keys else None
print('KEY:', key)
if key:
    data = r.hgetall(key)
    print('NUM KEYS:', len(data))
    print('ALL KEYS:', sorted(data.keys()))
    for f in ['obv','obv_ema','vpt','momentum_5','momentum_10','momentum_20','acceleration','volume_ratio']:
        v = data.get(f, 'MISSING')
        print(f'  {f} = {v}')
