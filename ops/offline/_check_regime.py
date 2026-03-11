import redis
r = redis.Redis(decode_responses=True)
keys = list(r.scan_iter("*regime*", count=50))
print("REGIME KEYS:", keys[:20])
keys2 = list(r.scan_iter("*market*state*", count=50))
print("MARKET STATE KEYS:", keys2[:10])
keys3 = list(r.scan_iter("quantum:market:*", count=50))
print("MARKET KEYS:", keys3[:15])
