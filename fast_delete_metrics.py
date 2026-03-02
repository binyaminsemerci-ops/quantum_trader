#!/usr/bin/env python3
"""Fast delete of quantum:metrics:exit:* keys using SCAN + pipeline batches of 500"""
import redis
import time

r = redis.Redis(unix_socket_path='/var/run/redis/redis-server.sock', decode_responses=True)
# fallback to TCP if socket not found
try:
    r.ping()
except:
    r = redis.Redis(host='127.0.0.1', port=6379, decode_responses=True)

pattern = 'quantum:metrics:exit:*'
deleted = 0
start = time.time()

cursor = 0
while True:
    cursor, keys = r.scan(cursor, match=pattern, count=2000)
    if keys:
        pipe = r.pipeline(transaction=False)
        for k in keys:
            pipe.delete(k)
        pipe.execute()
        deleted += len(keys)
        elapsed = time.time() - start
        print(f"  Deleted {deleted} (rate: {deleted/elapsed:.0f}/s)", flush=True)
    if cursor == 0:
        break

remaining = r.dbsize()
print(f"\nDone: {deleted} keys deleted in {time.time()-start:.1f}s")
print(f"Redis total keys now: {remaining}")
mem = r.info('memory')
print(f"Redis memory: {mem['used_memory_human']}")
