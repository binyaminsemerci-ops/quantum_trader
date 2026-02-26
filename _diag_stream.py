#!/usr/bin/env python3
"""See Redis stream structure and AI Engine signal publisher"""
import os, redis, json, time

# 1. Read full entry_scanner.py lines 95-200
with open("microservices/autonomous_trader/entry_scanner.py") as f:
    lines = f.readlines()
print("=== entry_scanner.py lines 95-200 ===")
for i, l in enumerate(lines[94:200], 95):
    print(f"{i:4d}: {l}", end="")

# 2. Check Redis stream
r = redis.Redis(unix_socket_path='/var/run/redis/redis.sock' if os.path.exists('/var/run/redis/redis.sock') else None,
                host='127.0.0.1', port=6379, decode_responses=True)
stream_key = "quantum:stream:ai.signal_generated"
try:
    entries = r.xrevrange(stream_key, count=5)
    print(f"\n=== Last 5 entries in {stream_key} ===")
    for entry_id, data in entries:
        print(f"  ID:{entry_id}  data:{data}")
    if not entries:
        print("  (empty stream)")
except Exception as e:
    print(f"\nRedis error: {e}")

# 3. Find where AI Engine publishes signals
for base in ["/home/qt/quantum_trader", "/opt/quantum"]:
    for dirpath, _, files in os.walk(f"{base}/microservices/ai_engine"):
        for fname in files:
            if fname.endswith('.py'):
                fpath = os.path.join(dirpath, fname)
                try:
                    with open(fpath) as f:
                        code = f.read()
                    if 'signal_generated' in code or 'xadd' in code.lower():
                        print(f"\n=== Publisher: {fpath} ===")
                        for i, line in enumerate(code.splitlines(), 1):
                            if any(w in line.lower() for w in ['signal_generated', 'xadd', 'confidence', 'action', 'regime']):
                                print(f"  {i:4d}: {line.strip()}")
                except:
                    pass
