#!/usr/bin/env python3
"""Check feature publisher health and stream data."""
import sys
import subprocess
import redis

sys.path.insert(0, '/home/qt/quantum_trader')
r = redis.Redis()

print("=== FEATURE PUBLISHER STATUS ===")
result = subprocess.run(
    ['systemctl', 'status', 'quantum-feature-publisher', '--no-pager', '-n', '5'],
    capture_output=True, text=True
)
# Find active/PID lines
for line in result.stdout.splitlines():
    if any(k in line for k in ['Active', 'PID', 'Memory', 'CGroup']):
        print(' ', line.strip())

print("\n=== FEATURE KEYS ===")
feat_keys = r.keys("quantum:features:*")
print(f"  quantum:features:* keys: {len(feat_keys)}")
for k in feat_keys[:5]:
    print(f"  {k.decode()}")

print("\n=== FEATURE STREAM ===")
# Check various possible stream keys
for key in ['quantum:stream:features', 'quantum:stream:feature', 'features', 'quantum:features']:
    try:
        length = r.xlen(key)
        entries = r.xrevrange(key, count=2)
        print(f"  {key}: {length} entries")
        for eid, fields in entries:
            sym = fields.get(b'symbol', b'?').decode()
            ts = eid.decode()[:13]
            rsival = fields.get(b'rsi', b'?').decode()
            print(f"    {ts} sym={sym} rsi={rsival}")
    except Exception as e:
        pass  # Key doesn't exist or not a stream

print("\n=== HARVEST BRAIN CURRENT STATE ===")
# Check harvest brain process
result2 = subprocess.run(
    ['systemctl', 'status', 'quantum-harvest-brain', '--no-pager', '-n', '3'],
    capture_output=True, text=True
)
for line in result2.stdout.splitlines():
    if any(k in line for k in ['Active', 'PID', 'CGroup', 'uptime', 'error', 'warn']):
        print(' ', line.strip())

# Check what features harvest brain actually uses
print("\n=== SAMPLE PROPOSAL DETAILS (BTCUSDT) ===")
data = r.hgetall("quantum:harvest:proposal:BTCUSDT")
for k, v in sorted(data.items()):
    print(f"  {k.decode()}: {v.decode()[:80]}")
