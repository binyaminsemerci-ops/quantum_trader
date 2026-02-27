#!/usr/bin/env python3
import redis, json
from datetime import datetime, timezone, timedelta

r = redis.Redis(host='localhost', port=6379, decode_responses=True)

print("=== RAW trade.closed stream entries ===")
entries = r.xrange("quantum:stream:trade.closed", count=10)
print(f"Total entries (all time): {r.xlen('quantum:stream:trade.closed')}")
print("")
for eid, fields in entries[-5:]:
    ts = datetime.fromtimestamp(int(eid.split('-')[0])/1000, tz=timezone.utc)
    print(f"ID: {eid}  Time: {ts}")
    for k, v in fields.items():
        if k == 'payload':
            try:
                print(f"  payload: {json.dumps(json.loads(v), indent=4)}")
            except:
                print(f"  payload (raw): {v[:500]}")
        else:
            print(f"  {k}: {v}")
    print()

print("=== KUMULATIV HARVEST METRICS ===")
result_line = None
import subprocess
out = subprocess.run(['journalctl','-u','quantum-intent-executor','-n','10','--no-pager'],
    capture_output=True, text=True)
for line in reversed(out.stdout.splitlines()):
    if 'harvest_executed=' in line:
        parts = line.split()
        for p in parts:
            if '=' in p:
                print(f"  {p}")
        break

print("\n=== HARVEST MISS BEREGNING ===")
# Count 401 HARVEST SKIP events = each one is a position that SHOULD have been closed
out2 = subprocess.run(['journalctl','-u','quantum-intent-executor','--since','24 hours ago','--no-pager'],
    capture_output=True, text=True)
logs = out2.stdout.splitlines()

harvest_success = [l for l in logs if 'HARVEST SUCCESS' in l]
harvest_skip = [l for l in logs if 'HARVEST SKIP' in l and '401' in ''.join(logs[max(0,logs.index(l)-2):logs.index(l)])]
harvest_401 = [l for l in logs if 'Unauthorized' in l or '401' in l]

# Get PnL from successful ones  
print(f"HARVEST SUCCESS siste 24t: {len(harvest_success)}")
print(f"401 Errors siste 24t: {len(harvest_401)}")
print()
print("Vellykkede closes:")
for l in harvest_success:
    if 'trade.closed' in l:
        ts = l.split()[2] if len(l.split()) > 2 else ""
        parts = l.split("trade.closed:")[1].strip() if 'trade.closed:' in l else ""
        print(f"  {ts}  {parts}")

# check stream length
print(f"\nquantum:stream:trade.closed total length: {r.xlen('quantum:stream:trade.closed')}")
