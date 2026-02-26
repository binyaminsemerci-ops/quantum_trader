#!/usr/bin/env python3
"""Check binance_pnl_tracker status and full RL reward pipeline"""
import subprocess, os, redis as rlib, json

r = rlib.Redis(host='127.0.0.1', port=6379, decode_responses=True)

# 1. Is binance_pnl_tracker running?
print("=== Service status ===")
services = subprocess.run(['systemctl', 'list-units', '--all', '--no-pager', '-q', '--type=service'],
                          capture_output=True, text=True)
for line in services.stdout.splitlines():
    if any(w in line.lower() for w in ['pnl', 'rl_', 'rl-', 'reward', 'feedback', 'calibrat']):
        print(f"  {line.strip()}")

# 2. Read binance_pnl_tracker.py
base = '/home/qt/quantum_trader'
tracker_path = f'{base}/microservices/binance_pnl_tracker/binance_pnl_tracker.py'
print(f"\n=== binance_pnl_tracker.py ===")
if os.path.exists(tracker_path):
    with open(tracker_path) as f:
        code = f.read()
    print(f"  File size: {len(code)} chars")
    for i, line in enumerate(code.splitlines(), 1):
        if any(w in line.lower() for w in ['usertrades', 'realized', 'redis', 'xadd', 'hset', 'reward', 'stream', 'api_key', 'base_url']):
            print(f"  L{i:4d}: {line.strip()[:110]}")
else:
    print("  NOT FOUND")

# 3. Check rl:reward keys in Redis
print("\n=== quantum:rl:reward:* keys ===")
keys = r.keys('quantum:rl:reward:*')
print(f"  Found {len(keys)} reward keys")
for k in sorted(keys)[:10]:
    t = r.type(k)
    if t == 'hash':
        d = r.hgetall(k)
        print(f"  {k}: {d}")
    elif t == 'string':
        print(f"  {k}: {r.get(k)[:100]}")

# 4. Check rl_rewards stream
print("\n=== quantum:stream:rl_rewards (last 5) ===")
msgs = r.xrevrange('quantum:stream:rl_rewards', count=5)
print(f"  {len(msgs)} entries")
for mid, d in msgs:
    import time
    ts = time.strftime('%H:%M:%S', time.gmtime(int(mid.split('-')[0])/1000))
    print(f"  {ts}: {d}")

# 5. What field does rl_calibrator use?
print("\n=== rl_calibrator/calibrate_tp_sl.py ===")
cal_path = f'{base}/microservices/rl_calibrator/calibrate_tp_sl.py'
if os.path.exists(cal_path):
    with open(cal_path) as f:
        print(f.read()[:3000])

# 6. Check how exit_manager publishes close event
print("\n=== exit_manager.py publish section ===")
em_path = f'{base}/microservices/autonomous_trader/exit_manager.py'
with open(em_path) as f:
    lines = f.readlines()
total = len(lines)
# Find publish/xadd/trade.closed section
for i, line in enumerate(lines):
    if any(w in line.lower() for w in ['trade.closed', 'xadd', 'publish_close', 'pnl_usd', 'order_id']):
        start = max(0, i-2)
        end = min(total, i+3)
        for j in range(start, end):
            print(f"  L{j+1:4d}: {lines[j].rstrip()}")
        print()
