#!/usr/bin/env python3
"""Read exit_manager.py close/publish section and rl_feedback_v2 bridge"""
import os

base = '/home/qt/quantum_trader'

# 1. Full exit_manager.py - find close order + publish section
print("=== exit_manager.py - close execution + publish ===")
with open(f'{base}/microservices/autonomous_trader/exit_manager.py') as f:
    lines = f.readlines()

# Find all relevant blocks
for i, line in enumerate(lines):
    if any(w in line for w in ['trade.closed', 'pnl_usd', 'xadd', 'publish', 'order_id', 
                                    'fapi', 'userTrades', 'realizedPnl', 'close_order',
                                    'place_close', 'execute_exit', 'fill_price']):
        start = max(0, i-1)
        end = min(len(lines), i+2)
        print(f"L{i+1:4d}: {lines[i].rstrip()}")

# 2. Read rl_feedback bridge v2 to understand what it reads
print("\n=== rl_feedback_bridge_v2/bridge_v2.py ===")
p2 = f'{base}/microservices/rl_feedback_bridge_v2/bridge_v2.py'
if os.path.exists(p2):
    with open(p2) as f:
        print(f.read())

# 3. Read quantum-rl-feedback-v2 systemd service to see what it runs
import subprocess
result = subprocess.run(['cat', '/etc/systemd/system/quantum-rl-feedback-v2.service'],
                       capture_output=True, text=True)
print(f"\n=== quantum-rl-feedback-v2.service ===\n{result.stdout}")

# 4. What stream does it read from?
import redis
r = redis.Redis(host='127.0.0.1', port=6379, decode_responses=True)
for key in ['quantum:stream:exitbrain.pnl', 'quantum:stream:rl_rewards', 
            'quantum:stream:trade.feedback']:
    t = r.type(key)
    msgs = r.xrevrange(key, count=3) if t == 'stream' else []
    if msgs:
        print(f"\n{key}: {len(msgs)} entries")
        for mid, d in msgs:
            print(f"  {mid}: {d}")
