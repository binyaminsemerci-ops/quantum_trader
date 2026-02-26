#!/usr/bin/env python3
"""Find how RL agent reads reward signal"""
import os, subprocess

base = '/home/qt/quantum_trader'

# 1. Where does RL agent read PnL/reward?
print("=== RL reward signal sources ===")
result = subprocess.run(
    ['grep', '-rn', 'pnl\|reward\|realized\|trade.closed\|feedback',
     f'{base}/microservices/'],
    capture_output=True, text=True
)
for line in result.stdout.splitlines():
    if any(w in line.lower() for w in ['reward', 'pnl_usd', 'realized_pnl', 'trade.closed', 'feedback']):
        if 'rl' in line.lower() or 'signal_manager' in line.lower() or 'feedback' in line.lower():
            print(f"  {line.strip()[:120]}")

# 2. Find RL signal manager / feedback files
print("\n=== RL feedback/signal_manager files ===")
for dirpath, dirs, files in os.walk(base):
    for f in files:
        if any(w in f.lower() for w in ['rl_signal', 'feedback', 'reward', 'calibrat']):
            fpath = os.path.join(dirpath, f)
            print(f"  {fpath}")
            with open(fpath) as fh:
                code = fh.read()
            for i, line in enumerate(code.splitlines(), 1):
                if any(w in line.lower() for w in ['pnl', 'reward', 'realized', 'trade.closed', 'feedback']):
                    print(f"    L{i:4d}: {line.strip()[:100]}")

# 3. Check exit_manager how it publishes close event
print("\n=== exit_manager.py close event ===")
with open(f'{base}/microservices/autonomous_trader/exit_manager.py') as f:
    for i, line in enumerate(f, 1):
        if any(w in line.lower() for w in ['pnl', 'trade.closed', 'publish', 'xadd', 'order_id', 'realized']):
            print(f"  L{i:4d}: {line.strip()[:110]}")
