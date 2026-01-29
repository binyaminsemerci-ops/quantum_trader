#!/usr/bin/env python3
"""
Exit Brain v3.5 Operational Audit - VPS Remote Execution
Connects via SSH and runs comprehensive audit
"""
import subprocess
import sys

def run_ssh_command(cmd):
    """Execute command on VPS via SSH"""
    full_cmd = [
        "wsl", "ssh", "-i", "~/.ssh/hetzner_fresh",
        "root@46.224.116.254",
        cmd
    ]
    try:
        result = subprocess.run(full_cmd, capture_output=True, text=True, timeout=60)
        return result.stdout + result.stderr
    except Exception as e:
        return f"ERROR: {e}"

print("=== EXIT BRAIN V3.5 OPERATIONAL AUDIT ===\n")

# STEP 0: Discover service
print("STEP 0: DISCOVER SERVICE")
print("-" * 60)
output = run_ssh_command("systemctl list-unit-files | grep position")
print(output)

# Try to find service name
service_name = None
for line in output.split('\n'):
    if 'position' in line.lower() and '.service' in line:
        service_name = line.split()[0]
        print(f"\n✅ Found service: {service_name}\n")
        break

if not service_name:
    print("❌ FAIL: No position monitor service found")
    sys.exit(1)

# STEP 1: Service health
print("\nSTEP 1: SERVICE HEALTH")
print("-" * 60)
output = run_ssh_command(f"systemctl is-active {service_name}")
print(f"Status: {output.strip()}")

output = run_ssh_command(f"systemctl status {service_name} --no-pager -l | head -60")
print(output)

print("\nRecent logs:")
output = run_ssh_command(f"journalctl -u {service_name} -n 100 --no-pager | tail -80")
print(output)

# STEP 2: Code verification
print("\nSTEP 2: CODE PATH VERIFICATION")
print("-" * 60)
output = run_ssh_command("grep -r 'ExitBrainV35' /home/qt/quantum_trader/backend/services/monitoring --include='*.py' | head -20")
print("ExitBrainV35 usage:")
print(output)

output = run_ssh_command("grep -r 'build_exit_plan' /home/qt/quantum_trader/backend/services/monitoring --include='*.py' | head -20")
print("\nbuild_exit_plan calls:")
print(output)

# STEP 3: Exchange check
print("\nSTEP 3: EXCHANGE REALITY CHECK")
print("-" * 60)

# Create and run position checker
checker_script = """
import os, sys
sys.path.insert(0, '/home/qt/quantum_trader')
from binance.client import Client

use_testnet = os.getenv('STAGING_MODE', 'false').lower() == 'true' or os.getenv('BINANCE_TESTNET', 'false').lower() == 'true'
api_key = os.getenv('BINANCE_API_KEY', '')
api_secret = os.getenv('BINANCE_API_SECRET', '')

if not api_key:
    print('No credentials')
    sys.exit(1)

client = Client(api_key, api_secret, testnet=use_testnet)
if use_testnet:
    client.API_URL = 'https://testnet.binancefuture.com'
    print('Mode: TESTNET')
else:
    print('Mode: LIVE')

positions = client.futures_position_information()
open_pos = [p for p in positions if float(p.get('positionAmt', 0)) != 0]

print(f'Open positions: {len(open_pos)}')
for pos in open_pos[:3]:
    sym = pos['symbol']
    qty = float(pos['positionAmt'])
    entry = float(pos['entryPrice'])
    pnl = float(pos['unRealizedProfit'])
    print(f'  {sym}: qty={qty:.4f}, entry=${entry:.2f}, PnL=${pnl:.2f}')
    
    orders = client.futures_get_open_orders(symbol=sym)
    tp = [o for o in orders if 'TAKE_PROFIT' in o.get('type', '')]
    sl = [o for o in orders if 'STOP' in o.get('type', '')]
    print(f'    Protective orders: {len(tp)} TP, {len(sl)} SL')
    
    for o in orders[:2]:
        print(f'      {o["type"]} {o["side"]} @ ${o.get("stopPrice", o.get("price", "N/A"))}')
"""

output = run_ssh_command(f"cd /home/qt/quantum_trader && python3 << 'EOF'\n{checker_script}\nEOF")
print(output)

# VERDICT
print("\n" + "=" * 60)
print("VERDICT ANALYSIS")
print("=" * 60)
print("\nCriteria:")
print("✓ Service active and running loop")
print("✓ ExitBrainV35 used in code")
print("✓ Open positions have protective orders (TP/SL)")
print("✓ Recent activity in logs")
print("\nReview output above to determine PASS/FAIL")
