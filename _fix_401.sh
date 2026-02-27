#!/bin/bash
set -e

echo "=== FIXING intent-executor.env: BINANCE_BASE_URL ==="
echo "Before:"
grep "BINANCE_BASE_URL" /etc/quantum/intent-executor.env || echo "(not found)"

# Use Python to do the replacement (avoids shell quoting issues)
python3 -c "
import re
with open('/etc/quantum/intent-executor.env', 'r') as f:
    content = f.read()

old = 'BINANCE_BASE_URL=https://fapi.binance.com'
new = 'BINANCE_BASE_URL=https://testnet.binancefuture.com'

if old in content:
    content = content.replace(old, new)
    with open('/etc/quantum/intent-executor.env', 'w') as f:
        f.write(content)
    print('FIXED: replaced fapi.binance.com with testnet.binancefuture.com')
else:
    print('NOT FOUND: BINANCE_BASE_URL=https://fapi.binance.com')
    print('Current URL lines:')
    for line in content.splitlines():
        if 'BASE_URL' in line or 'TESTNET' in line.upper():
            print(' ', line)
"

echo ""
echo "After:"
grep "BINANCE_BASE_URL\|BINANCE_TESTNET\|BINANCE_USE" /etc/quantum/intent-executor.env

echo ""
echo "=== RESTARTING quantum-intent-executor ==="
systemctl restart quantum-intent-executor
sleep 3

echo "=== STATUS ==="
systemctl is-active quantum-intent-executor
echo ""
echo "=== LAST 5 LOGS ==="
journalctl -u quantum-intent-executor -n 5 --no-pager
