#!/bin/bash
echo "=== get_authoritative_open_positions function ==="
python3 - << 'EOF'
import re

with open('/home/qt/quantum_trader/microservices/autonomous_trader/autonomous_trader.py', 'r') as f:
    lines = f.readlines()

# Find the function
in_func = False
func_lines = []
for i, line in enumerate(lines):
    if 'async def get_authoritative_open_positions' in line:
        in_func = True
    if in_func:
        func_lines.append(f"{i+1}: {line}", )
        # Stop after 40 lines or next top-level function
        if len(func_lines) > 1 and (line.startswith('async def') or line.startswith('def ')) and not 'get_authoritative' in line:
            break
        if len(func_lines) > 50:
            break

print(''.join(func_lines))
EOF

echo ""
echo "=== POSITION HASH KEYS TYPE DISTRIBUTION ==="
redis-cli keys "quantum:position:*" | sed 's/:/ /g' | awk '{print $3}' | sort | uniq -c | sort -rn | head -20

echo ""
echo "=== SAMPLE position hash ==="
redis-cli hgetall "$(redis-cli keys 'quantum:position:*' | grep -v 'ledger\|snapshot' | head -1)"
