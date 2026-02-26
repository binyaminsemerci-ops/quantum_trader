#!/usr/bin/env python3
"""Diagnose exact content around the anchor in rl_agent_daemon.py"""
DAEMON = "/home/qt/quantum_trader/microservices/rl_sizing_agent/rl_agent_daemon.py"
with open(DAEMON) as f:
    src = f.read()

# Find pnl_usd line number and print surrounding context
lines = src.split('\n')
for i, line in enumerate(lines):
    if 'pnl_usd' in line and 'fields.get' in line:
        print(f"Found pnl_usd at line {i+1}")
        # Print previous 3 and next 15 lines with repr
        for j in range(max(0, i-3), min(len(lines), i+16)):
            print(f"  {j+1}: {repr(lines[j])}")
        print()

# Also try the regex
import re
PATTERN = re.compile(
    r'                    try:\n'
    r'                        symbol = fields\.get\("symbol", ""\)\n'
    r'                        pnl = float\(fields\.get\("pnl_usd", 0\)\)',
    re.DOTALL
)
m = PATTERN.search(src)
print(f"Simple regex match: {m}")
