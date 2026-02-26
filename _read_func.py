#!/usr/bin/env python3
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
        func_lines.append(f"{i+1}: {line}")
        # Stop after next top-level def
        if len(func_lines) > 1 and re.match(r'^(async def|def )\w', line):
            break
        if len(func_lines) > 60:
            break

print(''.join(func_lines))
