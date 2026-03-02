#!/usr/bin/env python3
"""M3 fix: Increase PHASE 1/2B/2D/3A timeouts from 5s to 12s in service.py
   Also updates log messages to say (12s) instead of (5s)
"""

path = "/opt/quantum/microservices/ai_engine/service.py"

with open(path, 'r') as f:
    lines = f.readlines()

total_changed = 0
target_range = range(2069, 2305)  # 0-indexed lines 2070-2305

for i in target_range:
    if i >= len(lines):
        break
    line = lines[i]
    # Update timeout value
    if 'timeout=5.0' in line:
        lines[i] = line.replace('timeout=5.0', 'timeout=12.0')
        total_changed += 1
        print(f"  Line {i+1} timeout: {lines[i].rstrip()}")
    # Update log messages "(5s)" -> "(12s)"
    if 'timeout (5s)' in lines[i]:
        lines[i] = lines[i].replace('timeout (5s)', 'timeout (12s)')
        total_changed += 1
        print(f"  Line {i+1} logmsg: {lines[i].rstrip()}")

print(f"\nTotal changes: {total_changed}")

if total_changed > 0:
    with open(path, 'w') as f:
        f.writelines(lines)
    print("M3: PATCHED service.py - PHASE timeouts 5s -> 12s")
else:
    print("WARNING: No changes - check line range")
    # Debug: print lines 2088-2095
    for i in range(2087, 2095):
        if i < len(lines):
            print(f"  [{i+1}]: {lines[i].rstrip()}")
