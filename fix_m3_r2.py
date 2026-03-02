#!/usr/bin/env python3
"""Fix M3 round 2: Patch lines 2305-2360 in service.py (Funding check + Drift detection timeouts)"""

path = "/opt/quantum/microservices/ai_engine/service.py"

with open(path, 'r') as f:
    lines = f.readlines()

total_changed = 0
# Extend range to cover lines 2305-2370
target_range = range(2300, min(2375, len(lines)))

for i in target_range:
    line = lines[i]
    if 'timeout=5.0' in line or 'timeout=5)' in line or 'timeout = 5' in line:
        lines[i] = line.replace('timeout=5.0', 'timeout=12.0').replace('timeout=5)', 'timeout=12)').replace('timeout = 5', 'timeout = 12')
        total_changed += 1
        print(f"  Line {i+1} timeout: {lines[i].rstrip()}")
    if 'timeout (5s)' in lines[i]:
        lines[i] = lines[i].replace('timeout (5s)', 'timeout (12s)')
        total_changed += 1
        print(f"  Line {i+1} logmsg: {lines[i].rstrip()}")
    if 'Max 5 seconds' in lines[i]:
        lines[i] = lines[i].replace('Max 5 seconds', 'Max 12 seconds')
        total_changed += 1
        print(f"  Line {i+1} comment: {lines[i].rstrip()}")
    if 'TIMEOUT (5s)' in lines[i]:
        lines[i] = lines[i].replace('TIMEOUT (5s)', 'TIMEOUT (12s)')
        total_changed += 1
        print(f"  Line {i+1} TIMEOUT: {lines[i].rstrip()}")

print(f"\nTotal changes: {total_changed}")

if total_changed > 0:
    with open(path, 'w') as f:
        f.writelines(lines)
    print("M3 round 2: PATCHED - lines 2305-2370 (Funding check + Drift detection)")
else:
    # Check what's there
    for i in range(2310, 2360):
        if i < len(lines):
            print(f"[{i+1}]: {lines[i].rstrip()}")
