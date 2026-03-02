#!/usr/bin/env python3
"""
H1 permanent fix: Add TTL (86400s = 24h) to permit key hset calls that have no expire.
Files to fix:
- intent_bridge/main.py line ~832
- risk_brake_v1_patch.py line ~135
- harvest_brain/harvest_brain.py line ~939
"""
import os

TTL = 86400  # 24h

BASE_PATHS = [
    '/opt/quantum/microservices/',
    '/home/qt/quantum_trader/microservices/'
]

TARGETS = [
    ('intent_bridge/main.py', 831, 840),
    ('risk_brake_v1_patch.py', 130, 145),
    ('harvest_brain/harvest_brain.py', 935, 945),
]

total_changes = 0

for base in BASE_PATHS:
    for rel_path, start_line, end_line in TARGETS:
        path = os.path.join(base, rel_path)
        if not os.path.exists(path):
            print(f"  SKIP (not found): {path}")
            continue
        
        with open(path) as f:
            lines = f.readlines()
        
        # Scan range for hset(permit_key, ...) or hset(permit_key, mapping=...)
        changed = 0
        i = start_line - 5  # a bit before
        while i < min(end_line + 5, len(lines)):
            line = lines[i]
            if 'permit_key' in line and ('hset(' in line or 'hset(' in line.lower()):
                # Check if expire is already in the next 3 lines
                next_lines = ''.join(lines[i+1:i+4])
                if 'expire' not in next_lines and 'setex' not in next_lines:
                    indent = len(line) - len(line.lstrip())
                    spaces = ' ' * indent
                    # Determine redis object
                    redis_obj = 'self.redis'
                    if 'pipe.' in line:
                        redis_obj = 'pipe'
                    elif 'r.' in line:
                        redis_obj = 'r'
                    elif 'redis.' in line:
                        redis_obj = 'redis'
                    elif 'self.redis' in line or 'self.r.' in line:
                        redis_obj = line.split('.hset')[0].strip().lstrip()
                    
                    expire_line = f"{spaces}{redis_obj}.expire(permit_key, {TTL})  # H1 fix: 24h TTL\n"
                    lines.insert(i + 1, expire_line)
                    print(f"  ADDED to {path}:{i+1}: {expire_line.rstrip()}")
                    changed += 1
                    i += 2  # skip past inserted line
                    total_changes += 1
                else:
                    print(f"  SKIP {path}:{i+1}: already has expire nearby")
                    i += 1
            else:
                i += 1
        
        if changed > 0:
            with open(path, 'w') as f:
                f.writelines(lines)
            print(f"  PATCHED {path}: {changed} changes")
        else:
            print(f"  No changes: {path}")

print(f"\nTotal changes: {total_changes}")
print("H1 TTL fix: DONE")
