#!/usr/bin/env python3
"""
H1 permanent fix: Add TTL (86400s = 24h) to permit key creation in apply_layer/main.py
Also fixes the home path copy.
"""
import re

TTL = 86400  # 24 hours

def add_ttl_to_permit_sets(path):
    try:
        with open(path) as f:
            content = f.read()
    except FileNotFoundError:
        print(f"Not found: {path}")
        return 0

    lines = content.split('\n')
    changes = 0
    
    # Show the permit key creation context
    for i, line in enumerate(lines):
        if ('gov_key' in line or 'p33_key' in line or 'p26_key' in line or 'permit_key' in line) and '=' in line and 'quantum:permit' in line:
            print(f"  Found key def at line {i+1}: {line.strip()}")
            # Look for the setnx/set call in next 20 lines
            for j in range(i+1, min(i+25, len(lines))):
                ln = lines[j]
                # Look for: r.setnx(key, ...) or r.set(key, ...) or redis.set(key, ...)
                if ('setnx(' in ln or '.set(' in ln) and ('gov_key' in ln or 'p33_key' in ln or 'p26_key' in ln or 'permit_key' in ln):
                    print(f"    Redis set at line {j+1}: {ln.strip()}")
                    # Check if there's already an expire call after this
                    has_expire = any('expire' in lines[min(j+k, len(lines)-1)] and ('gov_key' in lines[min(j+k, len(lines)-1)] or 'p33_key' in lines[min(j+k, len(lines)-1)] or 'p26_key' in lines[min(j+k, len(lines)-1)] or 'permit_key' in lines[min(j+k, len(lines)-1)]) for k in range(1, 5))
                    if not has_expire:
                        # Add expire line after the set call
                        indent = len(ln) - len(ln.lstrip())
                        spaces = ' ' * indent
                        # Extract variable name
                        match = re.search(r'(gov_key|p33_key|p26_key|permit_key)', ln)
                        if match:
                            var_name = match.group(1)
                            # Get the redis object name
                            redis_match = re.match(r'\s*(\w+)\.(setnx|set)\(', ln)
                            if redis_match:
                                redis_obj = redis_match.group(1)
                                expire_line = f"{spaces}{redis_obj}.expire({var_name}, {TTL})  # H1 fix: 24h TTL"
                                lines.insert(j+1, expire_line)
                                print(f"    ADDED expire at line {j+2}: {expire_line.strip()}")
                                changes += 1
                                break
    
    if changes > 0:
        with open(path, 'w') as f:
            f.write('\n'.join(lines))
        print(f"  PATCHED {path}: {changes} TTL lines added")
    else:
        print(f"  No changes needed (already has TTL or pattern not matched)")
    return changes

print("=== H1: Adding TTL to permit key creation ===")
c1 = add_ttl_to_permit_sets('/opt/quantum/microservices/apply_layer/main.py')
c2 = add_ttl_to_permit_sets('/home/qt/quantum_trader/microservices/apply_layer/main.py')
print(f"\nTotal changes: {c1 + c2}")
