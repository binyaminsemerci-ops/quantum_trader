#!/usr/bin/env python3
"""Patch /home/qt/quantum_trader/microservices/intent_bridge/main.py with TTL fix"""
import os

path = '/home/qt/quantum_trader/microservices/intent_bridge/main.py'

old = (
    '        self.redis.hset(permit_key, mapping={\n'
    '            "allow": "true",\n'
    '            "safe_qty": "0",\n'
    '            "reason": "auto_bypass_no_p33",\n'
    '            "timestamp": str(int(time.time()))\n'
    '        })\n'
)

new = (
    '        self.redis.hset(permit_key, mapping={\n'
    '            "allow": "true",\n'
    '            "safe_qty": "0",\n'
    '            "reason": "auto_bypass_no_p33",\n'
    '            "timestamp": str(int(time.time()))\n'
    '        })\n'
    '        self.redis.expire(permit_key, 86400)  # H1 fix: 24h TTL\n'
)

with open(path) as f:
    content = f.read()

if old in content:
    content = content.replace(old, new, 1)
    with open(path, 'w') as f:
        f.write(content)
    print(f"FIXED: {path}")
elif 'H1 fix' in content:
    print(f"Already patched: {path}")
else:
    print(f"Pattern not found: {path}")
    # Show nearby lines for diagnosis
    for i, line in enumerate(content.split('\n'), 1):
        if 'permit_key' in line or 'auto_bypass' in line:
            print(f"  L{i}: {line}")
