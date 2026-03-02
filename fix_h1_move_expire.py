#!/usr/bin/env python3
"""Fix the incorrectly placed expire calls — move from inside hset dict to after })"""
import os

FILES = [
    (
        '/opt/quantum/microservices/intent_bridge/main.py',
        # old (wrong: expire inside dict)
        '        self.redis.hset(permit_key, mapping={\n'
        '        redis.expire(permit_key, 86400)  # H1 fix: 24h TTL\n'
        '            "allow": "true",\n'
        '            "safe_qty": "0",\n'
        '            "reason": "auto_bypass_no_p33",\n'
        '            "timestamp": str(int(time.time()))\n'
        '        })\n',
        # new (correct: expire after hset)
        '        self.redis.hset(permit_key, mapping={\n'
        '            "allow": "true",\n'
        '            "safe_qty": "0",\n'
        '            "reason": "auto_bypass_no_p33",\n'
        '            "timestamp": str(int(time.time()))\n'
        '        })\n'
        '        self.redis.expire(permit_key, 86400)  # H1 fix: 24h TTL\n',
    ),
    (
        '/opt/quantum/microservices/harvest_brain/harvest_brain.py',
        # old (wrong)
        '            self.redis.hset(permit_key, mapping={\n'
        '            redis.expire(permit_key, 86400)  # H1 fix: 24h TTL\n'
        '                "allow": "true",\n'
        '                "safe_qty": "0",\n'
        '                "reason": "harvest_brain_auto_permit",\n'
        '                "timestamp": str(ts_unix)\n'
        '            })\n',
        # new (correct)
        '            self.redis.hset(permit_key, mapping={\n'
        '                "allow": "true",\n'
        '                "safe_qty": "0",\n'
        '                "reason": "harvest_brain_auto_permit",\n'
        '                "timestamp": str(ts_unix)\n'
        '            })\n'
        '            self.redis.expire(permit_key, 86400)  # H1 fix: 24h TTL\n',
    ),
]

for path, old, new in FILES:
    if not os.path.exists(path):
        print(f"NOT FOUND: {path}")
        continue
    
    with open(path) as f:
        content = f.read()
    
    if old in content:
        content = content.replace(old, new, 1)
        with open(path, 'w') as f:
            f.write(content)
        print(f"FIXED: {path}")
    elif 'H1 fix: 24h TTL' in content:
        print(f"TTL line exists but pattern mismatch — manual check needed: {path}")
    else:
        print(f"Pattern not found (may need re-check): {path}")

print("\nDone — verify with verify_h1.sh")
