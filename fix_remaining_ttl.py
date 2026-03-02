#!/usr/bin/env python3
"""
Fix all remaining permit key writers without TTL:
1. harvest_brain home dir (line ~926)
2. auto_permit_p33.py (opt + home)
"""
import os

TTL = 86400  # 24h

PATCHES = [
    # (path, old_block, new_block)
    (
        '/home/qt/quantum_trader/microservices/harvest_brain/harvest_brain.py',
        (
            '            self.redis.hset(permit_key, mapping={\n'
            '                "allow": "true",\n'
            '                "safe_qty": "0",\n'
            '                "reason": "harvest_brain_auto_permit",\n'
            '                "timestamp": str(ts_unix)\n'
            '            })\n'
        ),
        (
            '            self.redis.hset(permit_key, mapping={\n'
            '                "allow": "true",\n'
            '                "safe_qty": "0",\n'
            '                "reason": "harvest_brain_auto_permit",\n'
            '                "timestamp": str(ts_unix)\n'
            '            })\n'
            '            self.redis.expire(permit_key, 86400)  # H1 fix: 24h TTL\n'
        ),
    ),
    # auto_permit_p33 — opt
    (
        '/opt/quantum/scripts/auto_permit_p33.py',
        (
            "        r.hset(permit_key, mapping={\n"
            "            'allow': 'true',\n"
            "            'safe_qty': '0',\n"
            "            'reason': 'auto_bypass',\n"
            "            'timestamp': str(int(time.time()))\n"
            "        })\n"
        ),
        (
            "        r.hset(permit_key, mapping={\n"
            "            'allow': 'true',\n"
            "            'safe_qty': '0',\n"
            "            'reason': 'auto_bypass',\n"
            "            'timestamp': str(int(time.time()))\n"
            "        })\n"
            "        r.expire(permit_key, 86400)  # H1 fix: 24h TTL\n"
        ),
    ),
    # auto_permit_p33 — home
    (
        '/home/qt/quantum_trader/scripts/auto_permit_p33.py',
        (
            "        r.hset(permit_key, mapping={\n"
            "            'allow': 'true',\n"
            "            'safe_qty': '0',\n"
            "            'reason': 'auto_bypass',\n"
            "            'timestamp': str(int(time.time()))\n"
            "        })\n"
        ),
        (
            "        r.hset(permit_key, mapping={\n"
            "            'allow': 'true',\n"
            "            'safe_qty': '0',\n"
            "            'reason': 'auto_bypass',\n"
            "            'timestamp': str(int(time.time()))\n"
            "        })\n"
            "        r.expire(permit_key, 86400)  # H1 fix: 24h TTL\n"
        ),
    ),
]

for path, old, new in PATCHES:
    if not os.path.exists(path):
        print(f"SKIP (not found): {path}")
        continue
    with open(path) as f:
        content = f.read()
    if old in content:
        content = content.replace(old, new, 1)
        with open(path, 'w') as f:
            f.write(content)
        print(f"PATCHED: {path}")
    elif 'H1 fix' in content:
        print(f"ALREADY PATCHED: {path}")
    else:
        print(f"PATTERN NOT FOUND: {path}")

print("Done")
