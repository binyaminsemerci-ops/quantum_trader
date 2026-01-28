#!/usr/bin/env python3
"""Upgrade exit guard from in-memory to Redis-based persistence"""
import re
import sys

FILE_PATH = "/home/qt/quantum_trader/services/exit_monitor_service.py"

def upgrade_to_redis():
    with open(FILE_PATH, "r", encoding="utf-8") as f:
        content = f.read()
    
    # Check if already upgraded
    if "REDIS_EXIT_GUARD_V1" in content:
        print("✅ Already has REDIS_EXIT_GUARD_V1")
        return 0
    
    # Ensure redis import exists
    if not re.search(r"^import redis\b", content, flags=re.MULTILINE):
        content = re.sub(r"(import logging\n)", r"\1import redis\n", content, count=1)
        print("✅ Added redis import")
    
    # Find and replace in-memory guard block
    pattern = r"_exit_processed\s*=\s*set\(\)\s*\n_exit_cooldown\s*=\s*\{\}\s*\n(?:.|[\n])*?def\s+check_exit_cooldown\(sym,side\):(?:.|[\n])*?return\s+False\s*"
    
    match = re.search(pattern, content)
    if not match:
        print("❌ Could not find in-memory guard block")
        return 1
    
    redis_block = '''# REDIS_EXIT_GUARD_V1
# Dedup + cooldown persisted in Redis (survives restarts)
EXIT_DEDUP_TTL_SEC = 300
EXIT_COOLDOWN_TTL_SEC = 30

_guard_redis = None
def _r():
    global _guard_redis
    if _guard_redis is not None:
        return _guard_redis
    try:
        _guard_redis = redis.Redis(host="localhost", port=6379, decode_responses=True)
        return _guard_redis
    except Exception as e:
        logger.error(f"❌ EXIT_GUARD redis init failed: {e}")
        _guard_redis = None
        return None

def check_exit_dedup(sym, oid):
    # returns True if should skip
    r = _r()
    if r is None:
        return False  # fail-open: allow exit
    key = f"quantum:dedup:exit:{sym}_{oid}"
    try:
        ok = r.set(key, "1", nx=True, ex=EXIT_DEDUP_TTL_SEC)
        if not ok:
            logger.info(f"🔴 EXIT_DEDUP skip={sym}_{oid}")
            return True
        return False
    except Exception as e:
        logger.error(f"❌ EXIT_DEDUP redis op failed: {e}")
        return False

def check_exit_cooldown(sym, side):
    # returns True if should skip
    r = _r()
    if r is None:
        return False  # fail-open
    key = f"quantum:cooldown:exit:{sym}:{side}"
    try:
        if r.exists(key):
            logger.info(f"⏸️ EXIT_COOLDOWN skip={sym}")
            return True
        r.set(key, "1", ex=EXIT_COOLDOWN_TTL_SEC)
        return False
    except Exception as e:
        logger.error(f"❌ EXIT_COOLDOWN redis op failed: {e}")
        return False
'''
    
    content = content[:match.start()] + redis_block + content[match.end():]
    
    with open(FILE_PATH, "w", encoding="utf-8") as f:
        f.write(content)
    
    print("✅ Upgraded to Redis-based guards")
    return 0

if __name__ == "__main__":
    sys.exit(upgrade_to_redis())
