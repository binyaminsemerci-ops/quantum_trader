#!/usr/bin/env python3
"""
Direct Policy Refresh - Updates valid_until_epoch only
Lightweight fix for policy expiry without full regeneration
"""
import redis
import time
import sys

# Config
REDIS_HOST = "127.0.0.1"
REDIS_PORT = 6379
REDIS_DB = 0
POLICY_KEY = "quantum:policy:current"
VALID_FOR_HOURS = 2  # Policy valid for 2 hours

def main():
    try:
        # Connect to Redis
        r = redis.Redis(host=REDIS_HOST, port=REDIS_PORT, db=REDIS_DB, decode_responses=True)
        r.ping()
        
        # Calculate new expiry
        new_expiry = time.time() + (VALID_FOR_HOURS * 3600)
        
        # Update policy expiry
        r.hset(POLICY_KEY, "valid_until_epoch", new_expiry)
        
        # Verify
        current_expiry = float(r.hget(POLICY_KEY, "valid_until_epoch"))
        policy_version = r.hget(POLICY_KEY, "policy_version")
        
        print(f"✅ Policy refresh successful")
        print(f"   Version: {policy_version}")
        print(f"   Valid until: {new_expiry} ({VALID_FOR_HOURS}h from now)")
        print(f"   Verified: {current_expiry}")
        
        return 0
        
    except Exception as e:
        print(f"❌ Policy refresh failed: {e}", file=sys.stderr)
        return 1

if __name__ == "__main__":
    sys.exit(main())
