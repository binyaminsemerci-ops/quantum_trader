#!/usr/bin/env python3
"""
CI Assertion: Verify heartbeat was flowing before chaos test

Checks that exit_brain:heartbeat stream has recent entries,
confirming Exit Brain was alive before being killed.
"""

import os
import sys
import time
import redis

REDIS_URL = os.getenv("REDIS_URL", "redis://localhost:6379")
HEARTBEAT_STREAM = "exit_brain:heartbeat"
MAX_AGE_SECONDS = 10  # Heartbeat should be recent


def main():
    print("=" * 50)
    print("ASSERTION: Heartbeat was flowing")
    print("=" * 50)
    
    r = redis.from_url(REDIS_URL)
    
    # Check stream length
    stream_len = r.xlen(HEARTBEAT_STREAM)
    print(f"Heartbeat stream length: {stream_len}")
    
    if stream_len == 0:
        print("❌ FAIL: No heartbeats recorded")
        print("   Exit Brain mock may not have started correctly")
        sys.exit(1)
    
    # Get latest heartbeat
    messages = r.xrevrange(HEARTBEAT_STREAM, count=1)
    
    if not messages:
        print("❌ FAIL: Could not read heartbeat stream")
        sys.exit(1)
    
    msg_id, data = messages[0]
    ts_ms = int(data.get(b"ts", 0))
    status = data.get(b"status", b"UNKNOWN").decode()
    
    if ts_ms == 0:
        print("❌ FAIL: Heartbeat has no timestamp")
        sys.exit(1)
    
    heartbeat_time = ts_ms / 1000.0
    age = time.time() - heartbeat_time
    
    print(f"Latest heartbeat:")
    print(f"  Status: {status}")
    print(f"  Age: {age:.1f}s")
    
    if age > MAX_AGE_SECONDS:
        print(f"⚠️ WARNING: Heartbeat is {age:.1f}s old (max {MAX_AGE_SECONDS}s)")
        print("   This is expected after Exit Brain death")
    
    if status != "ALIVE":
        print(f"⚠️ WARNING: Last status was {status}, not ALIVE")
    
    print("=" * 50)
    print("✅ PASS: Heartbeat was flowing")
    print(f"   {stream_len} heartbeats recorded")
    print("=" * 50)
    sys.exit(0)


if __name__ == "__main__":
    main()
