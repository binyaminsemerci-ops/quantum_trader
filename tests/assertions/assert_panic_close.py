#!/usr/bin/env python3
"""
CI Assertion: Verify system:panic_close was published

Confirms watchdog detected Exit Brain death and triggered panic_close.
"""

import os
import sys
import time
import redis

REDIS_URL = os.getenv("REDIS_URL", "redis://localhost:6379")
PANIC_CLOSE_STREAM = "system:panic_close"
MAX_EVENT_AGE_SECONDS = 30  # Event should be recent


def main():
    print("=" * 50)
    print("ASSERTION: panic_close was published")
    print("=" * 50)
    
    r = redis.from_url(REDIS_URL)
    
    # Check stream exists and has entries
    stream_len = r.xlen(PANIC_CLOSE_STREAM)
    print(f"panic_close stream length: {stream_len}")
    
    if stream_len == 0:
        print("❌ FAIL: No panic_close events found")
        print("   Watchdog may not have detected Exit Brain death")
        print("   Check:")
        print("   - Was Exit Brain running before kill?")
        print("   - Did watchdog start correctly?")
        print("   - Is heartbeat threshold correct (5s)?")
        sys.exit(1)
    
    # Get latest panic_close event
    messages = r.xrevrange(PANIC_CLOSE_STREAM, count=1)
    
    if not messages:
        print("❌ FAIL: Could not read panic_close stream")
        sys.exit(1)
    
    msg_id, data = messages[0]
    
    # Parse event
    event_id = data.get(b"event_id", b"").decode()
    reason = data.get(b"reason", b"").decode()
    severity = data.get(b"severity", b"").decode()
    issued_by = data.get(b"issued_by", b"").decode()
    ts_ms = int(data.get(b"ts", 0))
    
    print(f"Latest panic_close event:")
    print(f"  Event ID: {event_id}")
    print(f"  Reason: {reason}")
    print(f"  Severity: {severity}")
    print(f"  Issued by: {issued_by}")
    
    # Validate required fields
    errors = []
    
    if not event_id:
        errors.append("Missing event_id")
    
    if not reason:
        errors.append("Missing reason")
    
    if severity != "CRITICAL":
        errors.append(f"Severity should be CRITICAL, got {severity}")
    
    if issued_by != "watchdog":
        errors.append(f"issued_by should be watchdog, got {issued_by}")
    
    if ts_ms == 0:
        errors.append("Missing timestamp (ts)")
    else:
        event_time = ts_ms / 1000.0
        age = time.time() - event_time
        print(f"  Age: {age:.1f}s")
        
        if age > MAX_EVENT_AGE_SECONDS:
            errors.append(f"Event too old ({age:.1f}s > {MAX_EVENT_AGE_SECONDS}s)")
    
    if errors:
        print("=" * 50)
        print("❌ FAIL: Invalid panic_close event")
        for err in errors:
            print(f"   - {err}")
        sys.exit(1)
    
    print("=" * 50)
    print("✅ PASS: panic_close was published correctly")
    print(f"   Event ID: {event_id}")
    print(f"   Reason: {reason}")
    print("=" * 50)
    sys.exit(0)


if __name__ == "__main__":
    main()
