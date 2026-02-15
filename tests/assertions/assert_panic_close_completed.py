#!/usr/bin/env python3
"""
CI Assertion: Verify system:panic_close:completed was published

Confirms Emergency Exit Worker received panic_close and executed it.
"""

import os
import sys
import time
import json
import redis

REDIS_URL = os.getenv("REDIS_URL", "redis://localhost:6379")
PANIC_CLOSE_STREAM = "system:panic_close"
PANIC_COMPLETE_STREAM = "system:panic_close:completed"
MAX_EXECUTION_TIME_MS = 10000  # 10 seconds max


def main():
    print("=" * 50)
    print("ASSERTION: panic_close:completed was published")
    print("=" * 50)
    
    r = redis.from_url(REDIS_URL)
    
    # Check completion stream
    stream_len = r.xlen(PANIC_COMPLETE_STREAM)
    print(f"panic_close:completed stream length: {stream_len}")
    
    if stream_len == 0:
        print("❌ FAIL: No completion events found")
        print("   Emergency Exit Worker may not have executed")
        print("   Check:")
        print("   - Did EEW start correctly?")
        print("   - Did EEW receive panic_close?")
        print("   - Is consumer group set up?")
        sys.exit(1)
    
    # Get original panic_close event_id
    panic_messages = r.xrevrange(PANIC_CLOSE_STREAM, count=1)
    if not panic_messages:
        print("❌ FAIL: Could not read original panic_close")
        sys.exit(1)
    
    _, panic_data = panic_messages[0]
    original_event_id = panic_data.get(b"event_id", b"").decode()
    print(f"Original panic_close event_id: {original_event_id}")
    
    # Get completion event
    complete_messages = r.xrevrange(PANIC_COMPLETE_STREAM, count=1)
    if not complete_messages:
        print("❌ FAIL: Could not read completion stream")
        sys.exit(1)
    
    msg_id, data = complete_messages[0]
    
    # Parse completion
    event_id = data.get(b"event_id", b"").decode()
    positions_total = int(data.get(b"positions_total", 0))
    positions_closed = int(data.get(b"positions_closed", 0))
    positions_failed = int(data.get(b"positions_failed", 0))
    failed_symbols = data.get(b"failed_symbols", b"[]").decode()
    execution_time_ms = int(data.get(b"execution_time_ms", 0))
    
    print(f"Completion event:")
    print(f"  Event ID: {event_id}")
    print(f"  Positions total: {positions_total}")
    print(f"  Positions closed: {positions_closed}")
    print(f"  Positions failed: {positions_failed}")
    print(f"  Execution time: {execution_time_ms}ms")
    
    # Validate
    errors = []
    
    if event_id != original_event_id:
        errors.append(f"Event ID mismatch: {event_id} != {original_event_id}")
    
    if positions_failed > 0:
        errors.append(f"Positions failed to close: {positions_failed}")
        try:
            failed = json.loads(failed_symbols)
            if failed:
                errors.append(f"Failed symbols: {failed}")
        except:
            pass
    
    if positions_closed != positions_total:
        errors.append(f"Not all positions closed: {positions_closed}/{positions_total}")
    
    if execution_time_ms > MAX_EXECUTION_TIME_MS:
        errors.append(f"Execution too slow: {execution_time_ms}ms > {MAX_EXECUTION_TIME_MS}ms")
    
    if errors:
        print("=" * 50)
        print("❌ FAIL: Invalid completion event")
        for err in errors:
            print(f"   - {err}")
        sys.exit(1)
    
    print("=" * 50)
    print("✅ PASS: panic_close:completed published correctly")
    print(f"   All {positions_closed} positions closed")
    print(f"   Execution time: {execution_time_ms}ms")
    print("=" * 50)
    sys.exit(0)


if __name__ == "__main__":
    main()
