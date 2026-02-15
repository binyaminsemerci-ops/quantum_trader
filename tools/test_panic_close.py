#!/usr/bin/env python3
"""
Test Panic Close Trigger

USE ONLY ON TESTNET!

This script triggers a system:panic_close event for testing purposes.
"""

import sys
import time
import uuid
import argparse

try:
    import redis
except ImportError:
    print("Missing redis-py: pip install redis")
    sys.exit(1)


PANIC_CLOSE_STREAM = "system:panic_close"
PANIC_COMPLETE_STREAM = "system:panic_close:completed"


def trigger_panic_close(source: str, reason: str, redis_url: str = "redis://localhost:6379"):
    """Trigger a panic_close event per schema"""
    
    event_id = str(uuid.uuid4())
    ts = int(time.time() * 1000)  # Epoch ms
    
    print("=" * 60)
    print("⚠️  PANIC CLOSE TRIGGER")
    print("=" * 60)
    print(f"Event ID: {event_id}")
    print(f"Source: {source}")
    print(f"Reason: {reason}")
    print(f"Redis: {redis_url}")
    print("=" * 60)
    
    # Safety check
    if "TESTNET" not in reason.upper() and "TEST" not in reason.upper():
        print("\n⚠️  WARNING: Reason does not contain 'TEST' or 'TESTNET'")
        confirm = input("Are you SURE you want to trigger panic_close? (type 'YES' to confirm): ")
        if confirm != "YES":
            print("Cancelled.")
            return
    
    # Connect to Redis
    r = redis.from_url(redis_url)
    
    # Publish event per schema
    event = {
        "event_id": event_id,
        "reason": reason,
        "severity": "CRITICAL",
        "issued_by": source,
        "ts": str(ts)
    }
    
    msg_id = r.xadd(PANIC_CLOSE_STREAM, event)
    
    print(f"\n✅ Event published!")
    print(f"   Stream: {PANIC_CLOSE_STREAM}")
    print(f"   Message ID: {msg_id}")
    print(f"   Event ID: {event_id}")
    
    # Wait and check for completion
    print("\nWaiting for completion...")
    
    for i in range(15):
        time.sleep(1)
        
        # Check for completion
        messages = r.xrevrange(PANIC_COMPLETE_STREAM, count=1)
        if messages:
            msg_id, data = messages[0]
            recv_event_id = data.get(b'event_id', b'').decode()
            if recv_event_id == event_id:
                print("\n✅ Panic close completed!")
                print(f"   Event ID: {recv_event_id}")
                print(f"   Positions total: {data.get(b'positions_total', b'?').decode()}")
                print(f"   Positions closed: {data.get(b'positions_closed', b'?').decode()}")
                print(f"   Positions failed: {data.get(b'positions_failed', b'?').decode()}")
                print(f"   Execution time: {data.get(b'execution_time_ms', b'?').decode()}ms")
                return
        
        print(f"   Waiting... ({i+1}/15)")
    
    print("\n⚠️  No completion event received within 15 seconds")
    print("   Check quantum-emergency-exit-worker logs")


def main():
    parser = argparse.ArgumentParser(description="Test panic_close trigger")
    parser.add_argument(
        "--source", 
        choices=["risk_kernel", "exit_brain", "watchdog", "ops"],
        default="ops",
        help="Trigger source / issued_by (default: ops)"
    )
    parser.add_argument(
        "--reason",
        default="TESTNET_DEPLOYMENT_TEST",
        help="Trigger reason"
    )
    parser.add_argument(
        "--redis",
        default="redis://localhost:6379",
        help="Redis URL"
    )
    
    args = parser.parse_args()
    
    trigger_panic_close(args.source, args.reason, args.redis)


if __name__ == "__main__":
    main()
