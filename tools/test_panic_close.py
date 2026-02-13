#!/usr/bin/env python3
"""
Test Panic Close Trigger

USE ONLY ON TESTNET!

This script triggers a system.panic_close event for testing purposes.
"""

import sys
import time
import argparse

try:
    import redis
except ImportError:
    print("Missing redis-py: pip install redis")
    sys.exit(1)


PANIC_CLOSE_STREAM = "quantum:stream:system.panic_close"


def trigger_panic_close(source: str, reason: str, redis_url: str = "redis://localhost:6379"):
    """Trigger a panic_close event"""
    
    print("=" * 60)
    print("⚠️  PANIC CLOSE TRIGGER")
    print("=" * 60)
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
    
    # Publish event
    event = {
        "source": source,
        "reason": reason,
        "timestamp": str(time.time()),
        "severity": "CRITICAL"
    }
    
    msg_id = r.xadd(PANIC_CLOSE_STREAM, event)
    
    print(f"\n✅ Event published!")
    print(f"   Stream: {PANIC_CLOSE_STREAM}")
    print(f"   Message ID: {msg_id}")
    
    # Wait and check for completion
    print("\nWaiting for panic_close.completed...")
    
    for i in range(10):
        time.sleep(1)
        
        # Check for completion
        messages = r.xrevrange("quantum:stream:panic_close.completed", count=1)
        if messages:
            msg_id, data = messages[0]
            if float(data.get(b'timestamp', 0)) > time.time() - 15:
                print("\n✅ Panic close completed!")
                print(f"   Positions found: {data.get(b'positions_found', b'?').decode()}")
                print(f"   Positions closed: {data.get(b'positions_closed', b'?').decode()}")
                print(f"   Positions failed: {data.get(b'positions_failed', b'?').decode()}")
                print(f"   Execution time: {data.get(b'execution_time_ms', b'?').decode()}ms")
                return
        
        print(f"   Waiting... ({i+1}/10)")
    
    print("\n⚠️  No completion event received within 10 seconds")
    print("   Check quantum-emergency-exit-worker logs")


def main():
    parser = argparse.ArgumentParser(description="Test panic_close trigger")
    parser.add_argument(
        "--source", 
        choices=["risk_kernel", "exit_brain", "ops"],
        default="ops",
        help="Trigger source (default: ops)"
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
