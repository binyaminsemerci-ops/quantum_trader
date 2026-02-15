#!/usr/bin/env python3
"""
Emergency Exit Worker Mock for CI Testing

Listens for system:panic_close and:
1. "Closes" simulated positions
2. Publishes system:panic_close:completed
3. Sets system:state:trading to halted

NO REAL EXCHANGE API CALLS - pure behavior test.
"""

import os
import sys
import time
import signal
import json
import redis

# Configuration
REDIS_URL = os.getenv("REDIS_URL", "redis://localhost:6379")
PANIC_CLOSE_STREAM = "system:panic_close"
PANIC_COMPLETE_STREAM = "system:panic_close:completed"
TRADING_HALT_KEY = "system:state:trading"
POSITIONS_KEY = "mock:open_positions"  # CI-only key for tracking

CONSUMER_GROUP = "emergency_exit_worker"
CONSUMER_NAME = "eew_ci_mock"


def main():
    print("=" * 50)
    print("EMERGENCY EXIT WORKER MOCK STARTING")
    print("=" * 50)
    print(f"Redis: {REDIS_URL}")
    print(f"Listening on: {PANIC_CLOSE_STREAM}")
    print("=" * 50)
    
    # Connect to Redis
    r = redis.from_url(REDIS_URL)
    r.ping()
    print("✅ Connected to Redis")
    
    # Initialize simulated positions (3 mock positions)
    r.set(POSITIONS_KEY, "3")
    print("✅ Initialized 3 simulated positions")
    
    # Ensure consumer group exists
    try:
        r.xgroup_create(PANIC_CLOSE_STREAM, CONSUMER_GROUP, "$", mkstream=True)
    except redis.ResponseError as e:
        if "BUSYGROUP" not in str(e):
            raise
    
    running = True
    
    def shutdown(sig, frame):
        nonlocal running
        print(f"\n⚠️ Received signal {sig}")
        running = False
    
    signal.signal(signal.SIGTERM, shutdown)
    signal.signal(signal.SIGINT, shutdown)
    
    print("👂 Listening for panic_close events...")
    
    # Main listen loop
    while running:
        try:
            # Read from stream
            messages = r.xreadgroup(
                CONSUMER_GROUP,
                CONSUMER_NAME,
                {PANIC_CLOSE_STREAM: ">"},
                count=1,
                block=1000
            )
            
            if not messages:
                continue
            
            for stream_name, stream_messages in messages:
                for msg_id, data in stream_messages:
                    # Handle panic_close event
                    handle_panic_close(r, msg_id, data)
                    
                    # Acknowledge
                    r.xack(PANIC_CLOSE_STREAM, CONSUMER_GROUP, msg_id)
            
        except redis.ConnectionError as e:
            print(f"❌ Redis connection lost: {e}")
            time.sleep(1)
        except Exception as e:
            print(f"❌ Error: {e}")
            time.sleep(1)
    
    print("EEW mock stopped")


def handle_panic_close(r: redis.Redis, msg_id: str, data: dict):
    """Handle panic_close event - close all mock positions"""
    
    print("=" * 50)
    print("🚨 PANIC CLOSE EVENT RECEIVED 🚨")
    print("=" * 50)
    
    # Parse event
    event_id = data.get(b"event_id", b"unknown").decode()
    reason = data.get(b"reason", b"unknown").decode()
    issued_by = data.get(b"issued_by", b"unknown").decode()
    
    print(f"Event ID: {event_id}")
    print(f"Reason: {reason}")
    print(f"Issued by: {issued_by}")
    
    ts_started = int(time.time() * 1000)
    
    # Get simulated positions
    positions_total = int(r.get(POSITIONS_KEY) or 0)
    print(f"Simulated positions to close: {positions_total}")
    
    # "Close" all positions
    positions_closed = 0
    for i in range(positions_total):
        print(f"  ✅ Closing position {i+1}/{positions_total}")
        time.sleep(0.1)  # Simulate order latency
        positions_closed += 1
    
    # Set positions to 0
    r.set(POSITIONS_KEY, "0")
    
    ts_completed = int(time.time() * 1000)
    execution_time_ms = ts_completed - ts_started
    
    # Publish completion
    r.xadd(
        PANIC_COMPLETE_STREAM,
        {
            "event_id": event_id,
            "positions_total": str(positions_total),
            "positions_closed": str(positions_closed),
            "positions_failed": "0",
            "failed_symbols": "[]",
            "ts_started": str(ts_started),
            "ts_completed": str(ts_completed),
            "execution_time_ms": str(execution_time_ms)
        }
    )
    
    print(f"✅ Published to {PANIC_COMPLETE_STREAM}")
    
    # Halt trading
    r.hset(TRADING_HALT_KEY, mapping={
        "halted": "true",
        "reason": reason,
        "source": issued_by,
        "event_id": event_id,
        "ts": str(ts_completed),
        "positions_closed": str(positions_closed),
        "requires_manual_reset": "true"
    })
    
    print(f"✅ Trading halted in {TRADING_HALT_KEY}")
    
    print("=" * 50)
    print(f"🔒 PANIC CLOSE COMPLETE")
    print(f"   Positions closed: {positions_closed}/{positions_total}")
    print(f"   Execution time: {execution_time_ms}ms")
    print("=" * 50)


if __name__ == "__main__":
    main()
