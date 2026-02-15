#!/usr/bin/env python3
"""
Exit Brain Watchdog Mock for CI Testing

Monitors exit_brain:heartbeat and triggers system:panic_close
when heartbeat is missing for > 5 seconds.

THIS IS A SIMPLIFIED MOCK FOR CI - the real watchdog has more checks.
"""

import os
import sys
import time
import signal
import uuid
import redis

# Configuration
REDIS_URL = os.getenv("REDIS_URL", "redis://localhost:6379")
HEARTBEAT_STREAM = "exit_brain:heartbeat"
PANIC_CLOSE_STREAM = "system:panic_close"

# Thresholds (match production)
HEARTBEAT_MISSING_THRESHOLD = 5.0  # seconds
CHECK_INTERVAL = 1.0  # seconds


def main():
    print("=" * 50)
    print("WATCHDOG MOCK STARTING")
    print("=" * 50)
    print(f"Redis: {REDIS_URL}")
    print(f"Heartbeat threshold: {HEARTBEAT_MISSING_THRESHOLD}s")
    print("=" * 50)
    
    # Connect to Redis
    r = redis.from_url(REDIS_URL)
    r.ping()
    print("✅ Connected to Redis")
    
    # State
    last_heartbeat_ts = time.time()
    panic_triggered = False
    running = True
    
    def shutdown(sig, frame):
        nonlocal running
        print(f"\n⚠️ Received signal {sig}")
        running = False
    
    signal.signal(signal.SIGTERM, shutdown)
    signal.signal(signal.SIGINT, shutdown)
    
    # Main monitoring loop
    while running and not panic_triggered:
        try:
            # Read latest heartbeat
            messages = r.xrevrange(HEARTBEAT_STREAM, count=1)
            
            if messages:
                msg_id, data = messages[0]
                # Parse ts (int64 epoch ms)
                ts_ms = int(data.get(b"ts", 0))
                if ts_ms > 0:
                    last_heartbeat_ts = ts_ms / 1000.0
            
            # Check heartbeat age
            heartbeat_age = time.time() - last_heartbeat_ts
            
            if heartbeat_age > HEARTBEAT_MISSING_THRESHOLD:
                print("=" * 50)
                print("🚨 FAILURE DETECTED 🚨")
                print(f"Heartbeat missing: {heartbeat_age:.1f}s > {HEARTBEAT_MISSING_THRESHOLD}s")
                print("=" * 50)
                
                # Trigger panic_close
                event_id = str(uuid.uuid4())
                ts = int(time.time() * 1000)
                
                r.xadd(
                    PANIC_CLOSE_STREAM,
                    {
                        "event_id": event_id,
                        "reason": "EXIT_BRAIN_HEARTBEAT_MISSING",
                        "severity": "CRITICAL",
                        "issued_by": "watchdog",
                        "ts": str(ts)
                    }
                )
                
                print(f"✅ panic_close published")
                print(f"   Event ID: {event_id}")
                print(f"   Stream: {PANIC_CLOSE_STREAM}")
                
                panic_triggered = True
                break
            
            time.sleep(CHECK_INTERVAL)
            
        except redis.ConnectionError as e:
            print(f"❌ Redis connection lost: {e}")
            time.sleep(1)
        except Exception as e:
            print(f"❌ Error: {e}")
            time.sleep(1)
    
    # Keep running after trigger (for CI to read state)
    while running:
        time.sleep(1)
    
    print("Watchdog mock stopped")


if __name__ == "__main__":
    main()
