#!/usr/bin/env python3
"""
Exit Brain Mock for CI Testing

Simulates Exit Brain by:
1. Publishing heartbeats to exit_brain:heartbeat every 1 second
2. Tracking simulated open positions

This mock is killed by CI to simulate Exit Brain failure.
"""

import os
import sys
import time
import signal
import redis

# Configuration
REDIS_URL = os.getenv("REDIS_URL", "redis://localhost:6379")
HEARTBEAT_STREAM = "exit_brain:heartbeat"
HEARTBEAT_INTERVAL = 1.0  # seconds

# Simulated state
SIMULATED_POSITIONS = 3  # Simulated open positions


def main():
    print("=" * 50)
    print("EXIT BRAIN MOCK STARTING")
    print("=" * 50)
    print(f"Redis: {REDIS_URL}")
    print(f"Heartbeat interval: {HEARTBEAT_INTERVAL}s")
    print(f"Simulated positions: {SIMULATED_POSITIONS}")
    print("=" * 50)
    
    # Connect to Redis
    r = redis.from_url(REDIS_URL)
    
    # Verify connection
    r.ping()
    print("✅ Connected to Redis")
    
    # Signal handler for graceful shutdown
    running = True
    
    def shutdown(sig, frame):
        nonlocal running
        print(f"\n⚠️ Received signal {sig}, shutting down...")
        running = False
    
    signal.signal(signal.SIGTERM, shutdown)
    signal.signal(signal.SIGINT, shutdown)
    
    # Main heartbeat loop
    loop_count = 0
    while running:
        try:
            loop_count += 1
            ts = int(time.time() * 1000)  # Epoch ms
            
            # Publish heartbeat per schema
            r.xadd(
                HEARTBEAT_STREAM,
                {
                    "status": "ALIVE",
                    "active_positions": str(SIMULATED_POSITIONS),
                    "ts": str(ts),
                    "loop_cycle_ms": str(int(HEARTBEAT_INTERVAL * 1000))
                },
                maxlen=100
            )
            
            if loop_count % 5 == 0:
                print(f"💓 Heartbeat #{loop_count} published (ts={ts})")
            
            time.sleep(HEARTBEAT_INTERVAL)
            
        except redis.ConnectionError as e:
            print(f"❌ Redis connection lost: {e}")
            time.sleep(1)
        except Exception as e:
            print(f"❌ Error: {e}")
            time.sleep(1)
    
    print("Exit Brain mock stopped")


if __name__ == "__main__":
    main()
