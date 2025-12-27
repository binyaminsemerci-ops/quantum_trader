#!/usr/bin/env python3
"""Test ServiceHealth.create() with realistic dependencies."""

import sys
sys.path.insert(0, "/home/belen/quantum_trader")

from datetime import datetime, timezone
from backend.core.health_contract import ServiceHealth, DependencyHealth, DependencyStatus

# Simulate what the service does
dependencies = {
    "redis": DependencyHealth(
        name="redis",
        status=DependencyStatus.OK,
        latency_ms=1.2,
        details={"host": "localhost", "port": 6379}
    ),
    "eventbus": DependencyHealth(
        name="eventbus",
        status=DependencyStatus.OK,
        details={"subscriptions": 4}
    )
}

metrics = {
    "total_predictions": 0,
    "active_models": 0,
    "avg_inference_time": 0.0
}

start_time = datetime.now(timezone.utc)

print("=" * 60)
print("Testing ServiceHealth.create() with realistic data...")
print("=" * 60)

try:
    health = ServiceHealth.create(
        service_name="ai-engine-service",
        version="1.0.0",
        start_time=start_time,
        dependencies=dependencies,
        metrics=metrics
    )
    print("✅ SUCCESS!")
    print(f"Status: {health.status}")
    print(f"Service: {health.service}")
    print(f"Dependencies: {len(health.dependencies)}")
    
    # Test to_dict()
    print("\n" + "=" * 60)
    print("Testing to_dict()...")
    print("=" * 60)
    health_dict = health.to_dict()
    print("✅ to_dict() SUCCESS!")
    print(f"Keys: {list(health_dict.keys())}")
    
except Exception as e:
    print(f"❌ FAILED: {e}")
    import traceback
    traceback.print_exc()

print("=" * 60)
