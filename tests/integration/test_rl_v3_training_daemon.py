"""
Integration test for RL v3 Training Daemon.

Tests the production training daemon with EventBus, PolicyStore, and dashboard API.
"""

import asyncio
import pytest
from unittest.mock import Mock

from backend.domains.learning.rl_v3.rl_manager_v3 import RLv3Manager
from backend.domains.learning.rl_v3.config_v3 import RLv3Config
from backend.domains.learning.rl_v3.training_daemon_v3 import RLv3TrainingDaemon
from backend.domains.learning.rl_v3.metrics_v3 import RLv3MetricsStore


class MockPolicyStore:
    """Mock PolicyStore for testing."""
    def __init__(self):
        self.policies = {
            "rl_v3.training.enabled": True,
            "rl_v3.training.interval_minutes": 5,
            "rl_v3.training.episodes_per_run": 1
        }
    
    def get(self, key, default=None):
        return self.policies.get(key, default)


class MockEventBus:
    """Mock EventBus for testing."""
    def __init__(self):
        self.published_events = []
    
    async def publish(self, event_type, payload, trace_id=None):
        self.published_events.append({
            "event_type": event_type,
            "payload": payload,
            "trace_id": trace_id
        })


@pytest.mark.asyncio
async def test_daemon_instantiation():
    """Test 1: Daemon instantiates and reads PolicyStore."""
    print("\n🧪 Test 1: Daemon instantiation with PolicyStore")
    
    metrics = RLv3MetricsStore.instance()
    metrics.clear()
    
    rl_config = RLv3Config()
    rl_config.buffer_size = 10
    rl_manager = RLv3Manager(config=rl_config)
    
    policy_store = MockPolicyStore()
    daemon = RLv3TrainingDaemon(
        rl_manager=rl_manager,
        policy_store=policy_store
    )
    
    assert daemon.config["enabled"] == True
    assert daemon.config["interval_minutes"] == 5
    assert daemon.config["episodes_per_run"] == 1
    
    print("   ✅ Daemon created with PolicyStore config")


@pytest.mark.asyncio
async def test_manual_training_run():
    """Test 2: Manual run_once() executes training."""
    print("\n🧪 Test 2: Manual training run")
    
    metrics = RLv3MetricsStore.instance()
    metrics.clear()
    
    rl_config = RLv3Config()
    rl_config.buffer_size = 10
    rl_manager = RLv3Manager(config=rl_config)
    
    daemon = RLv3TrainingDaemon(
        rl_manager=rl_manager,
        policy_store=MockPolicyStore()
    )
    
    result = await daemon.run_once()
    
    assert result is not None
    assert "run_id" in result
    assert result["success"] == True
    assert result["episodes"] == 1
    
    summary = metrics.get_training_summary()
    assert summary["total_runs"] >= 1
    
    print(f"   ✅ Training run completed (run_id={result['run_id']})")


@pytest.mark.asyncio
async def test_eventbus_events():
    """Test 3: EventBus events are published."""
    print("\n🧪 Test 3: EventBus event publishing")
    
    metrics = RLv3MetricsStore.instance()
    metrics.clear()
    
    rl_config = RLv3Config()
    rl_config.buffer_size = 10
    rl_manager = RLv3Manager(config=rl_config)
    
    event_bus = MockEventBus()
    daemon = RLv3TrainingDaemon(
        rl_manager=rl_manager,
        event_bus=event_bus,
        policy_store=MockPolicyStore()
    )
    
    await daemon.run_once()
    
    assert len(event_bus.published_events) >= 2
    
    event_types = [e["event_type"] for e in event_bus.published_events]
    assert "rl_v3.training.started" in event_types
    assert "rl_v3.training.completed" in event_types
    
    print(f"   ✅ EventBus events published: {event_types}")


@pytest.mark.asyncio
async def test_daemon_shutdown():
    """Test 4: Daemon shutdown is clean."""
    print("\n🧪 Test 4: Daemon shutdown")
    
    rl_config = RLv3Config()
    rl_config.buffer_size = 10
    rl_manager = RLv3Manager(config=rl_config)
    
    daemon = RLv3TrainingDaemon(
        rl_manager=rl_manager,
        policy_store=MockPolicyStore()
    )
    
    await daemon.start()
    assert daemon._running == True
    
    await daemon.stop()
    assert daemon._running == False
    assert daemon._task is None
    
    print("   ✅ Daemon shutdown cleanly")


if __name__ == "__main__":
    print("=" * 60)
    print("🚀 RL v3 Training Daemon Integration Tests")
    print("=" * 60)
    
    asyncio.run(test_daemon_instantiation())
    asyncio.run(test_manual_training_run())
    asyncio.run(test_eventbus_events())
    asyncio.run(test_daemon_shutdown())
    
    print("\n" + "=" * 60)
    print("✅ ALL TESTS PASSED!")
    print("=" * 60)
