"""
Test PolicyStore integration with RL v3 Training Daemon.
"""

import asyncio
from backend.domains.learning.rl_v3.rl_manager_v3 import RLv3Manager
from backend.domains.learning.rl_v3.config_v3 import RLv3Config
from backend.domains.learning.rl_v3.training_config_v3 import load_training_config_from_policy_store
from backend.services.ai.rl_v3_training_daemon import RLv3TrainingDaemon
from backend.domains.learning.rl_v3.metrics_v3 import RLv3MetricsStore


class MockPolicyStore:
    """Mock PolicyStore for testing."""
    
    def __init__(self):
        self.policies = {
            "rl_v3.training.enabled": True,
            "rl_v3.training.interval_minutes": 5,
            "rl_v3.training.episodes_per_run": 2,
            "rl_v3.training.save_after_each_run": False
        }
    
    def get(self, key, default=None):
        return self.policies.get(key, default)
    
    def set(self, key, value):
        self.policies[key] = value


async def test_policy_store_integration():
    """Test that daemon loads config from PolicyStore."""
    
    print("🧪 Testing PolicyStore integration...")
    
    # Create mock policy store
    policy_store = MockPolicyStore()
    
    # Test 1: Load config from PolicyStore
    print("\n1️⃣ Testing config loading from PolicyStore...")
    config = load_training_config_from_policy_store(policy_store)
    
    assert config.schedule.enabled == True
    assert config.schedule.interval_minutes == 5
    assert config.schedule.episodes_per_run == 2
    assert config.schedule.save_after_each_run == False
    
    print(f"   ✅ Config loaded: interval={config.schedule.interval_minutes}min, episodes={config.schedule.episodes_per_run}")
    
    # Test 2: Create daemon with PolicyStore
    print("\n2️⃣ Testing daemon creation with PolicyStore...")
    rl_config = RLv3Config()
    rl_config.buffer_size = 10
    rl_manager = RLv3Manager(config=rl_config)
    
    daemon = RLv3TrainingDaemon(
        rl_manager=rl_manager,
        config=None,  # Should load from PolicyStore
        logger=None,
        policy_store=policy_store
    )
    
    assert daemon.config.schedule.interval_minutes == 5
    assert daemon.config.schedule.episodes_per_run == 2
    print(f"   ✅ Daemon config: interval={daemon.config.schedule.interval_minutes}min, episodes={daemon.config.schedule.episodes_per_run}")
    
    # Test 3: Update PolicyStore and refresh
    print("\n3️⃣ Testing config refresh from PolicyStore...")
    policy_store.set("rl_v3.training.interval_minutes", 10)
    policy_store.set("rl_v3.training.episodes_per_run", 5)
    
    daemon._refresh_config()
    
    assert daemon.config.schedule.interval_minutes == 10
    assert daemon.config.schedule.episodes_per_run == 5
    print(f"   ✅ Config refreshed: interval={daemon.config.schedule.interval_minutes}min, episodes={daemon.config.schedule.episodes_per_run}")
    
    # Test 4: Disable training via PolicyStore
    print("\n4️⃣ Testing disable training via PolicyStore...")
    policy_store.set("rl_v3.training.enabled", False)
    daemon._refresh_config()
    
    assert daemon.config.schedule.enabled == False
    print(f"   ✅ Training disabled: enabled={daemon.config.schedule.enabled}")
    
    print("\n✅ All PolicyStore integration tests passed!")


if __name__ == "__main__":
    asyncio.run(test_policy_store_integration())
