"""
Simple integration tests for RL v3 without Redis dependencies.
"""

import pytest
from backend.domains.learning.rl_v3.rl_manager_v3 import RLv3Manager
from backend.domains.learning.rl_v3.config_v3 import RLv3Config


def test_rl_v3_predict_basic():
    """Test RL v3 prediction with basic observation."""
    config = RLv3Config()
    manager = RLv3Manager(config)
    
    obs = {
        "price_change_1m": 0.002,
        "price_change_5m": 0.008,
        "price_change_15m": 0.015,
        "volatility": 0.025,
        "rsi": 60.0,
        "macd": 1.2,
        "position_size": 0.0,
        "position_side": 0.0,
        "balance": 10000.0,
        "equity": 10000.0,
        "regime": "TREND",
        "trend_strength": 0.8,
        "volume_ratio": 1.5,
        "spread": 0.001,
        "time_of_day": 0.6
    }
    
    result = manager.predict(obs)
    
    assert 'action' in result
    assert 'confidence' in result
    assert 'value' in result
    assert 0 <= result['action'] <= 5
    assert 0.0 <= result['confidence'] <= 1.0
    
    print(f"✅ Basic predict test passed: action={result['action']}, confidence={result['confidence']:.3f}")


def test_rl_v3_multiple_predictions():
    """Test RL v3 with multiple different observations."""
    config = RLv3Config()
    manager = RLv3Manager(config)
    
    observations = [
        {
            "price_change_1m": 0.005,
            "volatility": 0.03,
            "rsi": 70.0,
            "position_size": 0.0,
            "balance": 10000.0,
            "equity": 10000.0
        },
        {
            "price_change_1m": -0.003,
            "volatility": 0.02,
            "rsi": 30.0,
            "position_size": 0.1,
            "balance": 10000.0,
            "equity": 10050.0
        },
        {
            "price_change_1m": 0.001,
            "volatility": 0.015,
            "rsi": 50.0,
            "position_size": 0.05,
            "balance": 10000.0,
            "equity": 10100.0
        }
    ]
    
    results = []
    for obs in observations:
        result = manager.predict(obs)
        results.append(result)
        assert 'action' in result
        assert 0 <= result['action'] <= 5
    
    print(f"✅ Multiple predictions test passed: {len(results)} predictions")


def test_rl_v3_train_small_batch():
    """Test RL v3 training with small episode count."""
    config = RLv3Config()
    config.buffer_size = 128
    config.batch_size = 32
    config.n_epochs = 2
    config.max_steps_per_episode = 50
    
    manager = RLv3Manager(config)
    
    # Train for 2 episodes
    metrics = manager.train(num_episodes=2)
    
    assert 'total_rewards' in metrics
    assert 'policy_losses' in metrics
    assert 'value_losses' in metrics
    assert 'avg_reward' in metrics
    assert len(metrics['total_rewards']) == 2
    
    print(f"✅ Small batch training test passed: avg_reward={metrics['avg_reward']:.2f}")


def test_rl_v3_save_load():
    """Test RL v3 model save and load."""
    from pathlib import Path
    import tempfile
    
    config = RLv3Config()
    manager1 = RLv3Manager(config)
    
    obs = {
        "price_change_1m": 0.001,
        "volatility": 0.02,
        "rsi": 55.0,
        "position_size": 0.0,
        "balance": 10000.0,
        "equity": 10000.0
    }
    
    # Get prediction before save
    result1 = manager1.predict(obs)
    
    # Save model
    with tempfile.TemporaryDirectory() as tmpdir:
        model_path = Path(tmpdir) / "test_model.pt"
        manager1.save(model_path)
        
        assert model_path.exists()
        
        # Load model in new manager
        manager2 = RLv3Manager(config)
        manager2.load(model_path)
        
        # Get prediction from loaded model
        result2 = manager2.predict(obs)
        
        # Predictions should be identical
        assert result1['action'] == result2['action']
        assert abs(result1['confidence'] - result2['confidence']) < 0.001
        assert abs(result1['value'] - result2['value']) < 0.001
    
    print(f"✅ Save/load test passed")


def test_rl_v3_action_mapping():
    """Test action code to name mapping."""
    from backend.events.subscribers.rl_subscriber_v3 import RLSubscriberV3
    from backend.domains.learning.rl_v3.config_v3 import RLv3Config
    
    config = RLv3Config()
    
    # Create a mock event bus (we won't use it)
    class MockEventBus:
        def subscribe(self, *args, **kwargs):
            pass
    
    subscriber = RLSubscriberV3(MockEventBus(), config, shadow_mode=True)
    
    action_names = [
        subscriber._map_action_to_name(i) 
        for i in range(6)
    ]
    
    expected = ["HOLD", "LONG", "SHORT", "REDUCE", "CLOSE", "EMERGENCY_FLATTEN"]
    
    assert action_names == expected
    
    print(f"✅ Action mapping test passed: {action_names}")


def test_rl_v3_observation_builder():
    """Test observation building from partial data."""
    from backend.events.subscribers.rl_subscriber_v3 import RLSubscriberV3
    from backend.domains.learning.rl_v3.config_v3 import RLv3Config
    
    config = RLv3Config()
    
    class MockEventBus:
        def subscribe(self, *args, **kwargs):
            pass
    
    subscriber = RLSubscriberV3(MockEventBus(), config, shadow_mode=True)
    
    # Minimal data
    data = {
        "symbol": "BTCUSDT",
        "price_change_1m": 0.001
    }
    
    obs = subscriber._build_observation(data)
    
    # Should fill in defaults
    assert 'price_change_1m' in obs
    assert 'volatility' in obs
    assert 'rsi' in obs
    assert obs['price_change_1m'] == 0.001
    assert obs['volatility'] == 0.02  # default
    assert obs['rsi'] == 50.0  # default
    
    print(f"✅ Observation builder test passed")


if __name__ == '__main__':
    print("Running RL v3 simple integration tests...\n")
    
    test_rl_v3_predict_basic()
    test_rl_v3_multiple_predictions()
    test_rl_v3_train_small_batch()
    test_rl_v3_save_load()
    test_rl_v3_action_mapping()
    test_rl_v3_observation_builder()
    
    print("\n✅ All RL v3 simple integration tests passed!")
