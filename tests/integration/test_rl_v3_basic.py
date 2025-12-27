"""
Integration tests for RL v3 PPO system.
"""

import pytest
import numpy as np

from backend.domains.learning.rl_v3.rl_manager_v3 import RLv3Manager
from backend.domains.learning.rl_v3.config_v3 import RLv3Config


def test_rl_v3_predict():
    """Test RL v3 prediction."""
    # Create manager
    config = RLv3Config()
    manager = RLv3Manager(config)
    
    # Create fake observation
    obs_dict = {
        'price_change_1m': 0.001,
        'price_change_5m': 0.005,
        'price_change_15m': 0.01,
        'volatility': 0.02,
        'rsi': 55.0,
        'macd': 0.5,
        'position_size': 0.0,
        'position_side': 0.0,
        'balance': 10000.0,
        'equity': 10000.0,
        'regime': 'TREND',
        'trend_strength': 0.7,
        'volume_ratio': 1.2,
        'spread': 0.001,
        'time_of_day': 0.5
    }
    
    # Get prediction
    result = manager.predict(obs_dict)
    
    # Validate result
    assert 'action' in result
    assert 'confidence' in result
    assert 'value' in result
    
    # Validate action
    assert isinstance(result['action'], (int, np.integer))
    assert 0 <= result['action'] <= 5
    
    # Validate confidence
    assert isinstance(result['confidence'], float)
    assert 0.0 <= result['confidence'] <= 1.0
    
    print(f"✅ Prediction test passed: action={result['action']}, confidence={result['confidence']:.2f}")


def test_rl_v3_train_smoke():
    """Smoke test for RL v3 training."""
    # Create manager with small config
    config = RLv3Config()
    config.buffer_size = 128
    config.batch_size = 32
    config.n_epochs = 2
    config.max_steps_per_episode = 100
    
    manager = RLv3Manager(config)
    
    # Train for 2 episodes (quick test)
    metrics = manager.train(num_episodes=2)
    
    # Validate metrics
    assert 'total_rewards' in metrics
    assert 'policy_losses' in metrics
    assert 'value_losses' in metrics
    assert 'avg_reward' in metrics
    assert 'final_reward' in metrics
    
    assert len(metrics['total_rewards']) == 2
    assert len(metrics['policy_losses']) == 2
    assert len(metrics['value_losses']) == 2
    
    print(f"✅ Training smoke test passed: avg_reward={metrics['avg_reward']:.2f}")


if __name__ == '__main__':
    print("Running RL v3 integration tests...")
    print()
    
    test_rl_v3_predict()
    test_rl_v3_train_smoke()
    
    print()
    print("✅ All RL v3 tests passed!")
