"""
RL v3 Sandbox - Experimentation script for PPO-based RL system.
"""

from pathlib import Path
from backend.domains.learning.rl_v3.rl_manager_v3 import RLv3Manager
from backend.domains.learning.rl_v3.config_v3 import RLv3Config


def main():
    """Main sandbox function."""
    print("=" * 60)
    print("RL v3 PPO Sandbox")
    print("=" * 60)
    print()
    
    # Create custom config
    config = RLv3Config()
    config.buffer_size = 256
    config.batch_size = 64
    config.n_epochs = 5
    config.max_steps_per_episode = 200
    
    print("Configuration:")
    print(f"  State dim: {config.state_dim}")
    print(f"  Action dim: {config.action_dim}")
    print(f"  Hidden dim: {config.hidden_dim}")
    print(f"  Learning rate: {config.learning_rate}")
    print(f"  Buffer size: {config.buffer_size}")
    print(f"  Batch size: {config.batch_size}")
    print()
    
    # Create manager
    manager = RLv3Manager(config)
    print("✅ RL v3 Manager created")
    print()
    
    # Test prediction with fake observation
    print("Testing prediction...")
    obs_dict = {
        'price_change_1m': 0.002,
        'price_change_5m': 0.008,
        'price_change_15m': 0.015,
        'volatility': 0.025,
        'rsi': 60.0,
        'macd': 1.2,
        'position_size': 0.0,
        'position_side': 0.0,
        'balance': 10000.0,
        'equity': 10000.0,
        'regime': 'TREND',
        'trend_strength': 0.8,
        'volume_ratio': 1.5,
        'spread': 0.001,
        'time_of_day': 0.6
    }
    
    result = manager.predict(obs_dict)
    print(f"  Action: {result['action']}")
    print(f"  Confidence: {result['confidence']:.3f}")
    print(f"  Value: {result['value']:.3f}")
    print()
    
    # Train for a few episodes
    print("Training for 5 episodes...")
    metrics = manager.train(num_episodes=5)
    print()
    
    print("Training Results:")
    print(f"  Average reward: {metrics['avg_reward']:.2f}")
    print(f"  Final reward: {metrics['final_reward']:.2f}")
    print(f"  Avg policy loss: {sum(metrics['policy_losses']) / len(metrics['policy_losses']):.4f}")
    print(f"  Avg value loss: {sum(metrics['value_losses']) / len(metrics['value_losses']):.4f}")
    print()
    
    # Test prediction after training
    print("Testing prediction after training...")
    result = manager.predict(obs_dict)
    print(f"  Action: {result['action']}")
    print(f"  Confidence: {result['confidence']:.3f}")
    print(f"  Value: {result['value']:.3f}")
    print()
    
    # Save model
    model_path = Path("data/rl_v3/sandbox_model.pt")
    manager.save(model_path)
    print()
    
    # Test loading
    print("Testing model load...")
    manager2 = RLv3Manager(config)
    manager2.load(model_path)
    result2 = manager2.predict(obs_dict)
    print(f"  Loaded model action: {result2['action']}")
    print(f"  Loaded model confidence: {result2['confidence']:.3f}")
    print()
    
    print("=" * 60)
    print("✅ Sandbox test complete!")
    print("=" * 60)


if __name__ == '__main__':
    main()
