"""
Configuration for RL v3 (PPO)
"""

from dataclasses import dataclass


@dataclass
class RLv3Config:
    """PPO hyperparameters and system config."""
    
    # Network architecture
    state_dim: int = 64
    hidden_dim: int = 128
    action_dim: int = 6
    
    # PPO hyperparameters
    learning_rate: float = 3e-4
    gamma: float = 0.99
    lambda_gae: float = 0.95
    clip_range: float = 0.2
    entropy_coef: float = 0.01
    value_coef: float = 0.5
    max_grad_norm: float = 0.5
    
    # Training
    n_epochs: int = 10
    batch_size: int = 64
    buffer_size: int = 2048
    
    # Environment
    max_steps_per_episode: int = 1000
    initial_balance: float = 10000.0
    max_position_size: float = 1.0
    
    # Model persistence
    model_path: str = "data/rl_v3/ppo_model.pt"
    
    # Feature processing
    normalize_features: bool = True
    feature_clip_range: float = 5.0


DEFAULT_CONFIG = RLv3Config()
