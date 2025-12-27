"""
RL v3 - PPO-based Reinforcement Learning System
================================================

Complete PPO implementation for trading with:
- Policy gradient learning (PPO with GAE)
- Neural network policy and value functions
- Discrete action space (6 actions)
- Vector observation space
- Clean interface via RLv3Manager

Author: Quantum Trader AI Team
Date: December 2, 2025
Version: 3.0
"""

from backend.domains.learning.rl_v3.rl_manager_v3 import RLv3Manager
from backend.domains.learning.rl_v3.ppo_agent_v3 import PPOAgent
from backend.domains.learning.rl_v3.env_v3 import TradingEnvV3

__all__ = [
    "RLv3Manager",
    "PPOAgent",
    "TradingEnvV3",
]

__version__ = "3.0.0"
