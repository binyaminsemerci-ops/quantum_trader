"""
RL v2 Domain - Advanced Reinforcement Learning System
======================================================

Complete RL v2 implementation with:
- Reward Engine v2 (regime-aware, risk-aware)
- State Builder v2 (advanced state representation)
- Action Space v2 (expanded actions)
- Episode Tracker v2 (episodic rewards, discounting)
- Q-Learning Core (TD-learning, Q-tables)
- Meta Strategy Agent v2
- Position Sizing Agent v2

Author: Quantum Trader AI Team
Date: December 2, 2025
Version: 2.0
"""

from backend.domains.learning.rl_v2.reward_engine_v2 import RewardEngineV2
from backend.domains.learning.rl_v2.state_builder_v2 import StateBuilderV2
from backend.domains.learning.rl_v2.action_space_v2 import ActionSpaceV2
from backend.domains.learning.rl_v2.episode_tracker_v2 import EpisodeTrackerV2
from backend.domains.learning.rl_v2.q_learning_core import QLearningCore
from backend.domains.learning.rl_v2.meta_strategy_agent_v2 import MetaStrategyAgentV2
from backend.domains.learning.rl_v2.position_sizing_agent_v2 import PositionSizingAgentV2

__all__ = [
    "RewardEngineV2",
    "StateBuilderV2",
    "ActionSpaceV2",
    "EpisodeTrackerV2",
    "QLearningCore",
    "MetaStrategyAgentV2",
    "PositionSizingAgentV2",
]

__version__ = "2.0.0"
