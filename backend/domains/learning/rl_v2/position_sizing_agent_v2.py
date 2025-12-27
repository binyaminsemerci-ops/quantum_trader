"""
Position Sizing Agent v2 - Position Sizing Optimization
========================================================

Uses RL v2 to optimize position sizing.

Optimizes:
- Size multiplier (0.5 - 2.0)
- Leverage (5 - 50)

Author: Quantum Trader AI Team
Date: December 2, 2025
Version: 2.0
"""

from typing import Dict, Any, Optional
from pathlib import Path
import structlog

from backend.domains.learning.rl_v2.state_builder_v2 import StateBuilderV2
from backend.domains.learning.rl_v2.action_space_v2 import ActionSpaceV2, PositionSizingAction
from backend.domains.learning.rl_v2.reward_engine_v2 import RewardEngineV2
from backend.domains.learning.rl_v2.episode_tracker_v2 import EpisodeTrackerV2
from backend.domains.learning.rl_v2.q_learning_core import QLearningCore

logger = structlog.get_logger(__name__)


class PositionSizingAgentV2:
    """
    Position sizing agent using RL v2.
    
    Optimizes position sizing decisions:
    - Size multiplier (risk adjustment)
    - Leverage (capital efficiency)
    """
    
    def __init__(
        self,
        alpha: float = 0.01,
        gamma: float = 0.99,
        epsilon: float = 0.1,
        q_table_path: Optional[Path] = None
    ):
        """
        Initialize Position Sizing Agent v2.
        
        Args:
            alpha: Learning rate
            gamma: Discount factor
            epsilon: Exploration rate
            q_table_path: Path to Q-table file
        """
        self.state_builder = StateBuilderV2()
        self.action_space = ActionSpaceV2()
        self.reward_engine = RewardEngineV2()
        self.episode_tracker = EpisodeTrackerV2(gamma=gamma)
        self.q_learning = QLearningCore(alpha=alpha, gamma=gamma, epsilon=epsilon)
        
        self.q_table_path = q_table_path or Path("data/rl_v2/position_sizing_q_table.json")
        
        # Load Q-table if exists
        self.q_learning.load_q_table(self.q_table_path)
        
        # Current state and action
        self.current_state: Optional[Dict[str, Any]] = None
        self.current_action: Optional[PositionSizingAction] = None
        
        logger.info(
            "[Position Sizing Agent v2] Initialized",
            alpha=alpha,
            gamma=gamma,
            epsilon=epsilon,
            q_table_path=str(self.q_table_path)
        )
    
    def select_action(self, market_data: Dict[str, Any]) -> PositionSizingAction:
        """
        Select position sizing action.
        
        Args:
            market_data: Current market data
            
        Returns:
            Selected position sizing action
        """
        # Build state
        self.current_state = self.state_builder.build_position_sizing_state(market_data)
        
        # Get available actions
        available_actions = self.action_space.get_position_sizing_actions()
        
        # Convert to dictionaries for Q-learning
        action_dicts = [
            self.action_space.action_to_dict(a) for a in available_actions
        ]
        
        # Select action using Q-learning
        selected_dict = self.q_learning.select_action(
            self.current_state,
            action_dicts
        )
        
        # Convert back to action object
        self.current_action = self.action_space.dict_to_sizing_action(selected_dict)
        
        logger.info(
            "[Position Sizing Agent v2] Action selected",
            size_multiplier=self.current_action.size_multiplier,
            leverage=self.current_action.leverage
        )
        
        return self.current_action
    
    def update(self, result_data: Dict[str, Any]):
        """
        Update Q-values based on results.
        
        Args:
            result_data: Trading results
        """
        if not self.current_state or not self.current_action:
            logger.warning("[Position Sizing Agent v2] No current state/action")
            return
        
        # Calculate reward
        reward = self.reward_engine.calculate_position_sizing_reward(result_data)
        
        # Build next state
        next_state = self.state_builder.build_position_sizing_state(result_data)
        
        # Get best Q-value for next state
        _, next_best_q = self.q_learning.get_best_action_value(next_state)
        
        # Update Q-value
        action_dict = self.action_space.action_to_dict(self.current_action)
        
        self.q_learning.update_q_value(
            state=self.current_state,
            action=action_dict,
            reward=reward,
            next_state=next_state,
            next_best_q=next_best_q
        )
        
        logger.info(
            "[Position Sizing Agent v2] Q-value updated",
            reward=reward,
            next_best_q=next_best_q
        )
        
        # Save Q-table periodically
        if self.q_learning.update_count % 100 == 0:
            self.save_q_table()
    
    def save_q_table(self):
        """Save Q-table to disk."""
        self.q_learning.save_q_table(self.q_table_path)
    
    def get_stats(self) -> Dict[str, Any]:
        """
        Get agent statistics.
        
        Returns:
            Statistics
        """
        return {
            "agent_type": "position_sizing",
            "q_learning_stats": self.q_learning.get_stats(),
            "episode_stats": self.episode_tracker.get_episode_stats(),
            "action_space_size": self.action_space.get_position_sizing_action_count()
        }
    
    def reset(self):
        """Reset agent state."""
        self.current_state = None
        self.current_action = None
        self.episode_tracker.reset()
        logger.info("[Position Sizing Agent v2] Reset")
