"""
RL Episode Tracker v2 - Episode and TD-Learning Management
===========================================================

Implements:
- Episode tracking
- Episodic reward accumulation
- TD-learning updates (Q-learning)
- Discounted return calculation

Author: Quantum Trader AI Team
Date: December 2, 2025
Version: 2.0
"""

from typing import Dict, Any, List, Optional, Tuple
from dataclasses import dataclass, field
import numpy as np
import structlog

logger = structlog.get_logger(__name__)


@dataclass
class Episode:
    """
    Episode data structure.
    
    Tracks state-action-reward sequences for a complete trading episode.
    """
    episode_id: str
    start_time: float
    end_time: Optional[float] = None
    
    states: List[Dict[str, Any]] = field(default_factory=list)
    actions: List[Any] = field(default_factory=list)
    rewards: List[float] = field(default_factory=list)
    
    total_reward: float = 0.0
    discounted_return: float = 0.0
    is_complete: bool = False
    
    def add_step(
        self,
        state: Dict[str, Any],
        action: Any,
        reward: float
    ):
        """Add a step to the episode."""
        self.states.append(state)
        self.actions.append(action)
        self.rewards.append(reward)
        self.total_reward += reward
    
    def calculate_discounted_return(self, gamma: float) -> float:
        """
        Calculate discounted return for episode.
        
        G_t = Σ γ^k * r_{t+k}
        
        Args:
            gamma: Discount factor
            
        Returns:
            Discounted return
        """
        discounted_return = 0.0
        for t, reward in enumerate(self.rewards):
            discounted_return += (gamma ** t) * reward
        
        self.discounted_return = discounted_return
        return discounted_return


class EpisodeTrackerV2:
    """
    Episode tracker for RL v2.
    
    Manages:
    - Episode lifecycle
    - Reward accumulation
    - TD-learning updates
    - Q-table management
    """
    
    def __init__(
        self,
        gamma: float = 0.99,
        alpha: float = 0.01,
        max_episodes: int = 1000
    ):
        """
        Initialize Episode Tracker v2.
        
        Args:
            gamma: Discount factor for TD-learning
            alpha: Learning rate for Q-updates
            max_episodes: Maximum episodes to keep in memory
        """
        self.gamma = gamma
        self.alpha = alpha
        self.max_episodes = max_episodes
        
        # Episode storage
        self.episodes: Dict[str, Episode] = {}
        self.completed_episodes: List[Episode] = []
        
        # Q-tables (state-action values)
        self.meta_q_table: Dict[str, Dict[str, float]] = {}
        self.size_q_table: Dict[str, List[float]] = {}
        
        logger.info(
            "[Episode Tracker v2] Initialized",
            gamma=gamma,
            alpha=alpha,
            max_episodes=max_episodes
        )
    
    def start_episode(self, trace_id: str, start_time: float) -> Episode:
        """
        Start a new episode.
        
        Args:
            trace_id: Episode/trace ID
            start_time: Episode start timestamp
            
        Returns:
            New episode instance
        """
        episode = Episode(
            episode_id=trace_id,
            start_time=start_time
        )
        
        self.episodes[trace_id] = episode
        
        logger.debug(
            "[Episode Tracker v2] Episode started",
            trace_id=trace_id,
            start_time=start_time
        )
        
        return episode
    
    def add_step(
        self,
        trace_id: str,
        state: Dict[str, Any],
        action: Any,
        reward: float
    ):
        """
        Add a step to current episode.
        
        Args:
            trace_id: Episode ID
            state: Current state
            action: Action taken
            reward: Reward received
        """
        if trace_id not in self.episodes:
            logger.warning(
                "[Episode Tracker v2] Episode not found, creating new",
                trace_id=trace_id
            )
            import time
            self.start_episode(trace_id, time.time())
        
        episode = self.episodes[trace_id]
        episode.add_step(state, action, reward)
        
        logger.debug(
            "[Episode Tracker v2] Step added",
            trace_id=trace_id,
            reward=reward,
            total_steps=len(episode.rewards)
        )
    
    def end_episode(self, trace_id: str, end_time: float):
        """
        End an episode and calculate returns.
        
        Args:
            trace_id: Episode ID
            end_time: Episode end timestamp
        """
        if trace_id not in self.episodes:
            logger.warning(
                "[Episode Tracker v2] Episode not found for ending",
                trace_id=trace_id
            )
            return
        
        episode = self.episodes[trace_id]
        episode.end_time = end_time
        episode.is_complete = True
        
        # Calculate discounted return
        episode.calculate_discounted_return(self.gamma)
        
        # Move to completed episodes
        self.completed_episodes.append(episode)
        del self.episodes[trace_id]
        
        # Trim completed episodes if needed
        if len(self.completed_episodes) > self.max_episodes:
            self.completed_episodes = self.completed_episodes[-self.max_episodes:]
        
        logger.info(
            "[Episode Tracker v2] Episode completed",
            trace_id=trace_id,
            total_reward=episode.total_reward,
            discounted_return=episode.discounted_return,
            steps=len(episode.rewards)
        )
    
    def td_update_meta(
        self,
        state: Dict[str, Any],
        action: str,
        reward: float,
        next_state: Optional[Dict[str, Any]] = None,
        trace_id: str = ""
    ) -> float:
        """
        TD-learning update for meta strategy Q-table.
        
        Q(s,a) ← Q(s,a) + α * (reward + γ * max(Q(s', a')) - Q(s,a))
        
        Args:
            state: Current state
            action: Action taken
            reward: Reward received
            next_state: Next state (None if terminal)
            trace_id: Trace ID for logging
            
        Returns:
            Updated Q-value
        """
        # Convert state to hashable key
        state_key = self._state_to_key(state)
        
        # Initialize Q-values for state if not exists
        if state_key not in self.meta_q_table:
            self.meta_q_table[state_key] = {
                "TREND": 0.0,
                "RANGE": 0.0,
                "BREAKOUT": 0.0,
                "MEAN_REVERSION": 0.0
            }
        
        # Current Q-value
        current_q = self.meta_q_table[state_key][action]
        
        # Calculate TD target
        if next_state is None:
            # Terminal state
            td_target = reward
        else:
            # Non-terminal state
            next_state_key = self._state_to_key(next_state)
            if next_state_key not in self.meta_q_table:
                self.meta_q_table[next_state_key] = {
                    "TREND": 0.0,
                    "RANGE": 0.0,
                    "BREAKOUT": 0.0,
                    "MEAN_REVERSION": 0.0
                }
            
            max_next_q = max(self.meta_q_table[next_state_key].values())
            td_target = reward + self.gamma * max_next_q
        
        # TD error
        td_error = td_target - current_q
        
        # Update Q-value
        new_q = current_q + self.alpha * td_error
        self.meta_q_table[state_key][action] = new_q
        
        logger.debug(
            "[Episode Tracker v2] Meta TD-update",
            trace_id=trace_id,
            state_key=state_key,
            action=action,
            reward=reward,
            current_q=current_q,
            new_q=new_q,
            td_error=td_error
        )
        
        return new_q
    
    def td_update_size(
        self,
        state: Dict[str, Any],
        action_index: int,
        reward: float,
        next_state: Optional[Dict[str, Any]] = None,
        action_space_size: int = 56,  # 8 multipliers * 7 leverage levels
        trace_id: str = ""
    ) -> float:
        """
        TD-learning update for position sizing Q-table.
        
        Q(s,a) ← Q(s,a) + α * (reward + γ * max(Q(s', a')) - Q(s,a))
        
        Args:
            state: Current state
            action_index: Action index taken
            reward: Reward received
            next_state: Next state (None if terminal)
            action_space_size: Size of action space
            trace_id: Trace ID for logging
            
        Returns:
            Updated Q-value
        """
        # Convert state to hashable key
        state_key = self._state_to_key(state)
        
        # Initialize Q-values for state if not exists
        if state_key not in self.size_q_table:
            self.size_q_table[state_key] = [0.0] * action_space_size
        
        # Current Q-value
        current_q = self.size_q_table[state_key][action_index]
        
        # Calculate TD target
        if next_state is None:
            # Terminal state
            td_target = reward
        else:
            # Non-terminal state
            next_state_key = self._state_to_key(next_state)
            if next_state_key not in self.size_q_table:
                self.size_q_table[next_state_key] = [0.0] * action_space_size
            
            max_next_q = max(self.size_q_table[next_state_key])
            td_target = reward + self.gamma * max_next_q
        
        # TD error
        td_error = td_target - current_q
        
        # Update Q-value
        new_q = current_q + self.alpha * td_error
        self.size_q_table[state_key][action_index] = new_q
        
        logger.debug(
            "[Episode Tracker v2] Size TD-update",
            trace_id=trace_id,
            state_key=state_key,
            action_index=action_index,
            reward=reward,
            current_q=current_q,
            new_q=new_q,
            td_error=td_error
        )
        
        return new_q
    
    def get_meta_q_values(self, state: Dict[str, Any]) -> Dict[str, float]:
        """
        Get Q-values for meta strategy actions in given state.
        
        Args:
            state: Current state
            
        Returns:
            Q-values for each action
        """
        state_key = self._state_to_key(state)
        
        if state_key not in self.meta_q_table:
            return {
                "TREND": 0.0,
                "RANGE": 0.0,
                "BREAKOUT": 0.0,
                "MEAN_REVERSION": 0.0
            }
        
        return self.meta_q_table[state_key].copy()
    
    def get_size_q_values(
        self,
        state: Dict[str, Any],
        action_space_size: int = 56
    ) -> List[float]:
        """
        Get Q-values for position sizing actions in given state.
        
        Args:
            state: Current state
            action_space_size: Size of action space
            
        Returns:
            Q-values for each action
        """
        state_key = self._state_to_key(state)
        
        if state_key not in self.size_q_table:
            return [0.0] * action_space_size
        
        return self.size_q_table[state_key].copy()
    
    def _state_to_key(self, state: Dict[str, Any]) -> str:
        """
        Convert state dict to hashable string key.
        
        Args:
            state: State dictionary
            
        Returns:
            String key
        """
        # Sort keys for consistent hashing
        sorted_items = sorted(state.items())
        
        # Convert to string representation
        key_parts = []
        for k, v in sorted_items:
            if isinstance(v, float):
                # Round floats to 2 decimals for bucketing
                v_str = f"{v:.2f}"
            else:
                v_str = str(v)
            key_parts.append(f"{k}={v_str}")
        
        return "|".join(key_parts)
    
    def get_episode_stats(self) -> Dict[str, Any]:
        """
        Get episode statistics.
        
        Returns:
            Statistics dictionary
        """
        if not self.completed_episodes:
            return {
                "total_episodes": 0,
                "avg_reward": 0.0,
                "avg_discounted_return": 0.0,
                "avg_steps": 0.0
            }
        
        total_episodes = len(self.completed_episodes)
        avg_reward = np.mean([e.total_reward for e in self.completed_episodes])
        avg_return = np.mean([e.discounted_return for e in self.completed_episodes])
        avg_steps = np.mean([len(e.rewards) for e in self.completed_episodes])
        
        return {
            "total_episodes": total_episodes,
            "avg_reward": float(avg_reward),
            "avg_discounted_return": float(avg_return),
            "avg_steps": float(avg_steps)
        }
    
    def reset(self):
        """Reset all episode data."""
        self.episodes.clear()
        self.completed_episodes.clear()
        self.meta_q_table.clear()
        self.size_q_table.clear()
        logger.info("[Episode Tracker v2] Reset complete")


# Global singleton instance
_episode_tracker_instance: Optional[EpisodeTrackerV2] = None


def get_episode_tracker() -> EpisodeTrackerV2:
    """Get or create global EpisodeTrackerV2 instance."""
    global _episode_tracker_instance
    if _episode_tracker_instance is None:
        _episode_tracker_instance = EpisodeTrackerV2()
    return _episode_tracker_instance
