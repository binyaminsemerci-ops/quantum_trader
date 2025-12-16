"""
Episode Tracker v2 - Episode Lifecycle Management
==================================================

Manages RL episodes with TD-learning.

Features:
- Episode lifecycle (start, step, end)
- Reward tracking with discounting
- State-action-reward history

Author: Quantum Trader AI Team
Date: December 2, 2025
Version: 2.0
"""

from typing import Dict, Any, List, Optional
from dataclasses import dataclass, field
import structlog

logger = structlog.get_logger(__name__)


@dataclass
class EpisodeStep:
    """Single step in episode."""
    state: Dict[str, Any]
    action: Dict[str, Any]
    reward: float
    next_state: Optional[Dict[str, Any]] = None


@dataclass
class Episode:
    """Episode data."""
    episode_id: int
    steps: List[EpisodeStep] = field(default_factory=list)
    total_reward: float = 0.0
    discounted_return: float = 0.0
    is_active: bool = True


class EpisodeTrackerV2:
    """
    Episode tracker for RL v2 with TD-learning.
    
    Manages:
    - Episode lifecycle
    - Reward accumulation with discounting
    - State-action-reward sequences
    """
    
    def __init__(self, gamma: float = 0.99):
        """
        Initialize Episode Tracker v2.
        
        Args:
            gamma: Discount factor for future rewards
        """
        self.gamma = gamma
        self.current_episode: Optional[Episode] = None
        self.episode_count = 0
        self.completed_episodes: List[Episode] = []
        
        logger.info(
            "[Episode Tracker v2] Initialized",
            gamma=gamma
        )
    
    def start_episode(self) -> int:
        """
        Start new episode.
        
        Returns:
            Episode ID
        """
        if self.current_episode and self.current_episode.is_active:
            logger.warning("[Episode Tracker v2] Ending previous episode")
            self.end_episode()
        
        self.episode_count += 1
        self.current_episode = Episode(episode_id=self.episode_count)
        
        logger.info(
            "[Episode Tracker v2] Episode started",
            episode_id=self.episode_count
        )
        
        return self.episode_count
    
    def record_step(
        self,
        state: Dict[str, Any],
        action: Dict[str, Any],
        reward: float,
        next_state: Optional[Dict[str, Any]] = None
    ):
        """
        Record step in current episode.
        
        Args:
            state: Current state
            action: Action taken
            reward: Immediate reward
            next_state: Next state (optional)
        """
        if not self.current_episode:
            logger.warning("[Episode Tracker v2] No active episode")
            return
        
        step = EpisodeStep(
            state=state,
            action=action,
            reward=reward,
            next_state=next_state
        )
        
        self.current_episode.steps.append(step)
        self.current_episode.total_reward += reward
    
    def end_episode(self) -> float:
        """
        End current episode and calculate discounted return.
        
        Returns:
            Discounted return
        """
        if not self.current_episode:
            logger.warning("[Episode Tracker v2] No active episode")
            return 0.0
        
        # Calculate discounted return
        discounted_return = 0.0
        for i, step in enumerate(reversed(self.current_episode.steps)):
            discounted_return = step.reward + self.gamma * discounted_return
        
        self.current_episode.discounted_return = discounted_return
        self.current_episode.is_active = False
        
        # Store completed episode
        self.completed_episodes.append(self.current_episode)
        
        # Keep only last 100 episodes
        if len(self.completed_episodes) > 100:
            self.completed_episodes.pop(0)
        
        logger.info(
            "[Episode Tracker v2] Episode ended",
            episode_id=self.current_episode.episode_id,
            total_reward=self.current_episode.total_reward,
            discounted_return=discounted_return,
            steps=len(self.current_episode.steps)
        )
        
        result = discounted_return
        self.current_episode = None
        
        return result
    
    def get_current_episode(self) -> Optional[Episode]:
        """Get current active episode."""
        return self.current_episode
    
    def get_episode_stats(self) -> Dict[str, Any]:
        """
        Get episode statistics.
        
        Returns:
            Episode stats
        """
        if not self.completed_episodes:
            return {
                "total_episodes": 0,
                "avg_reward": 0.0,
                "avg_return": 0.0,
                "avg_steps": 0.0
            }
        
        total_reward = sum(ep.total_reward for ep in self.completed_episodes)
        total_return = sum(ep.discounted_return for ep in self.completed_episodes)
        total_steps = sum(len(ep.steps) for ep in self.completed_episodes)
        
        count = len(self.completed_episodes)
        
        return {
            "total_episodes": count,
            "avg_reward": total_reward / count,
            "avg_return": total_return / count,
            "avg_steps": total_steps / count,
            "recent_episodes": [
                {
                    "id": ep.episode_id,
                    "reward": ep.total_reward,
                    "return": ep.discounted_return,
                    "steps": len(ep.steps)
                }
                for ep in self.completed_episodes[-5:]
            ]
        }
    
    def reset(self):
        """Reset episode tracker."""
        self.current_episode = None
        self.episode_count = 0
        self.completed_episodes.clear()
        logger.info("[Episode Tracker v2] Reset")
