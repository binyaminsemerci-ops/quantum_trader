"""
Q-Learning Core v2 - TD-Learning Engine
========================================

Implements Q-learning with TD updates.

Features:
- Q-table management
- TD-learning updates
- Epsilon-greedy action selection
- Q-table persistence

Author: Quantum Trader AI Team
Date: December 2, 2025
Version: 2.0
"""

from typing import Dict, Any, Optional, List, Tuple
import json
from pathlib import Path
import numpy as np
import structlog

logger = structlog.get_logger(__name__)


class QLearningCore:
    """
    Q-learning core with TD updates.
    
    Implements:
    - Q-table storage (state -> action -> Q-value)
    - TD update: Q(s,a) ← Q(s,a) + α[r + γ·max Q(s',a') - Q(s,a)]
    - Epsilon-greedy exploration
    """
    
    def __init__(
        self,
        alpha: float = 0.01,
        gamma: float = 0.99,
        epsilon: float = 0.1,
        epsilon_decay: float = 0.999,
        epsilon_min: float = 0.01
    ):
        """
        Initialize Q-Learning Core.
        
        Args:
            alpha: Learning rate
            gamma: Discount factor
            epsilon: Exploration rate
            epsilon_decay: Epsilon decay rate
            epsilon_min: Minimum epsilon
        """
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.epsilon_min = epsilon_min
        
        self.q_table: Dict[str, Dict[str, float]] = {}
        self.update_count = 0
        
        logger.info(
            "[Q-Learning Core] Initialized",
            alpha=alpha,
            gamma=gamma,
            epsilon=epsilon
        )
    
    def _state_to_key(self, state: Dict[str, Any]) -> str:
        """
        Convert state to hashable key.
        
        Args:
            state: State dictionary
            
        Returns:
            State key
        """
        # Simple discretization for continuous values
        discretized = {}
        for k, v in state.items():
            if isinstance(v, (int, float)):
                discretized[k] = round(float(v), 2)
            else:
                discretized[k] = str(v)
        
        return json.dumps(discretized, sort_keys=True)
    
    def _action_to_key(self, action: Dict[str, Any]) -> str:
        """
        Convert action to hashable key.
        
        Args:
            action: Action dictionary
            
        Returns:
            Action key
        """
        return json.dumps(action, sort_keys=True)
    
    def get_q_value(self, state: Dict[str, Any], action: Dict[str, Any]) -> float:
        """
        Get Q-value for state-action pair.
        
        Args:
            state: State
            action: Action
            
        Returns:
            Q-value (default 0.0)
        """
        state_key = self._state_to_key(state)
        action_key = self._action_to_key(action)
        
        if state_key not in self.q_table:
            return 0.0
        
        return self.q_table[state_key].get(action_key, 0.0)
    
    def update_q_value(
        self,
        state: Dict[str, Any],
        action: Dict[str, Any],
        reward: float,
        next_state: Optional[Dict[str, Any]] = None,
        next_best_q: Optional[float] = None
    ):
        """
        TD update for Q-value.
        
        Q(s,a) ← Q(s,a) + α[r + γ·max Q(s',a') - Q(s,a)]
        
        Args:
            state: Current state
            action: Action taken
            reward: Immediate reward
            next_state: Next state (optional)
            next_best_q: Best Q-value for next state (optional)
        """
        state_key = self._state_to_key(state)
        action_key = self._action_to_key(action)
        
        # Initialize state if needed
        if state_key not in self.q_table:
            self.q_table[state_key] = {}
        
        # Get current Q-value
        current_q = self.q_table[state_key].get(action_key, 0.0)
        
        # Calculate TD target
        if next_best_q is not None:
            td_target = reward + self.gamma * next_best_q
        elif next_state is not None:
            next_state_key = self._state_to_key(next_state)
            if next_state_key in self.q_table:
                max_next_q = max(self.q_table[next_state_key].values())
            else:
                max_next_q = 0.0
            td_target = reward + self.gamma * max_next_q
        else:
            # Terminal state
            td_target = reward
        
        # TD update
        new_q = current_q + self.alpha * (td_target - current_q)
        
        # Store updated Q-value
        self.q_table[state_key][action_key] = new_q
        
        self.update_count += 1
        
        # Decay epsilon
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay
    
    def select_action(
        self,
        state: Dict[str, Any],
        available_actions: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """
        Select action using epsilon-greedy.
        
        Args:
            state: Current state
            available_actions: List of available actions
            
        Returns:
            Selected action
        """
        if not available_actions:
            logger.warning("[Q-Learning Core] No available actions")
            return {}
        
        # Exploration
        if np.random.random() < self.epsilon:
            return available_actions[np.random.randint(len(available_actions))]
        
        # Exploitation: choose action with highest Q-value
        state_key = self._state_to_key(state)
        
        best_action = available_actions[0]
        best_q = self.get_q_value(state, best_action)
        
        for action in available_actions[1:]:
            q_value = self.get_q_value(state, action)
            if q_value > best_q:
                best_q = q_value
                best_action = action
        
        return best_action
    
    def get_best_action_value(self, state: Dict[str, Any]) -> Tuple[Dict[str, Any], float]:
        """
        Get best action and its Q-value for state.
        
        Args:
            state: State
            
        Returns:
            (best_action, q_value)
        """
        state_key = self._state_to_key(state)
        
        if state_key not in self.q_table or not self.q_table[state_key]:
            return ({}, 0.0)
        
        best_action_key = max(
            self.q_table[state_key],
            key=lambda k: self.q_table[state_key][k]
        )
        best_q = self.q_table[state_key][best_action_key]
        best_action = json.loads(best_action_key)
        
        return (best_action, best_q)
    
    def save_q_table(self, filepath: Path):
        """
        Save Q-table to disk.
        
        Args:
            filepath: Path to save file
        """
        try:
            filepath.parent.mkdir(parents=True, exist_ok=True)
            
            data = {
                "q_table": self.q_table,
                "epsilon": self.epsilon,
                "update_count": self.update_count
            }
            
            with open(filepath, 'w') as f:
                json.dump(data, f, indent=2)
            
            logger.info(
                "[Q-Learning Core] Q-table saved",
                filepath=str(filepath),
                states=len(self.q_table)
            )
        except Exception as e:
            logger.error(
                "[Q-Learning Core] Failed to save Q-table",
                error=str(e)
            )
    
    def load_q_table(self, filepath: Path) -> bool:
        """
        Load Q-table from disk.
        
        Args:
            filepath: Path to load file
            
        Returns:
            True if loaded successfully
        """
        try:
            if not filepath.exists():
                logger.warning(
                    "[Q-Learning Core] Q-table file not found",
                    filepath=str(filepath)
                )
                return False
            
            with open(filepath, 'r') as f:
                data = json.load(f)
            
            self.q_table = data.get("q_table", {})
            self.epsilon = data.get("epsilon", self.epsilon)
            self.update_count = data.get("update_count", 0)
            
            logger.info(
                "[Q-Learning Core] Q-table loaded",
                filepath=str(filepath),
                states=len(self.q_table),
                epsilon=self.epsilon
            )
            
            return True
        except Exception as e:
            logger.error(
                "[Q-Learning Core] Failed to load Q-table",
                error=str(e)
            )
            return False
    
    def get_stats(self) -> Dict[str, Any]:
        """
        Get Q-learning statistics.
        
        Returns:
            Statistics
        """
        total_states = len(self.q_table)
        total_state_actions = sum(len(actions) for actions in self.q_table.values())
        
        return {
            "total_states": total_states,
            "total_state_actions": total_state_actions,
            "epsilon": self.epsilon,
            "update_count": self.update_count,
            "alpha": self.alpha,
            "gamma": self.gamma
        }
