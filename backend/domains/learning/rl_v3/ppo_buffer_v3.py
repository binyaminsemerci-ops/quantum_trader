"""
Experience buffer for PPO with GAE computation.
"""

import numpy as np
import torch
from typing import Tuple


class PPOBuffer:
    """
    Buffer for storing trajectories and computing advantages with GAE.
    """
    
    def __init__(self, size: int, state_dim: int, gamma: float = 0.99, lambda_gae: float = 0.95):
        """
        Initialize buffer.
        
        Args:
            size: Maximum buffer size
            state_dim: Dimension of state
            gamma: Discount factor
            lambda_gae: GAE lambda parameter
        """
        self.size = size
        self.state_dim = state_dim
        self.gamma = gamma
        self.lambda_gae = lambda_gae
        
        # Storage
        self.states = np.zeros((size, state_dim), dtype=np.float32)
        self.actions = np.zeros(size, dtype=np.int64)
        self.log_probs = np.zeros(size, dtype=np.float32)
        self.rewards = np.zeros(size, dtype=np.float32)
        self.values = np.zeros(size, dtype=np.float32)
        self.dones = np.zeros(size, dtype=np.float32)
        
        self.ptr = 0
        self.path_start = 0
    
    def store(
        self,
        state: np.ndarray,
        action: int,
        log_prob: float,
        reward: float,
        value: float,
        done: bool
    ):
        """Store one step of experience."""
        assert self.ptr < self.size
        
        self.states[self.ptr] = state
        self.actions[self.ptr] = action
        self.log_probs[self.ptr] = log_prob
        self.rewards[self.ptr] = reward
        self.values[self.ptr] = value
        self.dones[self.ptr] = float(done)
        
        self.ptr += 1
    
    def finish_path(self, last_value: float = 0.0):
        """
        Finish trajectory and compute advantages with GAE.
        
        Args:
            last_value: Value estimate for final state (0 if terminal)
        """
        path_slice = slice(self.path_start, self.ptr)
        rewards = np.append(self.rewards[path_slice], last_value)
        values = np.append(self.values[path_slice], last_value)
        
        # Compute GAE
        deltas = rewards[:-1] + self.gamma * values[1:] - values[:-1]
        
        advantages = np.zeros_like(deltas)
        last_gae = 0.0
        for t in reversed(range(len(deltas))):
            last_gae = deltas[t] + self.gamma * self.lambda_gae * last_gae
            advantages[t] = last_gae
        
        # Store advantages as returns (for value target)
        returns = advantages + values[:-1]
        
        # Create buffer for advantages and returns
        if not hasattr(self, 'advantages'):
            self.advantages = np.zeros(self.size, dtype=np.float32)
            self.returns = np.zeros(self.size, dtype=np.float32)
        
        self.advantages[path_slice] = advantages
        self.returns[path_slice] = returns
        
        self.path_start = self.ptr
    
    def get(self) -> Tuple[torch.Tensor, ...]:
        """
        Get all data as PyTorch tensors.
        
        CRITICAL: Buffer must be completely filled before calling this method.
        If buffer is not full, this will raise an assertion error.
        
        Returns:
            Tuple of (states, actions, log_probs, advantages, returns)
            
        Raises:
            AssertionError: If buffer is not full (ptr != size)
        """
        if self.ptr != self.size:
            raise AssertionError(
                f"Buffer must be full before calling get(). "
                f"Current: {self.ptr}/{self.size} steps. "
                f"Fill buffer completely using store() and finish_path()."
            )
        
        # Normalize advantages
        advantages = self.advantages[:self.ptr]
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        
        return (
            torch.from_numpy(self.states[:self.ptr]),
            torch.from_numpy(self.actions[:self.ptr]),
            torch.from_numpy(self.log_probs[:self.ptr]),
            torch.from_numpy(advantages),
            torch.from_numpy(self.returns[:self.ptr])
        )
    
    def clear(self):
        """Reset buffer."""
        self.ptr = 0
        self.path_start = 0
