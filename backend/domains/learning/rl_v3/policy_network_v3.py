"""
Policy network for PPO agent.
"""

import torch
import torch.nn as nn
from typing import Tuple


class PolicyNetwork(nn.Module):
    """
    Policy network that outputs action logits.
    """
    
    def __init__(self, state_dim: int, hidden_dim: int, action_dim: int):
        """
        Initialize policy network.
        
        Args:
            state_dim: Dimension of state/observation
            hidden_dim: Hidden layer size
            action_dim: Number of discrete actions
        """
        super().__init__()
        
        self.network = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim)
        )
        
        # Initialize weights
        self._initialize_weights()
    
    def _initialize_weights(self):
        """Initialize network weights with orthogonal initialization."""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.orthogonal_(module.weight, gain=0.01)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0.0)
    
    def forward(self, state: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.
        
        Args:
            state: State tensor [batch, state_dim]
            
        Returns:
            Action logits [batch, action_dim]
        """
        return self.network(state)
    
    def get_action_distribution(self, state: torch.Tensor) -> torch.distributions.Categorical:
        """
        Get categorical distribution over actions.
        
        Args:
            state: State tensor
            
        Returns:
            Categorical distribution
        """
        logits = self.forward(state)
        return torch.distributions.Categorical(logits=logits)
