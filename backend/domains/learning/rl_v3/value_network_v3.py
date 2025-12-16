"""
Value network for PPO agent.
"""

import torch
import torch.nn as nn


class ValueNetwork(nn.Module):
    """
    Value network that estimates state value V(s).
    """
    
    def __init__(self, state_dim: int, hidden_dim: int):
        """
        Initialize value network.
        
        Args:
            state_dim: Dimension of state/observation
            hidden_dim: Hidden layer size
        """
        super().__init__()
        
        self.network = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )
        
        # Initialize weights
        self._initialize_weights()
    
    def _initialize_weights(self):
        """Initialize network weights."""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.orthogonal_(module.weight, gain=1.0)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0.0)
    
    def forward(self, state: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.
        
        Args:
            state: State tensor [batch, state_dim]
            
        Returns:
            Value estimate [batch, 1]
        """
        return self.network(state)
