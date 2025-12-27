"""
PPO Agent with policy and value networks.
"""

import torch
import torch.nn as nn
import numpy as np
from typing import Tuple, Dict
from pathlib import Path

from backend.domains.learning.rl_v3.policy_network_v3 import PolicyNetwork
from backend.domains.learning.rl_v3.value_network_v3 import ValueNetwork
from backend.domains.learning.rl_v3.config_v3 import RLv3Config


class PPOAgent:
    """
    PPO agent with policy and value networks.
    """
    
    def __init__(self, config: RLv3Config):
        """
        Initialize PPO agent.
        
        Args:
            config: RL v3 configuration
        """
        self.config = config
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Create networks
        self.policy_net = PolicyNetwork(
            config.state_dim,
            config.hidden_dim,
            config.action_dim
        ).to(self.device)
        
        self.value_net = ValueNetwork(
            config.state_dim,
            config.hidden_dim
        ).to(self.device)
        
        # Optimizers
        self.policy_optimizer = torch.optim.Adam(
            self.policy_net.parameters(),
            lr=config.learning_rate
        )
        self.value_optimizer = torch.optim.Adam(
            self.value_net.parameters(),
            lr=config.learning_rate
        )
    
    def act(self, state: np.ndarray, deterministic: bool = False) -> Tuple[int, float, float]:
        """
        Select action given state.
        
        Args:
            state: State array [state_dim]
            deterministic: If True, select most probable action
            
        Returns:
            Tuple of (action, log_prob, value)
        """
        state_tensor = torch.from_numpy(state).float().unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            dist = self.policy_net.get_action_distribution(state_tensor)
            value = self.value_net(state_tensor).item()
            
            if deterministic:
                action = dist.logits.argmax(dim=-1).item()
                log_prob = dist.log_prob(torch.tensor([action])).item()
            else:
                action = dist.sample().item()
                log_prob = dist.log_prob(torch.tensor([action])).item()
        
        return action, log_prob, value
    
    def evaluate(
        self,
        states: torch.Tensor,
        actions: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Evaluate actions given states.
        
        Args:
            states: State batch [batch, state_dim]
            actions: Action batch [batch]
            
        Returns:
            Tuple of (log_probs, values, entropy)
        """
        states = states.to(self.device)
        actions = actions.to(self.device)
        
        # Policy evaluation
        dist = self.policy_net.get_action_distribution(states)
        log_probs = dist.log_prob(actions)
        entropy = dist.entropy()
        
        # Value evaluation
        values = self.value_net(states).squeeze(-1)
        
        return log_probs, values, entropy
    
    def save(self, path: Path):
        """Save agent networks."""
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save({
            'policy_state_dict': self.policy_net.state_dict(),
            'value_state_dict': self.value_net.state_dict(),
            'policy_optimizer_state_dict': self.policy_optimizer.state_dict(),
            'value_optimizer_state_dict': self.value_optimizer.state_dict(),
        }, path)
    
    def load(self, path: Path):
        """Load agent networks."""
        checkpoint = torch.load(path, map_location=self.device)
        self.policy_net.load_state_dict(checkpoint['policy_state_dict'])
        self.value_net.load_state_dict(checkpoint['value_state_dict'])
        self.policy_optimizer.load_state_dict(checkpoint['policy_optimizer_state_dict'])
        self.value_optimizer.load_state_dict(checkpoint['value_optimizer_state_dict'])
