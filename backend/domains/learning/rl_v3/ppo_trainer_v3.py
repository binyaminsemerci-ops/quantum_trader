"""
PPO trainer with clipped objective.
"""

import torch
import torch.nn as nn
import numpy as np
from typing import Tuple

from backend.domains.learning.rl_v3.ppo_agent_v3 import PPOAgent
from backend.domains.learning.rl_v3.ppo_buffer_v3 import PPOBuffer
from backend.domains.learning.rl_v3.config_v3 import RLv3Config


class PPOTrainer:
    """
    PPO trainer with clipped surrogate objective.
    """
    
    def __init__(self, agent: PPOAgent, config: RLv3Config):
        """
        Initialize trainer.
        
        Args:
            agent: PPO agent
            config: RL v3 configuration
        """
        self.agent = agent
        self.config = config
    
    def update(self, buffer: PPOBuffer) -> Tuple[float, float, float]:
        """
        Update policy and value networks using PPO.
        
        Args:
            buffer: Experience buffer
            
        Returns:
            Tuple of (policy_loss, value_loss, entropy)
        """
        # Get batch data
        states, actions, old_log_probs, advantages, returns = buffer.get()
        
        total_policy_loss = 0.0
        total_value_loss = 0.0
        total_entropy = 0.0
        n_updates = 0
        
        # Multi-epoch updates
        for _ in range(self.config.n_epochs):
            # Generate random mini-batches
            indices = np.arange(len(states))
            np.random.shuffle(indices)
            
            for start in range(0, len(states), self.config.batch_size):
                end = start + self.config.batch_size
                batch_indices = indices[start:end]
                
                # Get batch
                batch_states = states[batch_indices]
                batch_actions = actions[batch_indices]
                batch_old_log_probs = old_log_probs[batch_indices]
                batch_advantages = advantages[batch_indices]
                batch_returns = returns[batch_indices]
                
                # Evaluate actions
                log_probs, values, entropy = self.agent.evaluate(batch_states, batch_actions)
                
                # Policy loss (clipped surrogate objective)
                ratio = torch.exp(log_probs - batch_old_log_probs)
                clipped_ratio = torch.clamp(
                    ratio,
                    1.0 - self.config.clip_range,
                    1.0 + self.config.clip_range
                )
                policy_loss = -torch.min(
                    ratio * batch_advantages,
                    clipped_ratio * batch_advantages
                ).mean()
                
                # Value loss
                value_loss = nn.functional.mse_loss(values, batch_returns)
                
                # Entropy bonus (for exploration)
                entropy_loss = -entropy.mean()
                
                # Total loss
                loss = (
                    policy_loss
                    + self.config.value_coef * value_loss
                    + self.config.entropy_coef * entropy_loss
                )
                
                # Update networks
                self.agent.policy_optimizer.zero_grad()
                self.agent.value_optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(
                    self.agent.policy_net.parameters(),
                    self.config.max_grad_norm
                )
                torch.nn.utils.clip_grad_norm_(
                    self.agent.value_net.parameters(),
                    self.config.max_grad_norm
                )
                self.agent.policy_optimizer.step()
                self.agent.value_optimizer.step()
                
                # Track metrics
                total_policy_loss += policy_loss.item()
                total_value_loss += value_loss.item()
                total_entropy += entropy.mean().item()
                n_updates += 1
        
        # Return average losses
        return (
            total_policy_loss / n_updates,
            total_value_loss / n_updates,
            total_entropy / n_updates
        )
