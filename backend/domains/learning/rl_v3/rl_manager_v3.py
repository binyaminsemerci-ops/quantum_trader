"""
RL v3 Manager - Main interface for PPO-based reinforcement learning.
"""

from pathlib import Path
import numpy as np
from typing import Dict, Any, Optional

from backend.domains.learning.rl_v3.config_v3 import RLv3Config
from backend.domains.learning.rl_v3.ppo_agent_v3 import PPOAgent
from backend.domains.learning.rl_v3.ppo_trainer_v3 import PPOTrainer
from backend.domains.learning.rl_v3.ppo_buffer_v3 import PPOBuffer
from backend.domains.learning.rl_v3.env_v3 import TradingEnvV3
from backend.domains.learning.rl_v3.features_v3 import build_feature_vector
from backend.domains.learning.rl_v3.market_data_provider import MarketDataProvider


class RLv3Manager:
    """
    Main interface for RL v3 PPO system.
    """
    
    def __init__(
        self,
        config: RLv3Config = None,
        market_data_provider: Optional[MarketDataProvider] = None
    ):
        """
        Initialize RL v3 manager.
        
        Args:
            config: RL v3 configuration (uses default if None)
            market_data_provider: Optional market data provider for training.
                                 If None, uses synthetic prices.
        """
        self.config = config or RLv3Config()
        self.agent = PPOAgent(self.config)
        self.trainer = PPOTrainer(self.agent, self.config)
        self.env = TradingEnvV3(self.config, market_data_provider=market_data_provider)
    
    def train(self, num_episodes: int = 100) -> Dict[str, Any]:
        """
        Train the PPO agent.
        
        Uses the configured market data provider:
        - Synthetic prices if no provider specified
        - Real historical data if RealMarketDataProvider configured
        
        Args:
            num_episodes: Number of training episodes
            
        Returns:
            Training metrics with keys:
            - total_rewards: List of episode rewards
            - policy_losses: List of policy losses
            - value_losses: List of value losses
            - avg_reward: Average reward across episodes
            - final_reward: Last episode reward
        """
        total_rewards = []
        policy_losses = []
        value_losses = []
        
        for episode in range(num_episodes):
            buffer = PPOBuffer(
                self.config.buffer_size,
                self.config.state_dim,
                self.config.gamma,
                self.config.lambda_gae
            )
            
            state = self.env.reset()
            episode_reward = 0.0
            
            # Collect trajectory
            for _ in range(self.config.buffer_size):
                action, log_prob, value = self.agent.act(state)
                next_state, reward, done, info = self.env.step(action)
                
                buffer.store(state, action, log_prob, reward, value, done)
                episode_reward += reward
                
                if done:
                    buffer.finish_path(last_value=0.0)
                    state = self.env.reset()
                else:
                    state = next_state
            
            # Finish any incomplete trajectory
            if buffer.ptr > buffer.path_start:
                _, _, last_value = self.agent.act(state)
                buffer.finish_path(last_value=last_value)
            
            # Update agent
            policy_loss, value_loss, entropy = self.trainer.update(buffer)
            
            # Track metrics
            total_rewards.append(episode_reward)
            policy_losses.append(policy_loss)
            value_losses.append(value_loss)
            
            if (episode + 1) % 10 == 0:
                print(f"Episode {episode+1}/{num_episodes} - "
                      f"Reward: {episode_reward:.2f}, "
                      f"Policy Loss: {policy_loss:.4f}, "
                      f"Value Loss: {value_loss:.4f}")
        
        return {
            'total_rewards': total_rewards,
            'policy_losses': policy_losses,
            'value_losses': value_losses,
            'avg_reward': np.mean(total_rewards),
            'final_reward': total_rewards[-1]
        }
    
    def predict(self, obs_dict: Dict[str, Any]) -> Dict[str, Any]:
        """
        Predict action given observation.
        
        Args:
            obs_dict: Observation dictionary with market data
            
        Returns:
            Dictionary with action, confidence, and TP zone suggestion
        """
        # Convert observation dict to feature vector
        state = build_feature_vector(obs_dict)
        
        # Get action from agent (deterministic)
        action, log_prob, value = self.agent.act(state, deterministic=True)
        
        # Convert action to confidence
        confidence = min(1.0, abs(value) / 10.0)
        
        # [TP v3 ENHANCEMENT] Suggest TP zone based on value estimate
        # Positive value → wider TP, negative value → tighter TP
        tp_multiplier = 1.0 + (value / 20.0)  # Range: 0.5x to 1.5x
        tp_multiplier = max(0.5, min(1.5, tp_multiplier))
        
        return {
            'action': action,
            'confidence': confidence,
            'value': value,
            'tp_zone_multiplier': tp_multiplier,
            'suggested_tp_pct': 0.06 * tp_multiplier  # Base 6% TP * multiplier
        }
    
    def save(self, path: Path = None):
        """
        Save agent model.
        
        Args:
            path: Path to save model (uses config path if None)
        """
        save_path = path or Path(self.config.model_path)
        self.agent.save(save_path)
        print(f"Model saved to {save_path}")
    
    def load(self, path: Path = None):
        """
        Load agent model.
        
        Args:
            path: Path to load model (uses config path if None)
        """
        load_path = path or Path(self.config.model_path)
        if load_path.exists():
            self.agent.load(load_path)
            print(f"Model loaded from {load_path}")
        else:
            print(f"No model found at {load_path}, using untrained agent")
