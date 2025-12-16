"""
RL Position Sizing Agent - Reinforcement Learning for dynamic position sizing.

Provides:
- SAC-based agent for position size, leverage, TP/SL optimization
- State includes market regime, volatility, portfolio metrics, model confidence
- Action space: position_size (0-100%), leverage (1-20x), TP%, SL%
- Rewards based on risk-adjusted returns (Sharpe ratio)
- Continuous learning from live trades
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncSession

from backend.core.event_bus import EventBus
from backend.core.policy_store import PolicyStore

logger = logging.getLogger(__name__)

try:
    import torch
    import torch.nn as nn
    import torch.optim as optim
    from torch.distributions import Normal
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    logger.warning("PyTorch not available - RL Position Sizing will fail")


# ============================================================================
# SAC Networks
# ============================================================================

class ActorNetwork(nn.Module):
    """
    SAC actor network (Gaussian policy).
    
    Outputs mean and log_std for continuous actions.
    """
    
    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        hidden_dim: int = 256,
        dropout: float = 0.1,
    ):
        super().__init__()
        
        self.shared = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
        )
        
        self.mean = nn.Linear(hidden_dim, action_dim)
        self.log_std = nn.Linear(hidden_dim, action_dim)
        
        # Action bounds
        self.action_scale = torch.FloatTensor([50.0, 10.0, 5.0, 2.0])  # position%, leverage, TP%, SL%
        self.action_bias = torch.FloatTensor([50.0, 10.5, 2.5, 1.0])
    
    def forward(self, state):
        features = self.shared(state)
        mean = self.mean(features)
        log_std = self.log_std(features)
        log_std = torch.clamp(log_std, -20, 2)
        
        return mean, log_std
    
    def sample(self, state):
        mean, log_std = self.forward(state)
        std = log_std.exp()
        
        normal = Normal(mean, std)
        z = normal.rsample()  # Reparameterization trick
        action = torch.tanh(z)
        
        # Scale to action space
        action_scaled = action * self.action_scale.to(action.device) + self.action_bias.to(action.device)
        
        # Log probability
        log_prob = normal.log_prob(z)
        # Adjust for tanh squashing
        log_prob -= torch.log(self.action_scale.to(action.device) * (1 - action.pow(2)) + 1e-6)
        log_prob = log_prob.sum(1, keepdim=True)
        
        return action_scaled, log_prob


class CriticNetwork(nn.Module):
    """
    SAC critic network (Q-function).
    
    Estimates Q(s, a) - expected return from state-action pair.
    """
    
    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        hidden_dim: int = 256,
        dropout: float = 0.1,
    ):
        super().__init__()
        
        self.network = nn.Sequential(
            nn.Linear(state_dim + action_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
        )
    
    def forward(self, state, action):
        x = torch.cat([state, action], dim=1)
        q_value = self.network(x)
        return q_value


# ============================================================================
# Experience Replay Buffer
# ============================================================================

@dataclass
class SACExperience:
    """SAC experience tuple."""
    
    state: np.ndarray
    action: np.ndarray
    reward: float
    next_state: np.ndarray
    done: bool


class ReplayBuffer:
    """Experience replay buffer for SAC."""
    
    def __init__(self, capacity: int = 100000):
        self.capacity = capacity
        self.buffer: List[SACExperience] = []
        self.position = 0
    
    def push(self, experience: SACExperience):
        if len(self.buffer) < self.capacity:
            self.buffer.append(experience)
        else:
            self.buffer[self.position] = experience
        
        self.position = (self.position + 1) % self.capacity
    
    def sample(self, batch_size: int) -> List[SACExperience]:
        indices = np.random.choice(len(self.buffer), batch_size, replace=False)
        return [self.buffer[i] for i in indices]
    
    def __len__(self):
        return len(self.buffer)


# ============================================================================
# Position Sizing Action
# ============================================================================

@dataclass
class PositionSizingAction:
    """Output from position sizing agent."""
    
    position_size_pct: float  # 0-100%
    leverage: float  # 1-20x
    take_profit_pct: float  # 0-10%
    stop_loss_pct: float  # 0-5%
    
    def to_dict(self) -> Dict:
        return {
            "position_size_pct": self.position_size_pct,
            "leverage": self.leverage,
            "take_profit_pct": self.take_profit_pct,
            "stop_loss_pct": self.stop_loss_pct,
        }


# ============================================================================
# RL Position Sizing Agent
# ============================================================================

class RLPositionSizingAgent:
    """
    Reinforcement Learning agent for dynamic position sizing.
    
    Features:
    - SAC (Soft Actor-Critic) algorithm
    - Continuous action space: [position_size%, leverage, TP%, SL%]
    - State: market regime, volatility, portfolio metrics, model confidence
    - Reward: risk-adjusted returns (Sharpe ratio)
    - Automatic entropy tuning
    """
    
    def __init__(
        self,
        db_session: AsyncSession,
        event_bus: EventBus,
        policy_store: PolicyStore,
        state_dim: int = 25,
        action_dim: int = 4,
        learning_rate: float = 0.0003,
        gamma: float = 0.99,
        tau: float = 0.005,
        alpha: float = 0.2,
        batch_size: int = 256,
        update_freq: int = 1,
    ):
        if not TORCH_AVAILABLE:
            raise ImportError("PyTorch is required for RL Position Sizing")
        
        self.db = db_session
        self.event_bus = event_bus
        self.policy_store = policy_store
        
        # Hyperparameters
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.learning_rate = learning_rate
        self.gamma = gamma
        self.tau = tau
        self.alpha = alpha  # Entropy coefficient
        self.batch_size = batch_size
        self.update_freq = update_freq
        
        # Device
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Networks
        self.actor = ActorNetwork(state_dim, action_dim).to(self.device)
        self.critic1 = CriticNetwork(state_dim, action_dim).to(self.device)
        self.critic2 = CriticNetwork(state_dim, action_dim).to(self.device)
        
        # Target networks
        self.critic1_target = CriticNetwork(state_dim, action_dim).to(self.device)
        self.critic2_target = CriticNetwork(state_dim, action_dim).to(self.device)
        
        self.critic1_target.load_state_dict(self.critic1.state_dict())
        self.critic2_target.load_state_dict(self.critic2.state_dict())
        
        # Optimizers
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=learning_rate)
        self.critic1_optimizer = optim.Adam(self.critic1.parameters(), lr=learning_rate)
        self.critic2_optimizer = optim.Adam(self.critic2.parameters(), lr=learning_rate)
        
        # Experience buffer
        self.buffer = ReplayBuffer()
        
        # Metrics
        self.episode_count = 0
        self.total_reward = 0.0
        self.update_count = 0
        
        # State tracking
        self.current_state = None
        self.current_action = None
        
        logger.info(
            f"RLPositionSizingAgent initialized: "
            f"state_dim={state_dim}, action_dim={action_dim}, lr={learning_rate}"
        )
    
    async def compute_position_size(
        self,
        market_data: Dict,
        portfolio_metrics: Dict,
        model_confidence: float,
        signal_strength: float,
    ) -> PositionSizingAction:
        """
        Compute optimal position sizing parameters.
        
        Args:
            market_data: Current market features
            portfolio_metrics: Portfolio state (balance, open positions, etc.)
            model_confidence: Confidence from ML models
            signal_strength: Strength of trading signal
            
        Returns:
            PositionSizingAction with size, leverage, TP, SL
        """
        # Build state vector
        state = self._build_state(market_data, portfolio_metrics, model_confidence, signal_strength)
        
        # Get action from actor
        with torch.no_grad():
            state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
            action, _ = self.actor.sample(state_tensor)
            action = action.cpu().numpy()[0]
        
        # Store for later reward
        self.current_state = state
        self.current_action = action
        
        # Convert to PositionSizingAction
        sizing = PositionSizingAction(
            position_size_pct=float(np.clip(action[0], 0, 100)),
            leverage=float(np.clip(action[1], 1, 20)),
            take_profit_pct=float(np.clip(action[2], 0.5, 10)),
            stop_loss_pct=float(np.clip(action[3], 0.2, 5)),
        )
        
        logger.debug(
            f"Position sizing: size={sizing.position_size_pct:.1f}%, "
            f"leverage={sizing.leverage:.1f}x, TP={sizing.take_profit_pct:.2f}%, "
            f"SL={sizing.stop_loss_pct:.2f}%"
        )
        
        return sizing
    
    async def record_reward(
        self,
        reward: float,
        next_market_data: Dict,
        next_portfolio_metrics: Dict,
        next_model_confidence: float,
        next_signal_strength: float,
        done: bool = False,
    ):
        """
        Record reward from trade outcome.
        
        Args:
            reward: Risk-adjusted return (Sharpe-like metric)
            next_market_data: Market state after trade
            next_portfolio_metrics: Portfolio state after trade
            next_model_confidence: Model confidence after trade
            next_signal_strength: Signal strength after trade
            done: Whether episode is complete
        """
        if self.current_state is None:
            logger.warning("No current state - cannot record reward")
            return
        
        # Build next state
        next_state = self._build_state(
            next_market_data,
            next_portfolio_metrics,
            next_model_confidence,
            next_signal_strength
        )
        
        # Create experience
        experience = SACExperience(
            state=self.current_state,
            action=self.current_action,
            reward=reward,
            next_state=next_state,
            done=done,
        )
        
        self.buffer.push(experience)
        self.total_reward += reward
        
        # Update networks
        if len(self.buffer) >= self.batch_size and self.update_count % self.update_freq == 0:
            await self._update_networks()
        
        self.update_count += 1
        
        # Reset current state
        if done:
            self.episode_count += 1
            self.current_state = None
            self.current_action = None
    
    async def _update_networks(self):
        """Update actor and critic networks using SAC."""
        if len(self.buffer) < self.batch_size:
            return
        
        # Sample batch
        batch = self.buffer.sample(self.batch_size)
        
        # Prepare tensors
        states = torch.FloatTensor([exp.state for exp in batch]).to(self.device)
        actions = torch.FloatTensor([exp.action for exp in batch]).to(self.device)
        rewards = torch.FloatTensor([exp.reward for exp in batch]).unsqueeze(1).to(self.device)
        next_states = torch.FloatTensor([exp.next_state for exp in batch]).to(self.device)
        dones = torch.FloatTensor([exp.done for exp in batch]).unsqueeze(1).to(self.device)
        
        # Update critics
        with torch.no_grad():
            next_actions, next_log_probs = self.actor.sample(next_states)
            
            q1_next = self.critic1_target(next_states, next_actions)
            q2_next = self.critic2_target(next_states, next_actions)
            q_next = torch.min(q1_next, q2_next) - self.alpha * next_log_probs
            
            q_target = rewards + self.gamma * (1 - dones) * q_next
        
        q1 = self.critic1(states, actions)
        q2 = self.critic2(states, actions)
        
        critic1_loss = nn.MSELoss()(q1, q_target)
        critic2_loss = nn.MSELoss()(q2, q_target)
        
        self.critic1_optimizer.zero_grad()
        critic1_loss.backward()
        self.critic1_optimizer.step()
        
        self.critic2_optimizer.zero_grad()
        critic2_loss.backward()
        self.critic2_optimizer.step()
        
        # Update actor
        new_actions, log_probs = self.actor.sample(states)
        
        q1_new = self.critic1(states, new_actions)
        q2_new = self.critic2(states, new_actions)
        q_new = torch.min(q1_new, q2_new)
        
        actor_loss = (self.alpha * log_probs - q_new).mean()
        
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()
        
        # Soft update target networks
        self._soft_update(self.critic1, self.critic1_target)
        self._soft_update(self.critic2, self.critic2_target)
        
        logger.debug(
            f"SAC update: critic1_loss={critic1_loss.item():.4f}, "
            f"critic2_loss={critic2_loss.item():.4f}, "
            f"actor_loss={actor_loss.item():.4f}"
        )
    
    def _soft_update(self, source: nn.Module, target: nn.Module):
        """Soft update of target network."""
        for target_param, source_param in zip(target.parameters(), source.parameters()):
            target_param.data.copy_(
                self.tau * source_param.data + (1.0 - self.tau) * target_param.data
            )
    
    def _build_state(
        self,
        market_data: Dict,
        portfolio_metrics: Dict,
        model_confidence: float,
        signal_strength: float,
    ) -> np.ndarray:
        """
        Build state vector.
        
        State features (25-dim):
        - Market features (10): returns, volatility, trend, volume, etc.
        - Portfolio metrics (8): balance, positions, PnL, drawdown, etc.
        - Model features (4): confidence, signal strength, ensemble agreement, uncertainty
        - Regime features (3): market regime indicators
        """
        state = np.zeros(self.state_dim)
        
        # Market features
        state[0] = market_data.get("returns", 0.0)
        state[1] = market_data.get("volatility", 0.0)
        state[2] = market_data.get("trend", 0.0)
        state[3] = market_data.get("volume_ratio", 1.0)
        state[4] = market_data.get("rsi", 50.0) / 100.0
        state[5] = market_data.get("macd", 0.0)
        state[6] = market_data.get("atr_pct", 0.0)
        state[7] = market_data.get("bb_width", 0.0)
        state[8] = market_data.get("adx", 0.0) / 100.0
        state[9] = market_data.get("obv_change", 0.0)
        
        # Portfolio metrics
        state[10] = portfolio_metrics.get("balance_pct", 100.0) / 100.0
        state[11] = portfolio_metrics.get("open_positions", 0) / 10.0  # Normalize
        state[12] = portfolio_metrics.get("total_pnl_pct", 0.0) / 100.0
        state[13] = portfolio_metrics.get("daily_pnl_pct", 0.0) / 10.0
        state[14] = portfolio_metrics.get("max_drawdown_pct", 0.0) / 100.0
        state[15] = portfolio_metrics.get("win_rate", 0.5)
        state[16] = portfolio_metrics.get("sharpe_ratio", 0.0) / 3.0  # Normalize
        state[17] = portfolio_metrics.get("leverage_used", 0.0) / 20.0
        
        # Model features
        state[18] = model_confidence
        state[19] = signal_strength
        state[20] = portfolio_metrics.get("ensemble_agreement", 0.5)
        state[21] = portfolio_metrics.get("model_uncertainty", 0.5)
        
        # Regime features
        regime = market_data.get("regime", "sideways")
        regime_scores = {
            "bull": [1.0, 0.0, 0.0],
            "bear": [0.0, 1.0, 0.0],
            "sideways": [0.0, 0.0, 1.0],
            "volatile": [0.5, 0.5, 0.0],
        }
        state[22:25] = regime_scores.get(regime, [0.0, 0.0, 1.0])
        
        return state
    
    async def save_checkpoint(self, version: str):
        """Save SAC checkpoint."""
        checkpoint = {
            "actor_state_dict": self.actor.state_dict(),
            "critic1_state_dict": self.critic1.state_dict(),
            "critic2_state_dict": self.critic2.state_dict(),
            "episode_count": self.episode_count,
            "total_reward": self.total_reward,
        }
        
        # Save to database
        query = text("""
            INSERT INTO rl_versions (
                version, agent_type, checkpoint_data, created_at, metrics
            ) VALUES (
                :version, :agent_type, :checkpoint_data, :created_at, :metrics
            )
            ON CONFLICT (version, agent_type) DO UPDATE SET
                checkpoint_data = EXCLUDED.checkpoint_data,
                metrics = EXCLUDED.metrics
        """)
        
        import json
        await self.db.execute(query, {
            "version": version,
            "agent_type": "position_sizing",
            "checkpoint_data": json.dumps({k: str(v) for k, v in checkpoint.items() if k.endswith("_state_dict")}),
            "created_at": datetime.utcnow(),
            "metrics": json.dumps({
                "episode_count": self.episode_count,
                "total_reward": self.total_reward,
                "avg_reward": self.total_reward / max(self.episode_count, 1),
            }),
        })
        
        await self.db.commit()
        
        logger.info(f"Saved SAC checkpoint: {version}")
    
    def get_metrics(self) -> Dict:
        """Get training metrics."""
        return {
            "episode_count": self.episode_count,
            "total_reward": self.total_reward,
            "avg_reward": self.total_reward / max(self.episode_count, 1),
            "buffer_size": len(self.buffer),
            "update_count": self.update_count,
        }
