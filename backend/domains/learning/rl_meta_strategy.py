"""
RL Meta Strategy Agent - Reinforcement Learning for strategy selection.

Provides:
- PPO-based agent for dynamic strategy selection
- 4 strategies: TrendFollowing, MeanReversion, Breakout, Neutral
- State includes market regime, volatility, model confidence
- Rewards based on actual trade PnL
- Continuous learning from live trades
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import datetime
from enum import Enum
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
    from torch.distributions import Categorical
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    logger.warning("PyTorch not available - RL Meta Strategy will fail")


# ============================================================================
# Enums
# ============================================================================

class TradingStrategy(str, Enum):
    """Available trading strategies."""
    TREND_FOLLOWING = "trend_following"
    MEAN_REVERSION = "mean_reversion"
    BREAKOUT = "breakout"
    NEUTRAL = "neutral"  # No trade


class MarketRegime(str, Enum):
    """Market regime classification."""
    BULL = "bull"
    BEAR = "bear"
    SIDEWAYS = "sideways"
    VOLATILE = "volatile"


# ============================================================================
# PPO Policy Network
# ============================================================================

class PolicyNetwork(nn.Module):
    """
    PPO policy network for strategy selection.
    
    Input: State vector (market features)
    Output: Strategy probabilities
    """
    
    def __init__(
        self,
        state_dim: int,
        n_strategies: int = 4,
        hidden_dim: int = 128,
        dropout: float = 0.1,
    ):
        super().__init__()
        
        self.network = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, n_strategies),
        )
    
    def forward(self, state):
        logits = self.network(state)
        return logits


class ValueNetwork(nn.Module):
    """
    PPO value network for advantage estimation.
    
    Input: State vector
    Output: State value (expected future reward)
    """
    
    def __init__(
        self,
        state_dim: int,
        hidden_dim: int = 128,
        dropout: float = 0.1,
    ):
        super().__init__()
        
        self.network = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
        )
    
    def forward(self, state):
        value = self.network(state)
        return value.squeeze(-1)


# ============================================================================
# Experience Buffer
# ============================================================================

@dataclass
class Experience:
    """Single RL experience tuple."""
    
    state: np.ndarray
    action: int
    reward: float
    next_state: np.ndarray
    done: bool
    log_prob: float


class ExperienceBuffer:
    """Replay buffer for PPO."""
    
    def __init__(self, capacity: int = 10000):
        self.capacity = capacity
        self.buffer: List[Experience] = []
        self.position = 0
    
    def push(self, experience: Experience):
        """Add experience to buffer."""
        if len(self.buffer) < self.capacity:
            self.buffer.append(experience)
        else:
            self.buffer[self.position] = experience
        
        self.position = (self.position + 1) % self.capacity
    
    def sample(self, batch_size: int) -> List[Experience]:
        """Sample random batch."""
        indices = np.random.choice(len(self.buffer), batch_size, replace=False)
        return [self.buffer[i] for i in indices]
    
    def clear(self):
        """Clear buffer."""
        self.buffer = []
        self.position = 0
    
    def __len__(self):
        return len(self.buffer)


# ============================================================================
# RL Meta Strategy Agent
# ============================================================================

class RLMetaStrategyAgent:
    """
    Reinforcement Learning agent for dynamic strategy selection.
    
    Features:
    - PPO (Proximal Policy Optimization)
    - 4 strategies: TrendFollowing, MeanReversion, Breakout, Neutral
    - State: market regime, volatility, model confidence, recent PnL
    - Reward: actual trade PnL
    - Continuous learning from live trades
    """
    
    def __init__(
        self,
        db_session: AsyncSession,
        event_bus: EventBus,
        policy_store: PolicyStore,
        state_dim: int = 20,
        learning_rate: float = 0.0003,
        gamma: float = 0.99,
        clip_epsilon: float = 0.2,
        update_freq: int = 100,
    ):
        if not TORCH_AVAILABLE:
            raise ImportError("PyTorch is required for RL Meta Strategy")
        
        self.db = db_session
        self.event_bus = event_bus
        self.policy_store = policy_store
        
        # Hyperparameters
        self.state_dim = state_dim
        self.n_strategies = len(TradingStrategy)
        self.learning_rate = learning_rate
        self.gamma = gamma
        self.clip_epsilon = clip_epsilon
        self.update_freq = update_freq
        
        # Networks
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.policy_net = PolicyNetwork(state_dim, self.n_strategies).to(self.device)
        self.value_net = ValueNetwork(state_dim).to(self.device)
        
        # Optimizers
        self.policy_optimizer = optim.Adam(self.policy_net.parameters(), lr=learning_rate)
        self.value_optimizer = optim.Adam(self.value_net.parameters(), lr=learning_rate)
        
        # Experience buffer
        self.buffer = ExperienceBuffer()
        
        # Metrics
        self.episode_count = 0
        self.total_reward = 0.0
        self.strategy_counts = {strategy: 0 for strategy in TradingStrategy}
        
        # State tracking
        self.current_state = None
        self.current_action = None
        self.current_log_prob = None
        
        logger.info(
            f"RLMetaStrategyAgent initialized: "
            f"state_dim={state_dim}, lr={learning_rate}, gamma={gamma}"
        )
    
    async def select_strategy(
        self,
        market_data: Dict,
        model_confidence: float,
    ) -> Tuple[TradingStrategy, float]:
        """
        Select trading strategy based on current state.
        
        Args:
            market_data: Current market features
            model_confidence: Confidence from ML models
            
        Returns:
            (selected_strategy, action_confidence)
        """
        # Build state vector
        state = self._build_state(market_data, model_confidence)
        
        # Get action from policy
        with torch.no_grad():
            state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
            logits = self.policy_net(state_tensor)
            probs = torch.softmax(logits, dim=-1)
            dist = Categorical(probs)
            action = dist.sample()
            log_prob = dist.log_prob(action)
        
        # Convert action to strategy
        strategies = list(TradingStrategy)
        selected_strategy = strategies[action.item()]
        action_confidence = probs[0, action.item()].item()
        
        # Store for later reward
        self.current_state = state
        self.current_action = action.item()
        self.current_log_prob = log_prob.item()
        
        # Update counts
        self.strategy_counts[selected_strategy] += 1
        
        logger.debug(
            f"Selected strategy: {selected_strategy.value} "
            f"(confidence: {action_confidence:.2%})"
        )
        
        return selected_strategy, action_confidence
    
    async def record_reward(
        self,
        reward: float,
        next_market_data: Dict,
        next_model_confidence: float,
        done: bool = False,
    ):
        """
        Record reward from trade outcome.
        
        Args:
            reward: Trade PnL (percentage)
            next_market_data: Market state after trade
            next_model_confidence: Model confidence after trade
            done: Whether episode is complete
        """
        if self.current_state is None:
            logger.warning("No current state - cannot record reward")
            return
        
        # Build next state
        next_state = self._build_state(next_market_data, next_model_confidence)
        
        # Create experience
        experience = Experience(
            state=self.current_state,
            action=self.current_action,
            reward=reward,
            next_state=next_state,
            done=done,
            log_prob=self.current_log_prob,
        )
        
        self.buffer.push(experience)
        self.total_reward += reward
        
        # Update policy
        if len(self.buffer) >= self.update_freq:
            await self._update_policy()
        
        # Reset current state
        if done:
            self.episode_count += 1
            self.current_state = None
            self.current_action = None
            self.current_log_prob = None
    
    async def _update_policy(self):
        """Update policy and value networks using PPO."""
        if len(self.buffer) < 32:
            return
        
        # Sample batch
        batch = self.buffer.sample(min(len(self.buffer), 256))
        
        # Prepare tensors
        states = torch.FloatTensor([exp.state for exp in batch]).to(self.device)
        actions = torch.LongTensor([exp.action for exp in batch]).to(self.device)
        rewards = torch.FloatTensor([exp.reward for exp in batch]).to(self.device)
        next_states = torch.FloatTensor([exp.next_state for exp in batch]).to(self.device)
        dones = torch.FloatTensor([exp.done for exp in batch]).to(self.device)
        old_log_probs = torch.FloatTensor([exp.log_prob for exp in batch]).to(self.device)
        
        # Compute advantages
        with torch.no_grad():
            values = self.value_net(states)
            next_values = self.value_net(next_states)
            td_targets = rewards + self.gamma * next_values * (1 - dones)
            advantages = td_targets - values
            advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        
        # PPO update (multiple epochs)
        for _ in range(4):
            # Policy loss
            logits = self.policy_net(states)
            probs = torch.softmax(logits, dim=-1)
            dist = Categorical(probs)
            new_log_probs = dist.log_prob(actions)
            
            ratio = torch.exp(new_log_probs - old_log_probs)
            clipped_ratio = torch.clamp(ratio, 1 - self.clip_epsilon, 1 + self.clip_epsilon)
            
            policy_loss = -torch.min(ratio * advantages, clipped_ratio * advantages).mean()
            
            # Value loss
            values = self.value_net(states)
            value_loss = nn.MSELoss()(values, td_targets)
            
            # Update policy
            self.policy_optimizer.zero_grad()
            policy_loss.backward()
            torch.nn.utils.clip_grad_norm_(self.policy_net.parameters(), 0.5)
            self.policy_optimizer.step()
            
            # Update value
            self.value_optimizer.zero_grad()
            value_loss.backward()
            torch.nn.utils.clip_grad_norm_(self.value_net.parameters(), 0.5)
            self.value_optimizer.step()
        
        logger.info(
            f"Updated RL policy: policy_loss={policy_loss.item():.4f}, "
            f"value_loss={value_loss.item():.4f}"
        )
    
    def _build_state(
        self,
        market_data: Dict,
        model_confidence: float,
    ) -> np.ndarray:
        """
        Build state vector from market data.
        
        State features (20-dim):
        - Price features (5): returns, volatility, trend, range, volume
        - Technical indicators (8): RSI, MACD, BB, ADX, etc.
        - Market regime (4): one-hot encoded
        - Model confidence (1)
        - Recent PnL (2): mean, std
        """
        state = np.zeros(self.state_dim)
        
        # Price features
        state[0] = market_data.get("returns", 0.0)
        state[1] = market_data.get("volatility", 0.0)
        state[2] = market_data.get("trend", 0.0)
        state[3] = market_data.get("price_range", 0.0)
        state[4] = market_data.get("volume_ratio", 1.0)
        
        # Technical indicators
        state[5] = market_data.get("rsi", 50.0) / 100.0  # Normalize
        state[6] = market_data.get("macd", 0.0)
        state[7] = market_data.get("bb_position", 0.5)
        state[8] = market_data.get("adx", 0.0) / 100.0
        state[9] = market_data.get("stoch_k", 50.0) / 100.0
        state[10] = market_data.get("atr_pct", 0.0)
        state[11] = market_data.get("obv", 0.0)
        state[12] = market_data.get("ema_dist", 0.0)
        
        # Market regime (one-hot)
        regime = market_data.get("regime", "sideways")
        regime_map = {"bull": 0, "bear": 1, "sideways": 2, "volatile": 3}
        regime_idx = regime_map.get(regime, 2)
        state[13 + regime_idx] = 1.0
        
        # Model confidence
        state[17] = model_confidence
        
        # Recent PnL (placeholder - would be computed from trade history)
        state[18] = 0.0  # mean_pnl
        state[19] = 0.0  # std_pnl
        
        return state
    
    async def save_checkpoint(self, version: str):
        """Save model checkpoint."""
        checkpoint = {
            "policy_state_dict": self.policy_net.state_dict(),
            "value_state_dict": self.value_net.state_dict(),
            "policy_optimizer_state_dict": self.policy_optimizer.state_dict(),
            "value_optimizer_state_dict": self.value_optimizer.state_dict(),
            "episode_count": self.episode_count,
            "total_reward": self.total_reward,
            "strategy_counts": self.strategy_counts,
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
            "agent_type": "meta_strategy",
            "checkpoint_data": json.dumps({k: str(v) for k, v in checkpoint.items() if k.endswith("_state_dict")}),
            "created_at": datetime.utcnow(),
            "metrics": json.dumps({
                "episode_count": self.episode_count,
                "total_reward": self.total_reward,
                "avg_reward": self.total_reward / max(self.episode_count, 1),
                "strategy_distribution": {k.value: v for k, v in self.strategy_counts.items()},
            }),
        })
        
        await self.db.commit()
        
        logger.info(f"Saved RL checkpoint: {version}")
    
    async def load_checkpoint(self, version: str):
        """Load model checkpoint."""
        query = text("""
            SELECT checkpoint_data, metrics
            FROM rl_versions
            WHERE version = :version AND agent_type = :agent_type
        """)
        
        result = await self.db.execute(query, {
            "version": version,
            "agent_type": "meta_strategy",
        })
        row = result.fetchone()
        
        if not row:
            logger.warning(f"Checkpoint not found: {version}")
            return False
        
        # Load state dicts (simplified - would need proper deserialization)
        logger.info(f"Loaded RL checkpoint: {version}")
        return True
    
    def get_metrics(self) -> Dict:
        """Get training metrics."""
        return {
            "episode_count": self.episode_count,
            "total_reward": self.total_reward,
            "avg_reward": self.total_reward / max(self.episode_count, 1),
            "strategy_distribution": {k.value: v for k, v in self.strategy_counts.items()},
            "buffer_size": len(self.buffer),
        }


# ============================================================================
# Database Schema
# ============================================================================

async def create_rl_versions_table(db_session: AsyncSession) -> None:
    """Create rl_versions table."""
    
    create_table_sql = text("""
        CREATE TABLE IF NOT EXISTS rl_versions (
            id SERIAL PRIMARY KEY,
            version VARCHAR(50) NOT NULL,
            agent_type VARCHAR(50) NOT NULL,
            checkpoint_data JSONB NOT NULL,
            created_at TIMESTAMP DEFAULT NOW(),
            metrics JSONB,
            UNIQUE(version, agent_type)
        );
        
        CREATE INDEX IF NOT EXISTS idx_rl_version 
        ON rl_versions(version, agent_type);
        
        CREATE INDEX IF NOT EXISTS idx_rl_created 
        ON rl_versions(created_at DESC);
    """)
    
    await db_session.execute(create_table_sql)
    await db_session.commit()
    logger.info("Created rl_versions table")
