"""
RL Position Sizing Agent - Phase 4O+
Reinforcement Learning Agent for Adaptive Position Sizing

Uses policy gradient RL to learn optimal position sizing based on:
- Market state (volatility, divergence, funding)
- Signal confidence
- Historical PnL trend
- Cross-exchange metrics

State Vector:
    [confidence, volatility, pnl_trend, exch_divergence, funding_rate, margin_util]

Reward Function:
    reward = (pnl_pct × confidence)
             - 0.005 × |leverage - target_leverage|
             - 0.002 × exch_divergence
             + 0.003 × sign(pnl_trend)

Policy:
    Learns weighted priorities between stability, profit-rate, and cross-exchange alignment
"""

import numpy as np
import logging
import json
import time
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
from pathlib import Path

try:
    import torch
    import torch.nn as nn
    import torch.optim as optim
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    logger = logging.getLogger(__name__)
    logger.warning("[RL-Agent] PyTorch not available, using fallback policy")

logger = logging.getLogger(__name__)


@dataclass
class Experience:
    """Single RL experience/transition"""
    state: np.ndarray  # State vector
    action: float  # Position size/leverage adjustment
    reward: float  # Calculated reward
    next_state: np.ndarray  # Next state
    timestamp: float


class PolicyNetwork(nn.Module if TORCH_AVAILABLE else object):
    """Neural network policy for position sizing"""
    
    def __init__(self, state_dim: int = 6, hidden_dim: int = 64):
        """
        Initialize policy network
        
        Args:
            state_dim: Dimension of state vector (default 6)
            hidden_dim: Hidden layer size
        """
        if not TORCH_AVAILABLE:
            return
            
        super().__init__()
        
        self.network = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),  # Output: position size multiplier
            nn.Sigmoid()  # Constrain to [0, 1]
        )
    
    def forward(self, state):
        """Forward pass"""
        if not TORCH_AVAILABLE:
            return None
        return self.network(state)


class RLPositionSizingAgent:
    """
    Reinforcement Learning agent for adaptive position sizing
    
    Learns to optimize:
    - Risk-adjusted returns
    - Leverage stability
    - Cross-exchange alignment
    """
    
    def __init__(
        self,
        model_path: str = "/models/rl_sizing_agent_v3.pth",
        config: Optional[Dict] = None
    ):
        """
        Initialize RL agent
        
        Args:
            model_path: Path to save/load policy network
            config: Configuration overrides
        """
        self.model_path = Path(model_path)
        self.config = config or {}
        
        # State dimension
        self.state_dim = 6  # [confidence, volatility, pnl_trend, divergence, funding, margin_util]
        
        # Reward weights
        self.pnl_weight = self.config.get("pnl_weight", 1.0)
        self.leverage_penalty = self.config.get("leverage_penalty", 0.005)
        self.divergence_penalty = self.config.get("divergence_penalty", 0.002)
        self.trend_bonus = self.config.get("trend_bonus", 0.003)
        
        # Training parameters
        self.learning_rate = self.config.get("learning_rate", 0.001)
        self.retrain_interval = self.config.get("retrain_interval", 100)  # trades
        self.retrain_threshold = self.config.get("retrain_threshold", 0.001)  # mean abs PnL
        
        # Experience replay
        self.experiences: List[Experience] = []
        self.max_experiences = self.config.get("max_experiences", 1000)
        
        # Statistics
        self.trades_processed = 0
        self.policy_updates = 0
        self.avg_reward = 0.0
        self.reward_history = []
        
        # Initialize policy network
        if TORCH_AVAILABLE:
            self.policy = PolicyNetwork(state_dim=self.state_dim)
            self.optimizer = optim.Adam(self.policy.parameters(), lr=self.learning_rate)
            
            # Load existing policy if available
            if self.model_path.exists():
                self._load_policy()
                logger.info(f"[RL-Agent] Loaded policy from {self.model_path}")
            else:
                logger.info(f"[RL-Agent] Initialized new policy network")
        else:
            self.policy = None
            logger.warning("[RL-Agent] Using fallback (no learning)")
        
        logger.info(
            f"[RL-Agent] Initialized | "
            f"State dim: {self.state_dim} | "
            f"Retrain: every {self.retrain_interval} trades"
        )
    
    def get_position_size_multiplier(self, state: Dict) -> float:
        """
        Get position size multiplier from current policy
        
        Args:
            state: State dict with keys:
                - confidence: float
                - volatility: float
                - pnl_trend: float
                - exch_divergence: float
                - funding_rate: float
                - margin_util: float
        
        Returns:
            float: Position size multiplier [0.5, 1.5]
                  (multiply with base position size)
        """
        # Build state vector
        state_vector = np.array([
            state.get("confidence", 0.5),
            state.get("volatility", 1.0),
            state.get("pnl_trend", 0.0),
            state.get("exch_divergence", 0.0),
            state.get("funding_rate", 0.0),
            state.get("margin_util", 0.0)
        ], dtype=np.float32)
        
        # Get policy prediction
        if TORCH_AVAILABLE and self.policy:
            with torch.no_grad():
                state_tensor = torch.FloatTensor(state_vector).unsqueeze(0)
                multiplier = self.policy(state_tensor).item()
                
                # Scale from [0, 1] to [0.5, 1.5]
                multiplier = 0.5 + multiplier * 1.0
        else:
            # Fallback: Simple heuristic
            multiplier = (
                0.5 +
                state.get("confidence", 0.5) * 0.5 +
                max(0, state.get("pnl_trend", 0.0)) * 0.25 -
                state.get("volatility", 1.0) * 0.15 -
                state.get("exch_divergence", 0.0) * 0.2
            )
            multiplier = max(0.5, min(1.5, multiplier))
        
        return multiplier
    
    def calculate_reward(
        self,
        pnl_pct: float,
        confidence: float,
        leverage: float,
        target_leverage: float,
        exch_divergence: float,
        pnl_trend: float
    ) -> float:
        """
        Calculate reward for an experience
        
        Args:
            pnl_pct: PnL percentage [-1 to +1]
            confidence: Signal confidence [0-1]
            leverage: Actual leverage used
            target_leverage: Target leverage (from ILF)
            exch_divergence: Cross-exchange divergence [0-1]
            pnl_trend: Recent PnL trend [-1 to +1]
        
        Returns:
            float: Calculated reward
        """
        # Base reward: PnL weighted by confidence
        reward = pnl_pct * confidence * self.pnl_weight
        
        # Penalty for leverage deviation
        leverage_diff = abs(leverage - target_leverage)
        reward -= self.leverage_penalty * leverage_diff
        
        # Penalty for cross-exchange divergence
        reward -= self.divergence_penalty * exch_divergence
        
        # Bonus for positive trend
        reward += self.trend_bonus * np.sign(pnl_trend)
        
        return reward
    
    def record_experience(
        self,
        state: Dict,
        action: float,
        pnl_pct: float,
        next_state: Dict,
        leverage: float,
        target_leverage: float
    ):
        """
        Record a trading experience
        
        Args:
            state: State before trade
            action: Position size multiplier used
            pnl_pct: Resulting PnL percentage
            next_state: State after trade
            leverage: Actual leverage used
            target_leverage: Target leverage from ILF
        """
        # Calculate reward
        reward = self.calculate_reward(
            pnl_pct=pnl_pct,
            confidence=state.get("confidence", 0.5),
            leverage=leverage,
            target_leverage=target_leverage,
            exch_divergence=state.get("exch_divergence", 0.0),
            pnl_trend=state.get("pnl_trend", 0.0)
        )
        
        # Build state vectors
        state_vector = np.array([
            state.get("confidence", 0.5),
            state.get("volatility", 1.0),
            state.get("pnl_trend", 0.0),
            state.get("exch_divergence", 0.0),
            state.get("funding_rate", 0.0),
            state.get("margin_util", 0.0)
        ], dtype=np.float32)
        
        next_state_vector = np.array([
            next_state.get("confidence", 0.5),
            next_state.get("volatility", 1.0),
            next_state.get("pnl_trend", 0.0),
            next_state.get("exch_divergence", 0.0),
            next_state.get("funding_rate", 0.0),
            next_state.get("margin_util", 0.0)
        ], dtype=np.float32)
        
        # Create experience
        exp = Experience(
            state=state_vector,
            action=action,
            reward=reward,
            next_state=next_state_vector,
            timestamp=time.time()
        )
        
        # Add to buffer
        self.experiences.append(exp)
        if len(self.experiences) > self.max_experiences:
            self.experiences.pop(0)
        
        # Update statistics
        self.trades_processed += 1
        self.reward_history.append(reward)
        if len(self.reward_history) > 100:
            self.reward_history.pop(0)
        self.avg_reward = np.mean(self.reward_history)
        
        logger.debug(
            f"[RL-Agent] Experience recorded | "
            f"Reward: {reward:.4f} | "
            f"PnL: {pnl_pct*100:.2f}%"
        )
        
        # Check if retraining needed
        self._check_retrain()
    
    def update_policy(
        self,
        pnl_trend: float,
        leverage: float,
        confidence: float,
        exch_divergence: float,
        funding_rate: float
    ):
        """
        Update policy based on recent performance (deprecated - use record_experience)
        
        Kept for backwards compatibility with existing code
        """
        logger.warning("[RL-Agent] update_policy() is deprecated, use record_experience()")
    
    def _check_retrain(self):
        """Check if policy should be retrained"""
        if not TORCH_AVAILABLE or not self.policy:
            return
        
        # Retrain every N trades
        if self.trades_processed % self.retrain_interval == 0 and len(self.experiences) >= 50:
            mean_abs_pnl = np.mean([abs(exp.reward) for exp in self.experiences[-50:]])
            
            # Only retrain if performance below threshold
            if mean_abs_pnl < self.retrain_threshold:
                logger.info(
                    f"[RL-Agent] Low performance detected "
                    f"(mean PnL: {mean_abs_pnl:.4f}), retraining..."
                )
                self._retrain()
    
    def _retrain(self):
        """Retrain policy network using policy gradient"""
        if not TORCH_AVAILABLE or not self.policy:
            return
        
        if len(self.experiences) < 10:
            logger.warning("[RL-Agent] Insufficient experiences for retraining")
            return
        
        try:
            # Prepare batch
            states = torch.FloatTensor([exp.state for exp in self.experiences])
            rewards = torch.FloatTensor([exp.reward for exp in self.experiences])
            
            # Normalize rewards
            rewards = (rewards - rewards.mean()) / (rewards.std() + 1e-8)
            
            # Forward pass
            self.optimizer.zero_grad()
            predictions = self.policy(states).squeeze()
            
            # Policy gradient loss
            # Want to maximize reward, so minimize negative reward
            loss = -(predictions * rewards).mean()
            
            # Backward pass
            loss.backward()
            self.optimizer.step()
            
            self.policy_updates += 1
            
            logger.info(
                f"[RL-Agent] Policy updated | "
                f"Update #{self.policy_updates} | "
                f"Loss: {loss.item():.4f} | "
                f"Avg Reward: {self.avg_reward:.4f}"
            )
            
            # Save policy
            self._save_policy()
            
        except Exception as e:
            logger.error(f"[RL-Agent] Retraining failed: {e}")
    
    def _save_policy(self):
        """Save policy network to disk"""
        if not TORCH_AVAILABLE or not self.policy:
            return
        
        try:
            self.model_path.parent.mkdir(parents=True, exist_ok=True)
            
            torch.save({
                'policy_state_dict': self.policy.state_dict(),
                'optimizer_state_dict': self.optimizer.state_dict(),
                'policy_updates': self.policy_updates,
                'trades_processed': self.trades_processed,
                'avg_reward': self.avg_reward
            }, self.model_path)
            
            logger.debug(f"[RL-Agent] Policy saved to {self.model_path}")
            
        except Exception as e:
            logger.error(f"[RL-Agent] Failed to save policy: {e}")
    
    def _load_policy(self):
        """Load policy network from disk"""
        if not TORCH_AVAILABLE or not self.policy:
            return
        
        try:
            checkpoint = torch.load(self.model_path)
            
            self.policy.load_state_dict(checkpoint['policy_state_dict'])
            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            self.policy_updates = checkpoint.get('policy_updates', 0)
            self.trades_processed = checkpoint.get('trades_processed', 0)
            self.avg_reward = checkpoint.get('avg_reward', 0.0)
            
            logger.info(
                f"[RL-Agent] Policy loaded | "
                f"Updates: {self.policy_updates} | "
                f"Trades: {self.trades_processed}"
            )
            
        except Exception as e:
            logger.error(f"[RL-Agent] Failed to load policy: {e}")
    
    def get_statistics(self) -> Dict:
        """Get agent statistics"""
        return {
            "trades_processed": self.trades_processed,
            "policy_updates": self.policy_updates,
            "avg_reward": round(self.avg_reward, 4),
            "recent_rewards": [round(r, 4) for r in self.reward_history[-10:]],
            "experiences_buffered": len(self.experiences),
            "pytorch_available": TORCH_AVAILABLE,
            "policy_loaded": (self.policy is not None)
        }


# Global singleton
_rl_agent: Optional[RLPositionSizingAgent] = None


def get_rl_agent(
    model_path: str = "/models/rl_sizing_agent_v3.pth",
    config: Optional[Dict] = None
) -> RLPositionSizingAgent:
    """Get or create global RL agent instance"""
    global _rl_agent
    
    if _rl_agent is None:
        _rl_agent = RLPositionSizingAgent(model_path=model_path, config=config)
        logger.info("[RL-Agent] Global agent initialized")
    
    return _rl_agent
