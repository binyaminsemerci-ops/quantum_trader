"""
RL Position Sizing Agent v2 - Advanced Position Sizing Learning
================================================================

Upgraded position sizing agent with:
- State representation v2
- Reward engine v2
- Action space v2 (size multipliers × leverage levels)
- Episode tracking
- TD-learning

Author: Quantum Trader AI Team
Date: December 2, 2025
Version: 2.0
"""

from typing import Dict, Any, Optional, Tuple
import time
import structlog

from backend.services.ai.rl_state_manager_v2 import get_state_manager
from backend.services.ai.rl_reward_engine_v2 import get_reward_engine
from backend.services.ai.rl_action_space_v2 import get_action_space
from backend.services.ai.rl_episode_tracker_v2 import get_episode_tracker

logger = structlog.get_logger(__name__)


class RLPositionSizingAgentV2:
    """
    RL Position Sizing Agent v2.
    
    Learns optimal position sizing based on:
    - Signal confidence
    - Portfolio exposure
    - Recent win rate
    - Volatility
    - Equity curve slope
    """
    
    def __init__(
        self,
        epsilon: float = 0.1,
        epsilon_decay: float = 0.995,
        min_epsilon: float = 0.01
    ):
        """
        Initialize RL Position Sizing Agent v2.
        
        Args:
            epsilon: Initial exploration rate
            epsilon_decay: Epsilon decay rate per episode
            min_epsilon: Minimum epsilon value
        """
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.min_epsilon = min_epsilon
        
        # V2 components
        self.state_manager = get_state_manager()
        self.reward_engine = get_reward_engine()
        self.action_space = get_action_space()
        self.episode_tracker = get_episode_tracker()
        
        # Current episode tracking
        self.current_states: Dict[str, Dict[str, Any]] = {}
        self.current_actions: Dict[str, Tuple[float, int]] = {}
        
        logger.info(
            "[RL Position Sizing Agent v2] Initialized",
            epsilon=epsilon,
            epsilon_decay=epsilon_decay,
            min_epsilon=min_epsilon
        )
    
    def set_current_state(
        self,
        trace_id: str,
        state_data: Dict[str, Any]
    ):
        """
        Set current state for position sizing decision.
        
        Builds state representation v2 from signal/trade data.
        
        Args:
            trace_id: Trace ID
            state_data: Raw state data (confidence, exposure, volatility, etc.)
        """
        # Extract state components
        signal_confidence = state_data.get("confidence", 0.5)
        portfolio_exposure = state_data.get("portfolio_exposure", 0.0)
        market_volatility = state_data.get("volatility", 0.02)
        account_balance = state_data.get("account_balance", 10000.0)
        
        # Build state v2
        state = self.state_manager.build_position_sizing_state(
            signal_confidence=signal_confidence,
            portfolio_exposure=portfolio_exposure,
            market_volatility=market_volatility,
            account_balance=account_balance,
            trace_id=trace_id
        )
        
        self.current_states[trace_id] = state
        
        # Start episode if not already started
        if trace_id not in self.episode_tracker.episodes:
            self.episode_tracker.start_episode(trace_id, time.time())
        
        logger.debug(
            "[RL Position Sizing Agent v2] State set",
            trace_id=trace_id,
            state=state
        )
    
    def select_action(self, trace_id: str) -> Tuple[float, int]:
        """
        Select position sizing action using epsilon-greedy policy.
        
        Args:
            trace_id: Trace ID
            
        Returns:
            (size_multiplier, leverage)
        """
        if trace_id not in self.current_states:
            logger.warning(
                "[RL Position Sizing Agent v2] No state for action selection",
                trace_id=trace_id
            )
            return (1.0, 3)  # Default
        
        state = self.current_states[trace_id]
        
        # Get Q-values for current state
        action_space_size = self.action_space.get_size_action_space_size()
        q_values = self.episode_tracker.get_size_q_values(state, action_space_size)
        
        # Select action index using epsilon-greedy
        if self.epsilon > 0 and len(q_values) > 0:
            import numpy as np
            if np.random.random() < self.epsilon:
                action_idx = np.random.randint(action_space_size)
            else:
                action_idx = int(np.argmax(q_values))
        else:
            action_idx = 28  # Default: multiplier=1.0, leverage=3
        
        # Decode action
        size_multiplier, leverage = self.action_space.decode_size_action(action_idx)
        
        self.current_actions[trace_id] = (size_multiplier, leverage)
        
        logger.info(
            "[RL Position Sizing Agent v2] Action selected",
            trace_id=trace_id,
            size_multiplier=size_multiplier,
            leverage=leverage,
            action_idx=action_idx,
            epsilon=self.epsilon
        )
        
        return (size_multiplier, leverage)
    
    def set_executed_action(
        self,
        trace_id: str,
        action_data: Dict[str, Any]
    ):
        """
        Record executed action (trade execution).
        
        Args:
            trace_id: Trace ID
            action_data: Executed action data (leverage, size_usd)
        """
        leverage = action_data.get("leverage", 3)
        position_size_usd = action_data.get("size_usd", 1000.0)
        
        # Store executed action
        # For simplicity, we store leverage and infer multiplier
        # In production, this would come from actual execution
        self.current_actions[trace_id] = (1.0, leverage)
        
        logger.debug(
            "[RL Position Sizing Agent v2] Executed action recorded",
            trace_id=trace_id,
            leverage=leverage,
            position_size_usd=position_size_usd
        )
    
    def update(
        self,
        trace_id: str,
        reward: Optional[float] = None,
        pnl_pct: Optional[float] = None,
        leverage: Optional[float] = None,
        position_size_usd: Optional[float] = None,
        account_balance: Optional[float] = None,
        market_volatility: Optional[float] = None
    ):
        """
        Update agent with reward and perform TD-learning.
        
        Args:
            trace_id: Trace ID
            reward: Pre-calculated reward (optional)
            pnl_pct: P&L percentage (for reward calculation)
            leverage: Applied leverage
            position_size_usd: Position size
            account_balance: Account balance
            market_volatility: Market volatility
        """
        if trace_id not in self.current_states:
            logger.warning(
                "[RL Position Sizing Agent v2] No state for update",
                trace_id=trace_id
            )
            return
        
        state = self.current_states[trace_id]
        action = self.current_actions.get(trace_id, (1.0, 3))
        
        # Calculate reward if not provided
        if reward is None:
            if pnl_pct is None:
                logger.warning(
                    "[RL Position Sizing Agent v2] No reward or pnl_pct provided",
                    trace_id=trace_id
                )
                return
            
            reward = self.reward_engine.calculate_position_sizing_reward(
                pnl_pct=pnl_pct,
                leverage=leverage or action[1],
                position_size_usd=position_size_usd or 1000.0,
                account_balance=account_balance or 10000.0,
                market_volatility=market_volatility or 0.02,
                trace_id=trace_id
            )
        
        # Record trade outcome for win rate
        is_win = (pnl_pct or 0.0) > 0
        self.state_manager.record_trade_outcome(is_win)
        
        # Encode action to index
        action_idx = self.action_space.encode_size_action(
            size_multiplier=action[0],
            leverage=action[1]
        )
        
        # Add step to episode
        self.episode_tracker.add_step(
            trace_id=trace_id,
            state=state,
            action=action_idx,
            reward=reward
        )
        
        # Perform TD-update (terminal state)
        action_space_size = self.action_space.get_size_action_space_size()
        new_q = self.episode_tracker.td_update_size(
            state=state,
            action_index=action_idx,
            reward=reward,
            next_state=None,  # Terminal
            action_space_size=action_space_size,
            trace_id=trace_id
        )
        
        # End episode
        self.episode_tracker.end_episode(trace_id, time.time())
        
        # Decay epsilon
        self.epsilon = max(
            self.min_epsilon,
            self.epsilon * self.epsilon_decay
        )
        
        # Cleanup
        self.current_states.pop(trace_id, None)
        self.current_actions.pop(trace_id, None)
        
        logger.info(
            "[RL Position Sizing Agent v2] Update complete",
            trace_id=trace_id,
            reward=reward,
            new_q=new_q,
            epsilon=self.epsilon
        )
    
    def get_q_values(self, trace_id: str) -> list:
        """
        Get Q-values for current state.
        
        Args:
            trace_id: Trace ID
            
        Returns:
            Q-values list
        """
        if trace_id not in self.current_states:
            action_space_size = self.action_space.get_size_action_space_size()
            return [0.0] * action_space_size
        
        state = self.current_states[trace_id]
        action_space_size = self.action_space.get_size_action_space_size()
        return self.episode_tracker.get_size_q_values(state, action_space_size)
    
    def get_stats(self) -> Dict[str, Any]:
        """
        Get agent statistics.
        
        Returns:
            Statistics dict
        """
        episode_stats = self.episode_tracker.get_episode_stats()
        
        return {
            "agent_type": "position_sizing_v2",
            "epsilon": self.epsilon,
            "active_states": len(self.current_states),
            **episode_stats
        }
    
    def reset(self):
        """Reset agent state."""
        self.current_states.clear()
        self.current_actions.clear()
        self.epsilon = 0.1
        logger.info("[RL Position Sizing Agent v2] Reset complete")


# Global singleton instance
_size_agent_instance: Optional[RLPositionSizingAgentV2] = None


def get_size_agent() -> RLPositionSizingAgentV2:
    """Get or create global RLPositionSizingAgentV2 instance."""
    global _size_agent_instance
    if _size_agent_instance is None:
        _size_agent_instance = RLPositionSizingAgentV2()
    return _size_agent_instance
