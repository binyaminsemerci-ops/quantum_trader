"""
RL Meta Strategy Agent v2 - Advanced Meta Strategy Learning
=============================================================

Upgraded meta strategy agent with:
- State representation v2
- Reward engine v2
- Action space v2 (TREND, RANGE, BREAKOUT, MEAN_REVERSION)
- Episode tracking
- TD-learning

Author: Quantum Trader AI Team
Date: December 2, 2025
Version: 2.0
"""

from typing import Dict, Any, Optional
import time
import structlog

from backend.services.ai.rl_state_manager_v2 import get_state_manager
from backend.services.ai.rl_reward_engine_v2 import get_reward_engine
from backend.services.ai.rl_action_space_v2 import get_action_space
from backend.services.ai.rl_episode_tracker_v2 import get_episode_tracker

logger = structlog.get_logger(__name__)


class RLMetaStrategyAgentV2:
    """
    RL Meta Strategy Agent v2.
    
    Learns optimal strategy selection based on:
    - Market regime
    - Volatility
    - Market pressure
    - Signal confidence
    - Previous win rate
    - Account health
    """
    
    def __init__(
        self,
        epsilon: float = 0.1,
        epsilon_decay: float = 0.995,
        min_epsilon: float = 0.01
    ):
        """
        Initialize RL Meta Strategy Agent v2.
        
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
        self.current_actions: Dict[str, str] = {}
        
        logger.info(
            "[RL Meta Strategy Agent v2] Initialized",
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
        Set current state for meta strategy decision.
        
        Builds state representation v2 from signal data.
        
        Args:
            trace_id: Trace ID
            state_data: Raw state data (symbol, confidence, timeframe, etc.)
        """
        # Extract state components
        regime = state_data.get("regime", "UNKNOWN")
        confidence = state_data.get("confidence", 0.5)
        market_price = state_data.get("market_price", 50000.0)
        account_balance = state_data.get("account_balance", 10000.0)
        
        # Build state v2
        state = self.state_manager.build_meta_strategy_state(
            regime=regime,
            confidence=confidence,
            market_price=market_price,
            account_balance=account_balance,
            trace_id=trace_id
        )
        
        self.current_states[trace_id] = state
        
        # Start episode
        self.episode_tracker.start_episode(trace_id, time.time())
        
        logger.debug(
            "[RL Meta Strategy Agent v2] State set",
            trace_id=trace_id,
            state=state
        )
    
    def select_action(self, trace_id: str) -> str:
        """
        Select meta strategy action using epsilon-greedy policy.
        
        Args:
            trace_id: Trace ID
            
        Returns:
            Selected strategy (TREND, RANGE, BREAKOUT, MEAN_REVERSION)
        """
        if trace_id not in self.current_states:
            logger.warning(
                "[RL Meta Strategy Agent v2] No state for action selection",
                trace_id=trace_id
            )
            return "TREND"  # Default
        
        state = self.current_states[trace_id]
        
        # Get Q-values for current state
        q_values = self.episode_tracker.get_meta_q_values(state)
        
        # Select action using epsilon-greedy
        action = self.action_space.select_meta_strategy_action(
            q_values=q_values,
            epsilon=self.epsilon
        )
        
        self.current_actions[trace_id] = action
        
        logger.info(
            "[RL Meta Strategy Agent v2] Action selected",
            trace_id=trace_id,
            action=action,
            q_values=q_values,
            epsilon=self.epsilon
        )
        
        return action
    
    def update(
        self,
        trace_id: str,
        reward: Optional[float] = None,
        pnl_pct: Optional[float] = None,
        max_drawdown_pct: Optional[float] = None,
        current_regime: Optional[str] = None,
        predicted_regime: Optional[str] = None,
        confidence: Optional[float] = None
    ):
        """
        Update agent with reward and perform TD-learning.
        
        Args:
            trace_id: Trace ID
            reward: Pre-calculated reward (optional)
            pnl_pct: P&L percentage (for reward calculation)
            max_drawdown_pct: Max drawdown percentage
            current_regime: Actual market regime
            predicted_regime: Predicted regime
            confidence: Signal confidence
        """
        if trace_id not in self.current_states:
            logger.warning(
                "[RL Meta Strategy Agent v2] No state for update",
                trace_id=trace_id
            )
            return
        
        state = self.current_states[trace_id]
        action = self.current_actions.get(trace_id, "TREND")
        
        # Calculate reward if not provided
        if reward is None:
            if pnl_pct is None:
                logger.warning(
                    "[RL Meta Strategy Agent v2] No reward or pnl_pct provided",
                    trace_id=trace_id
                )
                return
            
            reward = self.reward_engine.calculate_meta_strategy_reward(
                pnl_pct=pnl_pct,
                max_drawdown_pct=max_drawdown_pct or 0.0,
                current_regime=current_regime or "UNKNOWN",
                predicted_regime=predicted_regime or action,
                confidence=confidence or 0.5,
                trace_id=trace_id
            )
        
        # Record trade outcome for win rate
        is_win = (pnl_pct or 0.0) > 0
        self.state_manager.record_trade_outcome(is_win)
        
        # Add step to episode
        self.episode_tracker.add_step(
            trace_id=trace_id,
            state=state,
            action=action,
            reward=reward
        )
        
        # Perform TD-update (terminal state)
        new_q = self.episode_tracker.td_update_meta(
            state=state,
            action=action,
            reward=reward,
            next_state=None,  # Terminal
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
            "[RL Meta Strategy Agent v2] Update complete",
            trace_id=trace_id,
            reward=reward,
            new_q=new_q,
            epsilon=self.epsilon
        )
    
    def get_q_values(self, trace_id: str) -> Dict[str, float]:
        """
        Get Q-values for current state.
        
        Args:
            trace_id: Trace ID
            
        Returns:
            Q-values dict
        """
        if trace_id not in self.current_states:
            return {
                "TREND": 0.0,
                "RANGE": 0.0,
                "BREAKOUT": 0.0,
                "MEAN_REVERSION": 0.0
            }
        
        state = self.current_states[trace_id]
        return self.episode_tracker.get_meta_q_values(state)
    
    def get_stats(self) -> Dict[str, Any]:
        """
        Get agent statistics.
        
        Returns:
            Statistics dict
        """
        episode_stats = self.episode_tracker.get_episode_stats()
        
        return {
            "agent_type": "meta_strategy_v2",
            "epsilon": self.epsilon,
            "active_states": len(self.current_states),
            **episode_stats
        }
    
    def reset(self):
        """Reset agent state."""
        self.current_states.clear()
        self.current_actions.clear()
        self.epsilon = 0.1
        logger.info("[RL Meta Strategy Agent v2] Reset complete")


# Global singleton instance
_meta_agent_instance: Optional[RLMetaStrategyAgentV2] = None


def get_meta_agent() -> RLMetaStrategyAgentV2:
    """Get or create global RLMetaStrategyAgentV2 instance."""
    global _meta_agent_instance
    if _meta_agent_instance is None:
        _meta_agent_instance = RLMetaStrategyAgentV2()
    return _meta_agent_instance
