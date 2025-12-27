"""
RL Sizing Agent - Reinforcement Learning-based Position Sizing

This module provides RL-based position sizing decisions using trained agents
and PnL feedback for continuous improvement.

Components:
- rl_agent.py: RL agent implementation and inference
- pnl_feedback_listener.py: Listens to PnL outcomes and feeds back to agent

Integration:
- AI Engine calls get_rl_agent() for sizing recommendations
- Training worker trains agent offline using historical data
- PnL feedback enables online learning
"""

from .rl_agent import get_rl_agent

__all__ = ["get_rl_agent"]
