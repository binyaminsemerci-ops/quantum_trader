"""
Win Rate Tracker v2 - Trailing Win Rate Calculation
====================================================

Tracks win rates over rolling windows.

Author: Quantum Trader AI Team
Date: December 2, 2025
Version: 2.0
"""

from collections import deque
import numpy as np
import structlog

logger = structlog.get_logger(__name__)


class WinRateTrackerV2:
    """
    Win rate tracker for RL v2.
    
    Tracks trailing win rate from recent trades.
    """
    
    def __init__(self, window_size: int = 20):
        """
        Initialize Win Rate Tracker v2.
        
        Args:
            window_size: Window for trailing win rate
        """
        self.window_size = window_size
        self.trade_outcomes: deque = deque(maxlen=window_size)
        
        logger.info(
            "[Win Rate Tracker v2] Initialized",
            window_size=window_size
        )
    
    def record_trade_outcome(self, is_win: bool):
        """
        Record trade outcome.
        
        Args:
            is_win: True if trade was profitable
        """
        self.trade_outcomes.append(1 if is_win else 0)
    
    def get_trailing_winrate(self) -> float:
        """
        Get trailing win rate from recent trades.
        
        Returns:
            Win rate (0-1)
        """
        if len(self.trade_outcomes) < 3:
            return 0.5  # Default to 50%
        
        winrate = np.mean(list(self.trade_outcomes))
        return float(winrate)
    
    def reset(self):
        """Reset trade outcomes."""
        self.trade_outcomes.clear()
        logger.info("[Win Rate Tracker v2] Reset")
