"""
Equity Curve Tools v2 - Equity Curve Analysis
==============================================

Analyzes equity curves for:
- Equity curve slope (trend)
- Account health (drawdown-based)

Author: Quantum Trader AI Team
Date: December 2, 2025
Version: 2.0
"""

from collections import deque
import numpy as np
import structlog

logger = structlog.get_logger(__name__)


class EquityCurveToolsV2:
    """
    Equity curve analysis tools for RL v2.
    
    Calculates:
    - Equity curve slope (linear regression)
    - Account health (drawdown-based)
    """
    
    def __init__(self, window_size: int = 30):
        """
        Initialize Equity Curve Tools v2.
        
        Args:
            window_size: Window for equity curve analysis
        """
        self.window_size = window_size
        self.equity_history: deque = deque(maxlen=window_size)
        
        logger.info(
            "[Equity Curve Tools v2] Initialized",
            window_size=window_size
        )
    
    def record_equity_point(self, equity: float):
        """
        Record equity point.
        
        Args:
            equity: Current equity/balance
        """
        self.equity_history.append(equity)
    
    def calculate_equity_curve_slope(self) -> float:
        """
        Calculate equity curve slope using linear regression.
        
        Positive slope = growing equity
        Negative slope = declining equity
        
        Returns:
            Equity curve slope (normalized)
        """
        if len(self.equity_history) < 5:
            return 0.0
        
        equity_array = np.array(list(self.equity_history))
        
        # Simple linear regression
        x = np.arange(len(equity_array))
        y = equity_array
        
        # Calculate slope
        n = len(x)
        slope = (n * np.sum(x * y) - np.sum(x) * np.sum(y)) / \
                (n * np.sum(x ** 2) - np.sum(x) ** 2)
        
        # Normalize by average equity
        avg_equity = np.mean(equity_array)
        normalized_slope = slope / max(avg_equity, 1.0)
        
        return float(normalized_slope)
    
    def calculate_account_health(self, current_balance: float) -> float:
        """
        Calculate account health score.
        
        Based on equity curve and recent performance.
        
        Args:
            current_balance: Current account balance
            
        Returns:
            Health score (0-1)
        """
        if len(self.equity_history) < 3:
            return 0.7  # Default healthy
        
        # Get historical equity
        equity_array = np.array(list(self.equity_history))
        
        # Calculate drawdown
        peak = np.maximum.accumulate(equity_array)
        drawdown = (peak - equity_array) / peak
        current_drawdown = drawdown[-1]
        
        # Health score based on drawdown
        if current_drawdown < 0.05:
            health = 1.0
        elif current_drawdown < 0.10:
            health = 0.8
        elif current_drawdown < 0.20:
            health = 0.6
        else:
            health = 0.4
        
        return float(health)
    
    def reset(self):
        """Reset equity history."""
        self.equity_history.clear()
        logger.info("[Equity Curve Tools v2] Reset")
