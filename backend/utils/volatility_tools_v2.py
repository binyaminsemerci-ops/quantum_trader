"""
Volatility Tools v2 - Volatility Calculation and Market Pressure
=================================================================

Provides volatility metrics and market pressure indicators.

Author: Quantum Trader AI Team
Date: December 2, 2025
Version: 2.0
"""

from typing import List
import numpy as np
import structlog

logger = structlog.get_logger(__name__)


class VolatilityToolsV2:
    """
    Volatility calculation tools for RL v2.
    
    Calculates:
    - Market volatility (std dev of returns)
    - Market pressure (buy/sell pressure)
    """
    
    def __init__(self):
        """Initialize Volatility Tools v2."""
        logger.info("[Volatility Tools v2] Initialized")
    
    def calculate_volatility(self, price_history: List[float]) -> float:
        """
        Calculate market volatility from price history.
        
        Uses standard deviation of returns.
        
        Args:
            price_history: Recent price history
            
        Returns:
            Volatility value
        """
        if len(price_history) < 3:
            return 0.02  # Default 2%
        
        prices = np.array(price_history)
        
        # Calculate returns
        returns = np.diff(prices) / prices[:-1]
        
        # Calculate volatility (std dev)
        volatility = np.std(returns)
        
        return float(volatility)
    
    def calculate_market_pressure(self, price_history: List[float]) -> float:
        """
        Calculate market pressure from price momentum.
        
        Positive = buying pressure
        Negative = selling pressure
        
        Args:
            price_history: Recent price history
            
        Returns:
            Market pressure (-1 to 1)
        """
        if len(price_history) < 5:
            return 0.0
        
        # Recent price change
        recent_change = (price_history[-1] - price_history[-5]) / price_history[-5]
        
        # Normalize to [-1, 1]
        pressure = np.tanh(recent_change * 20)
        
        return float(pressure)
