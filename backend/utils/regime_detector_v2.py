"""
Regime Detector v2 - Market Regime Classification
==================================================

Classifies market regimes:
- TREND: Strong directional movement
- RANGE: Sideways movement
- BREAKOUT: Breaking key levels
- MEAN_REVERSION: Reverting to mean

Author: Quantum Trader AI Team
Date: December 2, 2025
Version: 2.0
"""

from typing import List, Optional
import numpy as np
import structlog

logger = structlog.get_logger(__name__)


class RegimeDetectorV2:
    """
    Market regime detector for RL v2.
    
    Classifies market conditions into distinct regimes.
    """
    
    def __init__(self):
        """Initialize Regime Detector v2."""
        logger.info("[Regime Detector v2] Initialized")
    
    def detect_regime(
        self,
        price_history: List[float],
        volume_history: Optional[List[float]] = None
    ) -> str:
        """
        Detect market regime from price/volume data.
        
        Regimes:
        - TREND: Strong directional movement
        - RANGE: Sideways movement
        - BREAKOUT: Breaking key levels
        - MEAN_REVERSION: Reverting to mean
        
        Args:
            price_history: Recent price history
            volume_history: Recent volume history (optional)
            
        Returns:
            Regime label
        """
        if len(price_history) < 10:
            return "UNKNOWN"
        
        prices = np.array(price_history)
        
        # Calculate metrics
        returns = np.diff(prices) / prices[:-1]
        volatility = np.std(returns)
        trend_strength = abs(np.mean(returns))
        
        # Regime classification
        if trend_strength > 0.02 and volatility > 0.015:
            regime = "TREND"
        elif volatility < 0.01:
            regime = "RANGE"
        elif trend_strength > 0.03:
            regime = "BREAKOUT"
        else:
            regime = "MEAN_REVERSION"
        
        logger.debug(
            "[Regime Detector v2] Regime detected",
            regime=regime,
            trend_strength=trend_strength,
            volatility=volatility
        )
        
        return regime
