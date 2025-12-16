"""
RL State Manager v2 - Advanced State Representation System
===========================================================

Implements sophisticated state representation for:
- Meta Strategy RL (regime-aware, market pressure)
- Position Sizing RL (portfolio exposure, equity curve)

Author: Quantum Trader AI Team
Date: December 2, 2025
Version: 2.0
"""

from typing import Dict, Any, List, Optional
import numpy as np
from collections import deque
import structlog

logger = structlog.get_logger(__name__)


class StateManagerV2:
    """
    Advanced state management for RL v2.
    
    Computes:
    - Trailing win rates
    - Volatility metrics
    - Equity curve slopes
    - Market pressure
    - Regime labels
    """
    
    def __init__(
        self,
        winrate_window: int = 20,
        volatility_window: int = 14,
        equity_window: int = 30
    ):
        """
        Initialize State Manager v2.
        
        Args:
            winrate_window: Window for trailing win rate
            volatility_window: Window for volatility calculation
            equity_window: Window for equity curve slope
        """
        self.winrate_window = winrate_window
        self.volatility_window = volatility_window
        self.equity_window = equity_window
        
        # Historical buffers
        self.trade_outcomes: deque = deque(maxlen=winrate_window)
        self.price_history: deque = deque(maxlen=volatility_window)
        self.equity_history: deque = deque(maxlen=equity_window)
        
        logger.info(
            "[State Manager v2] Initialized",
            winrate_window=winrate_window,
            volatility_window=volatility_window,
            equity_window=equity_window
        )
    
    def build_meta_strategy_state(
        self,
        regime: str,
        confidence: float,
        market_price: float,
        account_balance: float,
        trace_id: str
    ) -> Dict[str, Any]:
        """
        Build Meta Strategy state representation v2.
        
        State includes:
        - regime: Market regime label
        - volatility: Market volatility
        - market_pressure: Buy/sell pressure
        - confidence: Signal confidence
        - previous_winrate: Trailing win rate
        - account_health: Account health score
        
        Args:
            regime: Current market regime
            confidence: Signal confidence
            market_price: Current market price
            account_balance: Current account balance
            trace_id: Trace ID for logging
            
        Returns:
            Meta strategy state dictionary
        """
        # Store price for volatility calculation
        self.price_history.append(market_price)
        
        # Calculate state components
        volatility = self._calculate_volatility()
        market_pressure = self._calculate_market_pressure()
        trailing_winrate = self._calculate_trailing_winrate()
        account_health = self._calculate_account_health(account_balance)
        
        state = {
            "regime": regime,
            "volatility": volatility,
            "market_pressure": market_pressure,
            "confidence": confidence,
            "previous_winrate": trailing_winrate,
            "account_health": account_health
        }
        
        logger.debug(
            "[State Manager v2] Meta strategy state built",
            trace_id=trace_id,
            state=state
        )
        
        return state
    
    def build_position_sizing_state(
        self,
        signal_confidence: float,
        portfolio_exposure: float,
        market_volatility: float,
        account_balance: float,
        trace_id: str
    ) -> Dict[str, Any]:
        """
        Build Position Sizing state representation v2.
        
        State includes:
        - signal_confidence: Signal confidence
        - portfolio_exposure: Current portfolio exposure
        - recent_winrate: Trailing win rate
        - volatility: Market volatility
        - equity_curve_slope: Equity curve slope
        
        Args:
            signal_confidence: Signal confidence
            portfolio_exposure: Portfolio exposure (0-1)
            market_volatility: Market volatility
            account_balance: Current account balance
            trace_id: Trace ID for logging
            
        Returns:
            Position sizing state dictionary
        """
        # Store equity for slope calculation
        self.equity_history.append(account_balance)
        
        # Calculate state components
        trailing_winrate = self._calculate_trailing_winrate()
        equity_slope = self._calculate_equity_curve_slope()
        
        state = {
            "signal_confidence": signal_confidence,
            "portfolio_exposure": portfolio_exposure,
            "recent_winrate": trailing_winrate,
            "volatility": market_volatility,
            "equity_curve_slope": equity_slope
        }
        
        logger.debug(
            "[State Manager v2] Position sizing state built",
            trace_id=trace_id,
            state=state
        )
        
        return state
    
    def record_trade_outcome(self, is_win: bool):
        """
        Record trade outcome for win rate calculation.
        
        Args:
            is_win: True if trade was profitable
        """
        self.trade_outcomes.append(1 if is_win else 0)
    
    def _calculate_trailing_winrate(self) -> float:
        """
        Calculate trailing win rate from recent trades.
        
        Returns:
            Win rate (0-1)
        """
        if len(self.trade_outcomes) < 3:
            return 0.5  # Default to 50%
        
        winrate = np.mean(list(self.trade_outcomes))
        return float(winrate)
    
    def _calculate_volatility(self) -> float:
        """
        Calculate market volatility from price history.
        
        Uses standard deviation of returns.
        
        Returns:
            Volatility value
        """
        if len(self.price_history) < 3:
            return 0.02  # Default 2%
        
        prices = np.array(list(self.price_history))
        
        # Calculate returns
        returns = np.diff(prices) / prices[:-1]
        
        # Calculate volatility (std dev)
        volatility = np.std(returns)
        
        return float(volatility)
    
    def _calculate_equity_curve_slope(self) -> float:
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
    
    def _calculate_market_pressure(self) -> float:
        """
        Calculate market pressure from price momentum.
        
        Positive = buying pressure
        Negative = selling pressure
        
        Returns:
            Market pressure (-1 to 1)
        """
        if len(self.price_history) < 5:
            return 0.0
        
        prices = list(self.price_history)
        
        # Recent price change
        recent_change = (prices[-1] - prices[-5]) / prices[-5]
        
        # Normalize to [-1, 1]
        pressure = np.tanh(recent_change * 20)
        
        return float(pressure)
    
    def _calculate_account_health(self, account_balance: float) -> float:
        """
        Calculate account health score.
        
        Based on equity curve and recent performance.
        
        Args:
            account_balance: Current account balance
            
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
    
    def label_regime(
        self,
        price_history: List[float],
        volume_history: Optional[List[float]] = None
    ) -> str:
        """
        Label market regime from price/volume data.
        
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
            return "TREND"
        elif volatility < 0.01:
            return "RANGE"
        elif trend_strength > 0.03:
            return "BREAKOUT"
        else:
            return "MEAN_REVERSION"
    
    def reset(self):
        """Reset all historical buffers."""
        self.trade_outcomes.clear()
        self.price_history.clear()
        self.equity_history.clear()
        logger.info("[State Manager v2] Buffers reset")


# Global singleton instance
_state_manager_instance: Optional[StateManagerV2] = None


def get_state_manager() -> StateManagerV2:
    """Get or create global StateManagerV2 instance."""
    global _state_manager_instance
    if _state_manager_instance is None:
        _state_manager_instance = StateManagerV2()
    return _state_manager_instance
