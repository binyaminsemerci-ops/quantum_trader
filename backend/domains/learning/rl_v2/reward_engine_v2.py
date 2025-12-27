"""
Reward Engine v2 - Advanced Reward Calculation System
======================================================

Implements sophisticated reward functions for:
- Meta Strategy RL (regime-aware, sharpe-aware)
- Position Sizing RL (risk-aware, volatility-aware)

Author: Quantum Trader AI Team
Date: December 2, 2025
Version: 2.0
"""

from typing import Dict, Any, Optional
import numpy as np
from collections import deque
import structlog

logger = structlog.get_logger(__name__)


class RewardEngineV2:
    """
    Advanced reward calculation engine for RL v2.
    
    Implements:
    - Meta strategy reward: pnl - 0.5×dd + 0.2×sharpe + 0.15×regime_align
    - Position sizing reward: pnl - 0.4×risk_penalty + 0.1×vol_adjust
    """
    
    def __init__(
        self,
        lookback_window: int = 20,
        risk_free_rate: float = 0.02
    ):
        """
        Initialize Reward Engine v2.
        
        Args:
            lookback_window: Number of periods for trailing calculations
            risk_free_rate: Annual risk-free rate for Sharpe ratio
        """
        self.lookback_window = lookback_window
        self.risk_free_rate = risk_free_rate
        
        # Historical data buffers
        self.pnl_history: deque = deque(maxlen=lookback_window)
        
        logger.info(
            "[Reward Engine v2] Initialized",
            lookback_window=lookback_window,
            risk_free_rate=risk_free_rate
        )
    
    def calculate_meta_strategy_reward(
        self,
        data: Dict[str, Any]
    ) -> float:
        """
        Calculate Meta Strategy RL reward with regime awareness.
        
        Formula:
        meta_reward = (
            pnl_pct
            - 0.5 * max_drawdown_pct
            + 0.2 * sharpe_signal
            + 0.15 * regime_alignment_score
        )
        
        Args:
            data: Dictionary containing:
                - pnl_percentage: Position P&L percentage (or pnl_pct)
                - drawdown: Maximum drawdown percentage (or max_drawdown_pct)
                - regime: Actual market regime (or current_regime)
                - predicted_regime: Predicted market regime
                - confidence: Signal confidence
                - trace_id: Trace ID for logging (optional)
            
        Returns:
            Meta strategy reward
        """
        # Extract parameters with aliases
        pnl_pct = data.get("pnl_percentage", data.get("pnl_pct", 0.0))
        max_drawdown_pct = data.get("drawdown", data.get("max_drawdown_pct", 0.0))
        current_regime = data.get("regime", data.get("current_regime", "UNKNOWN"))
        predicted_regime = data.get("predicted_regime", "UNKNOWN")
        confidence = data.get("confidence", 0.5)
        trace_id = data.get("trace_id", "")
        
        # Store P&L for Sharpe calculation
        self.pnl_history.append(pnl_pct)
        
        # Calculate components
        sharpe_signal = self._calculate_sharpe_signal()
        regime_alignment = self._calculate_regime_alignment_score(
            current_regime,
            predicted_regime,
            confidence
        )
        
        # Compute reward
        meta_reward = (
            pnl_pct
            - 0.5 * max_drawdown_pct
            + 0.2 * sharpe_signal
            + 0.15 * regime_alignment
        )
        
        logger.debug(
            "[Reward Engine v2] Meta strategy reward calculated",
            trace_id=trace_id,
            pnl_pct=pnl_pct,
            max_drawdown_pct=max_drawdown_pct,
            sharpe_signal=sharpe_signal,
            regime_alignment=regime_alignment,
            meta_reward=meta_reward
        )
        
        return meta_reward
    
    def calculate_position_sizing_reward(
        self,
        data: Dict[str, Any]
    ) -> float:
        """
        Calculate Position Sizing RL reward with risk awareness.
        
        Formula:
        size_reward = (
            pnl_pct
            - 0.4 * risk_penalty
            + 0.1 * volatility_adjustment
        )
        
        Args:
            data: Dictionary containing:
                - pnl_percentage: Position P&L percentage (or pnl_pct)
                - leverage: Applied leverage
                - position_size_usd: Position size in USD
                - account_balance: Current account balance
                - volatility: Market volatility level (or market_volatility)
                - trace_id: Trace ID for logging (optional)
            
        Returns:
            Position sizing reward value
        """
        # Extract parameters with aliases
        pnl_pct = data.get("pnl_percentage", data.get("pnl_pct", 0.0))
        leverage = data.get("leverage", 20.0)
        position_size_usd = data.get("position_size_usd", 1000.0)
        account_balance = data.get("account_balance", 10000.0)
        market_volatility = data.get("volatility", data.get("market_volatility", 0.02))
        trace_id = data.get("trace_id", "")
        
        # Calculate components
        risk_penalty = self._calculate_risk_penalty(
            leverage,
            position_size_usd,
            account_balance
        )
        
        volatility_adjustment = self._calculate_volatility_adjustment(
            market_volatility
        )
        
        # Compute reward
        size_reward = (
            pnl_pct
            - 0.4 * risk_penalty
            + 0.1 * volatility_adjustment
        )
        
        logger.debug(
            "[Reward Engine v2] Position sizing reward calculated",
            trace_id=trace_id,
            pnl_pct=pnl_pct,
            risk_penalty=risk_penalty,
            volatility_adjustment=volatility_adjustment,
            size_reward=size_reward
        )
        
        return size_reward
    
    def _calculate_sharpe_signal(self) -> float:
        """
        Calculate Sharpe ratio signal from recent P&L history.
        
        Returns:
            Sharpe signal value (normalized to [-1, 1])
        """
        if len(self.pnl_history) < 3:
            return 0.0
        
        pnl_array = np.array(list(self.pnl_history))
        
        # Calculate returns statistics
        mean_return = np.mean(pnl_array)
        std_return = np.std(pnl_array)
        
        if std_return < 1e-6:
            return 0.0
        
        # Annualized Sharpe ratio (approximate)
        sharpe = (mean_return - self.risk_free_rate / 252) / std_return
        
        # Normalize to [-1, 1] range
        sharpe_signal = np.tanh(sharpe)
        
        return float(sharpe_signal)
    
    def _calculate_regime_alignment_score(
        self,
        current_regime: str,
        predicted_regime: str,
        confidence: float
    ) -> float:
        """
        Calculate regime alignment score.
        
        Rewards correct regime prediction weighted by confidence.
        
        Args:
            current_regime: Actual market regime
            predicted_regime: Predicted market regime
            confidence: Prediction confidence
            
        Returns:
            Regime alignment score
        """
        if not current_regime or not predicted_regime:
            return 0.0
        
        # Check if prediction matches reality
        is_aligned = (current_regime.upper() == predicted_regime.upper())
        
        # Base alignment score
        alignment_score = 1.0 if is_aligned else -0.5
        
        # Weight by confidence
        weighted_score = alignment_score * confidence
        
        return weighted_score
    
    def _calculate_risk_penalty(
        self,
        leverage: float,
        position_size_usd: float,
        account_balance: float
    ) -> float:
        """
        Calculate risk penalty based on leverage and position sizing.
        
        Args:
            leverage: Applied leverage
            position_size_usd: Position size in USD
            account_balance: Current account balance
            
        Returns:
            Risk penalty value
        """
        # Calculate position exposure relative to account
        exposure_ratio = position_size_usd / max(account_balance, 1.0)
        
        # Leverage penalty (exponential)
        leverage_penalty = 0.0
        if leverage > 5:
            leverage_penalty = (leverage - 5) * 0.3
        
        # Over-exposure penalty
        exposure_penalty = 0.0
        if exposure_ratio > 0.5:
            exposure_penalty = (exposure_ratio - 0.5) * 2.0
        
        # Combined risk penalty
        risk_penalty = leverage_penalty + exposure_penalty
        
        return min(risk_penalty, 5.0)  # Cap at 5.0
    
    def _calculate_volatility_adjustment(
        self,
        current_volatility: float
    ) -> float:
        """
        Calculate volatility adjustment reward.
        
        Rewards trading in favorable volatility conditions.
        
        Args:
            current_volatility: Current market volatility
            
        Returns:
            Volatility adjustment value
        """
        # Optimal volatility range: 0.01 - 0.03 (1% - 3%)
        optimal_min = 0.01
        optimal_max = 0.03
        
        if optimal_min <= current_volatility <= optimal_max:
            # In optimal range - positive adjustment
            return 0.5
        elif current_volatility < optimal_min:
            # Too low volatility - small negative adjustment
            return -0.2
        else:
            # Too high volatility - larger negative adjustment
            excess = min((current_volatility - optimal_max) / optimal_max, 2.0)
            return -0.5 * excess
    
    def reset(self):
        """Reset all historical buffers."""
        self.pnl_history.clear()
        logger.info("[Reward Engine v2] Buffers reset")
