"""Risk Models - Statistical risk calculation models.

This module provides statistical models for calculating various risk metrics:
- Value at Risk (VaR)
- Expected Shortfall (ES / CVaR)
- Tail risk indicators
- Volatility-based exposure limits
- Correlation-adjusted risk
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass
from typing import Optional

import numpy as np
from scipy import stats

logger = logging.getLogger(__name__)


@dataclass
class VaRResult:
    """Value at Risk calculation result."""
    
    var_value: float              # VaR estimate (USD or percentage)
    expected_shortfall: float     # ES/CVaR estimate
    confidence_level: float       # Confidence level (e.g., 0.95)
    time_horizon: str             # Time horizon (e.g., "1d", "1h")
    method: str                   # Calculation method (historical, parametric, etc.)
    timestamp: str


@dataclass
class TailRiskMetrics:
    """Tail risk indicators."""
    
    tail_risk_score: float        # 0-100 scale
    max_loss_scenario: float      # Maximum loss in worst-case scenario (USD)
    skewness: float               # Distribution skewness
    kurtosis: float               # Distribution kurtosis (excess)
    extreme_event_prob: float     # Probability of extreme event (> 3 sigma)


class RiskModels:
    """
    Statistical risk models for portfolio risk assessment.
    
    Provides various risk calculation methods that can be used
    individually or combined for comprehensive risk analysis.
    """
    
    def __init__(self):
        """Initialize risk models."""
        logger.info("RiskModels initialized")
    
    def calculate_var(
        self,
        returns: np.ndarray,
        confidence_level: float = 0.95,
        method: str = "historical",
    ) -> VaRResult:
        """
        Calculate Value at Risk (VaR).
        
        Args:
            returns: Array of historical returns (percentage or absolute)
            confidence_level: Confidence level (default 0.95 = 95%)
            method: Calculation method ("historical", "parametric", "cornish_fisher")
        
        Returns:
            VaRResult with VaR and Expected Shortfall
        """
        if len(returns) == 0:
            return VaRResult(
                var_value=0.0,
                expected_shortfall=0.0,
                confidence_level=confidence_level,
                time_horizon="1d",
                method=method,
                timestamp="",
            )
        
        if method == "historical":
            return self._historical_var(returns, confidence_level)
        elif method == "parametric":
            return self._parametric_var(returns, confidence_level)
        elif method == "cornish_fisher":
            return self._cornish_fisher_var(returns, confidence_level)
        else:
            logger.warning(f"Unknown VaR method: {method}, using historical")
            return self._historical_var(returns, confidence_level)
    
    def _historical_var(self, returns: np.ndarray, confidence_level: float) -> VaRResult:
        """Calculate VaR using historical simulation."""
        sorted_returns = np.sort(returns)
        index = int((1 - confidence_level) * len(sorted_returns))
        var_value = abs(sorted_returns[index])
        
        # Expected Shortfall (average of losses beyond VaR)
        tail_losses = sorted_returns[:index + 1]
        expected_shortfall = abs(np.mean(tail_losses)) if len(tail_losses) > 0 else var_value
        
        return VaRResult(
            var_value=var_value,
            expected_shortfall=expected_shortfall,
            confidence_level=confidence_level,
            time_horizon="1d",
            method="historical",
            timestamp="",
        )
    
    def _parametric_var(self, returns: np.ndarray, confidence_level: float) -> VaRResult:
        """Calculate VaR using parametric (normal distribution) method."""
        mean = np.mean(returns)
        std = np.std(returns)
        
        # Z-score for confidence level
        z_score = stats.norm.ppf(1 - confidence_level)
        
        # VaR = mean + z_score * std (z_score is negative)
        var_value = abs(mean + z_score * std)
        
        # Expected Shortfall for normal distribution
        # ES = mean - std * phi(z) / (1 - confidence_level)
        phi_z = stats.norm.pdf(z_score)
        expected_shortfall = abs(mean - std * phi_z / (1 - confidence_level))
        
        return VaRResult(
            var_value=var_value,
            expected_shortfall=expected_shortfall,
            confidence_level=confidence_level,
            time_horizon="1d",
            method="parametric",
            timestamp="",
        )
    
    def _cornish_fisher_var(self, returns: np.ndarray, confidence_level: float) -> VaRResult:
        """
        Calculate VaR using Cornish-Fisher expansion.
        
        Adjusts for skewness and kurtosis in return distribution.
        """
        mean = np.mean(returns)
        std = np.std(returns)
        skew = stats.skew(returns)
        kurt = stats.kurtosis(returns)  # Excess kurtosis
        
        # Z-score for confidence level
        z = stats.norm.ppf(1 - confidence_level)
        
        # Cornish-Fisher adjustment
        z_cf = (
            z
            + (z**2 - 1) * skew / 6
            + (z**3 - 3 * z) * kurt / 24
            - (2 * z**3 - 5 * z) * skew**2 / 36
        )
        
        var_value = abs(mean + z_cf * std)
        
        # ES approximation (simplified)
        expected_shortfall = var_value * 1.2  # Conservative multiplier
        
        return VaRResult(
            var_value=var_value,
            expected_shortfall=expected_shortfall,
            confidence_level=confidence_level,
            time_horizon="1d",
            method="cornish_fisher",
            timestamp="",
        )
    
    def calculate_tail_risk(
        self,
        returns: np.ndarray,
        current_exposure: float,
    ) -> TailRiskMetrics:
        """
        Calculate tail risk metrics.
        
        Args:
            returns: Array of historical returns
            current_exposure: Current portfolio exposure (USD)
        
        Returns:
            TailRiskMetrics with various tail risk indicators
        """
        if len(returns) < 30:
            # Insufficient data - return conservative estimates
            return TailRiskMetrics(
                tail_risk_score=50.0,
                max_loss_scenario=current_exposure * 0.10,
                skewness=0.0,
                kurtosis=0.0,
                extreme_event_prob=0.01,
            )
        
        # Calculate distribution statistics
        skewness = stats.skew(returns)
        kurtosis = stats.kurtosis(returns)  # Excess kurtosis
        
        # Extreme event probability (> 3 sigma)
        std = np.std(returns)
        mean = np.mean(returns)
        extreme_threshold = mean - 3 * std
        extreme_events = np.sum(returns < extreme_threshold)
        extreme_event_prob = extreme_events / len(returns)
        
        # Max loss scenario (worst historical return applied to current exposure)
        worst_return = np.min(returns)
        max_loss_scenario = abs(worst_return * current_exposure)
        
        # Tail risk score (0-100)
        # Higher score = higher tail risk
        tail_risk_score = 50.0  # Base
        
        # Adjust for negative skewness (fat left tail)
        if skewness < 0:
            tail_risk_score += abs(skewness) * 10
        
        # Adjust for high kurtosis (fat tails)
        if kurtosis > 0:
            tail_risk_score += min(kurtosis * 5, 20)
        
        # Adjust for extreme event probability
        tail_risk_score += extreme_event_prob * 100
        
        # Cap at 100
        tail_risk_score = min(tail_risk_score, 100.0)
        
        return TailRiskMetrics(
            tail_risk_score=tail_risk_score,
            max_loss_scenario=max_loss_scenario,
            skewness=skewness,
            kurtosis=kurtosis,
            extreme_event_prob=extreme_event_prob,
        )
    
    def calculate_volatility_adjusted_leverage(
        self,
        current_volatility: float,
        base_leverage: float,
        vol_target: float = 0.02,
    ) -> float:
        """
        Calculate volatility-adjusted leverage.
        
        Higher volatility → lower leverage
        Lower volatility → higher leverage (up to base_leverage)
        
        Args:
            current_volatility: Current market volatility (std of returns)
            base_leverage: Base leverage setting
            vol_target: Target volatility (default 2% = 0.02)
        
        Returns:
            Adjusted leverage
        """
        if current_volatility <= 0:
            return base_leverage
        
        # Scale leverage inversely with volatility
        adjusted_leverage = base_leverage * (vol_target / current_volatility)
        
        # Cap at base leverage (don't increase beyond base)
        adjusted_leverage = min(adjusted_leverage, base_leverage)
        
        # Floor at 1.0
        adjusted_leverage = max(adjusted_leverage, 1.0)
        
        return adjusted_leverage
    
    def calculate_position_size_limit(
        self,
        volatility: float,
        max_risk_per_trade: float,
        account_size: float,
    ) -> float:
        """
        Calculate maximum position size based on volatility.
        
        Args:
            volatility: Asset volatility (std of returns)
            max_risk_per_trade: Max risk per trade (e.g., 0.02 = 2%)
            account_size: Total account size (USD)
        
        Returns:
            Maximum position size (USD)
        """
        if volatility <= 0:
            return account_size * max_risk_per_trade
        
        # Position size = (account_size * max_risk) / volatility
        # This ensures that a 1-sigma move = max_risk loss
        position_size = (account_size * max_risk_per_trade) / volatility
        
        # Cap at reasonable fraction of account
        max_position = account_size * 0.25  # Max 25% of account per position
        position_size = min(position_size, max_position)
        
        return position_size
    
    def estimate_correlation_risk(
        self,
        positions: list[dict],
        correlation_matrix: Optional[np.ndarray] = None,
    ) -> float:
        """
        Estimate portfolio risk from position correlations.
        
        Args:
            positions: List of position dicts with 'exposure' and 'volatility'
            correlation_matrix: Optional correlation matrix (if available)
        
        Returns:
            Correlation risk score (0-100)
        """
        if len(positions) < 2:
            return 0.0
        
        # If no correlation matrix, assume moderate positive correlation
        if correlation_matrix is None:
            avg_correlation = 0.5  # Assume 50% correlation
            n_positions = len(positions)
            
            # Higher correlation risk with more positions
            correlation_risk = avg_correlation * n_positions * 5
            return min(correlation_risk, 100.0)
        
        # With correlation matrix, calculate portfolio variance
        exposures = np.array([p.get('exposure', 0) for p in positions])
        volatilities = np.array([p.get('volatility', 0.02) for p in positions])
        
        # Position variances
        variances = (exposures * volatilities) ** 2
        
        # Portfolio variance = sum(variances) + sum(correlations * vol_i * vol_j)
        portfolio_var = np.sum(variances)
        
        for i in range(len(positions)):
            for j in range(i + 1, len(positions)):
                if i < correlation_matrix.shape[0] and j < correlation_matrix.shape[1]:
                    corr = correlation_matrix[i, j]
                    portfolio_var += 2 * corr * exposures[i] * volatilities[i] * exposures[j] * volatilities[j]
        
        portfolio_vol = math.sqrt(portfolio_var)
        
        # Convert to risk score (higher vol = higher risk)
        # Normalize to 0-100 scale
        risk_score = min(portfolio_vol * 1000, 100.0)
        
        return risk_score
