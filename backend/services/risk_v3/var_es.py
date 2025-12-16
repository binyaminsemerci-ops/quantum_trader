"""
VaR and Expected Shortfall Engine

EPIC-RISK3-001: Value at Risk and Expected Shortfall calculation

Implements:
- Delta-normal VaR (parametric method)
- Historical VaR (empirical method)
- Expected Shortfall / CVaR (tail risk measure)

TODO (RISK3-002):
- Add Monte Carlo VaR simulation
- Implement GARCH volatility forecasting
- Add multi-period VaR scaling
- Enhance with extreme value theory (EVT)
- Add stress testing scenarios
"""

import logging
from typing import Dict, List, Optional, Tuple
from datetime import datetime
import numpy as np

from .models import (
    RiskSnapshot,
    VaRResult,
    ESResult,
)

logger = logging.getLogger(__name__)


class VaRESEngine:
    """Value at Risk and Expected Shortfall calculation engine"""
    
    def __init__(
        self,
        lookback_periods: int = 30,
        time_horizon_hours: int = 24,
        confidence_levels: Tuple[float, float, float] = (0.95, 0.975, 0.99),
    ):
        """
        Initialize VaR/ES engine
        
        Args:
            lookback_periods: Historical periods for calculation
            time_horizon_hours: Time horizon for VaR (typically 24 hours)
            confidence_levels: Confidence levels (95%, 97.5%, 99%)
        """
        self.lookback_periods = lookback_periods
        self.time_horizon_hours = time_horizon_hours
        self.confidence_levels = confidence_levels
        
        logger.info(
            f"[RISK-V3] VaRESEngine initialized "
            f"(lookback={lookback_periods}, horizon={time_horizon_hours}h)"
        )
    
    def compute_var(
        self,
        returns: List[float],
        portfolio_value: float,
        method: str = "delta_normal",
        level: float = 0.95,
    ) -> float:
        """
        Compute Value at Risk
        
        VaR answers: "What is the maximum loss we expect with X% confidence?"
        
        Args:
            returns: Historical portfolio returns (as decimals, e.g., 0.02 = 2%)
            portfolio_value: Current portfolio value in USD
            method: Calculation method ('delta_normal' or 'historical')
            level: Confidence level (0.95 = 95%)
        
        Returns:
            VaR in USD (positive number represents potential loss)
        """
        if not returns or len(returns) < 5:
            logger.warning("[RISK-V3] Insufficient return data for VaR calculation")
            return 0.0
        
        if method == "delta_normal":
            return self._compute_var_delta_normal(returns, portfolio_value, level)
        elif method == "historical":
            return self._compute_var_historical(returns, portfolio_value, level)
        else:
            logger.error(f"[RISK-V3] Unknown VaR method: {method}")
            return 0.0
    
    def _compute_var_delta_normal(
        self,
        returns: List[float],
        portfolio_value: float,
        level: float,
    ) -> float:
        """
        Delta-normal (parametric) VaR
        
        Assumes returns are normally distributed
        VaR = Z_alpha * sigma * sqrt(T) * portfolio_value
        
        Where:
        - Z_alpha is the Z-score for confidence level
        - sigma is the standard deviation of returns
        - T is the time horizon (in years)
        """
        # Calculate return statistics
        mean_return = np.mean(returns)
        std_return = np.std(returns, ddof=1)  # Sample std dev
        
        # Z-scores for common confidence levels
        z_scores = {
            0.90: 1.282,
            0.95: 1.645,
            0.975: 1.960,
            0.99: 2.326,
        }
        
        z_alpha = z_scores.get(level, 1.645)  # Default to 95%
        
        # Time horizon scaling (assuming daily returns)
        # If time_horizon_hours = 24, then T = 1/365 years
        time_horizon_years = self.time_horizon_hours / (24 * 365)
        time_scaling = np.sqrt(time_horizon_years)
        
        # Calculate VaR
        # Note: We use (mean - z * std) because we want the loss side
        var_return = -(mean_return - z_alpha * std_return * time_scaling)
        var_usd = var_return * portfolio_value
        
        return max(var_usd, 0.0)  # VaR cannot be negative
    
    def _compute_var_historical(
        self,
        returns: List[float],
        portfolio_value: float,
        level: float,
    ) -> float:
        """
        Historical (empirical) VaR
        
        Uses actual historical returns distribution without assuming normality
        VaR = (1-level) percentile of historical losses
        """
        # Convert returns to losses (negative returns)
        losses = [-r for r in returns]
        
        # Calculate percentile
        percentile = (1 - level) * 100
        var_return = np.percentile(losses, percentile)
        
        # Convert to USD
        var_usd = var_return * portfolio_value
        
        return max(var_usd, 0.0)
    
    def compute_es(
        self,
        returns: List[float],
        portfolio_value: float,
        method: str = "historical",
        level: float = 0.975,
    ) -> float:
        """
        Compute Expected Shortfall (CVaR, Conditional VaR)
        
        ES answers: "If we breach VaR, what is the expected loss in the tail?"
        ES is more conservative than VaR as it captures tail risk
        
        Args:
            returns: Historical portfolio returns (as decimals)
            portfolio_value: Current portfolio value in USD
            method: Calculation method ('historical' or 'parametric')
            level: Confidence level (0.975 = 97.5%)
        
        Returns:
            ES in USD (positive number represents potential loss)
        """
        if not returns or len(returns) < 10:
            logger.warning("[RISK-V3] Insufficient return data for ES calculation")
            return 0.0
        
        if method == "historical":
            return self._compute_es_historical(returns, portfolio_value, level)
        else:
            logger.warning(f"[RISK-V3] Method {method} not implemented for ES, using historical")
            return self._compute_es_historical(returns, portfolio_value, level)
    
    def _compute_es_historical(
        self,
        returns: List[float],
        portfolio_value: float,
        level: float,
    ) -> float:
        """
        Historical Expected Shortfall
        
        ES = Average of losses beyond VaR threshold
        """
        # Convert returns to losses
        losses = np.array([-r for r in returns])
        
        # Find VaR threshold
        percentile = (1 - level) * 100
        var_threshold = np.percentile(losses, percentile)
        
        # Get tail losses (losses beyond VaR)
        tail_losses = losses[losses >= var_threshold]
        
        if len(tail_losses) == 0:
            # No tail events, use VaR as ES
            return var_threshold * portfolio_value
        
        # Average tail loss
        es_return = np.mean(tail_losses)
        es_usd = es_return * portfolio_value
        
        return max(es_usd, 0.0)
    
    def compute_var_result(
        self,
        snapshot: RiskSnapshot,
        returns_data: Optional[Dict[str, List[float]]] = None,
        method: str = "delta_normal",
        threshold_95: float = 1000.0,
        threshold_99: float = 2000.0,
    ) -> VaRResult:
        """
        Compute complete VaR result with multiple confidence levels
        
        Args:
            snapshot: Current risk snapshot
            returns_data: Historical returns per symbol (for portfolio VaR)
            method: Calculation method
            threshold_95: Max acceptable VaR at 95%
            threshold_99: Max acceptable VaR at 99%
        
        Returns:
            VaRResult with all metrics
        """
        # If no returns data, generate placeholder using portfolio volatility
        if not returns_data or len(returns_data) == 0:
            portfolio_returns = self._generate_placeholder_returns(snapshot)
        else:
            portfolio_returns = self._compute_portfolio_returns(snapshot, returns_data)
        
        # Compute VaR at different confidence levels
        var_95 = self.compute_var(
            portfolio_returns,
            snapshot.total_equity,
            method=method,
            level=0.95,
        )
        
        var_99 = self.compute_var(
            portfolio_returns,
            snapshot.total_equity,
            method=method,
            level=0.99,
        )
        
        # Calculate portfolio volatility (annualized)
        if len(portfolio_returns) > 1:
            daily_vol = np.std(portfolio_returns, ddof=1)
            annual_vol = daily_vol * np.sqrt(252)  # 252 trading days
        else:
            annual_vol = 0.0
        
        result = VaRResult(
            var_95=round(var_95, 2),
            var_99=round(var_99, 2),
            method=method,
            lookback_periods=self.lookback_periods,
            time_horizon_hours=self.time_horizon_hours,
            threshold_95=threshold_95,
            threshold_99=threshold_99,
            pass_95=var_95 <= threshold_95,
            pass_99=var_99 <= threshold_99,
            portfolio_volatility=round(annual_vol, 4),
            note=f"Based on {len(portfolio_returns)} return observations",
        )
        
        logger.info(
            f"[RISK-V3] VaR calculated ({method}):\n"
            f"  95% VaR: ${var_95:,.2f} (threshold ${threshold_95:,.2f}) - {'PASS' if result.pass_95 else 'FAIL'}\n"
            f"  99% VaR: ${var_99:,.2f} (threshold ${threshold_99:,.2f}) - {'PASS' if result.pass_99 else 'FAIL'}\n"
            f"  Portfolio Vol: {annual_vol:.2%}"
        )
        
        return result
    
    def compute_es_result(
        self,
        snapshot: RiskSnapshot,
        returns_data: Optional[Dict[str, List[float]]] = None,
        method: str = "historical",
        threshold_975: float = 2500.0,
    ) -> ESResult:
        """
        Compute Expected Shortfall result
        
        Args:
            snapshot: Current risk snapshot
            returns_data: Historical returns per symbol
            method: Calculation method
            threshold_975: Max acceptable ES at 97.5%
        
        Returns:
            ESResult with tail risk metrics
        """
        # Get portfolio returns
        if not returns_data or len(returns_data) == 0:
            portfolio_returns = self._generate_placeholder_returns(snapshot)
        else:
            portfolio_returns = self._compute_portfolio_returns(snapshot, returns_data)
        
        # Compute ES at 97.5% confidence
        es_975 = self.compute_es(
            portfolio_returns,
            snapshot.total_equity,
            method=method,
            level=0.975,
        )
        
        # Find worst case loss in historical data
        losses = [-r for r in portfolio_returns]
        worst_case_loss = max(losses) * snapshot.total_equity if losses else 0.0
        
        # Count tail events (beyond 97.5% threshold)
        percentile_975 = np.percentile(losses, 97.5)
        tail_events = sum(1 for loss in losses if loss >= percentile_975)
        
        result = ESResult(
            es_975=round(es_975, 2),
            method=method,
            lookback_periods=self.lookback_periods,
            threshold_975=threshold_975,
            pass_975=es_975 <= threshold_975,
            worst_case_loss=round(worst_case_loss, 2),
            tail_events_count=tail_events,
            note=f"Based on {len(portfolio_returns)} return observations",
        )
        
        logger.info(
            f"[RISK-V3] ES calculated ({method}):\n"
            f"  97.5% ES: ${es_975:,.2f} (threshold ${threshold_975:,.2f}) - {'PASS' if result.pass_975 else 'FAIL'}\n"
            f"  Worst Case: ${worst_case_loss:,.2f}\n"
            f"  Tail Events: {tail_events}"
        )
        
        return result
    
    def _generate_placeholder_returns(
        self,
        snapshot: RiskSnapshot,
    ) -> List[float]:
        """
        Generate placeholder returns when no historical data available
        
        Uses portfolio leverage and drawdown as proxy for volatility
        
        TODO (RISK3-002): Remove this and require real returns data
        """
        logger.warning("[RISK-V3] Using placeholder returns (no historical data)")
        
        # Estimate daily volatility based on leverage
        # Higher leverage → higher volatility
        base_vol = 0.02  # 2% base daily volatility
        leverage_factor = 1.0 + (snapshot.total_leverage - 1.0) * 0.5
        estimated_vol = base_vol * leverage_factor
        
        # Generate synthetic returns with normal distribution
        np.random.seed(42)  # For reproducibility
        returns = np.random.normal(0.001, estimated_vol, self.lookback_periods)  # Slight positive mean
        
        return returns.tolist()
    
    def _compute_portfolio_returns(
        self,
        snapshot: RiskSnapshot,
        returns_data: Dict[str, List[float]],
    ) -> List[float]:
        """
        Compute portfolio-level returns from symbol returns and weights
        
        Portfolio return = weighted sum of symbol returns
        
        TODO (RISK3-002): Enhance with:
        - Proper time alignment
        - Missing data handling
        - Dynamic weight adjustment
        """
        if not returns_data:
            return self._generate_placeholder_returns(snapshot)
        
        # Get weights from snapshot
        weights = {}
        total_notional = snapshot.total_notional
        
        if total_notional > 0:
            for symbol, notional in snapshot.symbol_exposure.items():
                weights[symbol] = notional / total_notional
        else:
            # Equal weights if no exposure
            n_symbols = len(returns_data)
            weights = {symbol: 1.0 / n_symbols for symbol in returns_data.keys()}
        
        # Compute weighted portfolio returns
        # Align periods (use minimum length)
        min_periods = min(len(returns) for returns in returns_data.values())
        min_periods = min(min_periods, self.lookback_periods)
        
        portfolio_returns = []
        for t in range(min_periods):
            period_return = 0.0
            for symbol, symbol_returns in returns_data.items():
                weight = weights.get(symbol, 0.0)
                period_return += weight * symbol_returns[-(min_periods - t)]
            portfolio_returns.append(period_return)
        
        return portfolio_returns


__all__ = [
    "VaRESEngine",
]
