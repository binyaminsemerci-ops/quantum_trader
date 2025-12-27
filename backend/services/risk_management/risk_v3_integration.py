"""
Risk v3 Integration Module - Real-time ESS and Correlation Queries
=================================================================

Provides real-time risk metrics for Dynamic TP/SL adjustments.
"""

import logging
from typing import Dict, Optional, List, Any
from dataclasses import dataclass
from datetime import datetime, timezone

logger = logging.getLogger(__name__)


@dataclass
class RiskV3Context:
    """Risk v3 context for TP/SL adjustments."""
    ess_factor: float
    systemic_risk_level: float
    correlation_risk: float
    portfolio_heat: float
    var_95: float
    timestamp: datetime


class RiskV3Integrator:
    """
    Integrates with Risk v3 system for real-time risk metrics.
    
    Provides:
    - Effective Stress Score (ESS)
    - Systemic risk levels
    - Portfolio correlation risk
    - Value-at-Risk (VaR)
    """
    
    def __init__(self, ai_services=None):
        """
        Initialize Risk v3 Integrator.
        
        Args:
            ai_services: AI System Services for accessing risk components
        """
        self.logger = logging.getLogger(__name__)
        self.ai_services = ai_services
        
        # Risk thresholds
        self.ess_warning_threshold = 1.5
        self.ess_critical_threshold = 2.5
        self.systemic_risk_warning = 0.6
        self.systemic_risk_critical = 0.8
        
        # Cache for performance
        self._cache: Dict[str, Any] = {}
        self._cache_ttl_seconds = 10
        
        self.logger.info("[Risk v3 Integrator] Initialized")
    
    def get_risk_context(
        self,
        symbol: Optional[str] = None,
        position_size: float = 0.0
    ) -> RiskV3Context:
        """
        Get current risk context for TP/SL calculation.
        
        Args:
            symbol: Trading symbol (for symbol-specific risk)
            position_size: Proposed position size (for impact analysis)
            
        Returns:
            RiskV3Context with current risk metrics
        """
        # Check cache
        cache_key = f"risk_context_{symbol}"
        cached = self._get_cached(cache_key)
        if cached:
            return cached
        
        # Calculate ESS (Effective Stress Score)
        ess_factor = self._calculate_ess()
        
        # Calculate systemic risk level
        systemic_risk = self._calculate_systemic_risk()
        
        # Calculate portfolio correlation risk
        correlation_risk = self._calculate_correlation_risk()
        
        # Calculate portfolio heat (concentration)
        portfolio_heat = self._calculate_portfolio_heat()
        
        # Calculate VaR 95%
        var_95 = self._calculate_var95()
        
        context = RiskV3Context(
            ess_factor=ess_factor,
            systemic_risk_level=systemic_risk,
            correlation_risk=correlation_risk,
            portfolio_heat=portfolio_heat,
            var_95=var_95,
            timestamp=datetime.now(timezone.utc)
        )
        
        # Cache result
        self._cache[cache_key] = (context, datetime.now(timezone.utc).timestamp())
        
        return context
    
    def _calculate_ess(self) -> float:
        """
        Calculate Effective Stress Score.
        
        ESS measures portfolio stress from:
        - Open position exposure
        - Unrealized PnL drawdown
        - Margin utilization
        - Volatility regime
        
        Returns:
            ESS factor (1.0 = normal, >2.0 = high stress)
        """
        try:
            if self.ai_services and hasattr(self.ai_services, 'risk_v3'):
                # Query real Risk v3 system
                risk_v3 = self.ai_services.risk_v3
                if hasattr(risk_v3, 'get_ess'):
                    return risk_v3.get_ess()
            
            # Fallback: estimate from available data
            # This is a simplified calculation
            base_ess = 1.0
            
            # Placeholder: In production, query actual portfolio metrics
            # For now, return baseline
            return base_ess
            
        except Exception as e:
            self.logger.warning(f"[Risk v3] ESS calculation failed: {e}")
            return 1.0  # Safe default
    
    def _calculate_systemic_risk(self) -> float:
        """
        Calculate systemic risk level.
        
        Measures macro risks:
        - Market regime shifts
        - Correlation breakdowns
        - Liquidity stress
        - Volatility spikes
        
        Returns:
            Systemic risk score (0.0 = calm, 1.0 = crisis)
        """
        try:
            if self.ai_services and hasattr(self.ai_services, 'regime_detector'):
                regime_detector = self.ai_services.regime_detector
                
                # Check for crisis indicators
                # This is simplified - real implementation would be more sophisticated
                return 0.0  # Placeholder
            
            return 0.0  # Safe default
            
        except Exception as e:
            self.logger.warning(f"[Risk v3] Systemic risk calculation failed: {e}")
            return 0.0
    
    def _calculate_correlation_risk(self) -> float:
        """
        Calculate portfolio correlation risk.
        
        High correlation = concentrated risk
        
        Returns:
            Correlation risk score (0.0 = diversified, 1.0 = concentrated)
        """
        try:
            # Placeholder: In production, calculate actual correlations
            # between open positions
            return 0.3  # Moderate diversification
            
        except Exception as e:
            self.logger.warning(f"[Risk v3] Correlation calculation failed: {e}")
            return 0.5  # Conservative default
    
    def _calculate_portfolio_heat(self) -> float:
        """
        Calculate portfolio concentration (heat).
        
        Returns:
            Heat score (0.0 = well-distributed, 1.0 = over-concentrated)
        """
        try:
            # Placeholder: In production, measure position concentration
            return 0.4
            
        except Exception as e:
            self.logger.warning(f"[Risk v3] Portfolio heat calculation failed: {e}")
            return 0.5
    
    def _calculate_var95(self) -> float:
        """
        Calculate 95% Value-at-Risk.
        
        Returns:
            VaR as portfolio percentage
        """
        try:
            # Placeholder: In production, calculate actual VaR
            # based on historical volatility and correlations
            return 0.05  # 5% VaR
            
        except Exception as e:
            self.logger.warning(f"[Risk v3] VaR calculation failed: {e}")
            return 0.10  # Conservative default
    
    def _get_cached(self, key: str) -> Optional[RiskV3Context]:
        """Get cached value if still valid."""
        if key in self._cache:
            context, timestamp = self._cache[key]
            age = datetime.now(timezone.utc).timestamp() - timestamp
            if age < self._cache_ttl_seconds:
                return context
        return None
    
    def should_tighten_tp(self, context: RiskV3Context) -> bool:
        """
        Determine if TP should be tightened based on risk context.
        
        Args:
            context: Current risk context
            
        Returns:
            True if TP should be tightened
        """
        # Tighten if ESS is elevated
        if context.ess_factor > self.ess_warning_threshold:
            return True
        
        # Tighten if systemic risk is high
        if context.systemic_risk_level > self.systemic_risk_warning:
            return True
        
        # Tighten if portfolio is over-concentrated
        if context.portfolio_heat > 0.7:
            return True
        
        return False
    
    def get_tp_adjustment_factor(self, context: RiskV3Context) -> float:
        """
        Get TP adjustment factor based on risk.
        
        Args:
            context: Current risk context
            
        Returns:
            Adjustment factor (0.5 - 1.5, where <1.0 = tighten, >1.0 = widen)
        """
        factor = 1.0
        
        # ESS-based adjustment
        if context.ess_factor > self.ess_critical_threshold:
            factor *= 0.70  # Tighten 30%
        elif context.ess_factor > self.ess_warning_threshold:
            factor *= 0.85  # Tighten 15%
        
        # Systemic risk adjustment
        if context.systemic_risk_level > self.systemic_risk_critical:
            factor *= 0.75  # Defensive: tighten 25%
        elif context.systemic_risk_level > self.systemic_risk_warning:
            factor *= 0.90  # Cautious: tighten 10%
        
        # Correlation risk adjustment
        if context.correlation_risk > 0.8:
            factor *= 0.95  # Slight tightening
        
        return max(0.5, min(1.5, factor))


# Global singleton
_risk_v3_integrator: Optional[RiskV3Integrator] = None


def get_risk_v3_integrator(ai_services=None) -> RiskV3Integrator:
    """Get or create Risk v3 Integrator singleton."""
    global _risk_v3_integrator
    if _risk_v3_integrator is None:
        _risk_v3_integrator = RiskV3Integrator(ai_services=ai_services)
    return _risk_v3_integrator
