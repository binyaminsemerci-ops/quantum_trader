"""Risk Brain - Core risk analysis and decision logic.

This module contains the intelligence for evaluating portfolio risk
and determining appropriate risk limits and recommendations.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import datetime
from typing import Optional

import numpy as np

from backend.ai_risk.risk_models import RiskModels, TailRiskMetrics, VaRResult

logger = logging.getLogger(__name__)


@dataclass
class PortfolioRiskData:
    """Input data for risk assessment."""
    
    # Historical data
    returns_history: np.ndarray          # Historical returns (percentage)
    
    # Current portfolio state
    total_exposure: float                # Total USD exposure
    open_positions: int                  # Number of open positions
    account_balance: float               # Current account balance
    
    # Performance metrics
    current_drawdown: float              # Current drawdown (0-1)
    max_drawdown_today: float            # Max drawdown today (0-1)
    total_pnl_today: float               # Today's PnL (USD)
    
    # Market state
    current_volatility: float            # Current market volatility
    regime: str                          # Market regime
    
    # Risk limits (current settings)
    max_leverage: float
    max_risk_per_trade: float
    max_daily_drawdown: float
    
    # Timestamp
    timestamp: datetime


@dataclass
class RiskAssessment:
    """Risk assessment output."""
    
    # Overall risk score (0-100)
    risk_score: float
    
    # Risk metrics
    var_result: VaRResult
    tail_risk_metrics: TailRiskMetrics
    
    # Recommended limits
    recommended_max_leverage: float
    recommended_max_risk_per_trade: float
    recommended_max_positions: int
    recommended_max_exposure: float
    
    # Risk state
    risk_level: str                      # "low", "moderate", "high", "critical"
    risk_alerts: list[str]               # List of risk warnings
    
    # Actions
    should_reduce_exposure: bool
    should_tighten_stops: bool
    should_pause_trading: bool
    
    # Metadata
    assessment_timestamp: datetime
    confidence: float                    # Assessment confidence (0-1)
    
    def to_dict(self) -> dict:
        """Convert to dictionary for event publishing."""
        return {
            "risk_score": self.risk_score,
            "var_estimate": self.var_result.var_value,
            "expected_shortfall": self.var_result.expected_shortfall,
            "tail_risk_score": self.tail_risk_metrics.tail_risk_score,
            "recommended_max_leverage": self.recommended_max_leverage,
            "recommended_max_risk_per_trade": self.recommended_max_risk_per_trade,
            "recommended_max_positions": self.recommended_max_positions,
            "recommended_max_exposure": self.recommended_max_exposure,
            "risk_level": self.risk_level,
            "risk_alerts": self.risk_alerts,
            "should_reduce_exposure": self.should_reduce_exposure,
            "should_tighten_stops": self.should_tighten_stops,
            "should_pause_trading": self.should_pause_trading,
            "assessment_timestamp": self.assessment_timestamp.isoformat(),
            "confidence": self.confidence,
        }


class RiskBrain:
    """
    Core risk analysis engine.
    
    Responsibilities:
    - Analyze portfolio risk using statistical models
    - Calculate VaR, ES, tail risk
    - Recommend risk limits
    - Generate risk alerts
    - Provide risk state for AI CEO
    """
    
    # Risk score thresholds
    RISK_LOW = 30.0
    RISK_MODERATE = 50.0
    RISK_HIGH = 70.0
    RISK_CRITICAL = 85.0
    
    def __init__(self, models: Optional[RiskModels] = None):
        """Initialize Risk Brain."""
        self.models = models or RiskModels()
        
        # Assessment history
        self._assessment_history: list[RiskAssessment] = []
        self._max_history = 1000
        
        logger.info("RiskBrain initialized")
    
    def assess_risk(
        self,
        data: PortfolioRiskData,
        confidence_level: float = 0.95,
    ) -> RiskAssessment:
        """
        Perform comprehensive risk assessment.
        
        Args:
            data: Portfolio risk data
            confidence_level: VaR confidence level (default 0.95)
        
        Returns:
            RiskAssessment with recommendations and alerts
        """
        logger.debug(
            f"RiskBrain assessing risk: exposure={data.total_exposure:.2f}, "
            f"drawdown={data.current_drawdown:.2%}, "
            f"positions={data.open_positions}"
        )
        
        # Calculate risk metrics
        var_result = self.models.calculate_var(
            returns=data.returns_history,
            confidence_level=confidence_level,
            method="cornish_fisher",
        )
        
        tail_risk_metrics = self.models.calculate_tail_risk(
            returns=data.returns_history,
            current_exposure=data.total_exposure,
        )
        
        # Calculate overall risk score
        risk_score = self._calculate_risk_score(data, var_result, tail_risk_metrics)
        
        # Determine risk level
        risk_level = self._determine_risk_level(risk_score)
        
        # Generate risk alerts
        risk_alerts = self._generate_risk_alerts(data, risk_score, var_result, tail_risk_metrics)
        
        # Recommend risk limits
        recommended_limits = self._recommend_limits(
            data=data,
            risk_score=risk_score,
            var_result=var_result,
        )
        
        # Determine required actions
        should_reduce_exposure = risk_score > self.RISK_HIGH or data.current_drawdown > data.max_daily_drawdown
        should_tighten_stops = risk_score > self.RISK_MODERATE or data.current_drawdown > data.max_daily_drawdown * 0.7
        should_pause_trading = risk_score > self.RISK_CRITICAL or data.current_drawdown > data.max_daily_drawdown * 1.2
        
        # Calculate assessment confidence
        confidence = self._calculate_confidence(data)
        
        # Create assessment
        assessment = RiskAssessment(
            risk_score=risk_score,
            var_result=var_result,
            tail_risk_metrics=tail_risk_metrics,
            recommended_max_leverage=recommended_limits["max_leverage"],
            recommended_max_risk_per_trade=recommended_limits["max_risk_per_trade"],
            recommended_max_positions=recommended_limits["max_positions"],
            recommended_max_exposure=recommended_limits["max_exposure"],
            risk_level=risk_level,
            risk_alerts=risk_alerts,
            should_reduce_exposure=should_reduce_exposure,
            should_tighten_stops=should_tighten_stops,
            should_pause_trading=should_pause_trading,
            assessment_timestamp=datetime.utcnow(),
            confidence=confidence,
        )
        
        # Store in history
        self._assessment_history.append(assessment)
        if len(self._assessment_history) > self._max_history:
            self._assessment_history.pop(0)
        
        logger.info(
            f"RiskBrain assessment complete: risk_score={risk_score:.1f}, "
            f"level={risk_level}, alerts={len(risk_alerts)}"
        )
        
        return assessment
    
    def _calculate_risk_score(
        self,
        data: PortfolioRiskData,
        var_result: VaRResult,
        tail_risk: TailRiskMetrics,
    ) -> float:
        """
        Calculate overall risk score (0-100).
        
        Higher score = higher risk
        """
        score = 50.0  # Base score
        
        # Factor 1: Drawdown (high weight)
        drawdown_score = data.current_drawdown / data.max_daily_drawdown * 30
        score += min(drawdown_score, 30)
        
        # Factor 2: Tail risk (from models)
        tail_score = tail_risk.tail_risk_score * 0.2
        score += min(tail_score, 20)
        
        # Factor 3: VaR relative to exposure
        if data.total_exposure > 0:
            var_ratio = var_result.var_value / data.total_exposure
            var_score = var_ratio * 100
            score += min(var_score, 15)
        
        # Factor 4: Volatility
        vol_score = data.current_volatility * 500  # Scale volatility
        score += min(vol_score, 15)
        
        # Factor 5: Position concentration
        if data.open_positions > 0:
            avg_position_size = data.total_exposure / data.open_positions
            if data.account_balance > 0:
                concentration = avg_position_size / data.account_balance
                concentration_score = concentration * 50
                score += min(concentration_score, 10)
        
        # Factor 6: Leverage usage
        if data.account_balance > 0:
            current_leverage = data.total_exposure / data.account_balance
            leverage_ratio = current_leverage / data.max_leverage
            leverage_score = leverage_ratio * 10
            score += min(leverage_score, 10)
        
        # Cap at 100
        score = min(score, 100.0)
        
        return score
    
    def _determine_risk_level(self, risk_score: float) -> str:
        """Determine risk level from risk score."""
        if risk_score < self.RISK_LOW:
            return "low"
        elif risk_score < self.RISK_MODERATE:
            return "moderate"
        elif risk_score < self.RISK_HIGH:
            return "high"
        else:
            return "critical"
    
    def _generate_risk_alerts(
        self,
        data: PortfolioRiskData,
        risk_score: float,
        var_result: VaRResult,
        tail_risk: TailRiskMetrics,
    ) -> list[str]:
        """Generate specific risk alert messages."""
        alerts = []
        
        # Drawdown alerts
        if data.current_drawdown > data.max_daily_drawdown:
            alerts.append(
                f"Drawdown exceeded limit: {data.current_drawdown:.2%} > {data.max_daily_drawdown:.2%}"
            )
        elif data.current_drawdown > data.max_daily_drawdown * 0.8:
            alerts.append(
                f"Drawdown approaching limit: {data.current_drawdown:.2%}"
            )
        
        # VaR alerts
        if data.account_balance > 0:
            var_as_pct = var_result.var_value / data.account_balance
            if var_as_pct > 0.10:
                alerts.append(
                    f"High VaR: {var_as_pct:.2%} of account at risk"
                )
        
        # Tail risk alerts
        if tail_risk.tail_risk_score > 75:
            alerts.append(
                f"Elevated tail risk: score {tail_risk.tail_risk_score:.1f}"
            )
        
        # Volatility alerts
        if data.current_volatility > 0.05:
            alerts.append(
                f"High market volatility: {data.current_volatility:.2%}"
            )
        
        # Position alerts
        if data.open_positions > 15:
            alerts.append(
                f"High position count: {data.open_positions}"
            )
        
        # Leverage alerts
        if data.account_balance > 0:
            current_leverage = data.total_exposure / data.account_balance
            if current_leverage > data.max_leverage * 0.9:
                alerts.append(
                    f"High leverage: {current_leverage:.1f}x (limit {data.max_leverage:.1f}x)"
                )
        
        # Overall risk score alert
        if risk_score > self.RISK_CRITICAL:
            alerts.append(
                f"CRITICAL RISK SCORE: {risk_score:.1f}"
            )
        elif risk_score > self.RISK_HIGH:
            alerts.append(
                f"High risk score: {risk_score:.1f}"
            )
        
        return alerts
    
    def _recommend_limits(
        self,
        data: PortfolioRiskData,
        risk_score: float,
        var_result: VaRResult,
    ) -> dict:
        """
        Recommend risk limits based on current conditions.
        
        Returns dict with recommended limits.
        """
        # Base limits (from current settings)
        rec_leverage = data.max_leverage
        rec_risk_per_trade = data.max_risk_per_trade
        rec_max_positions = 10
        rec_max_exposure = data.account_balance * data.max_leverage
        
        # Adjust based on risk score
        if risk_score > self.RISK_CRITICAL:
            # Critical risk - severe restrictions
            rec_leverage = max(data.max_leverage * 0.2, 1.0)
            rec_risk_per_trade = data.max_risk_per_trade * 0.2
            rec_max_positions = 2
            rec_max_exposure = data.account_balance * rec_leverage
        
        elif risk_score > self.RISK_HIGH:
            # High risk - strong restrictions
            rec_leverage = max(data.max_leverage * 0.5, 2.0)
            rec_risk_per_trade = data.max_risk_per_trade * 0.5
            rec_max_positions = 5
            rec_max_exposure = data.account_balance * rec_leverage
        
        elif risk_score > self.RISK_MODERATE:
            # Moderate risk - light restrictions
            rec_leverage = max(data.max_leverage * 0.7, 3.0)
            rec_risk_per_trade = data.max_risk_per_trade * 0.75
            rec_max_positions = 8
            rec_max_exposure = data.account_balance * rec_leverage
        
        else:
            # Low risk - use volatility-adjusted limits
            rec_leverage = self.models.calculate_volatility_adjusted_leverage(
                current_volatility=data.current_volatility,
                base_leverage=data.max_leverage,
            )
            rec_max_exposure = data.account_balance * rec_leverage
        
        return {
            "max_leverage": rec_leverage,
            "max_risk_per_trade": rec_risk_per_trade,
            "max_positions": rec_max_positions,
            "max_exposure": rec_max_exposure,
        }
    
    def _calculate_confidence(self, data: PortfolioRiskData) -> float:
        """
        Calculate confidence in risk assessment.
        
        Higher confidence with:
        - More historical data
        - Lower volatility
        - Clearer market regime
        """
        confidence_factors = []
        
        # Historical data availability
        if len(data.returns_history) > 100:
            confidence_factors.append(1.0)
        elif len(data.returns_history) > 50:
            confidence_factors.append(0.8)
        else:
            confidence_factors.append(0.5)
        
        # Volatility (lower vol = higher confidence)
        if data.current_volatility < 0.02:
            confidence_factors.append(0.9)
        elif data.current_volatility < 0.04:
            confidence_factors.append(0.7)
        else:
            confidence_factors.append(0.5)
        
        # Market regime clarity (if available)
        if data.regime in ["BULLISH", "BEARISH", "SIDEWAYS"]:
            confidence_factors.append(0.9)
        else:
            confidence_factors.append(0.6)
        
        return sum(confidence_factors) / len(confidence_factors)
    
    def get_assessment_history(self, limit: int = 100) -> list[RiskAssessment]:
        """Get recent assessment history."""
        return self._assessment_history[-limit:]
