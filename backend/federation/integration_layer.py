"""Integration Layer - API and state aggregation for Federation.

This module provides the interface layer that collects and aggregates
state from all AI agents.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import datetime
from typing import Optional

logger = logging.getLogger(__name__)


@dataclass
class CEOState:
    """Aggregated state from AI CEO."""
    
    operating_mode: str
    mode_changed_recently: bool
    last_decision_time: datetime
    decision_confidence: float
    primary_reason: str
    alerts: list[str]


@dataclass
class RiskState:
    """Aggregated state from AI Risk Officer."""
    
    risk_score: float
    risk_level: str
    var_estimate: float
    expected_shortfall: float
    tail_risk_score: float
    recommended_max_leverage: float
    recommended_max_risk_per_trade: float
    should_reduce_exposure: bool
    should_pause_trading: bool
    alerts: list[str]


@dataclass
class StrategyState:
    """Aggregated state from AI Strategy Officer."""
    
    primary_strategy: str
    fallback_strategies: list[str]
    disabled_strategies: list[str]
    meta_strategy_mode: str
    overall_win_rate: float
    overall_sharpe: float
    recommended_models: list[str]
    confidence: float
    alerts: list[str]


class IntegrationLayer:
    """
    Integration layer for aggregating AI agent states.
    
    Responsibilities:
    - Collect state from AI CEO, AI-RO, AI-SO
    - Aggregate states into unified snapshots
    - Provide query API for current state
    - Track state history
    - Handle missing/stale data gracefully
    """
    
    MAX_STATE_AGE_SECONDS = 300  # 5 minutes
    
    def __init__(self):
        """Initialize Integration Layer."""
        # Latest states from each agent
        self._latest_ceo_state: Optional[CEOState] = None
        self._latest_risk_state: Optional[RiskState] = None
        self._latest_strategy_state: Optional[StrategyState] = None
        
        # Timestamps
        self._ceo_state_time: Optional[datetime] = None
        self._risk_state_time: Optional[datetime] = None
        self._strategy_state_time: Optional[datetime] = None
        
        logger.info("IntegrationLayer initialized")
    
    def update_ceo_state(
        self,
        operating_mode: str,
        mode_changed: bool,
        decision_confidence: float,
        primary_reason: str,
        alerts: list[str],
    ) -> None:
        """Update CEO state from event or API call."""
        self._latest_ceo_state = CEOState(
            operating_mode=operating_mode,
            mode_changed_recently=mode_changed,
            last_decision_time=datetime.utcnow(),
            decision_confidence=decision_confidence,
            primary_reason=primary_reason,
            alerts=alerts,
        )
        self._ceo_state_time = datetime.utcnow()
        
        logger.debug(f"IntegrationLayer updated CEO state: mode={operating_mode}")
    
    def update_risk_state(
        self,
        risk_score: float,
        risk_level: str,
        var_estimate: float,
        expected_shortfall: float,
        tail_risk_score: float,
        recommended_max_leverage: float,
        recommended_max_risk_per_trade: float,
        should_reduce_exposure: bool,
        should_pause_trading: bool,
        alerts: list[str],
    ) -> None:
        """Update Risk Officer state from event or API call."""
        self._latest_risk_state = RiskState(
            risk_score=risk_score,
            risk_level=risk_level,
            var_estimate=var_estimate,
            expected_shortfall=expected_shortfall,
            tail_risk_score=tail_risk_score,
            recommended_max_leverage=recommended_max_leverage,
            recommended_max_risk_per_trade=recommended_max_risk_per_trade,
            should_reduce_exposure=should_reduce_exposure,
            should_pause_trading=should_pause_trading,
            alerts=alerts,
        )
        self._risk_state_time = datetime.utcnow()
        
        logger.debug(f"IntegrationLayer updated Risk state: score={risk_score:.1f}")
    
    def update_strategy_state(
        self,
        primary_strategy: str,
        fallback_strategies: list[str],
        disabled_strategies: list[str],
        meta_strategy_mode: str,
        overall_win_rate: float,
        overall_sharpe: float,
        recommended_models: list[str],
        confidence: float,
        alerts: list[str],
    ) -> None:
        """Update Strategy Officer state from event or API call."""
        self._latest_strategy_state = StrategyState(
            primary_strategy=primary_strategy,
            fallback_strategies=fallback_strategies,
            disabled_strategies=disabled_strategies,
            meta_strategy_mode=meta_strategy_mode,
            overall_win_rate=overall_win_rate,
            overall_sharpe=overall_sharpe,
            recommended_models=recommended_models,
            confidence=confidence,
            alerts=alerts,
        )
        self._strategy_state_time = datetime.utcnow()
        
        logger.debug(f"IntegrationLayer updated Strategy state: primary={primary_strategy}")
    
    def get_ceo_state(self) -> Optional[CEOState]:
        """Get latest CEO state if not stale."""
        if self._is_state_stale(self._ceo_state_time):
            logger.warning("CEO state is stale")
            return None
        return self._latest_ceo_state
    
    def get_risk_state(self) -> Optional[RiskState]:
        """Get latest Risk Officer state if not stale."""
        if self._is_state_stale(self._risk_state_time):
            logger.warning("Risk state is stale")
            return None
        return self._latest_risk_state
    
    def get_strategy_state(self) -> Optional[StrategyState]:
        """Get latest Strategy Officer state if not stale."""
        if self._is_state_stale(self._strategy_state_time):
            logger.warning("Strategy state is stale")
            return None
        return self._latest_strategy_state
    
    def _is_state_stale(self, state_time: Optional[datetime]) -> bool:
        """Check if state is stale based on age."""
        if state_time is None:
            return True
        
        age = (datetime.utcnow() - state_time).total_seconds()
        return age > self.MAX_STATE_AGE_SECONDS
    
    def get_health_status(self) -> dict:
        """Get health status of all integrated components."""
        now = datetime.utcnow()
        
        def get_age(timestamp: Optional[datetime]) -> Optional[float]:
            if timestamp is None:
                return None
            return (now - timestamp).total_seconds()
        
        return {
            "ceo_state": {
                "available": self._latest_ceo_state is not None,
                "age_seconds": get_age(self._ceo_state_time),
                "stale": self._is_state_stale(self._ceo_state_time),
            },
            "risk_state": {
                "available": self._latest_risk_state is not None,
                "age_seconds": get_age(self._risk_state_time),
                "stale": self._is_state_stale(self._risk_state_time),
            },
            "strategy_state": {
                "available": self._latest_strategy_state is not None,
                "age_seconds": get_age(self._strategy_state_time),
                "stale": self._is_state_stale(self._strategy_state_time),
            },
        }
