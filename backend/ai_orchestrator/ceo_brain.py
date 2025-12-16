"""CEO Brain - Core decision-making logic for AI CEO.

This module contains the intelligence that evaluates system state
and determines optimal operating mode and configuration.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import datetime
from typing import Optional

from backend.ai_orchestrator.ceo_policy import (
    CEOPolicy,
    MarketRegime,
    OperatingMode,
    SystemHealth,
)

logger = logging.getLogger(__name__)


@dataclass
class SystemState:
    """Snapshot of current system state for CEO evaluation."""
    
    # Risk metrics (from AI-RO)
    risk_score: float                    # 0-100 scale
    var_estimate: float                  # Value at Risk estimate
    expected_shortfall: float            # Expected Shortfall (CVaR)
    tail_risk_score: float               # Tail risk indicator (0-100)
    
    # Performance metrics (from AI-SO)
    win_rate: float                      # Recent win rate (0-1)
    sharpe_ratio: float                  # Recent Sharpe ratio
    profit_factor: float                 # Gross profit / gross loss
    avg_win_loss_ratio: float            # Average win / average loss
    
    # Portfolio metrics (from PBA/PAL)
    current_drawdown: float              # Current daily drawdown (0-1)
    max_drawdown_today: float            # Max drawdown today (0-1)
    total_pnl_today: float               # Total PnL today (USD)
    open_positions: int                  # Number of open positions
    total_exposure: float                # Total USD exposure
    
    # Market state (from Universe OS / Regime Detector)
    market_regime: MarketRegime          # Current market regime
    regime_confidence: float             # Confidence in regime (0-1)
    market_volatility: float             # Current volatility measure
    
    # System health (from Health Monitor)
    system_health: SystemHealth          # Overall system health
    ai_engine_healthy: bool              # AI Engine status
    rl_engine_healthy: bool              # RL Engine status
    execution_healthy: bool              # Execution service status
    risk_os_healthy: bool                # Risk OS status
    
    # Timing
    timestamp: datetime
    
    def __post_init__(self):
        """Validate state values."""
        assert 0 <= self.risk_score <= 100, "risk_score must be 0-100"
        assert 0 <= self.win_rate <= 1, "win_rate must be 0-1"
        assert 0 <= self.current_drawdown <= 1, "current_drawdown must be 0-1"
        assert 0 <= self.regime_confidence <= 1, "regime_confidence must be 0-1"


@dataclass
class CEODecision:
    """Decision output from CEO Brain."""
    
    # Decision
    operating_mode: OperatingMode
    mode_changed: bool
    
    # Reasoning
    primary_reason: str
    contributing_factors: list[str]
    
    # Actions to take
    update_policy_store: bool
    policy_updates: dict[str, any]
    
    # Alerts
    alert_level: str                     # "info", "warning", "critical"
    alert_message: Optional[str] = None
    
    # Metadata
    decision_timestamp: datetime
    decision_confidence: float           # 0-1 scale
    
    def to_dict(self) -> dict:
        """Convert decision to dictionary for event publishing."""
        return {
            "operating_mode": self.operating_mode.value,
            "mode_changed": self.mode_changed,
            "primary_reason": self.primary_reason,
            "contributing_factors": self.contributing_factors,
            "update_policy_store": self.update_policy_store,
            "policy_updates": self.policy_updates,
            "alert_level": self.alert_level,
            "alert_message": self.alert_message,
            "decision_timestamp": self.decision_timestamp.isoformat(),
            "decision_confidence": self.decision_confidence,
        }


class CEOBrain:
    """
    Core decision-making engine for AI CEO.
    
    Responsibilities:
    - Evaluate system state from all inputs
    - Apply CEO policy to determine operating mode
    - Generate configuration updates for PolicyStore
    - Provide reasoning and alerts
    - Track decision history
    """
    
    def __init__(self, policy: Optional[CEOPolicy] = None):
        """Initialize CEO Brain with policy."""
        self.policy = policy or CEOPolicy()
        
        # Decision history (for learning and analysis)
        self._decision_history: list[CEODecision] = []
        self._max_history = 1000
        
        # State tracking
        self._current_mode = OperatingMode.EXPANSION
        self._last_transition_time = datetime.utcnow().timestamp()
        
        logger.info("CEOBrain initialized")
    
    def evaluate(
        self,
        state: SystemState,
        current_mode: Optional[OperatingMode] = None,
    ) -> CEODecision:
        """
        Evaluate system state and make operating mode decision.
        
        Args:
            state: Current system state snapshot
            current_mode: Current operating mode (if known)
        
        Returns:
            CEODecision with mode, reasoning, and actions
        """
        current_mode = current_mode or self._current_mode
        current_time = datetime.utcnow()
        
        logger.debug(
            f"CEOBrain evaluating state: mode={current_mode.value}, "
            f"drawdown={state.current_drawdown:.2%}, "
            f"risk_score={state.risk_score:.1f}, "
            f"win_rate={state.win_rate:.2%}"
        )
        
        # Get policy recommendation
        recommended_mode, primary_reason = self.policy.recommend_mode(
            current_mode=current_mode,
            drawdown=state.current_drawdown,
            risk_score=state.risk_score,
            win_rate=state.win_rate,
            regime=state.market_regime,
            health=state.system_health,
            open_positions=state.open_positions,
        )
        
        # Check if transition is valid
        mode_changed = recommended_mode != current_mode
        
        if mode_changed:
            is_valid, validation_reason = self.policy.validate_transition(
                from_mode=current_mode,
                to_mode=recommended_mode,
                last_transition_time=self._last_transition_time,
                current_time=current_time.timestamp(),
            )
            
            if not is_valid:
                logger.info(
                    f"Mode transition blocked: {current_mode.value} → {recommended_mode.value}. "
                    f"Reason: {validation_reason}"
                )
                recommended_mode = current_mode
                primary_reason = f"Transition blocked: {validation_reason}"
                mode_changed = False
        
        # Gather contributing factors
        contributing_factors = self._analyze_contributing_factors(state)
        
        # Get policy updates for new mode
        policy_updates = {}
        update_policy_store = False
        
        if mode_changed:
            policy_updates = self.policy.get_mode_config(recommended_mode)
            update_policy_store = True
            self._last_transition_time = current_time.timestamp()
            self._current_mode = recommended_mode
        
        # Determine alert level
        alert_level, alert_message = self._determine_alert(
            state=state,
            mode=recommended_mode,
            mode_changed=mode_changed,
        )
        
        # Calculate decision confidence
        decision_confidence = self._calculate_confidence(state)
        
        # Create decision
        decision = CEODecision(
            operating_mode=recommended_mode,
            mode_changed=mode_changed,
            primary_reason=primary_reason,
            contributing_factors=contributing_factors,
            update_policy_store=update_policy_store,
            policy_updates=policy_updates,
            alert_level=alert_level,
            alert_message=alert_message,
            decision_timestamp=current_time,
            decision_confidence=decision_confidence,
        )
        
        # Store in history
        self._decision_history.append(decision)
        if len(self._decision_history) > self._max_history:
            self._decision_history.pop(0)
        
        logger.info(
            f"CEOBrain decision: mode={recommended_mode.value}, "
            f"changed={mode_changed}, "
            f"confidence={decision_confidence:.2f}, "
            f"alert={alert_level}"
        )
        
        return decision
    
    def _analyze_contributing_factors(self, state: SystemState) -> list[str]:
        """Analyze state and identify key contributing factors."""
        factors = []
        
        # Risk factors
        if state.risk_score > 80:
            factors.append(f"High risk score: {state.risk_score:.1f}")
        elif state.risk_score < 30:
            factors.append(f"Low risk score: {state.risk_score:.1f}")
        
        # Drawdown factors
        if state.current_drawdown > 0.03:
            factors.append(f"Elevated drawdown: {state.current_drawdown:.2%}")
        
        # Performance factors
        if state.win_rate > 0.60:
            factors.append(f"Strong win rate: {state.win_rate:.2%}")
        elif state.win_rate < 0.45:
            factors.append(f"Poor win rate: {state.win_rate:.2%}")
        
        if state.sharpe_ratio > 2.0:
            factors.append(f"Excellent Sharpe: {state.sharpe_ratio:.2f}")
        elif state.sharpe_ratio < 0.5:
            factors.append(f"Low Sharpe: {state.sharpe_ratio:.2f}")
        
        # Market factors
        if state.regime_confidence > 0.8:
            factors.append(f"Clear market regime: {state.market_regime.value}")
        elif state.regime_confidence < 0.5:
            factors.append("Uncertain market regime")
        
        if state.market_volatility > 0.05:
            factors.append("High volatility environment")
        
        # System health factors
        if state.system_health == SystemHealth.DEGRADED:
            factors.append("System degraded")
        elif state.system_health == SystemHealth.CRITICAL:
            factors.append("System critical")
        
        if not state.ai_engine_healthy:
            factors.append("AI Engine unhealthy")
        if not state.execution_healthy:
            factors.append("Execution service unhealthy")
        
        # Position factors
        if state.open_positions > 12:
            factors.append(f"High position count: {state.open_positions}")
        elif state.open_positions == 0:
            factors.append("No open positions")
        
        return factors
    
    def _determine_alert(
        self,
        state: SystemState,
        mode: OperatingMode,
        mode_changed: bool,
    ) -> tuple[str, Optional[str]]:
        """Determine alert level and message."""
        # Critical alerts
        if mode == OperatingMode.BLACK_SWAN:
            return "critical", f"BLACK SWAN MODE: {state.current_drawdown:.2%} drawdown"
        
        if state.system_health == SystemHealth.CRITICAL:
            return "critical", "System health critical"
        
        if state.current_drawdown > 0.06:
            return "critical", f"High drawdown: {state.current_drawdown:.2%}"
        
        # Warning alerts
        if mode == OperatingMode.CAPITAL_PRESERVATION:
            return "warning", "Capital preservation mode active"
        
        if mode_changed:
            return "warning", f"Operating mode changed to {mode.value}"
        
        if state.system_health == SystemHealth.DEGRADED:
            return "warning", "System health degraded"
        
        if state.current_drawdown > 0.03:
            return "warning", f"Elevated drawdown: {state.current_drawdown:.2%}"
        
        # Info (normal)
        return "info", None
    
    def _calculate_confidence(self, state: SystemState) -> float:
        """
        Calculate confidence in decision.
        
        High confidence when:
        - Market regime is clear
        - System health is good
        - Performance metrics are consistent
        - Risk metrics are stable
        """
        confidence_factors = []
        
        # Regime confidence contributes directly
        confidence_factors.append(state.regime_confidence)
        
        # System health contributes
        if state.system_health == SystemHealth.HEALTHY:
            confidence_factors.append(1.0)
        elif state.system_health == SystemHealth.DEGRADED:
            confidence_factors.append(0.6)
        else:
            confidence_factors.append(0.3)
        
        # Component health
        healthy_components = sum([
            state.ai_engine_healthy,
            state.rl_engine_healthy,
            state.execution_healthy,
            state.risk_os_healthy,
        ])
        confidence_factors.append(healthy_components / 4.0)
        
        # Performance consistency (if metrics are in reasonable ranges)
        if 0.45 <= state.win_rate <= 0.70 and 0.5 <= state.sharpe_ratio <= 5.0:
            confidence_factors.append(0.9)
        else:
            confidence_factors.append(0.6)
        
        # Average confidence
        return sum(confidence_factors) / len(confidence_factors)
    
    def get_decision_history(self, limit: int = 100) -> list[CEODecision]:
        """Get recent decision history."""
        return self._decision_history[-limit:]
    
    def get_current_mode(self) -> OperatingMode:
        """Get current operating mode."""
        return self._current_mode
    
    def reset_history(self) -> None:
        """Reset decision history (for testing)."""
        self._decision_history.clear()
        logger.info("CEOBrain decision history reset")
