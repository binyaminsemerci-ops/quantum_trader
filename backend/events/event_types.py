"""
Event Type Definitions
======================

Centralized event type constants for the trading system.

All events follow the pattern: {domain}.{action}
Examples: signal.generated, trade.executed, position.opened

Event Flow:
    1. signal.generated (AI Engine)
    2. trade.execution_requested (RiskGuard approved)
    3. trade.executed (Execution Engine filled order)
    4. position.opened (Position Monitor confirmed)
    5. position.closed (Exit executed)
    6. risk.alert (Safety Governor warnings)
    7. system.event_error (Error handling)

Author: Quantum Trader AI Team
Date: December 2, 2025
"""

from enum import Enum


class EventType(str, Enum):
    """Event type enumeration for type-safe event publishing/subscribing."""
    
    # Trading signals from AI models
    SIGNAL_GENERATED = "signal.generated"
    
    # Trade execution flow
    TRADE_EXECUTION_REQUESTED = "trade.execution_requested"
    TRADE_EXECUTED = "trade.executed"
    
    # Position lifecycle
    POSITION_OPENED = "position.opened"
    POSITION_CLOSED = "position.closed"
    POSITION_UPDATED = "position.updated"
    
    # Risk management
    RISK_ALERT = "risk.alert"
    RISK_LIMIT_BREACHED = "risk.limit_breached"
    
    # System events
    SYSTEM_EVENT_ERROR = "system.event_error"
    SYSTEM_HEALTH_DEGRADED = "system.health_degraded"
    SYSTEM_EMERGENCY_STOP = "system.emergency_stop"
    SYSTEM_EMERGENCY_TRIGGERED = "system.emergency.triggered"  # [P1-01] Added
    SYSTEM_EMERGENCY_RECOVERED = "system.emergency.recovered"  # [P1-01] Added
    
    # Model events (for CLM, RL, etc.)
    MODEL_PREDICTION_READY = "model.prediction_ready"
    MODEL_RETRAIN_REQUESTED = "model.retrain_requested"
    MODEL_DRIFT_DETECTED = "model.drift_detected"
    MODEL_PROMOTED = "model.promoted"  # [P1-01] Added
    
    # RL v3 events (PPO-based RL)
    RL_V3_DECISION = "rl_v3.decision"
    RL_V3_TRAINING_COMPLETE = "rl_v3.training_complete"
    RL_STRATEGY_SELECTED = "rl.strategy_selected"  # [P1-01] Added
    RL_REWARD_RECEIVED = "rl.reward_received"  # [P1-01] Added
    
    # Market data events
    MARKET_DATA_UPDATED = "market.data_updated"
    MARKET_TICK = "market.tick"  # [P1-01] Added
    MARKET_REGIME_CHANGED = "market.regime.changed"  # [P1-01] Added
    
    def __str__(self) -> str:
        return self.value


# Event routing configuration
EVENT_ROUTING = {
    EventType.SIGNAL_GENERATED: ["signal_subscriber"],
    EventType.TRADE_EXECUTION_REQUESTED: ["trade_subscriber"],
    EventType.TRADE_EXECUTED: ["position_subscriber"],
    EventType.POSITION_OPENED: ["position_subscriber", "metrics_subscriber"],
    EventType.POSITION_CLOSED: ["position_subscriber", "rl_subscriber", "clm_subscriber", "metrics_subscriber"],
    EventType.RISK_ALERT: ["risk_subscriber", "alert_subscriber"],
    EventType.SYSTEM_EVENT_ERROR: ["error_subscriber", "alert_subscriber"],
}
