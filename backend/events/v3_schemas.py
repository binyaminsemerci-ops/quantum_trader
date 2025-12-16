"""
Event Schemas v3 - Microservices Architecture
==============================================

Complete event definitions for inter-service communication in Quantum Trader v3.0.

All events follow standard structure:
{
    "trace_id": "uuid",
    "timestamp": "ISO8601",
    "source": "ai-service | exec-risk-service | analytics-os-service",
    "version": "3.0",
    "payload": { ... }
}

Author: Quantum Trader AI Team
Date: December 2, 2025
Version: 3.0.0
"""

from datetime import datetime, timezone
from typing import Optional, Dict, Any, Literal, List
from pydantic import BaseModel, Field
import uuid


# ============================================================================
# BASE EVENT STRUCTURE
# ============================================================================

class EventMetadata(BaseModel):
    """Standard metadata for all inter-service events"""
    trace_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    timestamp: str = Field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    source: Literal["ai-service", "exec-risk-service", "analytics-os-service", "quantum_trader"]
    version: str = "3.0"
    
    class Config:
        frozen = False


class BaseEventV3(BaseModel):
    """Base event with standard header"""
    metadata: EventMetadata
    payload: Dict[str, Any]
    
    class Config:
        frozen = False


# ============================================================================
# AI SERVICE EVENTS
# ============================================================================

class SignalGeneratedPayload(BaseModel):
    """Payload for signal.generated event from AI service"""
    symbol: str
    action: Literal["BUY", "SELL", "HOLD"]
    confidence: float = Field(ge=0.0, le=1.0)
    model_source: str  # "xgboost", "lightgbm", "nhits", "patchtst", "ensemble"
    score: float = Field(ge=0.0, le=1.0)
    
    # Feature context
    timeframe: str = "1m"
    price: Optional[float] = None
    volume: Optional[float] = None
    
    # Model metadata
    model_version: Optional[str] = None
    ensemble_agreement: Optional[float] = None  # For ensemble: 0.0-1.0
    models_voted: Optional[List[str]] = None
    
    # Risk hints for exec-risk-service
    suggested_leverage: Optional[float] = None
    suggested_size_usd: Optional[float] = None
    
    metadata: Dict[str, Any] = Field(default_factory=dict)


class RLDecisionPayload(BaseModel):
    """Payload for rl.decision event from RL agents"""
    decision_type: Literal["position_sizing", "meta_strategy", "dynamic_tpsl"]
    symbol: str
    
    # Position sizing decision
    recommended_size_usd: Optional[float] = None
    recommended_leverage: Optional[float] = None
    kelly_fraction: Optional[float] = None
    
    # Meta strategy decision
    selected_strategy: Optional[str] = None  # "aggressive", "conservative", "balanced"
    strategy_confidence: Optional[float] = None
    
    # Dynamic TP/SL decision
    take_profit_pct: Optional[float] = None
    stop_loss_pct: Optional[float] = None
    trailing_stop_enabled: Optional[bool] = None
    
    # RL context
    rl_agent_version: str
    state_features: Optional[Dict[str, float]] = None
    q_values: Optional[Dict[str, float]] = None
    exploration_epsilon: Optional[float] = None
    
    metadata: Dict[str, Any] = Field(default_factory=dict)


class ModelPredictionPayload(BaseModel):
    """Payload for model.prediction event (for CLM/drift detection)"""
    model_id: str
    model_type: Literal["xgboost", "lightgbm", "nhits", "patchtst", "ensemble", "rl_ppo"]
    prediction: Dict[str, Any]
    
    # Prediction metadata
    confidence: float = Field(ge=0.0, le=1.0)
    latency_ms: Optional[float] = None
    feature_count: Optional[int] = None
    
    # For continuous learning
    ground_truth_available: bool = False
    prediction_error: Optional[float] = None
    
    metadata: Dict[str, Any] = Field(default_factory=dict)


class UniverseOpportunityPayload(BaseModel):
    """Payload for universe.opportunity event from Opportunity Ranker"""
    ranked_symbols: List[Dict[str, Any]]  # [{symbol, score, reason}, ...]
    total_scanned: int
    timestamp: str
    
    # Ranking criteria
    criteria_weights: Dict[str, float]  # {"volume": 0.3, "volatility": 0.2, ...}
    
    metadata: Dict[str, Any] = Field(default_factory=dict)


# ============================================================================
# EXEC-RISK SERVICE EVENTS
# ============================================================================

class ExecutionRequestPayload(BaseModel):
    """Payload for execution.request event to exec-risk-service"""
    symbol: str
    side: Literal["BUY", "SELL"]
    
    # Position parameters
    position_size_usd: float = Field(gt=0)
    leverage: float = Field(ge=1.0, le=125.0)
    
    # Risk management
    stop_loss_pct: Optional[float] = None
    take_profit_pct: Optional[float] = None
    trailing_stop_pct: Optional[float] = None
    
    # Execution preferences
    execution_strategy: Literal["MARKET", "LIMIT", "TWAP", "SMART"] = "SMART"
    urgency: Literal["LOW", "NORMAL", "HIGH"] = "NORMAL"
    slippage_tolerance_pct: float = 0.5
    
    # Signal context
    signal_confidence: float
    signal_source: str
    
    metadata: Dict[str, Any] = Field(default_factory=dict)


class ExecutionResultPayload(BaseModel):
    """Payload for execution.result event from exec-risk-service"""
    symbol: str
    side: Literal["BUY", "SELL"]
    
    # Execution outcome
    status: Literal["SUCCESS", "PARTIAL", "FAILED", "REJECTED"]
    rejection_reason: Optional[str] = None
    
    # Filled details
    filled_size_usd: float
    filled_price: float
    filled_quantity: float
    leverage_applied: float
    
    # Execution metrics
    slippage_pct: Optional[float] = None
    execution_time_ms: Optional[float] = None
    commission_usd: Optional[float] = None
    
    # Order IDs
    exchange_order_id: Optional[str] = None
    internal_order_id: Optional[str] = None
    
    metadata: Dict[str, Any] = Field(default_factory=dict)


class PositionOpenedPayload(BaseModel):
    """Payload for position.opened event"""
    symbol: str
    entry_price: float
    size_usd: float
    leverage: float
    is_long: bool
    
    # Risk management
    stop_loss_price: Optional[float] = None
    take_profit_price: Optional[float] = None
    trailing_stop_active: bool = False
    
    # Position ID
    position_id: str
    
    # Entry context
    entry_confidence: Optional[float] = None
    entry_model: Optional[str] = None
    entry_timestamp: str = Field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    
    metadata: Dict[str, Any] = Field(default_factory=dict)


class PositionClosedPayload(BaseModel):
    """Payload for position.closed event (CRITICAL for learning)"""
    symbol: str
    position_id: str
    
    # Position details
    entry_price: float
    exit_price: float
    size_usd: float
    leverage: float
    is_long: bool
    
    # Performance metrics
    pnl_usd: float
    pnl_pct: float
    duration_seconds: float
    max_drawdown_pct: Optional[float] = None
    max_profit_pct: Optional[float] = None
    
    # Exit details
    exit_reason: Literal["TAKE_PROFIT", "STOP_LOSS", "TRAILING_STOP", "MANUAL", "SIGNAL_EXIT", "LIQUIDATION", "TIMEOUT"]
    
    # Entry context (for learning)
    entry_confidence: Optional[float] = None
    entry_model: Optional[str] = None
    entry_timestamp: str
    exit_timestamp: str = Field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    
    # Market context
    market_condition: Optional[str] = None
    volatility_at_entry: Optional[float] = None
    volatility_at_exit: Optional[float] = None
    
    metadata: Dict[str, Any] = Field(default_factory=dict)


class RiskAlertPayload(BaseModel):
    """Payload for risk.alert event"""
    severity: Literal["LOW", "MEDIUM", "HIGH", "CRITICAL"]
    alert_type: str
    message: str
    
    # Risk metrics
    current_drawdown_pct: Optional[float] = None
    max_allowed_drawdown_pct: Optional[float] = None
    open_positions_count: Optional[int] = None
    max_positions: Optional[int] = None
    total_exposure_usd: Optional[float] = None
    
    # Action taken
    action_taken: Optional[Literal["BLOCK_TRADES", "REDUCE_LEVERAGE", "CLOSE_POSITIONS", "EMERGENCY_STOP"]] = None
    affected_symbols: Optional[List[str]] = None
    
    risk_profile: Optional[str] = None
    
    metadata: Dict[str, Any] = Field(default_factory=dict)


class EmergencyStopPayload(BaseModel):
    """Payload for emergency.stop event (highest priority)"""
    trigger_reason: str
    severity: Literal["EMERGENCY", "CRITICAL"]
    
    # Actions required
    close_all_positions: bool = True
    cancel_all_orders: bool = True
    block_new_trades: bool = True
    
    # Emergency context
    drawdown_pct: Optional[float] = None
    consecutive_losses: Optional[int] = None
    system_health_status: Optional[str] = None
    
    # Recovery
    estimated_recovery_time_seconds: Optional[int] = None
    manual_intervention_required: bool = False
    
    metadata: Dict[str, Any] = Field(default_factory=dict)


# ============================================================================
# ANALYTICS-OS SERVICE EVENTS
# ============================================================================

class HealthStatusPayload(BaseModel):
    """Payload for health.status event"""
    service_name: str
    status: Literal["HEALTHY", "DEGRADED", "CRITICAL", "UNKNOWN"]
    
    # Health metrics
    cpu_percent: Optional[float] = None
    memory_percent: Optional[float] = None
    memory_used_mb: Optional[float] = None
    uptime_seconds: Optional[float] = None
    
    # Dependency status
    redis_healthy: bool = True
    postgres_healthy: bool = True
    binance_healthy: bool = True
    
    # Module health
    modules: Dict[str, str] = Field(default_factory=dict)  # {module_name: status}
    
    # Degradation details
    degradation_reason: Optional[str] = None
    recovery_action: Optional[str] = None
    
    metadata: Dict[str, Any] = Field(default_factory=dict)


class LearningEventPayload(BaseModel):
    """Payload for learning.event from Continuous Learning Manager"""
    event_type: Literal["DRIFT_DETECTED", "RETRAINING_STARTED", "RETRAINING_COMPLETED", "MODEL_UPDATED"]
    
    # Model context
    model_id: str
    model_type: str
    
    # Drift detection
    drift_score: Optional[float] = None
    drift_threshold: Optional[float] = None
    
    # Retraining
    training_samples: Optional[int] = None
    training_duration_seconds: Optional[float] = None
    new_model_accuracy: Optional[float] = None
    old_model_accuracy: Optional[float] = None
    
    # Model update
    model_version_before: Optional[str] = None
    model_version_after: Optional[str] = None
    deployment_timestamp: Optional[str] = None
    
    metadata: Dict[str, Any] = Field(default_factory=dict)


class SystemAlertPayload(BaseModel):
    """Payload for system.alert event"""
    alert_type: Literal["PERFORMANCE_DEGRADED", "MEMORY_HIGH", "CPU_HIGH", "REDIS_ERROR", "DATABASE_ERROR", "API_ERROR"]
    severity: Literal["INFO", "WARNING", "ERROR", "CRITICAL"]
    message: str
    
    # Alert context
    component: str
    metric_name: Optional[str] = None
    metric_value: Optional[float] = None
    threshold_value: Optional[float] = None
    
    # Recovery
    recovery_action: Optional[str] = None
    auto_recovery_attempted: bool = False
    
    metadata: Dict[str, Any] = Field(default_factory=dict)


class PortfolioBalancePayload(BaseModel):
    """Payload for portfolio.balance event from PBA"""
    action: Literal["REBALANCE_REQUIRED", "REBALANCE_COMPLETED", "EXPOSURE_WARNING"]
    
    # Portfolio state
    total_exposure_usd: float
    max_allowed_exposure_usd: float
    position_count: int
    max_positions: int
    
    # Sector/asset allocation
    sector_exposure: Dict[str, float] = Field(default_factory=dict)
    asset_allocation: Dict[str, float] = Field(default_factory=dict)
    
    # Rebalance actions
    symbols_to_reduce: Optional[List[str]] = None
    symbols_to_increase: Optional[List[str]] = None
    
    metadata: Dict[str, Any] = Field(default_factory=dict)


class ProfitAmplificationPayload(BaseModel):
    """Payload for profit.amplification event from PAL"""
    symbol: str
    action: Literal["AMPLIFY_POSITION", "ADD_TO_WINNER", "CUT_LOSER"]
    
    # Amplification parameters
    current_pnl_pct: float
    amplification_factor: float = Field(ge=1.0, le=3.0)
    additional_size_usd: Optional[float] = None
    
    # Risk-adjusted amplification
    risk_adjusted: bool = True
    max_total_size_usd: Optional[float] = None
    
    metadata: Dict[str, Any] = Field(default_factory=dict)


# ============================================================================
# SYSTEM-LEVEL EVENTS
# ============================================================================

class PolicyUpdatedPayload(BaseModel):
    """Payload for policy.updated event"""
    previous_mode: str
    new_mode: str
    updated_by: str
    
    # Policy changes
    changed_fields: List[str]
    effective_timestamp: str = Field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    
    # Impact
    requires_restart: bool = False
    requires_position_adjustment: bool = False
    
    metadata: Dict[str, Any] = Field(default_factory=dict)


class SystemModeChangedPayload(BaseModel):
    """Payload for system.mode_changed event"""
    previous_mode: Literal["NORMAL", "DEFENSIVE", "EMERGENCY", "MAINTENANCE"]
    new_mode: Literal["NORMAL", "DEFENSIVE", "EMERGENCY", "MAINTENANCE"]
    reason: str
    changed_by: str
    
    # Mode-specific actions
    allow_new_trades: bool
    allow_position_expansion: bool
    force_close_positions: bool = False
    
    metadata: Dict[str, Any] = Field(default_factory=dict)


class ServiceHeartbeatPayload(BaseModel):
    """Payload for service.heartbeat event (sent every 5s)"""
    service_name: Literal["ai-service", "exec-risk-service", "analytics-os-service"]
    status: Literal["HEALTHY", "DEGRADED", "CRITICAL"]
    uptime_seconds: float
    
    # Resource metrics
    cpu_percent: float
    memory_percent: float
    memory_used_mb: float
    
    # Service-specific metrics
    active_tasks: Optional[int] = None
    pending_events: Optional[int] = None
    processed_events_last_minute: Optional[int] = None
    
    metadata: Dict[str, Any] = Field(default_factory=dict)


# ============================================================================
# RPC REQUEST/RESPONSE SCHEMAS
# ============================================================================

class RPCRequestPayload(BaseModel):
    """Payload for inter-service RPC request"""
    service_target: Literal["ai-service", "exec-risk-service", "analytics-os-service"]
    command: str
    parameters: Dict[str, Any] = Field(default_factory=dict)
    
    # RPC metadata
    timeout_seconds: float = 30.0
    require_response: bool = True
    
    metadata: Dict[str, Any] = Field(default_factory=dict)


class RPCResponsePayload(BaseModel):
    """Payload for inter-service RPC response"""
    request_trace_id: str
    status: Literal["SUCCESS", "ERROR", "TIMEOUT"]
    
    # Response data
    result: Optional[Dict[str, Any]] = None
    error_message: Optional[str] = None
    error_code: Optional[str] = None
    
    # Timing
    processing_time_ms: Optional[float] = None
    
    metadata: Dict[str, Any] = Field(default_factory=dict)


# ============================================================================
# EVENT TYPE CONSTANTS
# ============================================================================

class EventTypes:
    """Event type constants for type-safe event handling"""
    
    # AI Service events
    SIGNAL_GENERATED = "signal.generated"
    RL_DECISION = "rl.decision"
    MODEL_PREDICTION = "model.prediction"
    UNIVERSE_OPPORTUNITY = "universe.opportunity"
    
    # Exec-Risk Service events
    EXECUTION_REQUEST = "execution.request"
    EXECUTION_RESULT = "execution.result"
    POSITION_OPENED = "position.opened"
    POSITION_CLOSED = "position.closed"
    RISK_ALERT = "risk.alert"
    EMERGENCY_STOP = "emergency.stop"
    
    # Analytics-OS Service events
    HEALTH_STATUS = "health.status"
    LEARNING_EVENT = "learning.event"
    SYSTEM_ALERT = "system.alert"
    PORTFOLIO_BALANCE = "portfolio.balance"
    PROFIT_AMPLIFICATION = "profit.amplification"
    
    # System events
    POLICY_UPDATED = "policy.updated"
    SYSTEM_MODE_CHANGED = "system.mode_changed"
    SERVICE_HEARTBEAT = "service.heartbeat"
    
    # RPC events
    RPC_REQUEST = "rpc.request"
    RPC_RESPONSE = "rpc.response"


# ============================================================================
# EVENT BUILDERS (Helper Functions)
# ============================================================================

def build_event(
    event_type: str,
    payload: BaseModel,
    source: str,
    trace_id: Optional[str] = None
) -> BaseEventV3:
    """
    Build complete event with metadata.
    
    Args:
        event_type: Event type constant from EventTypes
        payload: Pydantic payload model
        source: Source service name
        trace_id: Optional trace ID for distributed tracing
    
    Returns:
        Complete BaseEventV3 with metadata
    """
    metadata = EventMetadata(
        source=source,
        trace_id=trace_id or str(uuid.uuid4())
    )
    
    return BaseEventV3(
        metadata=metadata,
        payload=payload.dict()
    )


def parse_event(event_data: Dict[str, Any], expected_payload_class: type) -> tuple[EventMetadata, BaseModel]:
    """
    Parse event data into metadata and payload.
    
    Args:
        event_data: Raw event dictionary
        expected_payload_class: Pydantic payload class
    
    Returns:
        (metadata, payload)
    
    Raises:
        ValueError: If parsing fails
    """
    try:
        metadata = EventMetadata(**event_data.get("metadata", {}))
        payload = expected_payload_class(**event_data.get("payload", {}))
        return metadata, payload
    except Exception as e:
        raise ValueError(f"Failed to parse event: {e}")
