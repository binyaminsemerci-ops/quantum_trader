"""
Event-Driven Trading Flow v1
=============================

Complete event-driven architecture using EventBus v2 (Redis Streams).

This module provides:
- Event type definitions
- Pydantic schemas for type-safe payloads
- Publishers for emitting events
- Subscribers for consuming events
- Full trace_id propagation
- Logger v2 integration

Architecture:
    AI Engine → signal.generated
              ↓
    RiskGuard → trade.execution_requested
              ↓
    Execution → trade.executed
              ↓
    Monitor → position.opened
            ↓
    Exit → position.closed → RL/CLM/Supervisor

Author: Quantum Trader AI Team
Date: December 2, 2025
Version: 1.0
"""

from backend.events.event_types import EventType
from backend.events.schemas import (
    # Market events
    MarketTickEvent,
    MarketRegimeChangedEvent,
    # Signal events
    SignalGeneratedEvent,
    # Trade events
    TradeExecutionRequestedEvent,
    TradeExecutedEvent,
    # Position events
    PositionOpenedEvent,
    PositionClosedEvent,
    # Risk events
    RiskAlertEvent,
    # System events
    SystemEmergencyTriggeredEvent,
    SystemEmergencyRecoveredEvent,
    SystemEventErrorEvent,
    # Model events
    ModelPromotedEvent,
    ModelPredictionReadyEvent,
    ModelDriftDetectedEvent,
    # RL events
    RLStrategySelectedEvent,
    RLRewardReceivedEvent,
)
from backend.events.publishers import (
    publish_signal_generated,
    publish_execution_requested,
    publish_trade_executed,
    publish_position_opened,
    publish_position_closed,
    publish_risk_alert,
    publish_event_error,
)

__all__ = [
    # Event types
    "EventType",
    # Market schemas
    "MarketTickEvent",
    "MarketRegimeChangedEvent",
    # Signal schemas
    "SignalGeneratedEvent",
    # Trade schemas
    "TradeExecutionRequestedEvent",
    "TradeExecutedEvent",
    # Position schemas
    "PositionOpenedEvent",
    "PositionClosedEvent",
    # Risk schemas
    "RiskAlertEvent",
    # System schemas
    "SystemEmergencyTriggeredEvent",
    "SystemEmergencyRecoveredEvent",
    "SystemEventErrorEvent",
    # Model schemas
    "ModelPromotedEvent",
    "ModelPredictionReadyEvent",
    "ModelDriftDetectedEvent",
    # RL schemas
    "RLStrategySelectedEvent",
    "RLRewardReceivedEvent",
    # Publishers
    "publish_signal_generated",
    "publish_execution_requested",
    "publish_trade_executed",
    "publish_position_opened",
    "publish_position_closed",
    "publish_risk_alert",
    "publish_event_error",
]
