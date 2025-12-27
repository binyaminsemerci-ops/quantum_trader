"""
EventBus subsystem for Quantum Trader.

Provides internal async pub/sub messaging for system-wide event coordination.
"""

from .events import (
    Event,
    PolicyUpdatedEvent,
    StrategyPromotedEvent,
    ModelPromotedEvent,
    HealthStatusChangedEvent,
    OpportunitiesUpdatedEvent,
    TradeExecutedEvent,
    HealthStatus,
    RiskMode,
    StrategyLifecycle,
)
from .bus import EventBus, InMemoryEventBus, EventHandler

__all__ = [
    "Event",
    "PolicyUpdatedEvent",
    "StrategyPromotedEvent",
    "ModelPromotedEvent",
    "HealthStatusChangedEvent",
    "OpportunitiesUpdatedEvent",
    "TradeExecutedEvent",
    "HealthStatus",
    "RiskMode",
    "StrategyLifecycle",
    "EventBus",
    "InMemoryEventBus",
    "EventHandler",
]
