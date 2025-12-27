"""
Event Subscribers
=================

Production-ready async event listeners for EventBus v2.

All subscribers follow the pattern:
1. Listen on event type
2. Deserialize payload
3. Execute business logic
4. Publish downstream events
5. Log with trace_id
6. Handle errors gracefully

Author: Quantum Trader AI Team
Date: December 2, 2025
"""

from backend.events.subscribers.signal_subscriber import SignalSubscriber
from backend.events.subscribers.trade_subscriber import TradeSubscriber
from backend.events.subscribers.position_subscriber import PositionSubscriber
from backend.events.subscribers.risk_subscriber import RiskSubscriber
from backend.events.subscribers.error_subscriber import ErrorSubscriber

__all__ = [
    "SignalSubscriber",
    "TradeSubscriber",
    "PositionSubscriber",
    "RiskSubscriber",
    "ErrorSubscriber",
]
