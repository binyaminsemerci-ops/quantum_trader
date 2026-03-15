"""
Typed IPC Contracts for Redis Streams (OP 7D)

Pydantic v2 schemas for all core Redis stream messages.
All field values are strings (Redis stream requirement).
Use .to_redis() to get a Dict[str, str] for XADD.
Use .from_redis(fields) to parse XREADGROUP bytes-decoded output.

Core chain: trade.intent → apply.plan → apply.result
Exit chain: exit.intent → harvest.intent → trade.closed
"""

from shared.contracts.trade_intent import TradeIntentEvent
from shared.contracts.apply_plan import ApplyPlanEvent
from shared.contracts.apply_result import ApplyResultEvent
from shared.contracts.exit_intent import ExitIntentEvent
from shared.contracts.harvest_intent import HarvestIntentEvent
from shared.contracts.trade_closed import TradeClosedEvent
from shared.contracts.validation import validate_xadd, validate_xread

__all__ = [
    "TradeIntentEvent",
    "ApplyPlanEvent",
    "ApplyResultEvent",
    "ExitIntentEvent",
    "HarvestIntentEvent",
    "TradeClosedEvent",
    "validate_xadd",
    "validate_xread",
]
