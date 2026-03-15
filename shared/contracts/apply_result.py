"""apply.result stream contract — quantum:stream:apply.result

Publishers: Intent Executor, Apply Layer
Consumers: Governor, Exit Intelligence, Metricpack Builder, Trade History Logger, Harvest Brain
"""

from __future__ import annotations
from typing import Optional
from shared.contracts.base import StreamEvent


class ApplyResultEvent(StreamEvent):
    """An execution result published to quantum:stream:apply.result."""

    # --- Required ---
    event_type: Optional[str] = "apply.result"
    plan_id: str                                # Matches apply.plan plan_id
    symbol: str                                 # e.g. "BTCUSDT"
    executed: str                               # "true" | "false"
    source: Optional[str] = "intent_executor"
    timestamp: str                              # Unix epoch

    # --- When executed=true ---
    side: Optional[str] = None                  # "BUY" | "SELL"
    qty: Optional[str] = None
    order_id: Optional[str] = None
    filled_qty: Optional[str] = None
    order_status: Optional[str] = None          # "FILLED" | "PARTIALLY_FILLED" etc.
    permit: Optional[str] = None

    # --- When executed=false ---
    error: Optional[str] = None                 # Error description
    decision: Optional[str] = None              # "BLOCKED" | "ERROR"

    # --- Rich result (JSON blob) ---
    details: Optional[str] = None               # JSON with full result data
