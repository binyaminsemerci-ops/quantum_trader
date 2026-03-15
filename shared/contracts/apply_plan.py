"""apply.plan stream contract — quantum:stream:apply.plan

Publishers: Intent Bridge (primary), Apply Layer, Harvest Brain, Governor, Risk Guard
Consumers: Intent Executor, Apply Layer (disabled), Governor, Position State Brain
"""

from __future__ import annotations
from typing import Optional
from shared.contracts.base import StreamEvent


class ApplyPlanEvent(StreamEvent):
    """An execution plan published to quantum:stream:apply.plan.

    Two distinct schema variants exist on this stream:
    1. Intent Bridge entry/close plans (plan_id, decision, side, qty, reduceOnly)
    2. Apply Layer harvest plans (kill_score, R_net, reason_codes, steps)
    This schema covers the Intent Bridge variant which is the primary format.
    """

    # --- Required ---
    plan_id: str                                # SHA256 hash (16 chars), unique
    decision: str                               # "EXECUTE" | "BLOCKED" | "SKIP" | "HALT"
    symbol: str                                 # e.g. "BTCUSDT"
    side: str                                   # "BUY" | "SELL"
    type: Optional[str] = "MARKET"              # Order type
    qty: str                                    # Calculated quantity (float as string)
    reduceOnly: Optional[str] = "false"         # "true" | "false"
    source: str                                 # "intent_bridge" | "harvest_brain" etc.
    timestamp: str                              # Unix epoch (int as string)

    # --- Action context ---
    action: Optional[str] = None                # "ENTRY_PROPOSED" | "FULL_CLOSE_PROPOSED" etc.
    signature: Optional[str] = None             # Same as source (audit trail)

    # --- Forwarded from trade.intent ---
    leverage: Optional[str] = None
    stop_loss: Optional[str] = None
    take_profit: Optional[str] = None
    entry_price: Optional[str] = None
    breakeven_price: Optional[str] = None
    atr_value: Optional[str] = None
    volatility_factor: Optional[str] = None
    confidence: Optional[str] = None
    regime: Optional[str] = None
