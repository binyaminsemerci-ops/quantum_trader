"""exit.intent stream contract — quantum:stream:exit.intent

Publishers: Exit Management Agent (sole publisher)
Consumers: Exit Intent Gateway
"""

from __future__ import annotations
from typing import Optional
from shared.contracts.base import StreamEvent


class ExitIntentEvent(StreamEvent):
    """An exit decision published to quantum:stream:exit.intent.

    All fields are required for the gateway to parse successfully.
    The gateway's IntentMessage.from_redis_fields() calls _require() on most fields.
    """

    # --- Required ---
    intent_id: str                              # UUID hex
    symbol: str                                 # e.g. "BTCUSDT"
    action: str                                 # FULL_CLOSE | PARTIAL_CLOSE_25 | TIME_STOP_EXIT etc.
    urgency: str                                # LOW | MEDIUM | HIGH | EMERGENCY
    side: str                                   # LONG | SHORT (position side)
    qty_fraction: str                           # "1.0" for full, "0.25" for partial
    quantity: str                               # Total position quantity
    entry_price: str                            # Position entry price
    mark_price: str                             # Current mark price
    R_net: str                                  # Risk-adjusted net return
    confidence: str                             # 0.0–1.0
    reason: str                                 # Human-readable decision text
    source: str                                 # "exit_management_agent"
    ts_epoch: str                               # Unix epoch (float as string)

    # --- Optional ---
    loop_id: Optional[str] = None               # Hex loop identifier
    patch: Optional[str] = None                 # "PATCH-5A"
