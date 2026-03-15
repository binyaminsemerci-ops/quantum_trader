"""harvest.intent stream contract — quantum:stream:harvest.intent

Publishers: Exit Intent Gateway (after 9-gate validation)
Consumers: Intent Executor (harvest lane)
"""

from __future__ import annotations
from typing import Optional
from shared.contracts.base import StreamEvent


class HarvestIntentEvent(StreamEvent):
    """A validated harvest/exit command forwarded to intent_executor harvest lane.

    Fields vary by source; some are injected by the gateway,
    others pass through from the original exit.intent.
    """

    # --- Required ---
    symbol: str                                 # e.g. "BTCUSDT"
    action: str                                 # CLOSE | PARTIAL_CLOSE
    reason: str                                 # Human-readable reason

    # --- Optional ---
    percentage: Optional[str] = None            # "100" for full, "25" for partial
    R_net: Optional[str] = None                 # Risk-adjusted metric
    pnl_usd: Optional[str] = None              # Estimated PnL in USD
    entry_price: Optional[str] = None           # Position entry price
    exit_price: Optional[str] = None            # Target/mark exit price
    confidence: Optional[str] = None            # 0.0–1.0
    source: Optional[str] = None                # "exit_intent_gateway"
    urgency: Optional[str] = None               # Forwarded from exit intent
    side: Optional[str] = None                  # LONG | SHORT
    quantity: Optional[str] = None              # Position quantity
    intent_id: Optional[str] = None             # Correlated to exit.intent UUID
    ts_epoch: Optional[str] = None              # Unix epoch

    # --- Autonomous trader extras ---
    intent_type: Optional[str] = None           # "AUTONOMOUS_EXIT"
    hold_score: Optional[str] = None            # Hold-vs-exit score
    exit_score: Optional[str] = None            # Exit urgency score
    timestamp: Optional[str] = None             # Unix epoch (int as string)
