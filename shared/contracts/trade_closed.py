"""trade.closed stream contract — quantum:stream:trade.closed

Publishers:
  - Intent Executor main lane (in _commit_ledger_exactly_once, FLAT case)
  - Intent Executor harvest lane
Consumers:
  - CLM / performance tracking
  - Calibration pipeline
"""

from __future__ import annotations
from typing import Optional
from shared.contracts.base import StreamEvent


class TradeClosedEvent(StreamEvent):
    """Emitted when a position is fully closed (qty → 0)."""

    # --- Required ---
    event_type: str = "trade.closed"            # Always "trade.closed"
    timestamp: str                              # ISO-8601 or epoch
    symbol: str                                 # e.g. "BTCUSDT"
    side: str                                   # LONG | SHORT (the closed side)
    entry_price: str                            # Original entry price
    exit_price: str                             # Close/mark price
    pnl_percent: str                            # Percentage P&L
    source: str                                 # "intent_executor_main" | "intent_executor_harvest"

    # --- Optional ---
    confidence: Optional[str] = None            # Model confidence at entry
    model_id: Optional[str] = None              # "main_lane_close" | model name
    R_net: Optional[str] = None                 # Risk-adjusted return
    pnl_usd: Optional[str] = None              # Dollar P&L
    reason: Optional[str] = None                # Close reason text
    order_id: Optional[str] = None              # Binance order ID
