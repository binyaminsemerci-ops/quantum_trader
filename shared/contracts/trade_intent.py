"""trade.intent stream contract — quantum:stream:trade.intent

Publishers: AI Engine, AI Strategy Router, RL Orchestrator, Exit Intent Gateway
Consumers: Intent Bridge, Execution Service (legacy), Trade Intent Subscriber
"""

from __future__ import annotations
from typing import Optional
from shared.contracts.base import StreamEvent


class TradeIntentEvent(StreamEvent):
    """A trading signal published to quantum:stream:trade.intent.

    NOTE: Two wire formats exist on this stream:
    1. EventBus path wraps in {"payload": "<JSON>", "timestamp": ..., "source": ...}
    2. Direct XADD uses {"data": "<JSON>"}
    Intent Bridge handles both. This schema describes the INNER payload fields.
    """

    # --- Required ---
    symbol: str                                 # e.g. "BTCUSDT"
    side: str                                   # "BUY" | "SELL"
    confidence: str                             # 0.0–1.0 as string
    timestamp: str                              # ISO8601 or epoch

    # --- Sizing ---
    position_size_usd: Optional[str] = None     # USD notional
    leverage: Optional[str] = None              # 1–125
    entry_price: Optional[str] = None           # Market price at signal time

    # --- TP/SL ---
    stop_loss: Optional[str] = None             # Absolute SL price
    take_profit: Optional[str] = None           # Absolute TP price

    # --- Model metadata ---
    model: Optional[str] = None                 # "ensemble"
    meta_strategy: Optional[str] = None         # Strategy ID
    consensus_count: Optional[str] = None       # How many models agreed
    total_models: Optional[str] = None          # Total models

    # --- Volatility / features ---
    atr_value: Optional[str] = None
    volatility_factor: Optional[str] = None
    exchange_divergence: Optional[str] = None
    funding_rate: Optional[str] = None
    regime: Optional[str] = None                # Market regime

    # --- Autonomous trader extras ---
    intent_type: Optional[str] = None           # "AUTONOMOUS_ENTRY"
    action: Optional[str] = None                # "BUY" | "SELL"
    reason: Optional[str] = None                # Entry reason text
    reduceOnly: Optional[str] = None            # "true" | "false"
