"""
feeds/atr_provider.py
Extracts ATR from position data.
ATR ring buffers and regime detection live in engine/state.py.
"""

import logging
from typing import Optional
from feeds.position_provider import Position

logger = logging.getLogger("hv2.feeds.atr")


class ATRProvider:
    """
    Single responsibility: extract and validate atr_value from a position.

    Ring buffer maintenance and percentile-based regime detection are the
    responsibility of SymbolState (engine/state.py), which receives the
    raw atr_value from here each tick.
    """

    def get_atr(self, pos: Position) -> Optional[float]:
        """
        Returns atr_value if valid, None if ATR is zero/invalid.
        Caller must skip position on None return.
        """
        val = pos.atr_value
        if val <= 0.0:
            logger.info("[HV2] ATR_INVALID symbol=%s atr=%.6f", pos.symbol, val)
            return None
        return val
