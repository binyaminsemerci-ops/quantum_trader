"""
ShadowPublisher — Publishes Exit Brain v1 outputs to shadow Redis streams.

SAFETY: This module writes ONLY to shadow streams (*.shadow).
It has a hard-coded blocklist to prevent accidental writes to execution paths.

Streams written:
  quantum:stream:exit.state.shadow     — Full PositionExitState per position
  quantum:stream:exit.geometry.shadow  — Geometry scores per position
  quantum:stream:exit.regime.shadow    — Regime drift analysis per cycle
"""

from __future__ import annotations

import json
import logging
import time
from dataclasses import asdict
from typing import Any, Dict, Optional

from ..models.position_exit_state import PositionExitState
from ..engines.geometry_engine import GeometryResult
from ..engines.regime_drift_engine import RegimeState

logger = logging.getLogger(__name__)

# Maximum stream length (auto-trimmed)
STREAM_MAXLEN = 5000

# Hard-coded blocklist: these streams must NEVER be written to
_FORBIDDEN_STREAMS = frozenset({
    "quantum:stream:trade.intent",
    "quantum:stream:apply.plan",
    "quantum:stream:apply.plan.manual",
    "quantum:stream:apply.result",
    "quantum:stream:exit.intent",
    "quantum:stream:harvest.intent",
})


class ShadowPublisher:
    """
    Publishes exit brain outputs to Redis shadow streams.

    Every write is guarded by the forbidden-streams blocklist.
    If someone misconfigures a stream name, the write will be rejected.
    """

    STREAM_STATE = "quantum:stream:exit.state.shadow"
    STREAM_GEOMETRY = "quantum:stream:exit.geometry.shadow"
    STREAM_REGIME = "quantum:stream:exit.regime.shadow"

    def __init__(self, redis_client) -> None:
        """
        Args:
            redis_client: A synchronous redis.Redis instance.
        """
        self._r = redis_client

    # ── Public API ───────────────────────────────────────────────────────

    def publish_state(self, state: PositionExitState) -> Optional[str]:
        """
        Publish a PositionExitState to the shadow state stream.

        Args:
            state: Enriched position state.

        Returns:
            Stream entry ID, or None on failure.
        """
        data = state.to_dict()
        # Flatten nested types to JSON strings for Redis XADD
        data["source_timestamps"] = json.dumps(data["source_timestamps"])
        data["data_quality_flags"] = json.dumps(data["data_quality_flags"])
        # Convert None values to empty string (Redis doesn't accept None)
        flat = {k: ("" if v is None else str(v)) for k, v in data.items()}
        return self._xadd(self.STREAM_STATE, flat)

    def publish_geometry(
        self,
        symbol: str,
        side: str,
        result: GeometryResult,
    ) -> Optional[str]:
        """
        Publish geometry scores for a position.

        Args:
            symbol: Trading pair.
            side: Position direction.
            result: GeometryResult from geometry_engine.

        Returns:
            Stream entry ID, or None on failure.
        """
        data = {
            "symbol": symbol,
            "side": side,
            "mfe": str(result.mfe),
            "mae": str(result.mae),
            "drawdown_from_peak": str(result.drawdown_from_peak),
            "profit_protection_ratio": str(result.profit_protection_ratio),
            "momentum_decay": str(result.momentum_decay),
            "reward_to_risk_remaining": str(result.reward_to_risk_remaining),
            "ts": str(time.time()),
        }
        return self._xadd(self.STREAM_GEOMETRY, data)

    def publish_regime(self, regime_state: RegimeState) -> Optional[str]:
        """
        Publish regime analysis to the shadow regime stream.

        Args:
            regime_state: RegimeState from regime_drift_engine.

        Returns:
            Stream entry ID, or None on failure.
        """
        data: Dict[str, str] = {
            "regime_label": regime_state.regime_label,
            "regime_confidence": str(regime_state.regime_confidence),
            "trend_alignment": str(regime_state.trend_alignment),
            "reversal_risk": str(regime_state.reversal_risk),
            "chop_risk": str(regime_state.chop_risk),
            "mean_reversion_score": str(regime_state.mean_reversion_score),
            "ts": str(time.time()),
        }
        if regime_state.drift is not None:
            data["drift_detected"] = str(regime_state.drift.drifted)
            data["drift_magnitude"] = str(regime_state.drift.magnitude)
            data["drift_transition"] = regime_state.drift.transition

        return self._xadd(self.STREAM_REGIME, data)

    # ── Private ──────────────────────────────────────────────────────────

    def _xadd(self, stream: str, fields: Dict[str, str]) -> Optional[str]:
        """
        Safe XADD wrapper with forbidden-stream guard.

        Returns stream entry ID or None.
        """
        if stream in _FORBIDDEN_STREAMS:
            logger.error(
                "[ShadowPublisher] BLOCKED write to forbidden stream: %s", stream
            )
            return None

        if not stream.endswith(".shadow"):
            logger.error(
                "[ShadowPublisher] BLOCKED write to non-shadow stream: %s", stream
            )
            return None

        try:
            entry_id = self._r.xadd(stream, fields, maxlen=STREAM_MAXLEN)
            if isinstance(entry_id, bytes):
                entry_id = entry_id.decode("utf-8")
            return entry_id
        except Exception as e:
            logger.error("[ShadowPublisher] XADD to %s failed: %s", stream, e)
            return None
