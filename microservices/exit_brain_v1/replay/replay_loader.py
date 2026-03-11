"""
ReplayLoader — Reads decision traces and upstream snapshots from Redis.

Phase 5 replay component. Shadow-only. Read-only.

Reads from: Phase 1-4 shadow streams
Writes to: Nothing (pure reader)
"""

from __future__ import annotations

import json
import logging
import time
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)

# Shadow streams to read from (Phase 1-4 outputs)
STREAM_DECISION_TRACE = "quantum:stream:exit.decision.trace.shadow"
STREAM_POLICY = "quantum:stream:exit.policy.shadow"
STREAM_BELIEF = "quantum:stream:exit.belief.shadow"
STREAM_HAZARD = "quantum:stream:exit.hazard.shadow"
STREAM_UTILITY = "quantum:stream:exit.utility.shadow"
STREAM_STATE = "quantum:stream:exit.state.shadow"


class ReplayLoader:
    """
    Loads historical decision traces and upstream snapshots from Redis
    shadow streams for offline replay and evaluation.

    Read-only. Never writes to any stream.
    """

    def __init__(self, redis_client) -> None:
        self._r = redis_client

    def load_decision_traces(
        self,
        start_ts: float,
        end_ts: float,
        count: int = 1000,
    ) -> List[Dict[str, Any]]:
        """
        Load decision traces within a time window.

        Args:
            start_ts: Epoch start (inclusive).
            end_ts: Epoch end (inclusive).
            count: Max entries to return.

        Returns:
            List of decoded decision trace dicts.
        """
        return self._read_stream_window(STREAM_DECISION_TRACE, start_ts, end_ts, count)

    def load_policy_decisions(
        self,
        start_ts: float,
        end_ts: float,
        count: int = 1000,
    ) -> List[Dict[str, Any]]:
        """Load policy decisions within a time window."""
        return self._read_stream_window(STREAM_POLICY, start_ts, end_ts, count)

    def load_belief_snapshots(
        self,
        start_ts: float,
        end_ts: float,
        count: int = 1000,
    ) -> List[Dict[str, Any]]:
        """Load belief snapshots within a time window."""
        return self._read_stream_window(STREAM_BELIEF, start_ts, end_ts, count)

    def load_hazard_snapshots(
        self,
        start_ts: float,
        end_ts: float,
        count: int = 1000,
    ) -> List[Dict[str, Any]]:
        """Load hazard snapshots within a time window."""
        return self._read_stream_window(STREAM_HAZARD, start_ts, end_ts, count)

    def load_utility_snapshots(
        self,
        start_ts: float,
        end_ts: float,
        count: int = 1000,
    ) -> List[Dict[str, Any]]:
        """Load utility snapshots within a time window."""
        return self._read_stream_window(STREAM_UTILITY, start_ts, end_ts, count)

    def load_state_snapshots(
        self,
        start_ts: float,
        end_ts: float,
        count: int = 1000,
    ) -> List[Dict[str, Any]]:
        """Load position state snapshots within a time window."""
        return self._read_stream_window(STREAM_STATE, start_ts, end_ts, count)

    def load_decision_trace_by_id(
        self,
        trace_id: str,
        search_window: float = 86400.0,
    ) -> Optional[Dict[str, Any]]:
        """
        Find a specific decision trace by trace_id.

        Args:
            trace_id: The trace_id to search for.
            search_window: How far back to search (seconds). Default 24h.

        Returns:
            Decoded trace dict, or None if not found.
        """
        now = time.time()
        traces = self._read_stream_window(
            STREAM_DECISION_TRACE,
            now - search_window,
            now,
            count=5000,
        )
        for t in traces:
            if t.get("trace_id") == trace_id:
                return t
        return None

    def load_snapshots_for_decision(
        self,
        decision_timestamp: float,
        symbol: str,
        tolerance_sec: float = 5.0,
    ) -> Dict[str, Optional[Dict[str, Any]]]:
        """
        Load belief, hazard, utility, and state snapshots closest to a decision.

        Args:
            decision_timestamp: Epoch of the decision.
            symbol: Trading pair to filter for.
            tolerance_sec: Max time distance to accept.

        Returns:
            Dict with keys "belief", "hazard", "utility", "state" → matched entry or None.
        """
        window_start = decision_timestamp - tolerance_sec
        window_end = decision_timestamp + tolerance_sec

        result: Dict[str, Optional[Dict[str, Any]]] = {
            "belief": None,
            "hazard": None,
            "utility": None,
            "state": None,
        }

        for key, loader in [
            ("belief", self.load_belief_snapshots),
            ("hazard", self.load_hazard_snapshots),
            ("utility", self.load_utility_snapshots),
            ("state", self.load_state_snapshots),
        ]:
            entries = loader(window_start, window_end, count=50)
            best = self._find_closest_by_symbol(entries, symbol, decision_timestamp)
            result[key] = best

        return result

    # ── Private ──────────────────────────────────────────────────────────

    def _read_stream_window(
        self,
        stream: str,
        start_ts: float,
        end_ts: float,
        count: int,
    ) -> List[Dict[str, Any]]:
        """Read entries from a Redis stream within a time window."""
        try:
            start_id = f"{int(start_ts * 1000)}-0"
            end_id = f"{int(end_ts * 1000)}-18446744073709551615"
            raw = self._r.xrange(stream, min=start_id, max=end_id, count=count)
            return [self._decode_entry(entry_id, fields) for entry_id, fields in raw]
        except Exception as e:
            logger.error("[ReplayLoader] Failed to read %s: %s", stream, e)
            return []

    @staticmethod
    def _decode_entry(entry_id: Any, fields: Dict) -> Dict[str, Any]:
        """Decode Redis bytes to strings and parse JSON fields."""
        decoded: Dict[str, Any] = {
            "_stream_id": entry_id.decode("utf-8") if isinstance(entry_id, bytes) else str(entry_id),
        }
        for k, v in fields.items():
            key = k.decode("utf-8") if isinstance(k, bytes) else str(k)
            val = v.decode("utf-8") if isinstance(v, bytes) else str(v)
            # Attempt JSON parse for nested fields
            if val.startswith("{") or val.startswith("["):
                try:
                    decoded[key] = json.loads(val)
                except (json.JSONDecodeError, ValueError):
                    decoded[key] = val
            else:
                decoded[key] = val
        return decoded

    @staticmethod
    def _find_closest_by_symbol(
        entries: List[Dict[str, Any]],
        symbol: str,
        target_ts: float,
    ) -> Optional[Dict[str, Any]]:
        """Find the entry closest in time to target_ts for a given symbol."""
        best: Optional[Dict[str, Any]] = None
        best_dist = float("inf")
        for entry in entries:
            if entry.get("symbol") != symbol:
                continue
            # Try common timestamp field names
            for ts_key in ("belief_timestamp", "hazard_timestamp", "ts", "decision_timestamp"):
                ts_val = entry.get(ts_key)
                if ts_val is not None:
                    try:
                        dist = abs(float(ts_val) - target_ts)
                        if dist < best_dist:
                            best_dist = dist
                            best = entry
                    except (ValueError, TypeError):
                        continue
                    break
        return best
