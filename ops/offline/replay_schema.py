"""replay_schema: typed schema and JSONL serialization for exit.replay records.

PATCH-8D — offline replay export.

A ReplayRecord is the parsed, typed representation of a single Redis stream
entry from quantum:stream:exit.replay, as written by ReplayWriter._build_record
(PATCH-8C).

Fields
------
Redis stores all values as byte strings.  ``from_redis_entry`` handles both
bytes-keyed and str-keyed dicts and decodes all values to str before
type-parsing.

This module has no runtime service dependencies — it is safe to import and
use from any offline or analysis script.
"""
from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from typing import Optional


# ── Low-level decode helpers ────────────────────────────────────────────────────

def _decode(raw) -> str:
    """Decode bytes to str, or str-ify other types.  Never raises."""
    if isinstance(raw, bytes):
        return raw.decode("utf-8", errors="replace")
    if raw is None:
        return ""
    return str(raw)


def _opt_float(v: str) -> Optional[float]:
    """Parse a string to float; return None for absent / null values."""
    if not v or v in ("null", "None", "nan", ""):
        return None
    try:
        return float(v)
    except (ValueError, TypeError):
        return None


def _opt_int(v: str) -> Optional[int]:
    """Parse a string to int; return None for absent / null values."""
    if not v or v in ("null", "None", ""):
        return None
    try:
        return int(v)
    except (ValueError, TypeError):
        return None


def _bool_field(v: str) -> bool:
    """Parse a Redis 'true'/'false' string field to Python bool."""
    return v.strip().lower() == "true"


# ── Record dataclass ────────────────────────────────────────────────────────────

@dataclass
class ReplayRecord:
    """
    Parsed, typed representation of one quantum:stream:exit.replay entry.

    Fields map 1-to-1 to the keys written by ReplayWriter._build_record
    (PATCH-8C), plus ``stream_id`` from the Redis message envelope.

    Numeric/optional fields that carry a "null" string in Redis are stored as
    ``None``.  Use ``to_json_line()`` for JSONL serialization.
    """

    # ── Redis metadata ──────────────────────────────────────────────────────────
    stream_id: str                      # Redis message ID, e.g. "1741200000123-0"

    # ── Identity / metadata ─────────────────────────────────────────────────────
    decision_id: str
    symbol: str
    record_time_epoch: Optional[int]    # seconds since epoch, wall clock at write time
    patch: str                          # "PATCH-8C"
    source: str                         # "exit_management_agent"

    # ── PATCH-8A snapshot ───────────────────────────────────────────────────────
    live_action: str                    # "FULL_CLOSE" | "PARTIAL_CLOSE_25" | "HOLD" | ...
    formula_action: str                 # "null" if not captured at snapshot time
    qwen3_action: str                   # "null" if Qwen3 was inactive
    diverged: bool                      # True when formula and qwen3 actions disagreed
    exit_score: Optional[float]         # composite urgency score [0.0, 1.0]
    entry_price: Optional[float]
    side: str                           # "LONG" | "SHORT"
    quantity: Optional[float]

    # ── PATCH-8B outcome ────────────────────────────────────────────────────────
    hold_duration_sec: Optional[int]    # seconds from snapshot capture to position closure
    close_price: Optional[float]
    closed_by: str                      # "exit_management_agent" | "unknown"
    outcome_action: str

    # ── PATCH-8C reward ─────────────────────────────────────────────────────────
    reward: Optional[float]             # scalar in [-1.0, 1.0]
    regret_label: str                   # "late_hold" | "premature_close" | "divergence_regret" | "none"
    regret_score: Optional[float]       # severity in [0.0, 1.0]
    preferred_action: str

    # ── Constructors ────────────────────────────────────────────────────────────

    @classmethod
    def from_redis_entry(cls, stream_id, raw_fields: dict) -> "ReplayRecord":
        """
        Build a ReplayRecord from a raw Redis XRANGE / XREAD result entry.

        ``stream_id`` and keys/values in ``raw_fields`` may be bytes or str.
        Missing fields are silently defaulted; never raises.

        Parameters
        ----------
        stream_id:
            The Redis stream message ID (first element of the XRANGE tuple).
        raw_fields:
            The dict of field → value pairs (second element of the XRANGE tuple).
        """
        sid = _decode(stream_id)
        f: dict[str, str] = {_decode(k): _decode(v) for k, v in raw_fields.items()}

        return cls(
            stream_id=sid,
            decision_id=f.get("decision_id", ""),
            symbol=f.get("symbol", ""),
            record_time_epoch=_opt_int(f.get("record_time_epoch", "")),
            patch=f.get("patch", ""),
            source=f.get("source", ""),
            live_action=f.get("live_action", "UNKNOWN"),
            formula_action=f.get("formula_action", "null"),
            qwen3_action=f.get("qwen3_action", "null"),
            diverged=_bool_field(f.get("diverged", "false")),
            exit_score=_opt_float(f.get("exit_score", "")),
            entry_price=_opt_float(f.get("entry_price", "")),
            side=f.get("side", "UNKNOWN"),
            quantity=_opt_float(f.get("quantity", "")),
            hold_duration_sec=_opt_int(f.get("hold_duration_sec", "")),
            close_price=_opt_float(f.get("close_price", "")),
            closed_by=f.get("closed_by", "unknown"),
            outcome_action=f.get("outcome_action", "UNKNOWN"),
            reward=_opt_float(f.get("reward", "")),
            regret_label=f.get("regret_label", "none"),
            regret_score=_opt_float(f.get("regret_score", "")),
            preferred_action=f.get("preferred_action", "HOLD"),
        )

    # ── Serialization ────────────────────────────────────────────────────────────

    def to_dict(self) -> dict:
        """Return a JSON-serializable dict of all fields."""
        return asdict(self)

    def to_json_line(self) -> str:
        """Serialize to a single compact JSON string (no trailing newline)."""
        return json.dumps(self.to_dict(), ensure_ascii=False)
