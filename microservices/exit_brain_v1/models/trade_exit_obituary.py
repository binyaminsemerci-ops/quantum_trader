"""
TradeExitObituary — Post-mortem record for one exit decision.

Phase 5 data contract. Shadow-only. Offline evaluation.

Produced by: replay_obituary_writer.py
Published to: quantum:stream:exit.obituary.shadow
"""

from __future__ import annotations

import logging
import uuid
from dataclasses import dataclass, field
from typing import Dict, List

from .action_candidate import VALID_ACTIONS

logger = logging.getLogger(__name__)


@dataclass(frozen=False)
class TradeExitObituary:
    """
    Structured post-mortem for one exit decision cycle.

    Captures what the system believed, what it recommended,
    what actually happened, and quantified quality scores.

    shadow_only MUST be True.
    """

    # ── Identity (REQUIRED) ──────────────────────────────────────────────
    position_id: str
    symbol: str
    obituary_id: str                          # uuid4
    obituary_timestamp: float                 # Epoch seconds when obituary written

    # ── Position lifecycle (REQUIRED) ────────────────────────────────────
    open_timestamp: float                     # Position open epoch
    lifecycle_duration_seconds: float          # close - open (or now - open)

    # ── Close event (OPTIONAL — 0.0 if still open) ──────────────────────
    close_timestamp: float = 0.0
    actual_exit_action: str = ""              # What actually was executed ∪{""}
    actual_exit_timestamp: float = 0.0

    # ── Actual outcomes (OPTIONAL) ───────────────────────────────────────
    actual_realized_pnl: float = 0.0          # USD
    actual_realized_pnl_pct: float = 0.0      # Percentage

    # ── Peak / drawdown (REQUIRED) ───────────────────────────────────────
    peak_unrealized_pnl: float = 0.0
    peak_unrealized_pnl_pct: float = 0.0
    max_drawdown_after_peak: float = 0.0      # ≥ 0

    # ── Best exit window (OPTIONAL) ──────────────────────────────────────
    best_exit_window_start: float = 0.0
    best_exit_window_end: float = 0.0
    best_exit_window_pnl: float = 0.0
    best_exit_window_reason: str = ""

    # ── Exit Brain recommendation at decision time (REQUIRED) ────────────
    recommended_action_at_decision: str = ""   # VALID_ACTIONS
    recommended_action_utility: float = 0.0    # [0,1]
    recommended_action_confidence: float = 0.0 # [0,1]
    policy_passed: bool = False
    policy_reason_codes: List[str] = field(default_factory=list)

    # ── Upstream snapshots at decision time (REQUIRED) ───────────────────
    belief_snapshot: Dict[str, float] = field(default_factory=dict)
    hazard_snapshot: Dict[str, float] = field(default_factory=dict)
    utility_snapshot: List[Dict] = field(default_factory=list)  # Top-3 candidates

    # ── Trace references (REQUIRED) ──────────────────────────────────────
    decision_trace_ref: str = ""               # trace_id from DecisionTrace
    source_decision_id: str = ""               # decision_id from PolicyDecision

    # ── Post-decision path (OPTIONAL) ────────────────────────────────────
    post_decision_price_path: List[float] = field(default_factory=list)
    post_decision_pnl_path: List[float] = field(default_factory=list)
    evaluation_horizon_seconds: float = 14400.0  # Default 4h

    # ── Quality scores (REQUIRED) ────────────────────────────────────────
    regret_score: float = 0.0                  # [0,1] lower = better
    preservation_score: float = 0.0            # [0,1] higher = better
    opportunity_capture_score: float = 0.0     # [0,1] higher = better

    # ── Quality flags (OPTIONAL) ─────────────────────────────────────────
    quality_flags: List[str] = field(default_factory=list)
    upstream_quality_flags: List[str] = field(default_factory=list)

    # ── Safety (REQUIRED) ────────────────────────────────────────────────
    shadow_only: bool = True

    # ── Validation ───────────────────────────────────────────────────────

    def validate(self) -> List[str]:
        """Return list of error strings. Empty = valid."""
        errors: List[str] = []

        if not self.position_id:
            errors.append("position_id is empty")
        if not self.symbol:
            errors.append("symbol is empty")
        if not self.obituary_id:
            errors.append("obituary_id is empty")
        if self.obituary_timestamp <= 0:
            errors.append(f"obituary_timestamp must be > 0, got {self.obituary_timestamp}")
        if self.open_timestamp <= 0:
            errors.append(f"open_timestamp must be > 0, got {self.open_timestamp}")
        if self.lifecycle_duration_seconds < 0:
            errors.append(f"lifecycle_duration_seconds must be >= 0, got {self.lifecycle_duration_seconds}")

        if self.recommended_action_at_decision and self.recommended_action_at_decision not in VALID_ACTIONS:
            errors.append(
                f"recommended_action_at_decision '{self.recommended_action_at_decision}' "
                f"not in {VALID_ACTIONS}"
            )

        if self.actual_exit_action and self.actual_exit_action not in VALID_ACTIONS:
            errors.append(
                f"actual_exit_action '{self.actual_exit_action}' not in {VALID_ACTIONS}"
            )

        for name, val in [
            ("recommended_action_utility", self.recommended_action_utility),
            ("recommended_action_confidence", self.recommended_action_confidence),
            ("regret_score", self.regret_score),
            ("preservation_score", self.preservation_score),
            ("opportunity_capture_score", self.opportunity_capture_score),
        ]:
            if not (0.0 <= val <= 1.0):
                errors.append(f"{name} must be in [0, 1], got {val}")

        if self.max_drawdown_after_peak < 0:
            errors.append(f"max_drawdown_after_peak must be >= 0, got {self.max_drawdown_after_peak}")

        if not self.shadow_only:
            errors.append("shadow_only MUST be True in Phase 5")

        return errors

    def to_dict(self) -> Dict:
        """Serialize to flat dict for Redis XADD."""
        import json
        return {
            "position_id": self.position_id,
            "symbol": self.symbol,
            "obituary_id": self.obituary_id,
            "obituary_timestamp": self.obituary_timestamp,
            "open_timestamp": self.open_timestamp,
            "close_timestamp": self.close_timestamp,
            "lifecycle_duration_seconds": self.lifecycle_duration_seconds,
            "actual_exit_action": self.actual_exit_action,
            "actual_exit_timestamp": self.actual_exit_timestamp,
            "actual_realized_pnl": self.actual_realized_pnl,
            "actual_realized_pnl_pct": self.actual_realized_pnl_pct,
            "peak_unrealized_pnl": self.peak_unrealized_pnl,
            "peak_unrealized_pnl_pct": self.peak_unrealized_pnl_pct,
            "max_drawdown_after_peak": self.max_drawdown_after_peak,
            "best_exit_window_start": self.best_exit_window_start,
            "best_exit_window_end": self.best_exit_window_end,
            "best_exit_window_pnl": self.best_exit_window_pnl,
            "best_exit_window_reason": self.best_exit_window_reason,
            "recommended_action_at_decision": self.recommended_action_at_decision,
            "recommended_action_utility": self.recommended_action_utility,
            "recommended_action_confidence": self.recommended_action_confidence,
            "policy_passed": self.policy_passed,
            "policy_reason_codes": json.dumps(self.policy_reason_codes),
            "belief_snapshot": json.dumps(self.belief_snapshot),
            "hazard_snapshot": json.dumps(self.hazard_snapshot),
            "utility_snapshot": json.dumps(self.utility_snapshot),
            "decision_trace_ref": self.decision_trace_ref,
            "source_decision_id": self.source_decision_id,
            "post_decision_price_path": json.dumps(self.post_decision_price_path),
            "post_decision_pnl_path": json.dumps(self.post_decision_pnl_path),
            "evaluation_horizon_seconds": self.evaluation_horizon_seconds,
            "regret_score": self.regret_score,
            "preservation_score": self.preservation_score,
            "opportunity_capture_score": self.opportunity_capture_score,
            "quality_flags": ",".join(self.quality_flags),
            "upstream_quality_flags": ",".join(self.upstream_quality_flags),
            "shadow_only": self.shadow_only,
        }
