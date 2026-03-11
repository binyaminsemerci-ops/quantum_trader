"""
DecisionTrace — Full audit trace of one exit decision cycle.

Phase 4 data contract. Shadow-only. For diagnostics and replay.

Produced by: exit_agent_orchestrator.py
Published to: quantum:stream:exit.decision.trace.shadow
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Dict, List

logger = logging.getLogger(__name__)


@dataclass(frozen=False)
class DecisionTrace:
    """
    Complete audit trail for one exit decision cycle.

    Contains every input version, all scored candidates, rejection reasons,
    decisive factors, and final reasoning. Designed for post-hoc analysis
    and RL feedback.

    shadow_only MUST be True.
    """

    # ── Identity (REQUIRED) ──────────────────────────────────────────────
    position_id: str
    symbol: str
    trace_id: str                           # uuid4
    trace_timestamp: float                  # Epoch seconds
    source_decision_id: str                 # Links to PolicyDecision.decision_id

    # ── Upstream versions (REQUIRED) ─────────────────────────────────────
    upstream_versions: Dict[str, float] = field(default_factory=dict)
    # e.g. {"state_ts": 123.4, "ensemble_ts": 123.5, "belief_ts": 123.6, ...}

    # ── All candidates (REQUIRED) ────────────────────────────────────────
    all_candidates: List[Dict] = field(default_factory=list)
    # Each entry: ActionCandidate.to_dict()

    # ── Decision (REQUIRED) ──────────────────────────────────────────────
    chosen_action: str = ""
    rejected_actions: List[Dict] = field(default_factory=list)
    # Each: {"action": str, "reason": str, "block_code": str}

    # ── Decisive factors (REQUIRED) ──────────────────────────────────────
    decisive_factors: List[str] = field(default_factory=list)
    uncertainty_penalties: Dict[str, float] = field(default_factory=dict)
    constraint_effects: Dict[str, str] = field(default_factory=dict)
    # e.g. {"UNCERTAINTY_CEILING_BREACH": "blocked CLOSE_FULL"}

    # ── Final reasoning (REQUIRED) ───────────────────────────────────────
    final_reasoning: str = ""

    # ── Quality (OPTIONAL) ───────────────────────────────────────────────
    quality_flags: List[str] = field(default_factory=list)

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
        if not self.trace_id:
            errors.append("trace_id is empty")
        if self.trace_timestamp <= 0:
            errors.append(f"trace_timestamp must be > 0, got {self.trace_timestamp}")
        if not self.source_decision_id:
            errors.append("source_decision_id is empty")
        if not self.chosen_action:
            errors.append("chosen_action is empty")

        if not self.shadow_only:
            errors.append("shadow_only MUST be True in Phase 4")

        return errors

    def to_dict(self) -> Dict:
        """Serialize to flat dict for Redis XADD."""
        import json
        return {
            "position_id": self.position_id,
            "symbol": self.symbol,
            "trace_id": self.trace_id,
            "trace_timestamp": self.trace_timestamp,
            "source_decision_id": self.source_decision_id,
            "upstream_versions": json.dumps(self.upstream_versions),
            "all_candidates": json.dumps(self.all_candidates),
            "chosen_action": self.chosen_action,
            "rejected_actions": json.dumps(self.rejected_actions),
            "decisive_factors": json.dumps(self.decisive_factors),
            "uncertainty_penalties": json.dumps(self.uncertainty_penalties),
            "constraint_effects": json.dumps(self.constraint_effects),
            "final_reasoning": self.final_reasoning,
            "quality_flags": ",".join(self.quality_flags),
            "shadow_only": self.shadow_only,
        }
