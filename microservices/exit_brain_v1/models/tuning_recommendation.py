"""
TuningRecommendation — Proposed parameter change from offline evaluation.

Phase 5 data contract. Shadow-only. Requires human review.

Produced by: proposal_builder.py / threshold_tuner.py / weight_tuner.py
Published to: quantum:stream:exit.tuning.recommendation.shadow
"""

from __future__ import annotations

import logging
import uuid
from dataclasses import dataclass, field
from typing import Dict, List

logger = logging.getLogger(__name__)

VALID_DIRECTIONS = frozenset({"increase", "decrease", "no_change"})


@dataclass(frozen=False)
class TuningRecommendation:
    """
    Immutable proposal for one parameter change.

    Never applied automatically. requires_human_review is always True in v1.
    applied is always False at creation.

    shadow_only MUST be True.
    """

    # ── Identity (REQUIRED) ──────────────────────────────────────────────
    recommendation_id: str                     # uuid4
    created_at: float                          # Epoch seconds
    source_evaluation_run_id: str              # Links to OfflineEvaluationSummary

    # ── Target (REQUIRED) ────────────────────────────────────────────────
    component_name: str                        # e.g. "policy_constraints", "hazard_engine"
    parameter_name: str                        # e.g. "UNCERTAINTY_HARD_CEILING"
    current_value: float                       # Current threshold / weight value
    proposed_value: float                      # Suggested new value
    direction: str                             # "increase", "decrease", "no_change"

    # ── Rationale (REQUIRED) ─────────────────────────────────────────────
    rationale: str                             # Human-readable justification
    supporting_metrics: Dict[str, float] = field(default_factory=dict)
    expected_effect: str = ""                  # What we expect to improve

    # ── Confidence (REQUIRED) ────────────────────────────────────────────
    confidence: float = 0.0                    # [0,1]
    risk_of_change: str = ""                   # "low", "medium", "high"

    # ── Governance (REQUIRED) ────────────────────────────────────────────
    requires_human_review: bool = True         # MUST be True in v1
    applied: bool = False                      # MUST be False at creation
    applied_at: float = 0.0
    applied_by: str = ""

    # ── Quality (OPTIONAL) ───────────────────────────────────────────────
    quality_flags: List[str] = field(default_factory=list)

    # ── Safety (REQUIRED) ────────────────────────────────────────────────
    shadow_only: bool = True

    # ── Validation ───────────────────────────────────────────────────────

    def validate(self) -> List[str]:
        """Return list of error strings. Empty = valid."""
        errors: List[str] = []

        if not self.recommendation_id:
            errors.append("recommendation_id is empty")
        if self.created_at <= 0:
            errors.append(f"created_at must be > 0, got {self.created_at}")
        if not self.source_evaluation_run_id:
            errors.append("source_evaluation_run_id is empty")
        if not self.component_name:
            errors.append("component_name is empty")
        if not self.parameter_name:
            errors.append("parameter_name is empty")
        if not self.rationale:
            errors.append("rationale is empty")

        if self.direction not in VALID_DIRECTIONS:
            errors.append(f"direction '{self.direction}' not in {VALID_DIRECTIONS}")

        if not (0.0 <= self.confidence <= 1.0):
            errors.append(f"confidence must be in [0, 1], got {self.confidence}")

        if not self.requires_human_review:
            errors.append("requires_human_review MUST be True in v1")

        if self.applied:
            errors.append("applied MUST be False at creation")

        if not self.shadow_only:
            errors.append("shadow_only MUST be True in Phase 5")

        return errors

    def to_dict(self) -> Dict:
        """Serialize to flat dict for Redis XADD."""
        import json
        return {
            "recommendation_id": self.recommendation_id,
            "created_at": self.created_at,
            "source_evaluation_run_id": self.source_evaluation_run_id,
            "component_name": self.component_name,
            "parameter_name": self.parameter_name,
            "current_value": self.current_value,
            "proposed_value": self.proposed_value,
            "direction": self.direction,
            "rationale": self.rationale,
            "supporting_metrics": json.dumps(self.supporting_metrics),
            "expected_effect": self.expected_effect,
            "confidence": self.confidence,
            "risk_of_change": self.risk_of_change,
            "requires_human_review": self.requires_human_review,
            "applied": self.applied,
            "applied_at": self.applied_at,
            "applied_by": self.applied_by,
            "quality_flags": ",".join(self.quality_flags),
            "shadow_only": self.shadow_only,
        }
