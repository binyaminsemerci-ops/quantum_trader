"""
PolicyDecision — Policy-evaluated exit decision for one position.

Phase 4 data contract. Shadow-only. Fail-closed.

Produced by: exit_policy_engine.py (via orchestrator)
Consumed by: exit_agent_orchestrator.py (to build intent)
Published to: quantum:stream:exit.policy.shadow
"""

from __future__ import annotations

import logging
import uuid
from dataclasses import dataclass, field
from typing import Dict, List

from .action_candidate import VALID_ACTIONS

logger = logging.getLogger(__name__)


@dataclass(frozen=False)
class PolicyDecision:
    """
    Result of policy evaluation on the utility scorecard.

    If policy_passed is False, chosen_action MUST be HOLD.
    shadow_only MUST be True.
    """

    # ── Identity (REQUIRED) ──────────────────────────────────────────────
    position_id: str
    symbol: str
    decision_id: str                       # uuid4, unique per decision cycle
    decision_timestamp: float              # Epoch seconds

    # ── Chosen action (REQUIRED) ─────────────────────────────────────────
    chosen_action: str                     # One of VALID_ACTIONS
    chosen_action_rank: int                # Original rank from utility scoring
    chosen_action_utility: float           # [0,1] Net utility of chosen action

    # ── Confidence (REQUIRED) ────────────────────────────────────────────
    decision_confidence: float             # [0,1] Policy confidence
    decision_uncertainty: float            # [0,1] 1 - confidence

    # ── Policy evaluation (REQUIRED) ─────────────────────────────────────
    policy_passed: bool                    # True if policy allows action
    policy_blocks: List[str] = field(default_factory=list)    # Hard block codes
    policy_warnings: List[str] = field(default_factory=list)  # Soft warning codes

    # ── Upstream summaries (REQUIRED) ────────────────────────────────────
    hazard_summary: Dict[str, float] = field(default_factory=dict)
    belief_summary: Dict[str, float] = field(default_factory=dict)
    utility_summary: Dict[str, float] = field(default_factory=dict)

    # ── Explainability (REQUIRED) ────────────────────────────────────────
    reason_codes: List[str] = field(default_factory=list)
    explanation_tags: List[str] = field(default_factory=list)

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
        if not self.decision_id:
            errors.append("decision_id is empty")
        if self.decision_timestamp <= 0:
            errors.append(f"decision_timestamp must be > 0, got {self.decision_timestamp}")

        if self.chosen_action not in VALID_ACTIONS:
            errors.append(f"chosen_action '{self.chosen_action}' not in {VALID_ACTIONS}")
        if self.chosen_action_rank < 1:
            errors.append(f"chosen_action_rank must be >= 1, got {self.chosen_action_rank}")

        for name, val in [
            ("chosen_action_utility", self.chosen_action_utility),
            ("decision_confidence", self.decision_confidence),
            ("decision_uncertainty", self.decision_uncertainty),
        ]:
            if not (0.0 <= val <= 1.0):
                errors.append(f"{name} must be in [0, 1], got {val}")

        # Fail-closed invariant: if policy blocked, action must be HOLD
        if not self.policy_passed and self.chosen_action != "HOLD":
            errors.append(
                f"policy_passed=False but chosen_action='{self.chosen_action}', must be HOLD"
            )

        if not self.shadow_only:
            errors.append("shadow_only MUST be True in Phase 4")

        return errors

    def to_dict(self) -> Dict:
        """Serialize to flat dict for Redis XADD."""
        import json
        return {
            "position_id": self.position_id,
            "symbol": self.symbol,
            "decision_id": self.decision_id,
            "decision_timestamp": self.decision_timestamp,
            "chosen_action": self.chosen_action,
            "chosen_action_rank": self.chosen_action_rank,
            "chosen_action_utility": self.chosen_action_utility,
            "decision_confidence": self.decision_confidence,
            "decision_uncertainty": self.decision_uncertainty,
            "policy_passed": self.policy_passed,
            "policy_blocks": json.dumps(self.policy_blocks),
            "policy_warnings": json.dumps(self.policy_warnings),
            "hazard_summary": json.dumps(self.hazard_summary),
            "belief_summary": json.dumps(self.belief_summary),
            "utility_summary": json.dumps(self.utility_summary),
            "reason_codes": json.dumps(self.reason_codes),
            "explanation_tags": json.dumps(self.explanation_tags),
            "quality_flags": ",".join(self.quality_flags),
            "shadow_only": self.shadow_only,
        }
