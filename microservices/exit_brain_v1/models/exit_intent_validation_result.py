"""
ExitIntentValidationResult — Gateway validation result for an exit intent.

Phase 4 data contract. Shadow-only.

Produced by: exit_intent_gateway_validator.py
Consumed by: exit_agent_orchestrator.py (for publishing)
Published to: quantum:stream:exit.intent.validation.shadow
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Dict, List

logger = logging.getLogger(__name__)


@dataclass(frozen=False)
class ExitIntentValidationResult:
    """
    Result of gateway validation on an ExitIntentCandidate.

    is_valid=False means the intent has hard blocks and MUST NOT proceed
    to any execution path. It is still published to shadow for observability.

    shadow_only MUST be True.
    """

    # ── Identity (REQUIRED) ──────────────────────────────────────────────
    position_id: str
    symbol: str
    validation_timestamp: float            # Epoch seconds

    # ── Candidate reference (REQUIRED) ───────────────────────────────────
    candidate_action: str                  # Action that was validated
    source_intent_id: str = ""             # Links to ExitIntentCandidate.intent_id

    # ── Validation result (REQUIRED) ─────────────────────────────────────
    is_valid: bool = False
    hard_blocks: List[str] = field(default_factory=list)
    soft_warnings: List[str] = field(default_factory=list)
    violated_constraints: List[str] = field(default_factory=list)

    # ── Normalized payload (REQUIRED) ────────────────────────────────────
    normalized_candidate_payload: Dict = field(default_factory=dict)

    # ── Confidence (REQUIRED) ────────────────────────────────────────────
    validation_confidence: float = 0.0     # [0, 1]

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
        if self.validation_timestamp <= 0:
            errors.append(
                f"validation_timestamp must be > 0, got {self.validation_timestamp}"
            )
        if not self.candidate_action:
            errors.append("candidate_action is empty")

        if not (0.0 <= self.validation_confidence <= 1.0):
            errors.append(
                f"validation_confidence must be in [0, 1], got {self.validation_confidence}"
            )

        # If not valid, there must be at least one hard block
        if not self.is_valid and not self.hard_blocks:
            errors.append("is_valid=False but no hard_blocks specified")

        if not self.shadow_only:
            errors.append("shadow_only MUST be True in Phase 4")

        return errors

    def to_dict(self) -> Dict:
        """Serialize to flat dict for Redis XADD."""
        import json
        return {
            "position_id": self.position_id,
            "symbol": self.symbol,
            "validation_timestamp": self.validation_timestamp,
            "candidate_action": self.candidate_action,
            "source_intent_id": self.source_intent_id,
            "is_valid": self.is_valid,
            "hard_blocks": json.dumps(self.hard_blocks),
            "soft_warnings": json.dumps(self.soft_warnings),
            "violated_constraints": json.dumps(self.violated_constraints),
            "normalized_candidate_payload": json.dumps(self.normalized_candidate_payload),
            "validation_confidence": self.validation_confidence,
            "quality_flags": ",".join(self.quality_flags),
            "shadow_only": self.shadow_only,
        }
