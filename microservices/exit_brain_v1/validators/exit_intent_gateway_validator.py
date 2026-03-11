"""
ExitIntentGatewayValidator — Schema and constraint validation for exit intents.

Pure logic. No Redis. No IO. Fail-closed.
Input: ExitIntentCandidate
Output: ExitIntentValidationResult

Checks:
  1. Schema completeness (required fields)
  2. Payload normalization (clamp, strip)
  3. Action allowlist
  4. Percentage sanity
  5. Idempotency (optional, if tracker provided)
  6. Source freshness
  7. Evidence completeness

shadow_only — no execution writes, no order generation.
"""

from __future__ import annotations

import logging
import time
from typing import List, Optional

from ..models.action_candidate import VALID_ACTIONS, ACTION_EXIT_FRACTIONS
from ..models.exit_intent_candidate import ExitIntentCandidate
from ..models.exit_intent_validation_result import ExitIntentValidationResult

from .payload_normalizer import normalize_payload, check_required_fields
from .idempotency import IdempotencyTracker

from ..policy import reason_codes as RC
from ..policy import policy_constraints as PC

logger = logging.getLogger(__name__)

# Maximum age of intent before it's considered stale
MAX_INTENT_AGE_SEC = 60.0


class ExitIntentGatewayValidator:
    """
    Validates ExitIntentCandidate payloads before (shadow) publication.

    Produces ExitIntentValidationResult with hard_blocks and soft_warnings.
    If any hard_block exists, is_valid=False.
    """

    def __init__(
        self,
        idempotency_tracker: Optional[IdempotencyTracker] = None,
    ) -> None:
        self._idem = idempotency_tracker or IdempotencyTracker()

    def validate(self, candidate: ExitIntentCandidate) -> ExitIntentValidationResult:
        """
        Run all validation checks on an ExitIntentCandidate.

        Returns ExitIntentValidationResult. Always succeeds (never raises).
        """
        now = time.time()
        hard_blocks: List[str] = []
        soft_warnings: List[str] = []
        violated: List[str] = []

        # ── 1. Schema validation ─────────────────────────────────────────
        schema_errors = candidate.validate()
        if schema_errors:
            hard_blocks.append("SCHEMA_VALIDATION_FAILED")
            violated.extend(schema_errors)

        # ── 2. Normalize payload ─────────────────────────────────────────
        raw_payload = candidate.to_dict()
        normalized, norm_warnings = normalize_payload(raw_payload)
        soft_warnings.extend(norm_warnings)

        # ── 3. Check required fields ─────────────────────────────────────
        missing = check_required_fields(normalized)
        if missing:
            hard_blocks.append("MISSING_REQUIRED_FIELDS")
            violated.extend([f"missing:{f}" for f in missing])

        # ── 4. Action allowlist ──────────────────────────────────────────
        if candidate.action_name not in VALID_ACTIONS:
            hard_blocks.append("INVALID_ACTION")
            violated.append(f"action={candidate.action_name}")

        # ── 5. Percentage sanity ─────────────────────────────────────────
        expected_frac = ACTION_EXIT_FRACTIONS.get(candidate.action_name, 0.0)
        if abs(candidate.target_reduction_pct - expected_frac) > 0.01:
            soft_warnings.append(
                f"reduction_pct_mismatch: "
                f"expected={expected_frac}, got={candidate.target_reduction_pct}"
            )

        # ── 6. Idempotency check ────────────────────────────────────────
        if candidate.idempotency_key:
            if self._idem.is_duplicate(candidate.idempotency_key):
                hard_blocks.append("DUPLICATE_INTENT")
                violated.append(f"idem_key={candidate.idempotency_key}")

        # ── 7. Source freshness ──────────────────────────────────────────
        if candidate.intent_timestamp > 0:
            age = now - candidate.intent_timestamp
            if age > MAX_INTENT_AGE_SEC:
                soft_warnings.append(f"intent_age={age:.1f}s > {MAX_INTENT_AGE_SEC}s")

        # ── 8. Shadow-only enforcement ───────────────────────────────────
        if not candidate.shadow_only:
            hard_blocks.append(RC.SHADOW_ONLY_VIOLATION)
            violated.append("shadow_only=False")

        # ── 9. Evidence completeness ─────────────────────────────────────
        if not candidate.source_decision_id:
            soft_warnings.append("no_source_decision_id")
        if candidate.confidence <= 0:
            soft_warnings.append("zero_confidence")

        # ── Build result ─────────────────────────────────────────────────
        is_valid = len(hard_blocks) == 0
        confidence = 1.0 if is_valid else 0.0

        return ExitIntentValidationResult(
            position_id=candidate.position_id,
            symbol=candidate.symbol,
            validation_timestamp=now,
            candidate_action=candidate.action_name,
            source_intent_id=candidate.intent_id,
            is_valid=is_valid,
            hard_blocks=hard_blocks,
            soft_warnings=soft_warnings,
            violated_constraints=violated,
            normalized_candidate_payload=normalized,
            validation_confidence=confidence,
            shadow_only=True,
        )
