"""
ExitIntentCandidate — Shadow exit intent for one position.

Phase 4 data contract. Shadow-only. No execution writes.

Produced by: exit_agent_orchestrator.py
Consumed by: exit_intent_gateway_validator.py
Published to: quantum:stream:exit.intent.candidate.shadow
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Dict, List, Optional

from .action_candidate import VALID_ACTIONS

logger = logging.getLogger(__name__)


@dataclass(frozen=False)
class ExitIntentCandidate:
    """
    Candidate exit intent produced by the orchestrator.

    Represents what *would* be sent to ExitIntentGateway
    if this were in enforced mode. In Phase 4: shadow only.

    shadow_only MUST be True.
    """

    # ── Identity (REQUIRED) ──────────────────────────────────────────────
    position_id: str
    symbol: str
    intent_id: str                          # uuid4
    intent_timestamp: float                 # Epoch seconds

    # ── Action (REQUIRED) ────────────────────────────────────────────────
    action_name: str                        # One of VALID_ACTIONS
    intent_type: str                        # "SHADOW_EXIT"

    # ── Parameters (OPTIONAL, action-dependent) ──────────────────────────
    target_reduction_pct: float = 0.0       # [0, 1] for REDUCE/TAKE_PROFIT/CLOSE
    tighten_parameters: Dict = field(default_factory=dict)  # For TIGHTEN_EXIT

    # ── Justification (REQUIRED) ─────────────────────────────────────────
    justification_summary: str = ""         # Human-readable
    source_decision_id: str = ""            # Links to PolicyDecision.decision_id

    # ── Confidence (REQUIRED) ────────────────────────────────────────────
    confidence: float = 0.0                 # [0, 1]
    uncertainty: float = 1.0               # [0, 1]

    # ── Flags (OPTIONAL) ─────────────────────────────────────────────────
    constraint_flags: List[str] = field(default_factory=list)
    quality_flags: List[str] = field(default_factory=list)

    # ── Idempotency (REQUIRED) ───────────────────────────────────────────
    idempotency_key: str = ""               # Deterministic hash

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
        if not self.intent_id:
            errors.append("intent_id is empty")
        if self.intent_timestamp <= 0:
            errors.append(f"intent_timestamp must be > 0, got {self.intent_timestamp}")

        if self.action_name not in VALID_ACTIONS:
            errors.append(f"action_name '{self.action_name}' not in {VALID_ACTIONS}")
        if self.intent_type != "SHADOW_EXIT":
            errors.append(f"intent_type must be 'SHADOW_EXIT', got '{self.intent_type}'")

        if not (0.0 <= self.target_reduction_pct <= 1.0):
            errors.append(
                f"target_reduction_pct must be in [0, 1], got {self.target_reduction_pct}"
            )

        for name, val in [
            ("confidence", self.confidence),
            ("uncertainty", self.uncertainty),
        ]:
            if not (0.0 <= val <= 1.0):
                errors.append(f"{name} must be in [0, 1], got {val}")

        if not self.source_decision_id:
            errors.append("source_decision_id is empty")
        if not self.idempotency_key:
            errors.append("idempotency_key is empty")

        if not self.shadow_only:
            errors.append("shadow_only MUST be True in Phase 4")

        return errors

    def to_dict(self) -> Dict:
        """Serialize to flat dict for Redis XADD."""
        import json
        return {
            "position_id": self.position_id,
            "symbol": self.symbol,
            "intent_id": self.intent_id,
            "intent_timestamp": self.intent_timestamp,
            "action_name": self.action_name,
            "intent_type": self.intent_type,
            "target_reduction_pct": self.target_reduction_pct,
            "tighten_parameters": json.dumps(self.tighten_parameters),
            "justification_summary": self.justification_summary,
            "source_decision_id": self.source_decision_id,
            "confidence": self.confidence,
            "uncertainty": self.uncertainty,
            "constraint_flags": json.dumps(self.constraint_flags),
            "quality_flags": ",".join(self.quality_flags),
            "idempotency_key": self.idempotency_key,
            "shadow_only": self.shadow_only,
        }
