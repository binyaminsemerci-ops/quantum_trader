"""
CalibrationArtifact — Fitted calibration parameters from offline evaluation.

Phase 5 data contract. Shadow-only. Offline.

Produced by: weight_tuner.py / threshold_tuner.py
Stored in: quantum:kv:exit:calibration:artifacts (v2+)
"""

from __future__ import annotations

import logging
import uuid
from dataclasses import dataclass, field
from typing import Dict, List

logger = logging.getLogger(__name__)

VALID_CALIBRATION_TYPES = frozenset({
    "threshold_snapshot",
    "weight_snapshot",
    "temperature_scaling",
    "platt_scaling",
})


@dataclass(frozen=False)
class CalibrationArtifact:
    """
    Snapshot of calibration parameters after an evaluation run.

    Immutable once created. Used for reproducibility and rollback.

    shadow_only MUST be True.
    """

    # ── Identity (REQUIRED) ──────────────────────────────────────────────
    artifact_id: str                           # uuid4
    created_at: float                          # Epoch seconds
    source_evaluation_run_id: str              # Links to OfflineEvaluationSummary

    # ── Content (REQUIRED) ───────────────────────────────────────────────
    component_name: str                        # e.g. "policy_constraints", "belief_engine"
    calibration_type: str                      # One of VALID_CALIBRATION_TYPES
    parameters: Dict[str, float] = field(default_factory=dict)
    # e.g. {"UNCERTAINTY_HARD_CEILING": 0.70, "UNCERTAINTY_SOFT_CEILING": 0.50}

    # ── Fit statistics (REQUIRED) ────────────────────────────────────────
    fit_statistics: Dict[str, float] = field(default_factory=dict)
    # e.g. {"log_loss": 0.45, "brier_score": 0.18}

    # ── Scope (REQUIRED) ─────────────────────────────────────────────────
    sample_size: int = 0
    time_window_start: float = 0.0
    time_window_end: float = 0.0

    # ── Quality (OPTIONAL) ───────────────────────────────────────────────
    quality_flags: List[str] = field(default_factory=list)

    # ── Safety (REQUIRED) ────────────────────────────────────────────────
    shadow_only: bool = True

    # ── Validation ───────────────────────────────────────────────────────

    def validate(self) -> List[str]:
        """Return list of error strings. Empty = valid."""
        errors: List[str] = []

        if not self.artifact_id:
            errors.append("artifact_id is empty")
        if self.created_at <= 0:
            errors.append(f"created_at must be > 0, got {self.created_at}")
        if not self.source_evaluation_run_id:
            errors.append("source_evaluation_run_id is empty")
        if not self.component_name:
            errors.append("component_name is empty")

        if self.calibration_type not in VALID_CALIBRATION_TYPES:
            errors.append(
                f"calibration_type '{self.calibration_type}' not in {VALID_CALIBRATION_TYPES}"
            )

        if self.sample_size < 0:
            errors.append(f"sample_size must be >= 0, got {self.sample_size}")
        if self.time_window_start < 0:
            errors.append(f"time_window_start must be >= 0, got {self.time_window_start}")

        if not self.shadow_only:
            errors.append("shadow_only MUST be True in Phase 5")

        return errors

    def to_dict(self) -> Dict:
        """Serialize to flat dict for Redis XADD or KV store."""
        import json
        return {
            "artifact_id": self.artifact_id,
            "created_at": self.created_at,
            "source_evaluation_run_id": self.source_evaluation_run_id,
            "component_name": self.component_name,
            "calibration_type": self.calibration_type,
            "parameters": json.dumps(self.parameters),
            "fit_statistics": json.dumps(self.fit_statistics),
            "sample_size": self.sample_size,
            "time_window_start": self.time_window_start,
            "time_window_end": self.time_window_end,
            "quality_flags": ",".join(self.quality_flags),
            "shadow_only": self.shadow_only,
        }
