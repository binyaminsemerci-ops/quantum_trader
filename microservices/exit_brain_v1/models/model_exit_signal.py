"""
ModelExitSignal — Output from one ensemble model for one position.

Phase 2 data contract. Shadow-only. Fail-closed.

Produced by: ensemble_exit_adapter.py
Consumed by: ensemble_aggregator.py
"""

from __future__ import annotations

import time
import logging
from dataclasses import dataclass, field
from typing import Dict, List, Optional

logger = logging.getLogger(__name__)

VALID_MODELS = frozenset({
    "xgboost", "lightgbm", "nhits", "patchtst", "tft", "dlinear",
})

VALID_HORIZONS = frozenset({
    "immediate",   # < 60s
    "short",       # 1-5 min
    "medium",      # 5-30 min
    "long",        # 30+ min
    "unknown",
})


@dataclass(frozen=False)
class ModelExitSignal:
    """
    Normalised exit signal from a single ensemble model.

    Probabilities MUST sum to ~1.0 (tolerance ±0.02).
    shadow_only MUST be True.
    """

    # ── Identity (REQUIRED) ──────────────────────────────────────────────
    position_id: str
    symbol: str
    model_name: str
    model_version: str

    # ── Timing (REQUIRED) ────────────────────────────────────────────────
    inference_timestamp: float
    horizon_label: str

    # ── Raw probabilities (REQUIRED) ─────────────────────────────────────
    sell_probability: float
    hold_probability: float
    buy_probability: float

    # ── Derived directional (REQUIRED) ───────────────────────────────────
    continuation_probability: float
    reversal_probability: float

    # ── Confidence (REQUIRED) ────────────────────────────────────────────
    confidence: float
    freshness_seconds: float

    # ── Expected values (OPTIONAL) ───────────────────────────────────────
    expected_upside_remaining: float = 0.0
    expected_downside_if_hold: float = 0.0
    expected_drawdown_risk: float = 0.0

    # ── Uncertainty / Calibration (OPTIONAL) ─────────────────────────────
    uncertainty: Optional[float] = None
    calibration_score: Optional[float] = None

    # ── Provenance (OPTIONAL) ────────────────────────────────────────────
    feature_version: str = "v1"
    source_state_version: str = ""
    source_quality_flags: List[str] = field(default_factory=list)

    # ── Safety (REQUIRED) ────────────────────────────────────────────────
    shadow_only: bool = True

    def __post_init__(self) -> None:
        if self.uncertainty is None:
            self.uncertainty = 1.0 - self.confidence

    # ── Validation ───────────────────────────────────────────────────────

    def validate(self) -> List[str]:
        """Return list of error strings. Empty = valid."""
        errors: List[str] = []

        if not self.position_id:
            errors.append("position_id is empty")
        if not self.symbol:
            errors.append("symbol is empty")
        if self.model_name not in VALID_MODELS:
            errors.append(f"model_name '{self.model_name}' not in {VALID_MODELS}")
        if not self.model_version:
            errors.append("model_version is empty")
        if self.inference_timestamp <= 0:
            errors.append(f"inference_timestamp must be > 0, got {self.inference_timestamp}")
        if self.horizon_label not in VALID_HORIZONS:
            errors.append(f"horizon_label '{self.horizon_label}' not in {VALID_HORIZONS}")

        # Probability range checks
        for name, val in [
            ("sell_probability", self.sell_probability),
            ("hold_probability", self.hold_probability),
            ("buy_probability", self.buy_probability),
            ("continuation_probability", self.continuation_probability),
            ("reversal_probability", self.reversal_probability),
            ("confidence", self.confidence),
        ]:
            if not (0.0 <= val <= 1.0):
                errors.append(f"{name} must be in [0, 1], got {val}")

        # Proba sum check
        proba_sum = self.sell_probability + self.hold_probability + self.buy_probability
        if abs(proba_sum - 1.0) > 0.02:
            errors.append(f"probabilities sum to {proba_sum:.4f}, expected ~1.0")

        if self.freshness_seconds < 0:
            errors.append(f"freshness_seconds must be >= 0, got {self.freshness_seconds}")

        if not self.shadow_only:
            errors.append("shadow_only MUST be True in Phase 2")

        return errors

    def to_dict(self) -> Dict:
        """Serialize to flat dict for Redis XADD."""
        return {
            "position_id": self.position_id,
            "symbol": self.symbol,
            "model_name": self.model_name,
            "model_version": self.model_version,
            "inference_timestamp": self.inference_timestamp,
            "horizon_label": self.horizon_label,
            "sell_probability": self.sell_probability,
            "hold_probability": self.hold_probability,
            "buy_probability": self.buy_probability,
            "continuation_probability": self.continuation_probability,
            "reversal_probability": self.reversal_probability,
            "expected_upside_remaining": self.expected_upside_remaining,
            "expected_downside_if_hold": self.expected_downside_if_hold,
            "expected_drawdown_risk": self.expected_drawdown_risk,
            "confidence": self.confidence,
            "uncertainty": self.uncertainty,
            "calibration_score": self.calibration_score,
            "freshness_seconds": self.freshness_seconds,
            "feature_version": self.feature_version,
            "source_state_version": self.source_state_version,
            "source_quality_flags": ",".join(self.source_quality_flags),
            "shadow_only": self.shadow_only,
        }
