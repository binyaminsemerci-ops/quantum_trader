"""
AggregatedExitSignal — Combined output from ensemble aggregator.

Phase 2 data contract. Shadow-only. Fail-closed when < 2 models.

Produced by: ensemble_aggregator.py
Consumed by: (future) belief_engine, hazard_engine, utility_engine
Published to: quantum:stream:exit.ensemble.agg.shadow
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Dict, List, Optional

logger = logging.getLogger(__name__)

MIN_MODELS_REQUIRED = 2


@dataclass(frozen=False)
class AggregatedExitSignal:
    """
    Aggregated exit signal from multiple ensemble models.

    shadow_only MUST be True.
    """

    # ── Identity (REQUIRED) ──────────────────────────────────────────────
    position_id: str
    symbol: str
    ensemble_timestamp: float

    # ── Model coverage (REQUIRED) ────────────────────────────────────────
    participating_models: List[str]
    missing_models: List[str]
    stale_models: List[str]
    model_count_total: int
    model_count_used: int

    # ── Aggregated probabilities (REQUIRED) ──────────────────────────────
    sell_probability_agg: float
    hold_probability_agg: float
    buy_probability_agg: float
    continuation_probability_agg: float
    reversal_probability_agg: float

    # ── Aggregated confidence (REQUIRED) ─────────────────────────────────
    confidence_agg: float
    uncertainty_agg: float
    disagreement_score: float
    consensus_strength: float
    reliability_score: float

    # ── Aggregation meta (REQUIRED) ──────────────────────────────────────
    aggregation_method: str
    quality_flags: List[str] = field(default_factory=list)
    shadow_only: bool = True

    # ── Expected values (OPTIONAL) ───────────────────────────────────────
    expected_upside_remaining_agg: float = 0.0
    expected_downside_if_hold_agg: float = 0.0
    expected_drawdown_risk_agg: float = 0.0

    # ── Validation ───────────────────────────────────────────────────────

    def validate(self) -> List[str]:
        """Return list of error strings. Empty = valid."""
        errors: List[str] = []

        if not self.position_id:
            errors.append("position_id is empty")
        if not self.symbol:
            errors.append("symbol is empty")
        if self.ensemble_timestamp <= 0:
            errors.append(f"ensemble_timestamp must be > 0, got {self.ensemble_timestamp}")

        if self.model_count_used < MIN_MODELS_REQUIRED:
            errors.append(
                f"model_count_used={self.model_count_used} < minimum {MIN_MODELS_REQUIRED}"
            )

        for name, val in [
            ("sell_probability_agg", self.sell_probability_agg),
            ("hold_probability_agg", self.hold_probability_agg),
            ("buy_probability_agg", self.buy_probability_agg),
            ("continuation_probability_agg", self.continuation_probability_agg),
            ("reversal_probability_agg", self.reversal_probability_agg),
            ("confidence_agg", self.confidence_agg),
            ("uncertainty_agg", self.uncertainty_agg),
            ("disagreement_score", self.disagreement_score),
            ("consensus_strength", self.consensus_strength),
            ("reliability_score", self.reliability_score),
        ]:
            if not (0.0 <= val <= 1.0):
                errors.append(f"{name} must be in [0, 1], got {val}")

        proba_sum = self.sell_probability_agg + self.hold_probability_agg + self.buy_probability_agg
        if abs(proba_sum - 1.0) > 0.02:
            errors.append(f"aggregated probabilities sum to {proba_sum:.4f}, expected ~1.0")

        if self.model_count_total <= 0:
            errors.append(f"model_count_total must be > 0, got {self.model_count_total}")

        if not self.aggregation_method:
            errors.append("aggregation_method is empty")

        if not self.shadow_only:
            errors.append("shadow_only MUST be True in Phase 2")

        return errors

    def to_dict(self) -> Dict:
        """Serialize to flat dict for Redis XADD."""
        return {
            "position_id": self.position_id,
            "symbol": self.symbol,
            "ensemble_timestamp": self.ensemble_timestamp,
            "participating_models": ",".join(self.participating_models),
            "missing_models": ",".join(self.missing_models),
            "stale_models": ",".join(self.stale_models),
            "model_count_total": self.model_count_total,
            "model_count_used": self.model_count_used,
            "sell_probability_agg": self.sell_probability_agg,
            "hold_probability_agg": self.hold_probability_agg,
            "buy_probability_agg": self.buy_probability_agg,
            "continuation_probability_agg": self.continuation_probability_agg,
            "reversal_probability_agg": self.reversal_probability_agg,
            "expected_upside_remaining_agg": self.expected_upside_remaining_agg,
            "expected_downside_if_hold_agg": self.expected_downside_if_hold_agg,
            "expected_drawdown_risk_agg": self.expected_drawdown_risk_agg,
            "confidence_agg": self.confidence_agg,
            "uncertainty_agg": self.uncertainty_agg,
            "disagreement_score": self.disagreement_score,
            "consensus_strength": self.consensus_strength,
            "reliability_score": self.reliability_score,
            "aggregation_method": self.aggregation_method,
            "quality_flags": ",".join(self.quality_flags),
            "shadow_only": self.shadow_only,
        }


@dataclass
class EnsembleDiagnostics:
    """Debug/audit diagnostics for one ensemble evaluation cycle."""

    position_id: str
    symbol: str
    timestamp: float
    per_model_signals: Dict[str, Dict]     # model_name → ModelExitSignal.to_dict()
    per_model_weights: Dict[str, float]    # model_name → final normalised weight
    per_model_freshness: Dict[str, float]  # model_name → seconds since inference
    per_model_status: Dict[str, str]       # model_name → OK / STALE / MISSING / ERROR
    probability_spread: Dict[str, float]   # sell_std, hold_std, buy_std
    max_disagreement_pair: str             # e.g. "xgboost vs dlinear"
    aggregation_config: Dict               # thresholds, method, etc.
    calibration_metadata: Dict = field(default_factory=dict)

    def to_dict(self) -> Dict:
        """Serialize for Redis. Nested dicts are JSON-encoded by publisher."""
        import json
        return {
            "position_id": self.position_id,
            "symbol": self.symbol,
            "timestamp": self.timestamp,
            "per_model_signals": json.dumps(self.per_model_signals),
            "per_model_weights": json.dumps(self.per_model_weights),
            "per_model_freshness": json.dumps(self.per_model_freshness),
            "per_model_status": json.dumps(self.per_model_status),
            "probability_spread": json.dumps(self.probability_spread),
            "max_disagreement_pair": self.max_disagreement_pair,
            "aggregation_config": json.dumps(self.aggregation_config),
            "calibration_metadata": json.dumps(self.calibration_metadata),
        }
