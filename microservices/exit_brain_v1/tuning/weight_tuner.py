"""
WeightTuner — Proposes fusion weight adjustments for belief/hazard engines.

Phase 5 tuning. Shadow-only. All proposals require human review.

Reads from: OfflineEvaluationSummary
Writes to: TuningRecommendation objects (via ProposalBuilder)

Rules:
  - Max ±20% change per weight per run
  - Weights must remain normalised (sum = 1.0 within each engine)
  - Confidence capped at 0.3 if sample_size < 50
  - Never auto-applies
"""

from __future__ import annotations

import logging
import time
import uuid
from typing import Dict, List

from ..models.offline_evaluation_summary import OfflineEvaluationSummary
from ..models.tuning_recommendation import TuningRecommendation

logger = logging.getLogger(__name__)

# Safety: max ±20% relative change per weight
MAX_CHANGE_FRACTION = 0.20

# Low sample → cap confidence
LOW_SAMPLE_THRESHOLD = 50
LOW_SAMPLE_CONFIDENCE_CAP = 0.30

# Minimum MAE gap between best and worst axis to trigger rebalancing
MIN_MAE_GAP = 0.05

# Default engine weight sets (matches Phase 3 belief/hazard engine defaults)
DEFAULT_BELIEF_WEIGHTS: Dict[str, float] = {
    "exit_pressure": 0.35,
    "hold_conviction": 0.35,
    "directional_edge": 0.15,
    "uncertainty_total": 0.15,
}

DEFAULT_HAZARD_WEIGHTS: Dict[str, float] = {
    "drawdown_hazard": 1 / 6,
    "reversal_hazard": 1 / 6,
    "volatility_hazard": 1 / 6,
    "time_decay_hazard": 1 / 6,
    "regime_hazard": 1 / 6,
    "ensemble_hazard": 1 / 6,
}


class WeightTuner:
    """
    Analyzes per-axis calibration and proposes weight rebalancing.

    Shadow-only. Never modifies live engine weights.
    """

    def propose_adjustments(
        self,
        summary: OfflineEvaluationSummary,
    ) -> List[TuningRecommendation]:
        """
        Identify mis-calibrated axes and propose weight shifts.

        Returns:
            List of TuningRecommendation (possibly empty).
        """
        recommendations: List[TuningRecommendation] = []

        # 1. Hazard weight rebalancing
        recs = self._tune_hazard_weights(summary)
        recommendations.extend(recs)

        # 2. Belief weight rebalancing
        recs = self._tune_belief_weights(summary)
        recommendations.extend(recs)

        logger.info(
            "[WeightTuner] Generated %d weight recommendations from run %s",
            len(recommendations), summary.evaluation_run_id,
        )
        return recommendations

    def _tune_hazard_weights(
        self,
        summary: OfflineEvaluationSummary,
    ) -> List[TuningRecommendation]:
        """
        Propose hazard axis weight changes based on per-axis MAE.

        Logic: Reduce weight for axes with high MAE (low accuracy),
        increase weight for axes with low MAE (high accuracy).
        """
        recs: List[TuningRecommendation] = []

        # Extract per-axis MAE from hazard calibration summary
        axis_mae: Dict[str, float] = {}
        for key, val in summary.hazard_calibration_summary.items():
            if key.endswith("_mae"):
                axis_name = key.replace("_mae", "")
                axis_mae[axis_name] = val

        if len(axis_mae) < 2:
            return recs

        # Check if MAE gap is large enough to warrant rebalancing
        best_mae = min(axis_mae.values())
        worst_mae = max(axis_mae.values())
        if worst_mae - best_mae < MIN_MAE_GAP:
            return recs

        # Propose: shift weight from high-MAE to low-MAE axes
        mean_mae = sum(axis_mae.values()) / len(axis_mae)

        for axis, mae in axis_mae.items():
            weight_key = f"{axis}_hazard"
            current_weight = DEFAULT_HAZARD_WEIGHTS.get(weight_key, 1 / 6)

            if mae > mean_mae + MIN_MAE_GAP:
                # High MAE → reduce weight
                delta = min(
                    (mae - mean_mae) * 0.2,
                    current_weight * MAX_CHANGE_FRACTION,
                )
                proposed = max(0.05, current_weight - delta)
                direction = "decrease"
                rationale = (
                    f"Hazard axis '{axis}' has MAE={mae:.3f} > mean {mean_mae:.3f}. "
                    f"Reducing weight to decrease influence of poorly-calibrated axis."
                )
            elif mae < mean_mae - MIN_MAE_GAP:
                # Low MAE → increase weight
                delta = min(
                    (mean_mae - mae) * 0.2,
                    current_weight * MAX_CHANGE_FRACTION,
                )
                proposed = min(0.50, current_weight + delta)
                direction = "increase"
                rationale = (
                    f"Hazard axis '{axis}' has MAE={mae:.3f} < mean {mean_mae:.3f}. "
                    f"Increasing weight to leverage well-calibrated axis."
                )
            else:
                continue

            recs.append(self._build(
                summary=summary,
                component="hazard_engine",
                param=f"weight_{axis}",
                current=current_weight,
                proposed=proposed,
                direction=direction,
                rationale=rationale,
                metrics={f"{axis}_mae": mae, "mean_axis_mae": mean_mae},
            ))

        return recs

    def _tune_belief_weights(
        self,
        summary: OfflineEvaluationSummary,
    ) -> List[TuningRecommendation]:
        """Propose belief axis weight changes based on per-axis calibration."""
        recs: List[TuningRecommendation] = []

        # Extract per-field MAE from belief calibration summary
        field_mae: Dict[str, float] = {}
        for key, val in summary.belief_calibration_summary.items():
            if key.endswith("_mae"):
                field_name = key.replace("_mae", "")
                field_mae[field_name] = val

        if len(field_mae) < 2:
            return recs

        best_mae = min(field_mae.values())
        worst_mae = max(field_mae.values())
        if worst_mae - best_mae < MIN_MAE_GAP:
            return recs

        mean_mae = sum(field_mae.values()) / len(field_mae)

        for field_name, mae in field_mae.items():
            current_weight = DEFAULT_BELIEF_WEIGHTS.get(field_name, 0.25)

            if mae > mean_mae + MIN_MAE_GAP:
                delta = min(
                    (mae - mean_mae) * 0.2,
                    current_weight * MAX_CHANGE_FRACTION,
                )
                proposed = max(0.05, current_weight - delta)
                direction = "decrease"
                rationale = (
                    f"Belief field '{field_name}' has MAE={mae:.3f} > mean {mean_mae:.3f}. "
                    f"Reducing weight to decrease influence."
                )
            elif mae < mean_mae - MIN_MAE_GAP:
                delta = min(
                    (mean_mae - mae) * 0.2,
                    current_weight * MAX_CHANGE_FRACTION,
                )
                proposed = min(0.60, current_weight + delta)
                direction = "increase"
                rationale = (
                    f"Belief field '{field_name}' has MAE={mae:.3f} < mean {mean_mae:.3f}. "
                    f"Increasing weight to leverage well-calibrated field."
                )
            else:
                continue

            recs.append(self._build(
                summary=summary,
                component="belief_engine",
                param=f"weight_{field_name}",
                current=current_weight,
                proposed=proposed,
                direction=direction,
                rationale=rationale,
                metrics={f"{field_name}_mae": mae, "mean_field_mae": mean_mae},
            ))

        return recs

    def _build(
        self,
        summary: OfflineEvaluationSummary,
        component: str,
        param: str,
        current: float,
        proposed: float,
        direction: str,
        rationale: str,
        metrics: Dict[str, float],
    ) -> TuningRecommendation:
        """Build a TuningRecommendation with safety caps."""
        confidence = min(0.7, abs(proposed - current) / max(abs(current), 1e-10))
        if summary.decisions_covered < LOW_SAMPLE_THRESHOLD:
            confidence = min(confidence, LOW_SAMPLE_CONFIDENCE_CAP)

        risk = "low"
        change_pct = abs(proposed - current) / max(abs(current), 1e-10)
        if change_pct > 0.10:
            risk = "medium"
        if change_pct > MAX_CHANGE_FRACTION:
            risk = "high"

        return TuningRecommendation(
            recommendation_id=str(uuid.uuid4()),
            created_at=time.time(),
            source_evaluation_run_id=summary.evaluation_run_id,
            component_name=component,
            parameter_name=param,
            current_value=current,
            proposed_value=round(proposed, 6),
            direction=direction,
            rationale=rationale,
            supporting_metrics=metrics,
            expected_effect=f"Adjust {component}.{param}: {current:.4f} → {proposed:.4f}",
            confidence=round(confidence, 3),
            risk_of_change=risk,
            requires_human_review=True,
            applied=False,
            shadow_only=True,
        )
