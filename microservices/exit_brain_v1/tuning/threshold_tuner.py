"""
ThresholdTuner — Proposes policy_constraints threshold adjustments.

Phase 5 tuning. Shadow-only. All proposals require human review.

Reads from: OfflineEvaluationSummary
Writes to: TuningRecommendation objects (via ProposalBuilder)

Rules:
  - Max ±20% change per parameter per run
  - Confidence capped at 0.3 if sample_size < 50
  - Never auto-applies
"""

from __future__ import annotations

import logging
import time
import uuid
from typing import Dict, List, Optional, Tuple

from ..models.offline_evaluation_summary import OfflineEvaluationSummary
from ..models.tuning_recommendation import TuningRecommendation
from ..policy.policy_constraints import (
    UNCERTAINTY_HARD_CEILING,
    UNCERTAINTY_SOFT_CEILING,
    DATA_COMPLETENESS_HARD_FLOOR,
    DATA_COMPLETENESS_SOFT_FLOOR,
    MIN_ACTION_CONVICTION,
    PREFER_HOLD_THRESHOLD,
    EDGE_NEUTRAL_BAND,
    HAZARD_EMERGENCY_THRESHOLD,
    CLOSE_FULL_MIN_HAZARD,
    MAX_UPSTREAM_AGE_SEC,
)

logger = logging.getLogger(__name__)

# Safety: max ±20% change per parameter
MAX_CHANGE_FRACTION = 0.20

# Low sample → cap confidence
LOW_SAMPLE_THRESHOLD = 50
LOW_SAMPLE_CONFIDENCE_CAP = 0.30

# Minimum score delta to trigger a recommendation
MIN_SCORE_DEFICIT = 0.05

# Tuneable thresholds: (param_name, current_value, component)
_TUNEABLE_THRESHOLDS: List[Tuple[str, float, str]] = [
    ("UNCERTAINTY_HARD_CEILING", UNCERTAINTY_HARD_CEILING, "policy_constraints"),
    ("UNCERTAINTY_SOFT_CEILING", UNCERTAINTY_SOFT_CEILING, "policy_constraints"),
    ("DATA_COMPLETENESS_HARD_FLOOR", DATA_COMPLETENESS_HARD_FLOOR, "policy_constraints"),
    ("DATA_COMPLETENESS_SOFT_FLOOR", DATA_COMPLETENESS_SOFT_FLOOR, "policy_constraints"),
    ("MIN_ACTION_CONVICTION", MIN_ACTION_CONVICTION, "policy_constraints"),
    ("PREFER_HOLD_THRESHOLD", PREFER_HOLD_THRESHOLD, "policy_constraints"),
    ("EDGE_NEUTRAL_BAND", EDGE_NEUTRAL_BAND, "policy_constraints"),
    ("HAZARD_EMERGENCY_THRESHOLD", HAZARD_EMERGENCY_THRESHOLD, "policy_constraints"),
    ("CLOSE_FULL_MIN_HAZARD", CLOSE_FULL_MIN_HAZARD, "policy_constraints"),
    ("MAX_UPSTREAM_AGE_SEC", MAX_UPSTREAM_AGE_SEC, "policy_constraints"),
]


class ThresholdTuner:
    """
    Analyzes evaluation summaries and proposes threshold adjustments.

    Shadow-only. Never modifies live policy_constraints.
    """

    def propose_adjustments(
        self,
        summary: OfflineEvaluationSummary,
    ) -> List[TuningRecommendation]:
        """
        Identify underperforming components and propose threshold changes.

        Returns:
            List of TuningRecommendation (possibly empty).
        """
        recommendations: List[TuningRecommendation] = []

        # 1. Uncertainty thresholds — driven by belief calibration
        recs = self._tune_uncertainty_thresholds(summary)
        recommendations.extend(recs)

        # 2. Conviction thresholds — driven by policy choice quality
        recs = self._tune_conviction_thresholds(summary)
        recommendations.extend(recs)

        # 3. Hazard thresholds — driven by hazard calibration
        recs = self._tune_hazard_thresholds(summary)
        recommendations.extend(recs)

        logger.info(
            "[ThresholdTuner] Generated %d recommendations from run %s",
            len(recommendations), summary.evaluation_run_id,
        )
        return recommendations

    def _tune_uncertainty_thresholds(
        self,
        summary: OfflineEvaluationSummary,
    ) -> List[TuningRecommendation]:
        """Adjust uncertainty ceilings based on belief calibration."""
        recs: List[TuningRecommendation] = []

        ep_bias = summary.belief_calibration_summary.get("exit_pressure_bias", 0.0)

        # If exit_pressure is systematically over-predicted (bias > 0),
        # the brain is too conservative → widen the uncertainty ceiling
        if abs(ep_bias) > MIN_SCORE_DEFICIT:
            current = UNCERTAINTY_HARD_CEILING
            direction = "increase" if ep_bias > 0 else "decrease"
            delta = min(abs(ep_bias) * 0.5, current * MAX_CHANGE_FRACTION)
            proposed = current + delta if direction == "increase" else current - delta
            proposed = max(0.30, min(0.95, proposed))

            recs.append(self._build(
                summary=summary,
                param="UNCERTAINTY_HARD_CEILING",
                current=current,
                proposed=proposed,
                direction=direction,
                rationale=(
                    f"exit_pressure bias = {ep_bias:.3f}. "
                    f"{'Over-predicting' if ep_bias > 0 else 'Under-predicting'} exit pressure "
                    f"→ {'widen' if direction == 'increase' else 'tighten'} uncertainty ceiling."
                ),
                metrics={"exit_pressure_bias": ep_bias},
            ))

        return recs

    def _tune_conviction_thresholds(
        self,
        summary: OfflineEvaluationSummary,
    ) -> List[TuningRecommendation]:
        """Adjust conviction thresholds based on policy quality."""
        recs: List[TuningRecommendation] = []

        pass_quality = summary.policy_choice_summary.get("pass_quality", 0.0)
        block_quality = summary.policy_choice_summary.get("block_quality", 0.0)

        # If blocks are low quality (blocking good decisions), lower conviction
        if block_quality < (1.0 - MIN_SCORE_DEFICIT):
            false_block_rate = 1.0 - block_quality
            if false_block_rate > MIN_SCORE_DEFICIT:
                current = MIN_ACTION_CONVICTION
                delta = min(false_block_rate * 0.3, current * MAX_CHANGE_FRACTION)
                proposed = max(0.05, current - delta)

                recs.append(self._build(
                    summary=summary,
                    param="MIN_ACTION_CONVICTION",
                    current=current,
                    proposed=proposed,
                    direction="decrease",
                    rationale=(
                        f"Block quality = {block_quality:.3f}, false block rate = {false_block_rate:.3f}. "
                        f"Policy is blocking too many good decisions → lower conviction threshold."
                    ),
                    metrics={"block_quality": block_quality, "false_block_rate": false_block_rate},
                ))

        # If pass quality is low (passing bad decisions), raise conviction
        if pass_quality < (1.0 - MIN_SCORE_DEFICIT):
            false_pass_rate = 1.0 - pass_quality
            if false_pass_rate > MIN_SCORE_DEFICIT:
                current = MIN_ACTION_CONVICTION
                delta = min(false_pass_rate * 0.3, current * MAX_CHANGE_FRACTION)
                proposed = min(0.50, current + delta)

                recs.append(self._build(
                    summary=summary,
                    param="MIN_ACTION_CONVICTION",
                    current=current,
                    proposed=proposed,
                    direction="increase",
                    rationale=(
                        f"Pass quality = {pass_quality:.3f}, false pass rate = {false_pass_rate:.3f}. "
                        f"Policy is passing too many bad decisions → raise conviction threshold."
                    ),
                    metrics={"pass_quality": pass_quality, "false_pass_rate": false_pass_rate},
                ))

        return recs

    def _tune_hazard_thresholds(
        self,
        summary: OfflineEvaluationSummary,
    ) -> List[TuningRecommendation]:
        """Adjust hazard thresholds based on hazard calibration."""
        recs: List[TuningRecommendation] = []

        haz_bias = summary.hazard_calibration_summary.get("composite_hazard_bias", 0.0)

        if abs(haz_bias) > MIN_SCORE_DEFICIT:
            current = HAZARD_EMERGENCY_THRESHOLD
            direction = "increase" if haz_bias > 0 else "decrease"
            delta = min(abs(haz_bias) * 0.4, current * MAX_CHANGE_FRACTION)
            proposed = current + delta if direction == "increase" else current - delta
            proposed = max(0.50, min(0.95, proposed))

            recs.append(self._build(
                summary=summary,
                param="HAZARD_EMERGENCY_THRESHOLD",
                current=current,
                proposed=proposed,
                direction=direction,
                rationale=(
                    f"composite_hazard bias = {haz_bias:.3f}. "
                    f"{'Over-' if haz_bias > 0 else 'Under-'}predicting hazard "
                    f"→ {direction} emergency threshold."
                ),
                metrics={"composite_hazard_bias": haz_bias},
            ))

        return recs

    def _build(
        self,
        summary: OfflineEvaluationSummary,
        param: str,
        current: float,
        proposed: float,
        direction: str,
        rationale: str,
        metrics: Dict[str, float],
    ) -> TuningRecommendation:
        """Build a TuningRecommendation with safety caps."""
        confidence = min(0.8, abs(proposed - current) / max(abs(current), 1e-10))
        sample = summary.decisions_covered
        if sample < LOW_SAMPLE_THRESHOLD:
            confidence = min(confidence, LOW_SAMPLE_CONFIDENCE_CAP)

        risk = "low"
        if abs(proposed - current) / max(abs(current), 1e-10) > 0.15:
            risk = "medium"
        if abs(proposed - current) / max(abs(current), 1e-10) > MAX_CHANGE_FRACTION:
            risk = "high"

        return TuningRecommendation(
            recommendation_id=str(uuid.uuid4()),
            created_at=time.time(),
            source_evaluation_run_id=summary.evaluation_run_id,
            component_name="policy_constraints",
            parameter_name=param,
            current_value=current,
            proposed_value=round(proposed, 4),
            direction=direction,
            rationale=rationale,
            supporting_metrics=metrics,
            expected_effect=f"Adjust {param}: {current:.4f} → {proposed:.4f}",
            confidence=round(confidence, 3),
            risk_of_change=risk,
            requires_human_review=True,
            applied=False,
            shadow_only=True,
        )
