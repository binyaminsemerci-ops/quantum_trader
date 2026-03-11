"""
ProposalBuilder — Orchestrates tuning proposals and publishes results.

Phase 5 tuning. Shadow-only. All proposals require human review.

Reads from: OfflineEvaluationSummary
Writes to: quantum:stream:exit.tuning.recommendation.shadow
"""

from __future__ import annotations

import logging
import time
import uuid
from typing import List, Optional

from ..models.offline_evaluation_summary import OfflineEvaluationSummary
from ..models.tuning_recommendation import TuningRecommendation
from ..models.calibration_artifact import CalibrationArtifact
from .threshold_tuner import ThresholdTuner
from .weight_tuner import WeightTuner

logger = logging.getLogger(__name__)

# Maximum recommendations per run (prevent overtuning)
MAX_RECOMMENDATIONS_PER_RUN = 5


class ProposalBuilder:
    """
    Orchestrates threshold and weight tuners, builds final proposals,
    creates calibration snapshots, and publishes to shadow streams.

    Shadow-only. Never auto-applies.
    """

    def __init__(self, publisher) -> None:
        """
        Args:
            publisher: ShadowPublisher instance.
        """
        self._publisher = publisher
        self._threshold_tuner = ThresholdTuner()
        self._weight_tuner = WeightTuner()

    def build_tuning_proposals(
        self,
        summary: OfflineEvaluationSummary,
    ) -> List[TuningRecommendation]:
        """
        Generate proposals from all tuners, rank, cap, and publish.

        Args:
            summary: Completed OfflineEvaluationSummary.

        Returns:
            List of published TuningRecommendation (max MAX_RECOMMENDATIONS_PER_RUN).
        """
        # 1. Gather raw proposals
        threshold_recs = self._threshold_tuner.propose_adjustments(summary)
        weight_recs = self._weight_tuner.propose_adjustments(summary)
        all_recs = threshold_recs + weight_recs

        if not all_recs:
            logger.info("[ProposalBuilder] No tuning recommendations generated.")
            return []

        # 2. Rank by confidence (highest first)
        all_recs.sort(key=lambda r: r.confidence, reverse=True)

        # 3. Cap to prevent overtuning
        final_recs = all_recs[:MAX_RECOMMENDATIONS_PER_RUN]

        # 4. Publish each recommendation
        for rec in final_recs:
            self.publish_tuning_recommendation(rec)

        logger.info(
            "[ProposalBuilder] Published %d / %d recommendations for run %s",
            len(final_recs), len(all_recs), summary.evaluation_run_id,
        )
        return final_recs

    def build_calibration_snapshot(
        self,
        summary: OfflineEvaluationSummary,
        recommendations: List[TuningRecommendation],
    ) -> CalibrationArtifact:
        """
        Create a CalibrationArtifact capturing current parameter state.

        This snapshot enables rollback and reproducibility.

        Args:
            summary: The evaluation that triggered this snapshot.
            recommendations: Proposals generated in this run.

        Returns:
            CalibrationArtifact with current-state parameters.
        """
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

        params = {
            "UNCERTAINTY_HARD_CEILING": UNCERTAINTY_HARD_CEILING,
            "UNCERTAINTY_SOFT_CEILING": UNCERTAINTY_SOFT_CEILING,
            "DATA_COMPLETENESS_HARD_FLOOR": DATA_COMPLETENESS_HARD_FLOOR,
            "DATA_COMPLETENESS_SOFT_FLOOR": DATA_COMPLETENESS_SOFT_FLOOR,
            "MIN_ACTION_CONVICTION": MIN_ACTION_CONVICTION,
            "PREFER_HOLD_THRESHOLD": PREFER_HOLD_THRESHOLD,
            "EDGE_NEUTRAL_BAND": EDGE_NEUTRAL_BAND,
            "HAZARD_EMERGENCY_THRESHOLD": HAZARD_EMERGENCY_THRESHOLD,
            "CLOSE_FULL_MIN_HAZARD": CLOSE_FULL_MIN_HAZARD,
            "MAX_UPSTREAM_AGE_SEC": MAX_UPSTREAM_AGE_SEC,
        }

        fit_stats = {
            "mean_decision_quality": summary.mean_decision_quality_score,
            "mean_regret": summary.mean_regret_score,
            "num_recommendations": float(len(recommendations)),
        }

        return CalibrationArtifact(
            artifact_id=str(uuid.uuid4()),
            created_at=time.time(),
            source_evaluation_run_id=summary.evaluation_run_id,
            component_name="policy_constraints",
            calibration_type="threshold_snapshot",
            parameters=params,
            fit_statistics=fit_stats,
            sample_size=summary.decisions_covered,
            time_window_start=summary.time_window_start,
            time_window_end=summary.time_window_end,
            shadow_only=True,
        )

    def publish_tuning_recommendation(
        self,
        rec: TuningRecommendation,
    ) -> Optional[str]:
        """Publish a single recommendation to shadow stream."""
        errors = rec.validate()
        if errors:
            logger.warning("[ProposalBuilder] Recommendation validation: %s", errors)
        return self._publisher.publish_tuning_recommendation(rec)
