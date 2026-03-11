"""
UtilityRankingEvaluator — Evaluates utility ranking accuracy ex-post.

Phase 5 evaluator. Pure computation.

Reads from: ReplayEvaluationRecord batch (in-memory)
Writes to: Nothing (pure math)
"""

from __future__ import annotations

import logging
from typing import Dict, List

from ..models.replay_evaluation_record import ReplayEvaluationRecord

logger = logging.getLogger(__name__)


class UtilityRankingEvaluator:
    """
    Evaluates whether the utility scoring correctly ranked actions.

    Answers: Was the chosen action actually the best ex-post?
             Was it in the top-3?
             How far off was the ranking?

    Pure math. No IO.
    """

    def evaluate(
        self,
        records: List[ReplayEvaluationRecord],
    ) -> Dict[str, float]:
        """
        Compute utility ranking quality metrics.

        Returns:
            Dict with rank_accuracy, top3_accuracy, mean_rank_displacement, etc.
        """
        if not records:
            return {
                "rank_accuracy": 0.0,
                "top3_accuracy": 0.0,
                "mean_rank_displacement": 0.0,
                "sample_size": 0.0,
            }

        exact_matches = 0
        top3_matches = 0
        rank_displacements: List[float] = []

        for rec in records:
            if not rec.realized_utility_proxy_by_action:
                continue

            # Sort realized utilities to find ex-post ranking
            sorted_realized = sorted(
                rec.realized_utility_proxy_by_action.items(),
                key=lambda x: x[1],
                reverse=True,
            )
            ex_post_ranking = {action: i + 1 for i, (action, _) in enumerate(sorted_realized)}
            chosen_ex_post_rank = ex_post_ranking.get(rec.chosen_action, len(sorted_realized))

            # Exact match: chosen was #1 ex-post
            if chosen_ex_post_rank == 1:
                exact_matches += 1

            # Top-3: chosen was in top 3 ex-post
            if chosen_ex_post_rank <= 3:
                top3_matches += 1

            # Displacement: how far from #1
            rank_displacements.append(float(chosen_ex_post_rank - 1))

        n = len(rank_displacements) if rank_displacements else 1
        return {
            "rank_accuracy": exact_matches / max(n, 1),
            "top3_accuracy": top3_matches / max(n, 1),
            "mean_rank_displacement": sum(rank_displacements) / max(n, 1),
            "sample_size": float(len(records)),
        }

    def evaluate_per_action(
        self,
        records: List[ReplayEvaluationRecord],
    ) -> Dict[str, Dict[str, float]]:
        """
        Per-action ranking quality.

        Returns:
            Dict mapping chosen_action → quality metrics.
        """
        by_action: Dict[str, List[ReplayEvaluationRecord]] = {}
        for rec in records:
            by_action.setdefault(rec.chosen_action, []).append(rec)

        result: Dict[str, Dict[str, float]] = {}
        for action, recs in by_action.items():
            result[action] = self.evaluate(recs)
        return result
