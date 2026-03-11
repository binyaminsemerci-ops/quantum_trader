"""
OfflineEvaluator — Orchestrator for full offline evaluation pipeline.

Phase 5 evaluator. Shadow-only.

Reads from: TradeExitObituary stream, upstream outcomes
Writes to: quantum:stream:exit.replay.eval.shadow,
           quantum:stream:exit.eval.summary.shadow
"""

from __future__ import annotations

import logging
import time
import uuid
from typing import Any, Dict, List, Optional

from ..models.trade_exit_obituary import TradeExitObituary
from ..models.replay_evaluation_record import ReplayEvaluationRecord
from ..models.offline_evaluation_summary import OfflineEvaluationSummary
from ..models.action_candidate import VALID_ACTIONS
from ..replay.replay_loader import ReplayLoader
from ..replay.outcome_reconstructor import OutcomeReconstructor, OutcomePathResult
from ..replay.counterfactual_evaluator import CounterfactualEvaluator
from .belief_calibration_evaluator import BeliefCalibrationEvaluator
from .hazard_calibration_evaluator import HazardCalibrationEvaluator
from .utility_ranking_evaluator import UtilityRankingEvaluator
from .policy_choice_evaluator import PolicyChoiceEvaluator, BASELINE_DEFINITIONS

logger = logging.getLogger(__name__)

# Minimum samples for valid evaluation
MIN_SAMPLES_FOR_EVAL = 5
LOW_SAMPLE_THRESHOLD = 30

# Default horizon
DEFAULT_HORIZON_SEC = 14400.0

# Stream for reading obituaries
STREAM_OBITUARY = "quantum:stream:exit.obituary.shadow"

_EPS = 1e-10


class OfflineEvaluator:
    """
    Runs the complete offline evaluation pipeline:
    1. Load obituaries for a time window
    2. Build ReplayEvaluationRecord per obituary
    3. Run sub-evaluators (belief, hazard, utility, policy)
    4. Compare against baselines
    5. Build OfflineEvaluationSummary
    6. Publish results to shadow streams

    Shadow-only. Never writes to execution paths.
    """

    def __init__(
        self,
        redis_client,
        publisher,
        horizon_seconds: float = DEFAULT_HORIZON_SEC,
    ) -> None:
        self._r = redis_client
        self._publisher = publisher
        self._loader = ReplayLoader(redis_client)
        self._reconstructor = OutcomeReconstructor(redis_client)
        self._counterfactual = CounterfactualEvaluator()
        self._belief_eval = BeliefCalibrationEvaluator()
        self._hazard_eval = HazardCalibrationEvaluator()
        self._utility_eval = UtilityRankingEvaluator()
        self._policy_eval = PolicyChoiceEvaluator()
        self._horizon = horizon_seconds

    def run_evaluation(
        self,
        start_ts: float,
        end_ts: float,
    ) -> Optional[OfflineEvaluationSummary]:
        """
        Run full offline evaluation for a time window.

        Args:
            start_ts: Window start (epoch).
            end_ts: Window end (epoch).

        Returns:
            OfflineEvaluationSummary, or None if insufficient data.
        """
        logger.info("[OfflineEvaluator] Starting evaluation for window %.0f → %.0f", start_ts, end_ts)

        # 1. Load obituaries
        obituaries = self.load_replay_records(start_ts, end_ts)
        if len(obituaries) < MIN_SAMPLES_FOR_EVAL:
            logger.warning(
                "[OfflineEvaluator] Only %d obituaries, minimum %d required",
                len(obituaries), MIN_SAMPLES_FOR_EVAL,
            )
            return None

        # 2. Reconstruct outcomes per position
        outcomes: Dict[str, OutcomePathResult] = {}
        for obit in obituaries:
            if obit.position_id in outcomes:
                continue
            entry_price = self._infer_entry_price(obit)
            side = self._infer_side(obit)
            outcome = self._reconstructor.reconstruct(
                symbol=obit.symbol,
                side=side,
                entry_price=entry_price,
                decision_timestamp=obit.obituary_timestamp - self._horizon,
                horizon_seconds=self._horizon,
            )
            outcomes[obit.position_id] = outcome

        # 3. Build replay evaluation records
        replay_records = self._build_replay_records(obituaries, outcomes)

        # 4. Publish replay records
        for rec in replay_records:
            self._publisher.publish_replay_evaluation(rec)

        # 5. Run sub-evaluators
        belief_cal = self.evaluate_belief_calibration(obituaries, outcomes)
        hazard_cal = self.evaluate_hazard_calibration(obituaries, outcomes)
        utility_rank = self.evaluate_utility_ranking(replay_records)
        policy_quality = self.evaluate_policy_quality(obituaries)
        baseline_cmp = self.compare_against_baselines(obituaries, outcomes)

        # 6. Build summary
        summary = self.build_offline_evaluation_summary(
            obituaries=obituaries,
            replay_records=replay_records,
            belief_cal=belief_cal,
            hazard_cal=hazard_cal,
            utility_rank=utility_rank,
            policy_quality=policy_quality,
            baseline_cmp=baseline_cmp,
            start_ts=start_ts,
            end_ts=end_ts,
        )

        # 7. Publish summary
        self.publish_evaluation_summary(summary)

        logger.info(
            "[OfflineEvaluator] Evaluation complete: %d obituaries, quality=%.3f",
            len(obituaries), summary.mean_decision_quality_score,
        )
        return summary

    def load_replay_records(
        self,
        start_ts: float,
        end_ts: float,
    ) -> List[TradeExitObituary]:
        """
        Load obituaries from shadow stream.

        Returns:
            List of deserialized TradeExitObituary objects.
        """
        raw_entries = self._read_obituary_stream(start_ts, end_ts)
        obituaries: List[TradeExitObituary] = []

        for entry in raw_entries:
            try:
                obit = self._deserialize_obituary(entry)
                if obit is not None:
                    obituaries.append(obit)
            except Exception as e:
                logger.warning("[OfflineEvaluator] Failed to deserialize obituary: %s", e)
                continue

        return obituaries

    def evaluate_belief_calibration(
        self,
        obituaries: List[TradeExitObituary],
        outcomes: Dict[str, OutcomePathResult],
    ) -> Dict[str, float]:
        """Delegate to BeliefCalibrationEvaluator."""
        return self._belief_eval.evaluate(obituaries, outcomes)

    def evaluate_hazard_calibration(
        self,
        obituaries: List[TradeExitObituary],
        outcomes: Dict[str, OutcomePathResult],
    ) -> Dict[str, float]:
        """Delegate to HazardCalibrationEvaluator."""
        return self._hazard_eval.evaluate(obituaries, outcomes)

    def evaluate_utility_ranking(
        self,
        records: List[ReplayEvaluationRecord],
    ) -> Dict[str, float]:
        """Delegate to UtilityRankingEvaluator."""
        return self._utility_eval.evaluate(records)

    def evaluate_policy_quality(
        self,
        obituaries: List[TradeExitObituary],
    ) -> Dict[str, float]:
        """Delegate to PolicyChoiceEvaluator."""
        return self._policy_eval.evaluate(obituaries)

    def compare_against_baselines(
        self,
        obituaries: List[TradeExitObituary],
        outcomes: Dict[str, OutcomePathResult],
    ) -> Dict[str, Dict[str, float]]:
        """Delegate to PolicyChoiceEvaluator baseline comparison."""
        return self._policy_eval.compare_against_baselines(obituaries, outcomes)

    def build_offline_evaluation_summary(
        self,
        obituaries: List[TradeExitObituary],
        replay_records: List[ReplayEvaluationRecord],
        belief_cal: Dict[str, float],
        hazard_cal: Dict[str, float],
        utility_rank: Dict[str, float],
        policy_quality: Dict[str, float],
        baseline_cmp: Dict[str, Dict[str, float]],
        start_ts: float,
        end_ts: float,
    ) -> OfflineEvaluationSummary:
        """Build the aggregated summary from sub-evaluator results."""
        # Collect unique symbols and positions
        symbols = sorted(set(o.symbol for o in obituaries))
        positions = set(o.position_id for o in obituaries)

        # Action distribution
        action_dist: Dict[str, int] = {}
        for obit in obituaries:
            action_dist[obit.recommended_action_at_decision] = (
                action_dist.get(obit.recommended_action_at_decision, 0) + 1
            )

        # Aggregate scores
        regret_scores = [o.regret_score for o in obituaries]
        preservation_scores = [o.preservation_score for o in obituaries]
        opportunity_scores = [o.opportunity_capture_score for o in obituaries]
        dq_scores = [r.decision_quality_score for r in replay_records]

        # Warnings
        warnings: List[str] = []
        if len(obituaries) < LOW_SAMPLE_THRESHOLD:
            warnings.append(f"LOW_SAMPLE_SIZE: {len(obituaries)} < {LOW_SAMPLE_THRESHOLD}")

        return OfflineEvaluationSummary(
            evaluation_run_id=str(uuid.uuid4()),
            run_timestamp=time.time(),
            time_window_start=start_ts,
            time_window_end=end_ts,
            symbols_covered=symbols,
            positions_covered=len(positions),
            decisions_covered=len(replay_records),
            obituaries_covered=len(obituaries),
            baseline_definitions=dict(BASELINE_DEFINITIONS),
            action_distribution=action_dist,
            mean_decision_quality_score=self._mean(dq_scores),
            median_decision_quality_score=self._median(dq_scores),
            mean_regret_score=self._mean(regret_scores),
            mean_preservation_score=self._mean(preservation_scores),
            mean_opportunity_capture_score=self._mean(opportunity_scores),
            belief_calibration_summary=belief_cal,
            hazard_calibration_summary=hazard_cal,
            utility_ranking_summary=utility_rank,
            policy_choice_summary=policy_quality,
            baseline_comparison=baseline_cmp,
            sample_size_warnings=warnings,
        )

    def publish_evaluation_summary(
        self,
        summary: OfflineEvaluationSummary,
    ) -> Optional[str]:
        """Publish to shadow stream."""
        errors = summary.validate()
        if errors:
            logger.warning("[OfflineEvaluator] Summary validation: %s", errors)
        return self._publisher.publish_evaluation_summary(summary)

    # ── Private ──────────────────────────────────────────────────────────

    def _build_replay_records(
        self,
        obituaries: List[TradeExitObituary],
        outcomes: Dict[str, OutcomePathResult],
    ) -> List[ReplayEvaluationRecord]:
        """Build per-decision evaluation records."""
        records: List[ReplayEvaluationRecord] = []

        for obit in obituaries:
            outcome = outcomes.get(obit.position_id)
            if not outcome:
                continue

            entry_price = self._infer_entry_price(obit)
            side = self._infer_side(obit)

            # Counterfactual analysis
            realized_by_action = self._counterfactual.evaluate_all_actions(
                outcome, entry_price, side, quantity=1.0,
            )
            best_action, best_utility = self._counterfactual.find_ex_post_best_action(
                realized_by_action
            )

            # Predicted values from snapshots
            pred_ep = obit.belief_snapshot.get("exit_pressure", 0.0)
            pred_hc = obit.belief_snapshot.get("hold_conviction", 0.0)
            pred_hazard = obit.hazard_snapshot.get("composite_hazard", 0.0)

            # Realized proxies
            real_ep = self._belief_eval._compute_realized_exit_pressure(outcome)
            real_hc = 1.0 - real_ep
            real_hazard = self._hazard_eval._compute_realized_composite_hazard(outcome)

            # Predicted utility per action from utility_snapshot
            pred_util: Dict[str, float] = {}
            for cand in obit.utility_snapshot:
                if isinstance(cand, dict):
                    pred_util[cand.get("action", "")] = float(cand.get("net_utility", 0))

            # Scores
            dq_score = self._counterfactual.compute_decision_quality_score(
                obit.recommended_action_at_decision,
                realized_by_action,
                obit.regret_score,
                obit.preservation_score,
                obit.opportunity_capture_score,
            )
            consistency = self._counterfactual.compute_explanation_consistency_score(
                pred_ep, real_ep, pred_hazard, real_hazard,
            )

            rec = ReplayEvaluationRecord(
                record_id=str(uuid.uuid4()),
                position_id=obit.position_id,
                symbol=obit.symbol,
                replay_timestamp=time.time(),
                source_decision_timestamp=obit.obituary_timestamp,
                evaluated_horizon_seconds=obit.evaluation_horizon_seconds,
                chosen_action=obit.recommended_action_at_decision or "HOLD",
                actual_action=obit.actual_exit_action,
                action_rank_at_decision=1,  # from top of utility_snapshot
                counterfactual_actions_evaluated=sorted(VALID_ACTIONS),
                predicted_exit_pressure=pred_ep,
                realized_exit_pressure_proxy=real_ep,
                predicted_hold_conviction=pred_hc,
                realized_hold_conviction_proxy=real_hc,
                predicted_composite_hazard=pred_hazard,
                realized_hazard_proxy=real_hazard,
                predicted_utility_by_action=pred_util,
                realized_utility_proxy_by_action=realized_by_action,
                ex_post_best_action=best_action,
                ex_post_best_utility_proxy=best_utility,
                decision_quality_score=dq_score,
                explanation_consistency_score=consistency,
                calibration_errors={
                    "exit_pressure": pred_ep - real_ep,
                    "hold_conviction": pred_hc - real_hc,
                    "composite_hazard": pred_hazard - real_hazard,
                },
                quality_flags=list(obit.quality_flags),
            )
            records.append(rec)

        return records

    def _read_obituary_stream(
        self,
        start_ts: float,
        end_ts: float,
    ) -> List[Dict[str, Any]]:
        """Read obituary entries from Redis stream."""
        try:
            start_id = f"{int(start_ts * 1000)}-0"
            end_id = f"{int(end_ts * 1000)}-18446744073709551615"
            raw = self._r.xrange(STREAM_OBITUARY, min=start_id, max=end_id, count=5000)
            return [self._decode_entry(fields) for _, fields in raw]
        except Exception as e:
            logger.error("[OfflineEvaluator] Failed to read obituary stream: %s", e)
            return []

    @staticmethod
    def _decode_entry(fields: Dict) -> Dict[str, Any]:
        """Decode Redis bytes to strings."""
        import json
        decoded: Dict[str, Any] = {}
        for k, v in fields.items():
            key = k.decode("utf-8") if isinstance(k, bytes) else str(k)
            val = v.decode("utf-8") if isinstance(v, bytes) else str(v)
            if val.startswith("{") or val.startswith("["):
                try:
                    decoded[key] = json.loads(val)
                except (json.JSONDecodeError, ValueError):
                    decoded[key] = val
            else:
                decoded[key] = val
        return decoded

    @staticmethod
    def _deserialize_obituary(data: Dict[str, Any]) -> Optional[TradeExitObituary]:
        """Reconstruct TradeExitObituary from Redis entry dict."""
        try:
            return TradeExitObituary(
                position_id=str(data.get("position_id", "")),
                symbol=str(data.get("symbol", "")),
                obituary_id=str(data.get("obituary_id", "")),
                obituary_timestamp=float(data.get("obituary_timestamp", 0)),
                open_timestamp=float(data.get("open_timestamp", 0)),
                lifecycle_duration_seconds=float(data.get("lifecycle_duration_seconds", 0)),
                close_timestamp=float(data.get("close_timestamp", 0)),
                actual_exit_action=str(data.get("actual_exit_action", "")),
                actual_exit_timestamp=float(data.get("actual_exit_timestamp", 0)),
                actual_realized_pnl=float(data.get("actual_realized_pnl", 0)),
                actual_realized_pnl_pct=float(data.get("actual_realized_pnl_pct", 0)),
                peak_unrealized_pnl=float(data.get("peak_unrealized_pnl", 0)),
                peak_unrealized_pnl_pct=float(data.get("peak_unrealized_pnl_pct", 0)),
                max_drawdown_after_peak=float(data.get("max_drawdown_after_peak", 0)),
                recommended_action_at_decision=str(data.get("recommended_action_at_decision", "")),
                recommended_action_utility=float(data.get("recommended_action_utility", 0)),
                recommended_action_confidence=float(data.get("recommended_action_confidence", 0)),
                policy_passed=str(data.get("policy_passed", "False")).lower() in ("true", "1"),
                policy_reason_codes=data.get("policy_reason_codes", []) if isinstance(data.get("policy_reason_codes"), list) else [],
                belief_snapshot=data.get("belief_snapshot", {}) if isinstance(data.get("belief_snapshot"), dict) else {},
                hazard_snapshot=data.get("hazard_snapshot", {}) if isinstance(data.get("hazard_snapshot"), dict) else {},
                utility_snapshot=data.get("utility_snapshot", []) if isinstance(data.get("utility_snapshot"), list) else [],
                decision_trace_ref=str(data.get("decision_trace_ref", "")),
                source_decision_id=str(data.get("source_decision_id", "")),
                evaluation_horizon_seconds=float(data.get("evaluation_horizon_seconds", DEFAULT_HORIZON_SEC)),
                regret_score=float(data.get("regret_score", 0)),
                preservation_score=float(data.get("preservation_score", 0)),
                opportunity_capture_score=float(data.get("opportunity_capture_score", 0)),
                quality_flags=data.get("quality_flags", "").split(",") if isinstance(data.get("quality_flags"), str) and data.get("quality_flags") else [],
            )
        except Exception as e:
            logger.warning("[OfflineEvaluator] Obituary deserialization failed: %s", e)
            return None

    @staticmethod
    def _infer_entry_price(obit: TradeExitObituary) -> float:
        """Infer entry price. ASSUMPTION: Not directly stored in obituary v1."""
        return 1.0  # Normalized; actual price via outcome reconstruction

    @staticmethod
    def _infer_side(obit: TradeExitObituary) -> str:
        """Infer side from belief_snapshot."""
        if isinstance(obit.belief_snapshot, dict):
            return obit.belief_snapshot.get("side", "LONG")
        return "LONG"

    @staticmethod
    def _mean(values: List[float]) -> float:
        if not values:
            return 0.0
        return sum(values) / len(values)

    @staticmethod
    def _median(values: List[float]) -> float:
        if not values:
            return 0.0
        s = sorted(values)
        n = len(s)
        if n % 2 == 1:
            return s[n // 2]
        return (s[n // 2 - 1] + s[n // 2]) / 2.0
