"""
ShadowPublisher — Publishes Exit Brain v1 outputs to shadow Redis streams.

SAFETY: This module writes ONLY to shadow streams (*.shadow).
It has a hard-coded blocklist to prevent accidental writes to execution paths.

Streams written:
  quantum:stream:exit.state.shadow     — Full PositionExitState per position
  quantum:stream:exit.geometry.shadow  — Geometry scores per position
  quantum:stream:exit.regime.shadow    — Regime drift analysis per cycle
"""

from __future__ import annotations

import json
import logging
import time
from dataclasses import asdict
from typing import Any, Dict, Optional

from ..models.position_exit_state import PositionExitState
from ..models.belief_state import BeliefState
from ..models.hazard_assessment import HazardAssessment
from ..models.action_candidate import ActionCandidate
from ..models.policy_decision import PolicyDecision
from ..models.exit_intent_candidate import ExitIntentCandidate
from ..models.exit_intent_validation_result import ExitIntentValidationResult
from ..models.decision_trace import DecisionTrace
from ..models.trade_exit_obituary import TradeExitObituary
from ..models.replay_evaluation_record import ReplayEvaluationRecord
from ..models.offline_evaluation_summary import OfflineEvaluationSummary
from ..models.tuning_recommendation import TuningRecommendation
from ..models.calibration_artifact import CalibrationArtifact
from ..engines.geometry_engine import GeometryResult
from ..engines.regime_drift_engine import RegimeState

logger = logging.getLogger(__name__)

# Maximum stream length (auto-trimmed)
STREAM_MAXLEN = 5000

# Hard-coded blocklist: these streams must NEVER be written to
_FORBIDDEN_STREAMS = frozenset({
    "quantum:stream:trade.intent",
    "quantum:stream:apply.plan",
    "quantum:stream:apply.plan.manual",
    "quantum:stream:apply.result",
    "quantum:stream:exit.intent",
    "quantum:stream:harvest.intent",
})


class ShadowPublisher:
    """
    Publishes exit brain outputs to Redis shadow streams.

    Every write is guarded by the forbidden-streams blocklist.
    If someone misconfigures a stream name, the write will be rejected.
    """

    STREAM_STATE = "quantum:stream:exit.state.shadow"
    STREAM_GEOMETRY = "quantum:stream:exit.geometry.shadow"
    STREAM_REGIME = "quantum:stream:exit.regime.shadow"

    # Phase 2 ensemble streams
    STREAM_ENSEMBLE_RAW = "quantum:stream:exit.ensemble.raw.shadow"
    STREAM_ENSEMBLE_AGG = "quantum:stream:exit.ensemble.agg.shadow"
    STREAM_ENSEMBLE_DIAG = "quantum:stream:exit.ensemble.diag.shadow"

    # Phase 3 reasoning streams
    STREAM_BELIEF = "quantum:stream:exit.belief.shadow"
    STREAM_HAZARD = "quantum:stream:exit.hazard.shadow"
    STREAM_UTILITY = "quantum:stream:exit.utility.shadow"

    # Phase 4 policy/orchestrator/validator streams
    STREAM_POLICY = "quantum:stream:exit.policy.shadow"
    STREAM_INTENT_CANDIDATE = "quantum:stream:exit.intent.candidate.shadow"
    STREAM_INTENT_VALIDATION = "quantum:stream:exit.intent.validation.shadow"
    STREAM_DECISION_TRACE = "quantum:stream:exit.decision.trace.shadow"

    # Phase 5 replay/evaluation/tuning streams
    STREAM_OBITUARY = "quantum:stream:exit.obituary.shadow"
    STREAM_REPLAY_EVAL = "quantum:stream:exit.replay.eval.shadow"
    STREAM_EVAL_SUMMARY = "quantum:stream:exit.eval.summary.shadow"
    STREAM_TUNING_RECOMMENDATION = "quantum:stream:exit.tuning.recommendation.shadow"

    def __init__(self, redis_client) -> None:
        """
        Args:
            redis_client: A synchronous redis.Redis instance.
        """
        self._r = redis_client

    # ── Public API ───────────────────────────────────────────────────────

    def publish_state(self, state: PositionExitState) -> Optional[str]:
        """
        Publish a PositionExitState to the shadow state stream.

        Args:
            state: Enriched position state.

        Returns:
            Stream entry ID, or None on failure.
        """
        data = state.to_dict()
        # Flatten nested types to JSON strings for Redis XADD
        data["source_timestamps"] = json.dumps(data["source_timestamps"])
        data["data_quality_flags"] = json.dumps(data["data_quality_flags"])
        # Convert None values to empty string (Redis doesn't accept None)
        flat = {k: ("" if v is None else str(v)) for k, v in data.items()}
        return self._xadd(self.STREAM_STATE, flat)

    def publish_geometry(
        self,
        symbol: str,
        side: str,
        result: GeometryResult,
    ) -> Optional[str]:
        """
        Publish geometry scores for a position.

        Args:
            symbol: Trading pair.
            side: Position direction.
            result: GeometryResult from geometry_engine.

        Returns:
            Stream entry ID, or None on failure.
        """
        data = {
            "symbol": symbol,
            "side": side,
            "mfe": str(result.mfe),
            "mae": str(result.mae),
            "drawdown_from_peak": str(result.drawdown_from_peak),
            "profit_protection_ratio": str(result.profit_protection_ratio),
            "momentum_decay": str(result.momentum_decay),
            "reward_to_risk_remaining": str(result.reward_to_risk_remaining),
            "ts": str(time.time()),
        }
        return self._xadd(self.STREAM_GEOMETRY, data)

    def publish_regime(self, regime_state: RegimeState) -> Optional[str]:
        """
        Publish regime analysis to the shadow regime stream.

        Args:
            regime_state: RegimeState from regime_drift_engine.

        Returns:
            Stream entry ID, or None on failure.
        """
        data: Dict[str, str] = {
            "regime_label": regime_state.regime_label,
            "regime_confidence": str(regime_state.regime_confidence),
            "trend_alignment": str(regime_state.trend_alignment),
            "reversal_risk": str(regime_state.reversal_risk),
            "chop_risk": str(regime_state.chop_risk),
            "mean_reversion_score": str(regime_state.mean_reversion_score),
            "ts": str(time.time()),
        }
        if regime_state.drift is not None:
            data["drift_detected"] = str(regime_state.drift.drifted)
            data["drift_magnitude"] = str(regime_state.drift.magnitude)
            data["drift_transition"] = regime_state.drift.transition

        return self._xadd(self.STREAM_REGIME, data)

    def publish_belief(self, belief: BeliefState) -> Optional[str]:
        """
        Publish a BeliefState to the shadow belief stream.

        Args:
            belief: Fused belief from belief_engine.

        Returns:
            Stream entry ID, or None on failure.
        """
        data = belief.to_dict()
        flat = {k: ("" if v is None else str(v)) for k, v in data.items()}
        return self._xadd(self.STREAM_BELIEF, flat)

    def publish_hazard(self, hazard: HazardAssessment) -> Optional[str]:
        """
        Publish a HazardAssessment to the shadow hazard stream.

        Args:
            hazard: Multi-dimensional risk from hazard_engine.

        Returns:
            Stream entry ID, or None on failure.
        """
        data = hazard.to_dict()
        flat = {k: ("" if v is None else str(v)) for k, v in data.items()}
        return self._xadd(self.STREAM_HAZARD, flat)

    def publish_utility(
        self,
        candidates: list,
        position_id: str,
        symbol: str,
    ) -> Optional[str]:
        """
        Publish scored ActionCandidate list to the shadow utility stream.

        Serialises the ranked list as a single stream entry with
        the top action highlighted and full breakdown in JSON.

        Args:
            candidates: Sorted list of ActionCandidate from action_utility_engine.
            position_id: Position identity.
            symbol: Trading pair.

        Returns:
            Stream entry ID, or None on failure.
        """
        if not candidates:
            return None

        top = candidates[0]
        data: Dict[str, str] = {
            "position_id": position_id,
            "symbol": symbol,
            "top_action": top.action,
            "top_net_utility": str(top.net_utility),
            "top_rank": str(top.rank),
            "top_rationale": top.rationale,
            "candidate_count": str(len(candidates)),
            "all_candidates": json.dumps([c.to_dict() for c in candidates]),
            "ts": str(time.time()),
        }
        return self._xadd(self.STREAM_UTILITY, data)

    def publish_policy_decision(self, decision: PolicyDecision) -> Optional[str]:
        """
        Publish a PolicyDecision to the shadow policy stream.

        Args:
            decision: Policy-evaluated exit decision.

        Returns:
            Stream entry ID, or None on failure.
        """
        data = decision.to_dict()
        flat = {k: ("" if v is None else str(v)) for k, v in data.items()}
        return self._xadd(self.STREAM_POLICY, flat)

    def publish_intent_candidate(self, intent: ExitIntentCandidate) -> Optional[str]:
        """
        Publish an ExitIntentCandidate to the shadow intent candidate stream.

        Args:
            intent: Shadow exit intent candidate.

        Returns:
            Stream entry ID, or None on failure.
        """
        data = intent.to_dict()
        flat = {k: ("" if v is None else str(v)) for k, v in data.items()}
        return self._xadd(self.STREAM_INTENT_CANDIDATE, flat)

    def publish_intent_validation(
        self, result: ExitIntentValidationResult
    ) -> Optional[str]:
        """
        Publish an ExitIntentValidationResult to the shadow validation stream.

        Args:
            result: Gateway validation result.

        Returns:
            Stream entry ID, or None on failure.
        """
        data = result.to_dict()
        flat = {k: ("" if v is None else str(v)) for k, v in data.items()}
        return self._xadd(self.STREAM_INTENT_VALIDATION, flat)

    def publish_decision_trace(self, trace: DecisionTrace) -> Optional[str]:
        """
        Publish a DecisionTrace to the shadow decision trace stream.

        Args:
            trace: Full audit trail for one decision cycle.

        Returns:
            Stream entry ID, or None on failure.
        """
        data = trace.to_dict()
        flat = {k: ("" if v is None else str(v)) for k, v in data.items()}
        return self._xadd(self.STREAM_DECISION_TRACE, flat)

    # ── Phase 5: Replay / Evaluation / Tuning ────────────────────────────

    def publish_obituary(self, obituary: TradeExitObituary) -> Optional[str]:
        """
        Publish a TradeExitObituary to the shadow obituary stream.

        Args:
            obituary: Complete post-mortem of a position lifecycle.

        Returns:
            Stream entry ID, or None on failure.
        """
        data = obituary.to_dict()
        flat = {k: ("" if v is None else str(v)) for k, v in data.items()}
        return self._xadd(self.STREAM_OBITUARY, flat)

    def publish_replay_evaluation(
        self, record: ReplayEvaluationRecord
    ) -> Optional[str]:
        """
        Publish a ReplayEvaluationRecord to the shadow replay eval stream.

        Args:
            record: Per-decision replay evaluation.

        Returns:
            Stream entry ID, or None on failure.
        """
        data = record.to_dict()
        flat = {k: ("" if v is None else str(v)) for k, v in data.items()}
        return self._xadd(self.STREAM_REPLAY_EVAL, flat)

    def publish_evaluation_summary(
        self, summary: OfflineEvaluationSummary
    ) -> Optional[str]:
        """
        Publish an OfflineEvaluationSummary to the shadow eval summary stream.

        Args:
            summary: Aggregated evaluation metrics for one run.

        Returns:
            Stream entry ID, or None on failure.
        """
        data = summary.to_dict()
        flat = {k: ("" if v is None else str(v)) for k, v in data.items()}
        return self._xadd(self.STREAM_EVAL_SUMMARY, flat)

    def publish_tuning_recommendation(
        self, recommendation: TuningRecommendation
    ) -> Optional[str]:
        """
        Publish a TuningRecommendation to the shadow tuning stream.

        Args:
            recommendation: Proposed parameter change.

        Returns:
            Stream entry ID, or None on failure.
        """
        data = recommendation.to_dict()
        flat = {k: ("" if v is None else str(v)) for k, v in data.items()}
        return self._xadd(self.STREAM_TUNING_RECOMMENDATION, flat)

    # ── Private ──────────────────────────────────────────────────────────

    def _xadd(self, stream: str, fields: Dict[str, str]) -> Optional[str]:
        """
        Safe XADD wrapper with forbidden-stream guard.

        Returns stream entry ID or None.
        """
        if stream in _FORBIDDEN_STREAMS:
            logger.error(
                "[ShadowPublisher] BLOCKED write to forbidden stream: %s", stream
            )
            return None

        if not stream.endswith(".shadow"):
            logger.error(
                "[ShadowPublisher] BLOCKED write to non-shadow stream: %s", stream
            )
            return None

        try:
            entry_id = self._r.xadd(stream, fields, maxlen=STREAM_MAXLEN)
            if isinstance(entry_id, bytes):
                entry_id = entry_id.decode("utf-8")
            return entry_id
        except Exception as e:
            logger.error("[ShadowPublisher] XADD to %s failed: %s", stream, e)
            return None
