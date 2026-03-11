"""
ExitAgentOrchestrator — IO coordination for Phase 4 exit decisions.

This is the ONLY module with side effects (Redis reads + shadow publishes).
Binds together:
  - ActionUtilityEngine (Phase 3) → scored candidates
  - ExitPolicyEngine (Phase 4) → policy decision
  - ExitIntentGatewayValidator (Phase 4) → validated intent
  - ShadowPublisher → shadow stream output

Call chain:
  run_decision_cycle() →
    1. validate upstream freshness
    2. evaluate policy
    3. build exit intent candidate
    4. validate intent via gateway
    5. build decision trace
    6. publish all to shadow

shadow_only — no execution writes, no order generation.
"""

from __future__ import annotations

import hashlib
import logging
import time
import uuid
from typing import Dict, List, Optional

from ..models.action_candidate import ActionCandidate, ACTION_EXIT_FRACTIONS
from ..models.belief_state import BeliefState
from ..models.decision_trace import DecisionTrace
from ..models.exit_intent_candidate import ExitIntentCandidate
from ..models.exit_intent_validation_result import ExitIntentValidationResult
from ..models.hazard_assessment import HazardAssessment
from ..models.policy_decision import PolicyDecision
from ..models.position_exit_state import PositionExitState

from ..policy.exit_policy_engine import ExitPolicyEngine
from ..validators.exit_intent_gateway_validator import ExitIntentGatewayValidator
from ..publishers.shadow_publisher import ShadowPublisher

logger = logging.getLogger(__name__)


class ExitAgentOrchestrator:
    """
    IO coordinator for one exit decision cycle per position.

    Owns:
    - ExitPolicyEngine (pure logic)
    - ExitIntentGatewayValidator (pure logic)
    - ShadowPublisher (IO)

    All outputs are shadow-only. Fail-closed to HOLD on any error.
    """

    def __init__(self, publisher: ShadowPublisher) -> None:
        self._publisher = publisher
        self._policy_engine = ExitPolicyEngine()
        self._gateway_validator = ExitIntentGatewayValidator()

    def run_decision_cycle(
        self,
        candidates: List[ActionCandidate],
        belief: BeliefState,
        hazard: HazardAssessment,
        state: PositionExitState,
    ) -> Optional[PolicyDecision]:
        """
        Execute one complete Phase 4 decision cycle for a position.

        Args:
            candidates: Sorted list from ActionUtilityEngine.
            belief: Fused belief from BeliefEngine.
            hazard: Assessment from HazardEngine.
            state: Enriched position state.

        Returns:
            PolicyDecision, or None on catastrophic failure.
        """
        now = time.time()

        try:
            # ── Step 1: Policy evaluation ────────────────────────────────
            decision = self._policy_engine.evaluate(
                candidates=candidates,
                belief=belief,
                hazard=hazard,
                state=state,
            )

            # Validate the decision
            errors = decision.validate()
            if errors:
                logger.error(
                    "[Orchestrator] PolicyDecision validation failed for %s: %s",
                    state.symbol, errors,
                )
                return None

            # ── Step 2: Build exit intent candidate ──────────────────────
            intent = self._build_intent_candidate(decision, belief, state, now)

            # ── Step 3: Validate intent via gateway ──────────────────────
            validation = self._gateway_validator.validate(intent)

            # ── Step 4: Build decision trace ─────────────────────────────
            trace = self._build_decision_trace(
                decision=decision,
                candidates=candidates,
                belief=belief,
                hazard=hazard,
                state=state,
                now=now,
            )

            # ── Step 5: Publish all to shadow streams ────────────────────
            self._publish_all(decision, intent, validation, trace)

            return decision

        except Exception:
            logger.exception(
                "[Orchestrator] Decision cycle failed for %s — fail-closed",
                state.symbol,
            )
            return None

    # ── Intent builder ───────────────────────────────────────────────────

    def _build_intent_candidate(
        self,
        decision: PolicyDecision,
        belief: BeliefState,
        state: PositionExitState,
        now: float,
    ) -> ExitIntentCandidate:
        """Build an ExitIntentCandidate from the policy decision."""
        action = decision.chosen_action
        exit_frac = ACTION_EXIT_FRACTIONS.get(action, 0.0)

        idem_key = self._compute_idempotency_key(
            position_id=state.position_id,
            symbol=state.symbol,
            action=action,
            belief_ts=belief.belief_timestamp,
        )

        justification_parts = []
        if decision.explanation_tags:
            justification_parts.extend(decision.explanation_tags[:3])
        if decision.reason_codes:
            justification_parts.append(f"codes={decision.reason_codes[:3]}")
        justification = " | ".join(justification_parts) if justification_parts else action

        tighten_params = {}
        if action == "TIGHTEN_EXIT":
            tighten_params = {
                "volatility_hazard": round(belief.uncertainty_total, 4),
                "mode": "shadow",
            }

        return ExitIntentCandidate(
            position_id=state.position_id,
            symbol=state.symbol,
            intent_id=str(uuid.uuid4()),
            intent_timestamp=now,
            action_name=action,
            intent_type="SHADOW_EXIT",
            target_reduction_pct=exit_frac,
            tighten_parameters=tighten_params,
            justification_summary=justification,
            source_decision_id=decision.decision_id,
            confidence=decision.decision_confidence,
            uncertainty=decision.decision_uncertainty,
            constraint_flags=list(decision.policy_blocks),
            quality_flags=list(decision.quality_flags),
            idempotency_key=idem_key,
            shadow_only=True,
        )

    # ── Trace builder ────────────────────────────────────────────────────

    def _build_decision_trace(
        self,
        decision: PolicyDecision,
        candidates: List[ActionCandidate],
        belief: BeliefState,
        hazard: HazardAssessment,
        state: PositionExitState,
        now: float,
    ) -> DecisionTrace:
        """Build a full audit trace for this decision cycle."""
        # Upstream versions
        upstream_versions: Dict[str, float] = {}
        if state.source_timestamps:
            upstream_versions.update(state.source_timestamps)
        upstream_versions["belief_ts"] = belief.belief_timestamp
        upstream_versions["hazard_ts"] = hazard.hazard_timestamp

        # All candidates serialized
        all_cands = [c.to_dict() for c in candidates]

        # Rejected actions
        rejected: List[Dict] = []
        for c in candidates:
            if c.action != decision.chosen_action:
                reason = "outranked"
                block = ""
                if c.action in [b for b in decision.policy_blocks]:
                    reason = "policy_blocked"
                    block = c.action
                rejected.append({
                    "action": c.action,
                    "reason": reason,
                    "block_code": block,
                    "net_utility": c.net_utility,
                })

        # Decisive factors
        decisive: List[str] = list(decision.reason_codes[:5])
        if decision.policy_passed:
            decisive.append(f"chosen={decision.chosen_action}")
        else:
            decisive.append("policy_blocked→HOLD")

        # Uncertainty penalties from belief
        unc_penalties: Dict[str, float] = {
            "uncertainty_total": belief.uncertainty_total,
        }

        # Constraint effects
        constraint_fx: Dict[str, str] = {}
        for block in decision.policy_blocks:
            constraint_fx[block] = f"forced HOLD (blocked {decision.chosen_action})"

        return DecisionTrace(
            position_id=state.position_id,
            symbol=state.symbol,
            trace_id=str(uuid.uuid4()),
            trace_timestamp=now,
            source_decision_id=decision.decision_id,
            upstream_versions=upstream_versions,
            all_candidates=all_cands,
            chosen_action=decision.chosen_action,
            rejected_actions=rejected,
            decisive_factors=decisive,
            uncertainty_penalties=unc_penalties,
            constraint_effects=constraint_fx,
            final_reasoning=(
                f"{decision.chosen_action} (rank={decision.chosen_action_rank}, "
                f"utility={decision.chosen_action_utility:.3f}, "
                f"conf={decision.decision_confidence:.3f})"
            ),
            shadow_only=True,
        )

    # ── Publisher ─────────────────────────────────────────────────────────

    def _publish_all(
        self,
        decision: PolicyDecision,
        intent: ExitIntentCandidate,
        validation: ExitIntentValidationResult,
        trace: DecisionTrace,
    ) -> None:
        """Publish all Phase 4 outputs to shadow streams."""
        self._publisher.publish_policy_decision(decision)
        self._publisher.publish_intent_candidate(intent)
        self._publisher.publish_intent_validation(validation)
        self._publisher.publish_decision_trace(trace)

    # ── Idempotency key ──────────────────────────────────────────────────

    @staticmethod
    def _compute_idempotency_key(
        position_id: str,
        symbol: str,
        action: str,
        belief_ts: float,
    ) -> str:
        """
        Deterministic hash for duplicate detection.
        Same inputs on the same belief cycle → same key.
        """
        raw = f"{position_id}|{symbol}|{action}|{belief_ts:.2f}"
        return hashlib.sha256(raw.encode()).hexdigest()[:16]
