"""
ExitPolicyEngine — Policy evaluation over scored action candidates.

Pure logic. No Redis. No IO. No side effects. Fail-closed.
Input: List[ActionCandidate] + BeliefState + HazardAssessment + PositionExitState
Output: PolicyDecision

Default action is HOLD. Every path that fails or is uncertain falls back to HOLD.
shadow_only — no execution writes, no order generation.
"""

from __future__ import annotations

import logging
import time
import uuid
from typing import Dict, List, Optional, Tuple

from ..models.action_candidate import (
    ActionCandidate,
    ACTION_EXIT_FRACTIONS,
    VALID_ACTIONS,
)
from ..models.belief_state import BeliefState
from ..models.hazard_assessment import HazardAssessment
from ..models.policy_decision import PolicyDecision
from ..models.position_exit_state import PositionExitState

from . import policy_constraints as PC
from . import reason_codes as RC

logger = logging.getLogger(__name__)


def _clamp(value: float, lo: float = 0.0, hi: float = 1.0) -> float:
    return max(lo, min(hi, value))


class ExitPolicyEngine:
    """
    Stateless policy evaluator. Takes utility-scored candidates and applies
    hard blocks, soft warnings, overrides, and fail-closed defaults.

    Call order: evaluate() → PolicyDecision
    """

    def evaluate(
        self,
        candidates: List[ActionCandidate],
        belief: BeliefState,
        hazard: HazardAssessment,
        state: PositionExitState,
    ) -> PolicyDecision:
        """
        Main entry point. Applies policy constraints and returns a decision.

        Args:
            candidates: Sorted list (rank 1 first) from ActionUtilityEngine.
            belief: Fused belief state.
            hazard: Multi-dimensional hazard assessment.
            state: Enriched position state.

        Returns:
            PolicyDecision. If anything fails, chosen_action=HOLD.
        """
        now = time.time()
        decision_id = str(uuid.uuid4())

        hard_blocks: List[str] = []
        soft_warnings: List[str] = []
        reason_codes: List[str] = []
        explanation_tags: List[str] = []

        # ── Guard: empty candidates → fail-closed HOLD ──────────────────
        if not candidates:
            hard_blocks.append(RC.MISSING_UPSTREAM_DATA)
            explanation_tags.append(RC.TAG_INSUFFICIENT_DATA)
            return self._build_hold_decision(
                position_id=state.position_id,
                symbol=state.symbol,
                decision_id=decision_id,
                timestamp=now,
                hard_blocks=hard_blocks,
                soft_warnings=soft_warnings,
                reason_codes=[RC.POLICY_FALLBACK_HOLD],
                explanation_tags=explanation_tags,
                belief=belief,
                hazard=hazard,
                candidates=candidates,
            )

        # ── Step 1: Check upstream freshness ─────────────────────────────
        stale_blocks = self._check_upstream_freshness(belief, hazard, state, now)
        hard_blocks.extend(stale_blocks)

        # ── Step 2: Check data completeness ──────────────────────────────
        completeness_blocks, completeness_warns = self._check_data_completeness(belief)
        hard_blocks.extend(completeness_blocks)
        soft_warnings.extend(completeness_warns)

        # ── Step 3: Check uncertainty ceiling ────────────────────────────
        uncertainty_blocks, uncertainty_warns = self._check_uncertainty(belief)
        hard_blocks.extend(uncertainty_blocks)
        soft_warnings.extend(uncertainty_warns)

        # ── If any hard blocks exist, force HOLD ─────────────────────────
        if hard_blocks:
            reason_codes.extend(hard_blocks)
            reason_codes.append(RC.POLICY_FALLBACK_HOLD)
            explanation_tags.extend(
                self._derive_explanation_tags_from_blocks(hard_blocks)
            )
            return self._build_hold_decision(
                position_id=state.position_id,
                symbol=state.symbol,
                decision_id=decision_id,
                timestamp=now,
                hard_blocks=hard_blocks,
                soft_warnings=soft_warnings,
                reason_codes=reason_codes,
                explanation_tags=explanation_tags,
                belief=belief,
                hazard=hazard,
                candidates=candidates,
            )

        # ── Step 4: Select top candidate and apply policy filters ────────
        chosen = candidates[0]
        action = chosen.action
        policy_passed = True

        # 4a. Emergency hazard override → boost CLOSE_FULL
        emergency_override = False
        if self._is_hazard_emergency(hazard):
            emergency_override = True
            reason_codes.append(RC.HAZARD_EMERGENCY_OVERRIDE)
            explanation_tags.append(RC.TAG_EMERGENCY_EXIT)
            # Find CLOSE_FULL among candidates or keep chosen
            close_full = self._find_candidate(candidates, "CLOSE_FULL")
            if close_full is not None:
                chosen = close_full
                action = "CLOSE_FULL"

        # 4b. Apply action-specific constraints
        action_blocks, action_warns = self._check_action_constraints(
            action, belief, hazard, state
        )
        if action_blocks and not emergency_override:
            # Blocked action → demote to HOLD
            hard_blocks.extend(action_blocks)
            soft_warnings.extend(action_warns)
            reason_codes.extend(action_blocks)
            reason_codes.append(RC.POLICY_FALLBACK_HOLD)
            explanation_tags.extend(
                self._derive_explanation_tags_from_blocks(action_blocks)
            )
            return self._build_hold_decision(
                position_id=state.position_id,
                symbol=state.symbol,
                decision_id=decision_id,
                timestamp=now,
                hard_blocks=hard_blocks,
                soft_warnings=soft_warnings,
                reason_codes=reason_codes,
                explanation_tags=explanation_tags,
                belief=belief,
                hazard=hazard,
                candidates=candidates,
            )
        soft_warnings.extend(action_warns)

        # 4c. Conviction check
        if (
            action != "HOLD"
            and chosen.net_utility < PC.MIN_ACTION_CONVICTION
        ):
            soft_warnings.append(RC.INSUFFICIENT_CONVICTION)
            reason_codes.append(RC.HOLD_PROMOTED_LOW_CONVICTION)
            explanation_tags.append(RC.TAG_UNCERTAIN)
            # Demote to HOLD
            chosen = self._find_candidate(candidates, "HOLD") or chosen
            action = "HOLD"

        # 4d. Edge-neutral HOLD bias
        if (
            action != "HOLD"
            and abs(belief.directional_edge) < PC.EDGE_NEUTRAL_BAND
            and chosen.net_utility < PC.PREFER_HOLD_THRESHOLD
        ):
            soft_warnings.append(RC.EDGE_NEUTRAL_HOLD_PREFERRED)
            reason_codes.append(RC.HOLD_PROMOTED_EDGE_NEUTRAL)
            chosen = self._find_candidate(candidates, "HOLD") or chosen
            action = "HOLD"

        # 4e. Hazard close boost (non-emergency)
        if (
            action == "CLOSE_FULL"
            and not emergency_override
            and hazard.composite_hazard > PC.HAZARD_CLOSE_BOOST_THRESHOLD
        ):
            reason_codes.append(RC.HAZARD_CLOSE_BOOST)

        # ── Step 5: Uncertainty dampening on remaining actions ───────────
        if (
            belief.uncertainty_total > PC.UNCERTAINTY_SOFT_CEILING
            and action not in PC.SAFE_ACTIONS_HIGH_UNCERTAINTY
        ):
            soft_warnings.append(RC.HIGH_UNCERTAINTY_DAMPENING)

        # ── Step 6: Build explanation tags ───────────────────────────────
        explanation_tags.extend(self._build_explanation_tags(
            action, belief, hazard, emergency_override
        ))

        # ── Step 7: Compute confidence ───────────────────────────────────
        confidence = self._compute_decision_confidence(
            chosen, belief, hazard, len(hard_blocks), len(soft_warnings)
        )

        # ── Final build ──────────────────────────────────────────────────
        reason_codes = list(dict.fromkeys(reason_codes))
        explanation_tags = list(dict.fromkeys(explanation_tags))
        soft_warnings = list(dict.fromkeys(soft_warnings))

        return PolicyDecision(
            position_id=state.position_id,
            symbol=state.symbol,
            decision_id=decision_id,
            decision_timestamp=now,
            chosen_action=action,
            chosen_action_rank=chosen.rank,
            chosen_action_utility=chosen.net_utility,
            decision_confidence=confidence,
            decision_uncertainty=_clamp(1.0 - confidence),
            policy_passed=policy_passed,
            policy_blocks=hard_blocks,
            policy_warnings=soft_warnings,
            hazard_summary=self._summarize_hazard(hazard),
            belief_summary=self._summarize_belief(belief),
            utility_summary=self._summarize_utility(candidates),
            reason_codes=reason_codes,
            explanation_tags=explanation_tags,
            shadow_only=True,
        )

    # ── Constraint checks ────────────────────────────────────────────────

    def _check_upstream_freshness(
        self,
        belief: BeliefState,
        hazard: HazardAssessment,
        state: PositionExitState,
        now: float,
    ) -> List[str]:
        """Check if upstream data is fresh enough."""
        blocks: List[str] = []
        if state.feature_freshness_seconds > PC.MAX_UPSTREAM_AGE_SEC:
            blocks.append(RC.STALE_UPSTREAM_DATA)
        return blocks

    def _check_data_completeness(
        self,
        belief: BeliefState,
    ) -> Tuple[List[str], List[str]]:
        """Check data completeness thresholds."""
        blocks: List[str] = []
        warns: List[str] = []
        if belief.data_completeness < PC.DATA_COMPLETENESS_HARD_FLOOR:
            blocks.append(RC.DATA_COMPLETENESS_FLOOR)
        elif belief.data_completeness < PC.DATA_COMPLETENESS_SOFT_FLOOR:
            warns.append(RC.LOW_DATA_COMPLETENESS)
        return blocks, warns

    def _check_uncertainty(
        self,
        belief: BeliefState,
    ) -> Tuple[List[str], List[str]]:
        """Check uncertainty ceiling."""
        blocks: List[str] = []
        warns: List[str] = []
        if belief.uncertainty_total > PC.UNCERTAINTY_HARD_CEILING:
            blocks.append(RC.UNCERTAINTY_CEILING_BREACH)
        elif belief.uncertainty_total > PC.UNCERTAINTY_SOFT_CEILING:
            warns.append(RC.HIGH_UNCERTAINTY_DAMPENING)
        return blocks, warns

    def _check_action_constraints(
        self,
        action: str,
        belief: BeliefState,
        hazard: HazardAssessment,
        state: PositionExitState,
    ) -> Tuple[List[str], List[str]]:
        """Check action-specific constraints."""
        blocks: List[str] = []
        warns: List[str] = []

        # Profit-taking actions need positive PnL
        if action in PC.PROFIT_REQUIRED_ACTIONS:
            if state.unrealized_pnl <= 0:
                blocks.append(RC.PROFIT_TAKING_NO_PROFIT)

        # CLOSE_FULL needs sufficient hazard/pressure
        if action == "CLOSE_FULL":
            if (
                hazard.composite_hazard < PC.CLOSE_FULL_MIN_HAZARD
                and belief.exit_pressure < PC.CLOSE_FULL_MIN_EXIT_PRESSURE
            ):
                blocks.append(RC.CLOSE_FULL_INSUFFICIENT_HAZARD)

        # Quality flags present
        if belief.quality_flags:
            warns.append(RC.QUALITY_FLAGS_PRESENT)

        return blocks, warns

    def _is_hazard_emergency(self, hazard: HazardAssessment) -> bool:
        """Check if composite hazard triggers emergency override."""
        if hazard.composite_hazard >= PC.HAZARD_EMERGENCY_THRESHOLD:
            return True
        if hazard.reversal_hazard >= PC.REVERSAL_EMERGENCY_THRESHOLD:
            return True
        if hazard.drawdown_hazard >= PC.DRAWDOWN_EMERGENCY_THRESHOLD:
            return True
        return False

    # ── Helpers ──────────────────────────────────────────────────────────

    def _find_candidate(
        self,
        candidates: List[ActionCandidate],
        action_name: str,
    ) -> Optional[ActionCandidate]:
        """Find a specific action among candidates."""
        for c in candidates:
            if c.action == action_name:
                return c
        return None

    def _build_hold_decision(
        self,
        position_id: str,
        symbol: str,
        decision_id: str,
        timestamp: float,
        hard_blocks: List[str],
        soft_warnings: List[str],
        reason_codes: List[str],
        explanation_tags: List[str],
        belief: BeliefState,
        hazard: HazardAssessment,
        candidates: List[ActionCandidate],
    ) -> PolicyDecision:
        """Build a HOLD PolicyDecision (fail-closed default)."""
        hold_candidate = self._find_candidate(candidates, "HOLD")
        rank = hold_candidate.rank if hold_candidate else 1
        utility = hold_candidate.net_utility if hold_candidate else 0.0

        return PolicyDecision(
            position_id=position_id,
            symbol=symbol,
            decision_id=decision_id,
            decision_timestamp=timestamp,
            chosen_action="HOLD",
            chosen_action_rank=rank,
            chosen_action_utility=utility,
            decision_confidence=_clamp(1.0 - belief.uncertainty_total),
            decision_uncertainty=belief.uncertainty_total,
            policy_passed=False,
            policy_blocks=hard_blocks,
            policy_warnings=soft_warnings,
            hazard_summary=self._summarize_hazard(hazard),
            belief_summary=self._summarize_belief(belief),
            utility_summary=self._summarize_utility(candidates),
            reason_codes=list(dict.fromkeys(reason_codes)),
            explanation_tags=list(dict.fromkeys(explanation_tags)),
            shadow_only=True,
        )

    def _compute_decision_confidence(
        self,
        chosen: ActionCandidate,
        belief: BeliefState,
        hazard: HazardAssessment,
        num_blocks: int,
        num_warnings: int,
    ) -> float:
        """
        Confidence = f(utility, belief completeness, block/warning count).
        More blocks/warnings → lower confidence.
        """
        base = chosen.net_utility * belief.data_completeness
        warning_penalty = min(num_warnings * 0.05, 0.30)
        return _clamp(base - warning_penalty)

    # ── Explanation builders ─────────────────────────────────────────────

    def _derive_explanation_tags_from_blocks(
        self, blocks: List[str]
    ) -> List[str]:
        """Map hard block codes to explanation tags."""
        tags: List[str] = []
        mapping = {
            RC.UNCERTAINTY_CEILING_BREACH: RC.TAG_UNCERTAIN,
            RC.DATA_COMPLETENESS_FLOOR: RC.TAG_INSUFFICIENT_DATA,
            RC.STALE_UPSTREAM_DATA: RC.TAG_STALE_DATA,
            RC.MISSING_UPSTREAM_DATA: RC.TAG_INSUFFICIENT_DATA,
        }
        for b in blocks:
            tag = mapping.get(b)
            if tag and tag not in tags:
                tags.append(tag)
        return tags

    def _build_explanation_tags(
        self,
        action: str,
        belief: BeliefState,
        hazard: HazardAssessment,
        emergency: bool,
    ) -> List[str]:
        """Build context-sensitive explanation tags for the chosen action."""
        tags: List[str] = []

        if action == "HOLD":
            if hazard.composite_hazard < 0.30 and belief.directional_edge > 0:
                tags.append(RC.TAG_SAFE_HOLD)
            elif belief.uncertainty_total > PC.UNCERTAINTY_SOFT_CEILING:
                tags.append(RC.TAG_UNCERTAIN)

        elif action == "CLOSE_FULL":
            if emergency:
                tags.append(RC.TAG_EMERGENCY_EXIT)
            else:
                tags.append(RC.TAG_CONVICTION_EXIT)

        elif action in ("TAKE_PROFIT_PARTIAL", "TAKE_PROFIT_LARGE"):
            tags.append(RC.TAG_PROFIT_LOCK)

        elif action in ("REDUCE_SMALL", "REDUCE_MEDIUM"):
            tags.append(RC.TAG_RISK_TRIM)

        elif action == "TIGHTEN_EXIT":
            tags.append(RC.TAG_DEFENSIVE)

        return tags

    # ── Summary builders ─────────────────────────────────────────────────

    @staticmethod
    def _summarize_hazard(hazard: HazardAssessment) -> Dict[str, float]:
        return {
            "composite": hazard.composite_hazard,
            "dominant": hazard.dominant_hazard,
            "drawdown": hazard.drawdown_hazard,
            "reversal": hazard.reversal_hazard,
            "volatility": hazard.volatility_hazard,
        }

    @staticmethod
    def _summarize_belief(belief: BeliefState) -> Dict[str, float]:
        return {
            "exit_pressure": belief.exit_pressure,
            "hold_conviction": belief.hold_conviction,
            "directional_edge": belief.directional_edge,
            "uncertainty": belief.uncertainty_total,
            "completeness": belief.data_completeness,
        }

    @staticmethod
    def _summarize_utility(candidates: List[ActionCandidate]) -> Dict[str, float]:
        if not candidates:
            return {}
        return {
            c.action: c.net_utility for c in candidates
        }
