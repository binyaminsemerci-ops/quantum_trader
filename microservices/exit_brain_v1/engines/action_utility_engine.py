"""
ActionUtilityEngine — Scores exit action candidates for one position.

Pure math. No Redis. No IO. No side effects.
Input: Phase 1 state, Phase 3 belief + hazard, Phase 2 ensemble.
Output: List of ActionCandidate, sorted by net_utility descending.

Seven action candidates:
  HOLD               (0%)   — keep position as-is
  REDUCE_SMALL       (10%)  — trim small slice
  REDUCE_MEDIUM      (25%)  — moderate reduction
  TAKE_PROFIT_PARTIAL(50%)  — exit half when profitable
  TAKE_PROFIT_LARGE  (75%)  — exit most when profitable
  TIGHTEN_EXIT       (0%)   — tighten stop, no immediate exit
  CLOSE_FULL         (100%) — exit entire position

shadow_only — no execution writes, no order generation.
"""

from __future__ import annotations

import time
from typing import List

from ..models.position_exit_state import PositionExitState
from ..models.aggregated_exit_signal import AggregatedExitSignal
from ..models.belief_state import BeliefState
from ..models.hazard_assessment import HazardAssessment
from ..models.action_candidate import (
    ActionCandidate,
    ACTION_EXIT_FRACTIONS,
    VALID_ACTIONS,
)


def _clamp(value: float, lo: float = 0.0, hi: float = 1.0) -> float:
    return max(lo, min(hi, value))


class ActionUtilityEngine:
    """
    Stateless calculator that scores each action candidate.

    Methods are static / pure. Instance exists for future config.
    """

    # ── Per-action base utility ──────────────────────────────────────────

    @staticmethod
    def _utility_hold(
        belief: BeliefState,
        hazard: HazardAssessment,
    ) -> tuple[float, dict[str, float]]:
        """
        HOLD utility: high when conviction is strong, hazard is low,
        and directional edge is positive.
        """
        # Scale edge from [-1,+1] to [0,1] for multiplication
        edge_factor = _clamp(0.5 + belief.directional_edge * 0.5)
        safety_factor = 1.0 - hazard.composite_hazard

        base = belief.hold_conviction * safety_factor * edge_factor
        components = {
            "hold_conviction": belief.hold_conviction,
            "safety_factor": round(safety_factor, 4),
            "edge_factor": round(edge_factor, 4),
        }
        return (_clamp(base), components)

    @staticmethod
    def _utility_reduce_small(
        belief: BeliefState,
        hazard: HazardAssessment,
    ) -> tuple[float, dict[str, float]]:
        """
        REDUCE_SMALL utility: Low-risk hedge. Useful when exit pressure
        or uncertainty is moderate — trim a little to reduce exposure.
        """
        base = (
            belief.exit_pressure * 0.50
            + hazard.composite_hazard * 0.30
            + belief.uncertainty_total * 0.20
        )
        components = {
            "exit_pressure_contrib": round(belief.exit_pressure * 0.50, 4),
            "hazard_contrib": round(hazard.composite_hazard * 0.30, 4),
            "uncertainty_contrib": round(belief.uncertainty_total * 0.20, 4),
        }
        return (_clamp(base), components)

    @staticmethod
    def _utility_reduce_medium(
        belief: BeliefState,
        hazard: HazardAssessment,
    ) -> tuple[float, dict[str, float]]:
        """
        REDUCE_MEDIUM utility: Requires more conviction than REDUCE_SMALL.
        """
        base = (
            belief.exit_pressure * 0.60
            + hazard.composite_hazard * 0.40
        )
        components = {
            "exit_pressure_contrib": round(belief.exit_pressure * 0.60, 4),
            "hazard_contrib": round(hazard.composite_hazard * 0.40, 4),
        }
        return (_clamp(base), components)

    @staticmethod
    def _utility_take_profit_partial(
        belief: BeliefState,
        hazard: HazardAssessment,
        state: PositionExitState,
    ) -> tuple[float, dict[str, float]]:
        """
        TAKE_PROFIT_PARTIAL utility: Lock in 50% when profitable.
        Driven by profit protection ratio (geometry) + exit pressure.
        """
        # profit_protection_ratio: how much profit is still preserved
        # We want to take profit when we HAVE profits to protect
        ppr = _clamp(1.0 - hazard.hazard_components.get("drawdown", 0.0))
        base = (
            ppr * belief.exit_pressure * 0.70
            + hazard.drawdown_hazard * 0.30
        )
        components = {
            "profit_factor": round(ppr, 4),
            "exit_pressure": belief.exit_pressure,
            "drawdown_hazard": hazard.drawdown_hazard,
        }
        return (_clamp(base), components)

    @staticmethod
    def _utility_take_profit_large(
        belief: BeliefState,
        hazard: HazardAssessment,
        state: PositionExitState,
    ) -> tuple[float, dict[str, float]]:
        """
        TAKE_PROFIT_LARGE utility: Lock in 75%. Needs higher conviction
        than partial take-profit.
        """
        ppr = _clamp(1.0 - hazard.hazard_components.get("drawdown", 0.0))
        base = (
            ppr * belief.exit_pressure * 0.80
            + hazard.drawdown_hazard * 0.20
        )
        components = {
            "profit_factor": round(ppr, 4),
            "exit_pressure": belief.exit_pressure,
            "drawdown_hazard": hazard.drawdown_hazard,
        }
        return (_clamp(base), components)

    @staticmethod
    def _utility_tighten_exit(
        belief: BeliefState,
        hazard: HazardAssessment,
    ) -> tuple[float, dict[str, float]]:
        """
        TIGHTEN_EXIT utility: Defensive risk management without exiting.
        Useful when hazard is moderate but exit pressure isn't critical.
        """
        # High when hazard is notable but exit pressure is not extreme
        hold_back = _clamp(1.0 - belief.exit_pressure)
        base = (
            hazard.composite_hazard * hold_back * 0.70
            + hazard.volatility_hazard * 0.30
        )
        components = {
            "composite_hazard": hazard.composite_hazard,
            "hold_back_factor": round(hold_back, 4),
            "volatility_hazard": hazard.volatility_hazard,
        }
        return (_clamp(base), components)

    @staticmethod
    def _utility_close_full(
        belief: BeliefState,
        hazard: HazardAssessment,
    ) -> tuple[float, dict[str, float]]:
        """
        CLOSE_FULL utility: Nuclear option. Requires high exit pressure
        combined with high hazard. Reversal hazard amplifies.
        """
        reversal_boost = 1.0 + hazard.reversal_hazard
        base = (
            belief.exit_pressure * hazard.composite_hazard * reversal_boost / 2.0
        )
        components = {
            "exit_pressure": belief.exit_pressure,
            "composite_hazard": hazard.composite_hazard,
            "reversal_boost": round(reversal_boost, 4),
        }
        return (_clamp(base), components)

    # ── Penalty computation ──────────────────────────────────────────────

    @staticmethod
    def _compute_penalties(
        action: str,
        state: PositionExitState,
        belief: BeliefState,
        hazard: HazardAssessment,
    ) -> tuple[float, dict[str, float]]:
        """
        Compute penalties that reduce base utility.

        Returns (total_penalty, {component_name: value}).
        """
        penalties: dict[str, float] = {}

        # 1. Profit-taking actions penalised when not profitable
        if action in ("TAKE_PROFIT_PARTIAL", "TAKE_PROFIT_LARGE"):
            if state.unrealized_pnl <= 0:
                penalties["not_profitable"] = 0.60

        # 2. CLOSE_FULL penalised when hazard is low (don't nuke a safe position)
        if action == "CLOSE_FULL" and hazard.composite_hazard < 0.30:
            penalties["low_hazard_close"] = 0.40 * (0.30 - hazard.composite_hazard) / 0.30

        # 3. HOLD penalised when hazard is very high (dangerous to sit still)
        if action == "HOLD" and hazard.composite_hazard > 0.70:
            penalties["high_hazard_hold"] = 0.40 * (hazard.composite_hazard - 0.70) / 0.30

        # 4. Uncertainty attenuation: aggressive actions dampened by uncertainty
        #    (but HOLD and TIGHTEN_EXIT are not aggressive exits)
        if action not in ("HOLD", "TIGHTEN_EXIT"):
            aggressiveness = ACTION_EXIT_FRACTIONS.get(action, 0.0)
            unc_penalty = belief.uncertainty_total * aggressiveness * 0.30
            if unc_penalty > 0.01:
                penalties["uncertainty_dampening"] = round(unc_penalty, 4)

        total = _clamp(sum(penalties.values()))
        return (total, penalties)

    # ── Rationale generation ─────────────────────────────────────────────

    @staticmethod
    def _generate_rationale(
        action: str,
        base_utility: float,
        penalty_total: float,
        net_utility: float,
        belief: BeliefState,
        hazard: HazardAssessment,
    ) -> str:
        """One-line human-readable explanation."""
        if action == "HOLD":
            return (
                f"Hold conviction={belief.hold_conviction:.2f}, "
                f"hazard={hazard.composite_hazard:.2f}"
            )
        if action == "CLOSE_FULL":
            return (
                f"Exit pressure={belief.exit_pressure:.2f}, "
                f"composite hazard={hazard.composite_hazard:.2f}, "
                f"dominant={hazard.dominant_hazard}"
            )
        if action.startswith("TAKE_PROFIT"):
            return (
                f"Profit taking: exit_pressure={belief.exit_pressure:.2f}, "
                f"drawdown_hazard={hazard.drawdown_hazard:.2f}"
            )
        if action.startswith("REDUCE"):
            return (
                f"Risk reduction: exit_pressure={belief.exit_pressure:.2f}, "
                f"uncertainty={belief.uncertainty_total:.2f}"
            )
        if action == "TIGHTEN_EXIT":
            return (
                f"Defensive: hazard={hazard.composite_hazard:.2f}, "
                f"volatility_hazard={hazard.volatility_hazard:.2f}"
            )
        return f"net_utility={net_utility:.4f}"

    # ── Full scoring ─────────────────────────────────────────────────────

    @staticmethod
    def score_all(
        state: PositionExitState,
        belief: BeliefState,
        hazard: HazardAssessment,
    ) -> List[ActionCandidate]:
        """
        Score all 7 action candidates and return sorted by net_utility.

        Args:
            state: Enriched position state from Phase 1.
            belief: BeliefState from Phase 3 belief_engine.
            hazard: HazardAssessment from Phase 3 hazard_engine.

        Returns:
            List of ActionCandidate sorted descending by net_utility.
            Rank 1 = best action.
        """
        now = time.time()

        # Compute base utility for each action
        utility_fns = {
            "HOLD": lambda: ActionUtilityEngine._utility_hold(belief, hazard),
            "REDUCE_SMALL": lambda: ActionUtilityEngine._utility_reduce_small(belief, hazard),
            "REDUCE_MEDIUM": lambda: ActionUtilityEngine._utility_reduce_medium(belief, hazard),
            "TAKE_PROFIT_PARTIAL": lambda: ActionUtilityEngine._utility_take_profit_partial(
                belief, hazard, state
            ),
            "TAKE_PROFIT_LARGE": lambda: ActionUtilityEngine._utility_take_profit_large(
                belief, hazard, state
            ),
            "TIGHTEN_EXIT": lambda: ActionUtilityEngine._utility_tighten_exit(belief, hazard),
            "CLOSE_FULL": lambda: ActionUtilityEngine._utility_close_full(belief, hazard),
        }

        candidates: list[ActionCandidate] = []

        for action in VALID_ACTIONS:
            fn = utility_fns[action]
            base_utility, utility_components = fn()
            penalty_total, penalty_components = ActionUtilityEngine._compute_penalties(
                action, state, belief, hazard,
            )
            net_utility = _clamp(base_utility - penalty_total)

            rationale = ActionUtilityEngine._generate_rationale(
                action, base_utility, penalty_total, net_utility, belief, hazard,
            )

            candidates.append(ActionCandidate(
                position_id=state.position_id,
                symbol=state.symbol,
                action=action,
                exit_fraction=ACTION_EXIT_FRACTIONS[action],
                base_utility=round(base_utility, 6),
                penalty_total=round(penalty_total, 6),
                net_utility=round(net_utility, 6),
                rank=0,  # computed after sort
                utility_components=utility_components,
                penalty_components=penalty_components,
                rationale=rationale,
                scoring_timestamp=now,
                shadow_only=True,
            ))

        # Sort descending by net_utility, then by action name for stability
        candidates.sort(key=lambda c: (-c.net_utility, c.action))

        # Assign ranks
        for i, c in enumerate(candidates):
            c.rank = i + 1

        return candidates
