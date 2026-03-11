"""
CounterfactualEvaluator — Simulates alternative actions on actual price path.

Phase 5 replay component. Pure computation.

Reads from: OutcomePathResult (in-memory)
Writes to: Nothing (pure math)
"""

from __future__ import annotations

import logging
from typing import Dict, List, Tuple

from ..models.action_candidate import ACTION_EXIT_FRACTIONS, VALID_ACTIONS
from .outcome_reconstructor import OutcomePathResult

logger = logging.getLogger(__name__)

# Epsilon to avoid division by zero
_EPS = 1e-10


class CounterfactualEvaluator:
    """
    Given an actual post-decision price path, evaluates what would
    have happened under each alternative action.

    Pure math. No IO. No state.
    """

    def evaluate_all_actions(
        self,
        outcome: OutcomePathResult,
        entry_price: float,
        side: str,
        quantity: float,
    ) -> Dict[str, float]:
        """
        Compute realized utility proxy for every action in VALID_ACTIONS.

        For each action:
        - HOLD: Final PnL normalized
        - CLOSE_FULL: PnL at decision moment (first price in path)
        - Partial exits (REDUCE_SMALL/MEDIUM, TAKE_PROFIT_*):
          Fraction exited at first price + remainder at final price
        - TIGHTEN_EXIT: Simulated as HOLD with trailing stop at 2%

        Args:
            outcome: Post-decision price/PnL path.
            entry_price: Position entry price.
            side: "LONG" or "SHORT".
            quantity: Position size.

        Returns:
            Dict mapping action → realized utility proxy (normalized PnL).
        """
        if not outcome.price_path or entry_price <= 0:
            return {action: 0.0 for action in VALID_ACTIONS}

        first_price = outcome.price_path[0]
        final_price = outcome.price_path[-1]

        results: Dict[str, float] = {}
        for action in sorted(VALID_ACTIONS):
            fraction = ACTION_EXIT_FRACTIONS.get(action, 0.0)
            pnl = self._simulate_action_pnl(
                action, fraction, first_price, final_price,
                outcome.price_path, entry_price, side, quantity,
            )
            # Normalize to utility-like scale [roughly 0-1]
            # Using entry_price * quantity as normalization factor
            norm = entry_price * quantity
            results[action] = pnl / max(norm, _EPS)

        return results

    def find_ex_post_best_action(
        self,
        realized_utility_by_action: Dict[str, float],
    ) -> Tuple[str, float]:
        """
        Determine which action would have been best in hindsight.

        Returns:
            (best_action, best_utility_proxy)
        """
        if not realized_utility_by_action:
            return "HOLD", 0.0
        best = max(realized_utility_by_action.items(), key=lambda x: x[1])
        return best[0], best[1]

    def compute_decision_quality_score(
        self,
        chosen_action: str,
        realized_utility_by_action: Dict[str, float],
        regret_score: float,
        preservation_score: float,
        opportunity_capture_score: float,
    ) -> float:
        """
        Composite decision quality score.

        Weights: rank_component (0.4), regret_inverse (0.3),
                 preservation (0.2), opportunity (0.1).

        Returns:
            [0, 1] quality score. Higher = better decision.
        """
        # Rank component: was chosen action the best?
        sorted_actions = sorted(
            realized_utility_by_action.items(), key=lambda x: x[1], reverse=True
        )
        chosen_rank = 1
        for i, (action, _) in enumerate(sorted_actions, start=1):
            if action == chosen_action:
                chosen_rank = i
                break
        # Normalize: rank 1 = 1.0, rank 7 = ~0.0
        num_actions = max(len(sorted_actions), 1)
        rank_component = max(0.0, 1.0 - (chosen_rank - 1) / max(num_actions - 1, 1))

        regret_inverse = 1.0 - regret_score

        quality = (
            0.40 * rank_component
            + 0.30 * regret_inverse
            + 0.20 * preservation_score
            + 0.10 * opportunity_capture_score
        )
        return max(0.0, min(1.0, quality))

    def compute_explanation_consistency_score(
        self,
        predicted_exit_pressure: float,
        realized_exit_pressure_proxy: float,
        predicted_composite_hazard: float,
        realized_hazard_proxy: float,
    ) -> float:
        """
        Did the system's explanation (beliefs, hazards) match reality?

        Lower absolute error → higher consistency.

        Returns:
            [0, 1] consistency score.
        """
        pressure_error = abs(predicted_exit_pressure - realized_exit_pressure_proxy)
        hazard_error = abs(predicted_composite_hazard - realized_hazard_proxy)
        avg_error = (pressure_error + hazard_error) / 2.0
        return max(0.0, min(1.0, 1.0 - avg_error))

    # ── Private ──────────────────────────────────────────────────────────

    def _simulate_action_pnl(
        self,
        action: str,
        exit_fraction: float,
        first_price: float,
        final_price: float,
        price_path: List[float],
        entry_price: float,
        side: str,
        quantity: float,
    ) -> float:
        """Simulate PnL for a specific action on the actual price path."""
        if action == "HOLD":
            return self._directional_pnl(final_price, entry_price, side, quantity)

        if action == "CLOSE_FULL":
            return self._directional_pnl(first_price, entry_price, side, quantity)

        if action == "TIGHTEN_EXIT":
            # Simulate trailing stop at 2% from running high
            return self._simulate_trailing_stop(
                price_path, entry_price, side, quantity, trail_pct=0.02
            )

        # Partial exits: fraction at first_price, remainder at final_price
        exited_pnl = self._directional_pnl(
            first_price, entry_price, side, quantity * exit_fraction
        )
        remaining_pnl = self._directional_pnl(
            final_price, entry_price, side, quantity * (1.0 - exit_fraction)
        )
        return exited_pnl + remaining_pnl

    @staticmethod
    def _directional_pnl(
        exit_price: float,
        entry_price: float,
        side: str,
        qty: float,
    ) -> float:
        """Compute directional PnL."""
        if side == "LONG":
            return (exit_price - entry_price) * qty
        return (entry_price - exit_price) * qty

    def _simulate_trailing_stop(
        self,
        price_path: List[float],
        entry_price: float,
        side: str,
        quantity: float,
        trail_pct: float = 0.02,
    ) -> float:
        """Simulate a trailing stop exit on the price path."""
        if not price_path:
            return 0.0

        if side == "LONG":
            running_high = price_path[0]
            for price in price_path:
                if price > running_high:
                    running_high = price
                stop_price = running_high * (1.0 - trail_pct)
                if price <= stop_price:
                    return (price - entry_price) * quantity
            return (price_path[-1] - entry_price) * quantity
        else:
            running_low = price_path[0]
            for price in price_path:
                if price < running_low:
                    running_low = price
                stop_price = running_low * (1.0 + trail_pct)
                if price >= stop_price:
                    return (entry_price - price) * quantity
            return (entry_price - price_path[-1]) * quantity
