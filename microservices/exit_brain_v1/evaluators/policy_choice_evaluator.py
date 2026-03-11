"""
PolicyChoiceEvaluator — Evaluates policy decision quality + baseline comparison.

Phase 5 evaluator. Pure computation.

Reads from: TradeExitObituary batch, baseline outcomes (in-memory)
Writes to: Nothing (pure math)
"""

from __future__ import annotations

import logging
from typing import Dict, List

from ..models.trade_exit_obituary import TradeExitObituary
from ..models.action_candidate import ACTION_EXIT_FRACTIONS
from ..replay.outcome_reconstructor import OutcomePathResult

logger = logging.getLogger(__name__)

_EPS = 1e-10


# v1 baseline definitions
BASELINE_DEFINITIONS = {
    "always_hold": "Never exit; measure final PnL at evaluation horizon",
    "fixed_trailing_2pct": "Exit when price drops 2% from running peak",
    "fixed_tp_3pct": "Exit when unrealized PnL reaches +3%",
    "naive_partial_50pct": "Take 50% off at every decision point",
}


class PolicyChoiceEvaluator:
    """
    Evaluates policy-level decision quality and compares
    Exit Brain decisions against naive baselines.

    Pure math. No IO.
    """

    def evaluate(
        self,
        obituaries: List[TradeExitObituary],
    ) -> Dict[str, float]:
        """
        Compute policy choice quality metrics.

        Returns:
            Dict with pass_quality, block_quality, mean_decision_quality, etc.
        """
        if not obituaries:
            return {"sample_size": 0.0}

        pass_outcomes: List[float] = []
        block_outcomes: List[float] = []

        for obit in obituaries:
            # For policy-passed decisions: was the action good?
            # Proxy: opportunity capture + preservation
            quality = 0.5 * obit.preservation_score + 0.5 * obit.opportunity_capture_score

            if obit.policy_passed:
                pass_outcomes.append(quality)
            else:
                block_outcomes.append(quality)

        return {
            "pass_quality": self._mean(pass_outcomes),
            "block_quality": self._mean(block_outcomes),
            "pass_count": float(len(pass_outcomes)),
            "block_count": float(len(block_outcomes)),
            "sample_size": float(len(obituaries)),
        }

    def compare_against_baselines(
        self,
        obituaries: List[TradeExitObituary],
        outcomes: Dict[str, OutcomePathResult],
    ) -> Dict[str, Dict[str, float]]:
        """
        Compare Exit Brain decisions against naive baselines.

        Args:
            obituaries: Batch of obituaries.
            outcomes: Dict mapping position_id → OutcomePathResult.

        Returns:
            Dict mapping baseline_name → {mean_pnl, vs_exit_brain_pnl_delta}.
        """
        exit_brain_pnls: List[float] = []
        baseline_pnls: Dict[str, List[float]] = {n: [] for n in BASELINE_DEFINITIONS}

        for obit in obituaries:
            outcome = outcomes.get(obit.position_id)
            if not outcome or not outcome.price_path:
                continue

            entry_price = self._infer_entry_price(obit, outcome)
            side = self._infer_side(obit)
            quantity = 1.0  # Normalized for comparison

            # Exit Brain actual PnL (from obituary)
            eb_pnl = outcome.final_pnl
            exit_brain_pnls.append(eb_pnl)

            # HOLD baseline: final PnL
            baseline_pnls["always_hold"].append(eb_pnl)  # Same as holding

            # Trailing stop 2%
            ts_pnl = self._simulate_trailing_stop(
                outcome.price_path, entry_price, side, quantity, trail_pct=0.02
            )
            baseline_pnls["fixed_trailing_2pct"].append(ts_pnl)

            # Take-profit 3%
            tp_pnl = self._simulate_take_profit(
                outcome.price_path, entry_price, side, quantity, tp_pct=0.03
            )
            baseline_pnls["fixed_tp_3pct"].append(tp_pnl)

            # Naive partial 50%
            np_pnl = self._simulate_naive_partial(
                outcome.price_path, entry_price, side, quantity, fraction=0.50
            )
            baseline_pnls["naive_partial_50pct"].append(np_pnl)

        eb_mean = self._mean(exit_brain_pnls)
        result: Dict[str, Dict[str, float]] = {}
        for name in BASELINE_DEFINITIONS:
            bl_mean = self._mean(baseline_pnls[name])
            result[name] = {
                "mean_pnl": bl_mean,
                "vs_exit_brain_pnl_delta": eb_mean - bl_mean,
            }
        return result

    # ── Baseline simulations ─────────────────────────────────────────────

    @staticmethod
    def _simulate_trailing_stop(
        prices: List[float],
        entry_price: float,
        side: str,
        quantity: float,
        trail_pct: float,
    ) -> float:
        if not prices:
            return 0.0
        if side == "LONG":
            running_high = prices[0]
            for p in prices:
                if p > running_high:
                    running_high = p
                if p <= running_high * (1.0 - trail_pct):
                    return (p - entry_price) * quantity
            return (prices[-1] - entry_price) * quantity
        else:
            running_low = prices[0]
            for p in prices:
                if p < running_low:
                    running_low = p
                if p >= running_low * (1.0 + trail_pct):
                    return (entry_price - p) * quantity
            return (entry_price - prices[-1]) * quantity

    @staticmethod
    def _simulate_take_profit(
        prices: List[float],
        entry_price: float,
        side: str,
        quantity: float,
        tp_pct: float,
    ) -> float:
        if not prices or entry_price <= 0:
            return 0.0
        target = entry_price * (1.0 + tp_pct) if side == "LONG" else entry_price * (1.0 - tp_pct)
        for p in prices:
            if side == "LONG" and p >= target:
                return (p - entry_price) * quantity
            if side == "SHORT" and p <= target:
                return (entry_price - p) * quantity
        # Never hit TP → final PnL
        if side == "LONG":
            return (prices[-1] - entry_price) * quantity
        return (entry_price - prices[-1]) * quantity

    @staticmethod
    def _simulate_naive_partial(
        prices: List[float],
        entry_price: float,
        side: str,
        quantity: float,
        fraction: float,
    ) -> float:
        if not prices:
            return 0.0
        # Exit fraction at first price, keep remainder to end
        exit_price = prices[0]
        final_price = prices[-1]
        if side == "LONG":
            exited = (exit_price - entry_price) * quantity * fraction
            remaining = (final_price - entry_price) * quantity * (1.0 - fraction)
        else:
            exited = (entry_price - exit_price) * quantity * fraction
            remaining = (entry_price - final_price) * quantity * (1.0 - fraction)
        return exited + remaining

    # ── Helpers ──────────────────────────────────────────────────────────

    @staticmethod
    def _infer_entry_price(obit: TradeExitObituary, outcome: OutcomePathResult) -> float:
        """Infer entry price from obituary belief_snapshot or first price."""
        if outcome.price_path:
            return outcome.price_path[0]
        return 0.0

    @staticmethod
    def _infer_side(obit: TradeExitObituary) -> str:
        """Infer side from belief_snapshot or default LONG."""
        return obit.belief_snapshot.get("side", "LONG") if obit.belief_snapshot else "LONG"

    @staticmethod
    def _mean(values: List[float]) -> float:
        if not values:
            return 0.0
        return sum(values) / len(values)
