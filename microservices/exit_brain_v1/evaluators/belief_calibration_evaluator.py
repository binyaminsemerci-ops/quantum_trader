"""
BeliefCalibrationEvaluator — Evaluates belief accuracy vs realized outcomes.

Phase 5 evaluator. Pure computation.

Reads from: TradeExitObituary batch (in-memory)
Writes to: Nothing (pure math)
"""

from __future__ import annotations

import logging
from typing import Dict, List, Tuple

from ..models.trade_exit_obituary import TradeExitObituary
from ..replay.outcome_reconstructor import OutcomePathResult

logger = logging.getLogger(__name__)

_EPS = 1e-10


class BeliefCalibrationEvaluator:
    """
    Evaluates whether beliefs (exit_pressure, hold_conviction, etc.)
    were well-calibrated against what actually happened.

    Pure math. No IO.
    """

    def evaluate(
        self,
        obituaries: List[TradeExitObituary],
        outcomes: Dict[str, OutcomePathResult],
    ) -> Dict[str, float]:
        """
        Compute belief calibration metrics across a batch of obituaries.

        Args:
            obituaries: List of completed obituaries.
            outcomes: Dict mapping position_id → OutcomePathResult.

        Returns:
            Calibration summary dict with bias and MAE per belief field.
        """
        exit_pressure_errors: List[float] = []
        hold_conviction_errors: List[float] = []

        for obit in obituaries:
            outcome = outcomes.get(obit.position_id)
            if not outcome or not obit.belief_snapshot:
                continue

            predicted_ep = obit.belief_snapshot.get("exit_pressure", 0.0)
            predicted_hc = obit.belief_snapshot.get("hold_conviction", 0.0)

            realized_ep = self._compute_realized_exit_pressure(outcome)
            realized_hc = 1.0 - realized_ep

            exit_pressure_errors.append(predicted_ep - realized_ep)
            hold_conviction_errors.append(predicted_hc - realized_hc)

        return {
            "exit_pressure_bias": self._mean(exit_pressure_errors),
            "exit_pressure_mae": self._mean_abs(exit_pressure_errors),
            "hold_conviction_bias": self._mean(hold_conviction_errors),
            "hold_conviction_mae": self._mean_abs(hold_conviction_errors),
            "sample_size": float(len(exit_pressure_errors)),
        }

    def evaluate_per_symbol(
        self,
        obituaries: List[TradeExitObituary],
        outcomes: Dict[str, OutcomePathResult],
        min_samples: int = 5,
    ) -> Dict[str, Dict[str, float]]:
        """Per-symbol belief calibration breakdown."""
        by_symbol: Dict[str, List[TradeExitObituary]] = {}
        for obit in obituaries:
            by_symbol.setdefault(obit.symbol, []).append(obit)

        result: Dict[str, Dict[str, float]] = {}
        for symbol, obits in by_symbol.items():
            if len(obits) < min_samples:
                continue
            result[symbol] = self.evaluate(obits, outcomes)
        return result

    @staticmethod
    def _compute_realized_exit_pressure(outcome: OutcomePathResult) -> float:
        """
        Derive realized exit pressure from actual price path.

        ASSUMPTION: If price moved significantly against position
        (drawdown > 2% of peak), exit pressure was high.
        """
        if outcome.peak_pnl <= _EPS:
            # No meaningful gains → pressure depends on loss magnitude
            if outcome.final_pnl < 0:
                return min(1.0, abs(outcome.final_pnl) / max(abs(outcome.trough_pnl), _EPS))
            return 0.2  # Neutral

        drawdown_ratio = outcome.max_drawdown_after_peak / max(outcome.peak_pnl, _EPS)
        return max(0.0, min(1.0, drawdown_ratio))

    @staticmethod
    def _mean(values: List[float]) -> float:
        if not values:
            return 0.0
        return sum(values) / len(values)

    @staticmethod
    def _mean_abs(values: List[float]) -> float:
        if not values:
            return 0.0
        return sum(abs(v) for v in values) / len(values)
