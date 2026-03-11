"""
HazardCalibrationEvaluator — Evaluates hazard predictions vs realized risk.

Phase 5 evaluator. Pure computation.

Reads from: TradeExitObituary batch (in-memory)
Writes to: Nothing (pure math)
"""

from __future__ import annotations

import logging
from typing import Dict, List

from ..models.trade_exit_obituary import TradeExitObituary
from ..replay.outcome_reconstructor import OutcomePathResult

logger = logging.getLogger(__name__)

_EPS = 1e-10

# Hazard axes to evaluate
HAZARD_AXES = [
    "drawdown_hazard",
    "reversal_hazard",
    "volatility_hazard",
    "time_decay_hazard",
    "regime_hazard",
    "ensemble_hazard",
    "composite_hazard",
]


class HazardCalibrationEvaluator:
    """
    Evaluates whether hazard predictions matched realized risk.

    ASSUMPTION: Realized hazard proxies are derived from actual
    post-decision price behavior. This is an approximation.

    Pure math. No IO.
    """

    def evaluate(
        self,
        obituaries: List[TradeExitObituary],
        outcomes: Dict[str, OutcomePathResult],
    ) -> Dict[str, float]:
        """
        Compute hazard calibration metrics across a batch.

        Returns:
            Dict with bias and MAE per hazard axis + composite.
        """
        errors_by_axis: Dict[str, List[float]] = {a: [] for a in HAZARD_AXES}

        for obit in obituaries:
            outcome = outcomes.get(obit.position_id)
            if not outcome or not obit.hazard_snapshot:
                continue

            realized_composite = self._compute_realized_composite_hazard(outcome)
            realized_drawdown = self._compute_realized_drawdown_hazard(outcome)

            for axis in HAZARD_AXES:
                predicted = obit.hazard_snapshot.get(axis, 0.0)
                if axis == "composite_hazard":
                    realized = realized_composite
                elif axis == "drawdown_hazard":
                    realized = realized_drawdown
                else:
                    # ASSUMPTION: Use composite as proxy for axes lacking
                    # direct realized proxy. Tagged for v2 refinement.
                    realized = realized_composite
                errors_by_axis[axis].append(predicted - realized)

        result: Dict[str, float] = {}
        for axis, errs in errors_by_axis.items():
            result[f"{axis}_bias"] = self._mean(errs)
            result[f"{axis}_mae"] = self._mean_abs(errs)
        result["sample_size"] = float(
            len(errors_by_axis.get("composite_hazard", []))
        )
        return result

    @staticmethod
    def _compute_realized_composite_hazard(outcome: OutcomePathResult) -> float:
        """
        Derive realized hazard from actual outcome.

        ASSUMPTION: If position experienced significant drawdown or
        adverse movement, hazard was high.
        """
        if not outcome.pnl_path:
            return 0.5  # Unknown → moderate

        # Measure: how bad did it get relative to how good it was?
        peak = max(max(outcome.pnl_path), _EPS)
        trough = min(outcome.pnl_path)
        if peak <= 0:
            # Never profitable → high hazard proxy
            return min(1.0, abs(trough) / max(abs(trough) + _EPS, _EPS))

        # Drawdown severity
        dd_ratio = outcome.max_drawdown_after_peak / max(peak, _EPS)
        return max(0.0, min(1.0, dd_ratio))

    @staticmethod
    def _compute_realized_drawdown_hazard(outcome: OutcomePathResult) -> float:
        """Realized drawdown hazard from actual drawdown."""
        if outcome.peak_pnl <= _EPS:
            return 0.5 if outcome.final_pnl < 0 else 0.1
        ratio = outcome.max_drawdown_after_peak / max(outcome.peak_pnl, _EPS)
        return max(0.0, min(1.0, ratio))

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
