"""
HazardEngine — Multi-dimensional risk assessment for exit decisions.

Pure math. No Redis. No IO. No side effects.
Input: Phase 1 state + geometry + regime, Phase 2 ensemble, Phase 3 belief.
Output: HazardAssessment.

Six independent hazard axes:
  1. drawdown_hazard   — profit giveback risk
  2. reversal_hazard   — market reversal risk
  3. volatility_hazard — volatility-driven risk
  4. time_decay_hazard — holding duration risk
  5. regime_hazard     — unfavorable regime risk
  6. ensemble_hazard   — direct ensemble sell signal

Each axis is [0,1]. Composite is a weighted sum.
"""

from __future__ import annotations

import math
import time
from typing import Optional

from ..models.position_exit_state import PositionExitState
from ..models.aggregated_exit_signal import AggregatedExitSignal
from ..models.belief_state import BeliefState
from ..models.hazard_assessment import HazardAssessment
from ..engines.geometry_engine import GeometryResult
from ..engines.regime_drift_engine import RegimeState


def _clamp(value: float, lo: float = 0.0, hi: float = 1.0) -> float:
    return max(lo, min(hi, value))


# ── Default weights for composite hazard (sum to 1.0) ───────────────────

HAZARD_WEIGHTS = {
    "drawdown": 1.0 / 6.0,
    "reversal": 1.0 / 6.0,
    "volatility": 1.0 / 6.0,
    "time_decay": 1.0 / 6.0,
    "regime": 1.0 / 6.0,
    "ensemble": 1.0 / 6.0,
}

# Time decay: half-life in seconds (4 hours)
TIME_DECAY_HALF_LIFE = 14400.0

# Volatility reference: ATR / price above this is considered high risk
VOLATILITY_HIGH_REF = 0.02  # 2% ATR/price = high risk cap


class HazardEngine:
    """
    Stateless calculator for multi-dimensional position risk.

    All methods are pure functions. Instance exists for future config.
    """

    # ── Individual hazard axes ───────────────────────────────────────────

    @staticmethod
    def compute_drawdown_hazard(
        geometry: GeometryResult,
        state: PositionExitState,
    ) -> float:
        """
        Risk from giving back profits.

        High when drawdown_from_peak is large relative to peak PnL or notional.
        Zero when position was never profitable.
        """
        if geometry.mfe <= 0:
            # Position never moved favorably; drawdown hazard not applicable
            # but MAE relative to notional can still be a concern
            if state.notional > 0 and state.max_adverse_excursion > 0:
                return _clamp(state.max_adverse_excursion / (state.notional * 0.05))
            return 0.0

        # Reference: max(peak_pnl, 1% of notional) avoids division by near-zero
        reference = max(state.peak_unrealized_pnl, state.notional * 0.01)
        if reference <= 0:
            return 0.0

        return _clamp(geometry.drawdown_from_peak / reference)

    @staticmethod
    def compute_reversal_hazard(
        regime: RegimeState,
        ensemble: AggregatedExitSignal,
    ) -> float:
        """
        Risk of market reversing against the position.

        Blends regime reversal_risk with ensemble reversal probability.
        """
        regime_component = regime.reversal_risk * regime.regime_confidence
        ensemble_component = (
            ensemble.reversal_probability_agg * ensemble.confidence_agg
        )
        return _clamp(regime_component * 0.6 + ensemble_component * 0.4)

    @staticmethod
    def compute_volatility_hazard(state: PositionExitState) -> float:
        """
        Risk from high market volatility.

        Uses ATR/price ratio when available, falls back to volatility_short.
        """
        if state.atr is not None and state.entry_price > 0:
            atr_ratio = state.atr / state.entry_price
            return _clamp(atr_ratio / VOLATILITY_HIGH_REF)

        if state.volatility_short is not None:
            # Normalize sigma: >4% short-term sigma = max hazard
            return _clamp(state.volatility_short / 0.04)

        # No volatility data — conservative moderate hazard
        return 0.5

    @staticmethod
    def compute_time_decay_hazard(state: PositionExitState) -> float:
        """
        Risk from holding a position too long.

        Exponential decay: hazard = 1 - exp(-hold_seconds / half_life).
        Approaches 1.0 asymptotically. At half_life, hazard ≈ 0.5.
        """
        hold_sec = state.hold_seconds
        if hold_sec <= 0:
            return 0.0
        return _clamp(1.0 - math.exp(-hold_sec / TIME_DECAY_HALF_LIFE))

    @staticmethod
    def compute_regime_hazard(regime: RegimeState) -> float:
        """
        Risk from being in an unfavorable regime.

        High in chop (whipsaw), high when counter-trend.
        """
        chop_component = regime.chop_risk * 0.5
        # Counter-trend: negative trend_alignment → unfavorable
        counter_trend_component = max(0.0, -regime.trend_alignment) * 0.5
        return _clamp(chop_component + counter_trend_component)

    @staticmethod
    def compute_ensemble_hazard(ensemble: AggregatedExitSignal) -> float:
        """
        Direct hazard from ensemble sell signal.

        sell_probability * confidence * (1 - uncertainty) to avoid
        acting on uncertain sell signals.
        """
        certainty = _clamp(1.0 - ensemble.uncertainty_agg)
        return _clamp(
            ensemble.sell_probability_agg * ensemble.confidence_agg * certainty
        )

    # ── Composite ────────────────────────────────────────────────────────

    @staticmethod
    def compute_composite(
        hazards: dict[str, float],
        weights: Optional[dict[str, float]] = None,
    ) -> tuple[float, str]:
        """
        Weighted sum of individual hazards + identify dominant.

        Args:
            hazards: {axis_name: hazard_value}
            weights: Optional custom weights. Defaults to equal weight.

        Returns:
            (composite_hazard, dominant_hazard_name)
        """
        w = weights or HAZARD_WEIGHTS
        total_weight = sum(w.get(k, 0.0) for k in hazards)
        if total_weight <= 0:
            return (0.0, "none")

        composite = sum(
            hazards[k] * w.get(k, 0.0) for k in hazards
        ) / total_weight

        dominant = max(hazards, key=hazards.get)
        return (_clamp(composite), dominant)

    # ── Full assessment ──────────────────────────────────────────────────

    @staticmethod
    def assess(
        state: PositionExitState,
        geometry: GeometryResult,
        regime: RegimeState,
        ensemble: AggregatedExitSignal,
    ) -> HazardAssessment:
        """
        Compute full multi-dimensional hazard assessment.

        Args:
            state: Enriched position state from Phase 1.
            geometry: Geometry scores from Phase 1.
            regime: Regime analysis from Phase 1.
            ensemble: Aggregated ensemble signal from Phase 2.

        Returns:
            HazardAssessment with all six axes + composite.
        """
        drawdown_h = HazardEngine.compute_drawdown_hazard(geometry, state)
        reversal_h = HazardEngine.compute_reversal_hazard(regime, ensemble)
        volatility_h = HazardEngine.compute_volatility_hazard(state)
        time_decay_h = HazardEngine.compute_time_decay_hazard(state)
        regime_h = HazardEngine.compute_regime_hazard(regime)
        ensemble_h = HazardEngine.compute_ensemble_hazard(ensemble)

        hazards = {
            "drawdown": drawdown_h,
            "reversal": reversal_h,
            "volatility": volatility_h,
            "time_decay": time_decay_h,
            "regime": regime_h,
            "ensemble": ensemble_h,
        }

        composite, dominant = HazardEngine.compute_composite(hazards)

        quality_flags: list[str] = []
        if state.atr is None and state.volatility_short is None:
            quality_flags.append("NO_VOLATILITY_DATA")

        return HazardAssessment(
            position_id=state.position_id,
            symbol=state.symbol,
            drawdown_hazard=round(drawdown_h, 6),
            reversal_hazard=round(reversal_h, 6),
            volatility_hazard=round(volatility_h, 6),
            time_decay_hazard=round(time_decay_h, 6),
            regime_hazard=round(regime_h, 6),
            ensemble_hazard=round(ensemble_h, 6),
            composite_hazard=round(composite, 6),
            dominant_hazard=dominant,
            hazard_timestamp=time.time(),
            hazard_components=hazards,
            quality_flags=quality_flags,
            shadow_only=True,
        )
