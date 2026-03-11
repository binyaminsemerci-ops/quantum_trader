"""
BeliefEngine — Fuses geometry, regime, and ensemble into a unified belief.

Pure math. No Redis. No IO. No side effects.
Input: Phase 1 state + geometry + regime, Phase 2 aggregated ensemble.
Output: BeliefState.

Fail-closed: returns None if critical inputs are missing.
"""

from __future__ import annotations

import time
from typing import Optional

from ..models.position_exit_state import PositionExitState
from ..models.aggregated_exit_signal import AggregatedExitSignal
from ..models.belief_state import BeliefState
from ..engines.geometry_engine import GeometryResult
from ..engines.regime_drift_engine import RegimeState


def _clamp(value: float, lo: float = 0.0, hi: float = 1.0) -> float:
    return max(lo, min(hi, value))


# ── Fusion weights (v1 defaults) ────────────────────────────────────────

# Exit pressure weights
W_EXIT_ENSEMBLE = 0.50
W_EXIT_REVERSAL = 0.25
W_EXIT_GEOMETRY = 0.25

# Hold conviction weights
W_HOLD_ENSEMBLE = 0.50
W_HOLD_TREND = 0.25
W_HOLD_GEOMETRY = 0.25

# Directional edge weights
W_EDGE_ENSEMBLE = 0.60
W_EDGE_REGIME = 0.40

# Uncertainty blending
W_UNC_ENSEMBLE = 0.50
W_UNC_DISAGREEMENT = 0.30
W_UNC_DATA = 0.20

# Maximum quality flags before data_completeness is heavily penalised
MAX_QUALITY_FLAGS = 10


class BeliefEngine:
    """
    Stateless calculator that fuses Phase 1 + Phase 2 signals
    into a single BeliefState per position.

    All methods are pure functions. Instance exists for future config.
    """

    # ── Public API ───────────────────────────────────────────────────────

    @staticmethod
    def compute(
        state: PositionExitState,
        geometry: GeometryResult,
        regime: RegimeState,
        ensemble: Optional[AggregatedExitSignal],
    ) -> Optional[BeliefState]:
        """
        Fuse all inputs into a BeliefState.

        Fail-closed: returns None if ensemble is None (not enough models).

        Args:
            state: Enriched position state from Phase 1.
            geometry: Geometry scores from Phase 1.
            regime: Regime analysis from Phase 1.
            ensemble: Aggregated ensemble signal from Phase 2 (None if <2 models).

        Returns:
            BeliefState or None if critical data is missing.
        """
        if ensemble is None:
            return None

        quality_flags: list[str] = list(state.data_quality_flags)
        quality_flags.extend(ensemble.quality_flags)

        # ── Exit pressure ────────────────────────────────────────────

        ens_exit = ensemble.sell_probability_agg * ensemble.confidence_agg
        rev_exit = regime.reversal_risk * regime.regime_confidence
        # Profit erosion: high when protection ratio is low (giving back profits)
        # Only triggers when position was profitable (mfe > 0)
        geo_exit = (1.0 - geometry.profit_protection_ratio) if geometry.mfe > 0 else 0.0

        exit_pressure = _clamp(
            W_EXIT_ENSEMBLE * ens_exit
            + W_EXIT_REVERSAL * rev_exit
            + W_EXIT_GEOMETRY * geo_exit
        )

        # ── Hold conviction ──────────────────────────────────────────

        ens_hold = ensemble.hold_probability_agg * ensemble.confidence_agg
        trend_hold = max(0.0, regime.trend_alignment)  # only positive alignment helps hold
        # Geometry hold: protection ratio boosted by non-negative momentum
        momentum_factor = _clamp(1.0 + geometry.momentum_decay, 0.0, 1.0)
        geo_hold = geometry.profit_protection_ratio * momentum_factor

        hold_conviction = _clamp(
            W_HOLD_ENSEMBLE * ens_hold
            + W_HOLD_TREND * trend_hold
            + W_HOLD_GEOMETRY * geo_hold
        )

        # ── Directional edge ─────────────────────────────────────────

        ens_edge = ensemble.continuation_probability_agg - ensemble.reversal_probability_agg
        regime_edge = regime.trend_alignment * regime.regime_confidence

        directional_edge = _clamp(
            W_EDGE_ENSEMBLE * ens_edge + W_EDGE_REGIME * regime_edge,
            lo=-1.0, hi=1.0,
        )

        # ── Uncertainty ──────────────────────────────────────────────

        ens_unc = ensemble.uncertainty_agg
        disagreement_unc = ensemble.disagreement_score
        # Data quality uncertainty: coverage + flag count
        coverage = ensemble.model_count_used / max(ensemble.model_count_total, 1)
        flag_penalty = min(1.0, len(quality_flags) / MAX_QUALITY_FLAGS)
        data_unc = 1.0 - coverage * (1.0 - flag_penalty * 0.5)

        uncertainty_total = _clamp(
            W_UNC_ENSEMBLE * ens_unc
            + W_UNC_DISAGREEMENT * disagreement_unc
            + W_UNC_DATA * data_unc
        )

        # ── Data completeness ────────────────────────────────────────

        data_completeness = _clamp(
            coverage * (1.0 - flag_penalty * 0.3)
        )

        # ── Assemble belief ──────────────────────────────────────────

        components = {
            "ens_exit": round(ens_exit, 4),
            "rev_exit": round(rev_exit, 4),
            "geo_exit": round(geo_exit, 4),
            "ens_hold": round(ens_hold, 4),
            "trend_hold": round(trend_hold, 4),
            "geo_hold": round(geo_hold, 4),
            "ens_edge": round(ens_edge, 4),
            "regime_edge": round(regime_edge, 4),
            "ens_unc": round(ens_unc, 4),
            "disagreement_unc": round(disagreement_unc, 4),
            "data_unc": round(data_unc, 4),
            "coverage": round(coverage, 4),
        }

        return BeliefState(
            position_id=state.position_id,
            symbol=state.symbol,
            side=state.side,
            exit_pressure=round(exit_pressure, 6),
            hold_conviction=round(hold_conviction, 6),
            directional_edge=round(directional_edge, 6),
            uncertainty_total=round(uncertainty_total, 6),
            data_completeness=round(data_completeness, 6),
            belief_timestamp=time.time(),
            belief_components=components,
            quality_flags=quality_flags,
            shadow_only=True,
        )
