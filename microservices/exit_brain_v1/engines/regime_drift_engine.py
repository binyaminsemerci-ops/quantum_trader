"""
RegimeDriftEngine — Pure math for regime analysis in exit context.

All functions are stateless and side-effect-free.
No Redis. No network. No file IO.

Consumes MarketState regime_probs and raw metrics (sigma, mu, ts)
to produce exit-relevant signals: drift detection, trend alignment,
reversal risk, chop risk.

Compatible with:
- MarketState regime_probs: {"TREND": p, "MR": p, "CHOP": p}
- meta_regime labels: BULL, BEAR, RANGE, VOLATILE, UNCERTAIN
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, List, Literal, Optional


@dataclass(frozen=True)
class RegimeDrift:
    """Result of drift detection between two regime snapshots."""

    drifted: bool                    # True if regime changed meaningfully
    magnitude: float                 # KL-divergence or L1 distance [0..2]
    old_dominant: str                # Dominant regime before
    new_dominant: str                # Dominant regime now
    transition: str                  # e.g. "TREND→CHOP"


@dataclass(frozen=True)
class RegimeState:
    """Aggregated regime analysis for exit decision-making."""

    regime_label: str                # Dominant regime (highest prob)
    regime_confidence: float         # Probability of dominant regime [0..1]
    trend_alignment: float           # Position alignment with trend [-1..+1]
    reversal_risk: float             # Probability of reversal [0..1]
    chop_risk: float                 # Probability of chop [0..1]
    mean_reversion_score: float      # MR regime strength [0..1]
    drift: Optional[RegimeDrift]     # None if no history


class RegimeDriftEngine:
    """
    Stateless calculator for regime-related exit signals.

    All methods are static for purity. Instance exists for future config.
    """

    # ── Drift Detection ──────────────────────────────────────────────────

    @staticmethod
    def detect_regime_drift(
        prev_probs: Dict[str, float],
        curr_probs: Dict[str, float],
        threshold: float = 0.20,
    ) -> RegimeDrift:
        """
        Detect if the regime has shifted meaningfully between two snapshots.

        Uses L1 distance (sum of absolute differences) between probability
        distributions. L1 in [0, 2] for normalized distributions.

        Args:
            prev_probs: Previous regime probabilities, e.g. {"TREND": 0.6, "MR": 0.2, "CHOP": 0.2}
            curr_probs: Current regime probabilities.
            threshold: L1 distance above which we declare drift (default 0.20).

        Returns:
            RegimeDrift with drifted flag and metadata.
        """
        # Unify keys
        all_keys = set(prev_probs) | set(curr_probs)
        if not all_keys:
            return RegimeDrift(
                drifted=False, magnitude=0.0,
                old_dominant="UNKNOWN", new_dominant="UNKNOWN",
                transition="UNKNOWN→UNKNOWN",
            )

        # L1 distance
        l1 = sum(
            abs(prev_probs.get(k, 0.0) - curr_probs.get(k, 0.0))
            for k in all_keys
        )

        old_dom = max(prev_probs, key=prev_probs.get) if prev_probs else "UNKNOWN"
        new_dom = max(curr_probs, key=curr_probs.get) if curr_probs else "UNKNOWN"

        return RegimeDrift(
            drifted=l1 >= threshold,
            magnitude=l1,
            old_dominant=old_dom,
            new_dominant=new_dom,
            transition=f"{old_dom}→{new_dom}",
        )

    # ── Trend Alignment ──────────────────────────────────────────────────

    @staticmethod
    def compute_trend_alignment(
        side: Literal["LONG", "SHORT"],
        mu: float,
        ts: float,
        mu_threshold: float = 1e-6,
    ) -> float:
        """
        How well the position direction aligns with the current trend.

        +1.0 = fully aligned (LONG in uptrend, SHORT in downtrend)
        -1.0 = fully counter-trend
         0.0 = no discernible trend (|mu| < threshold)

        Scaled by trend strength (ts) so weak trends give near-zero values.

        Formula:
            sign = +1 if (LONG and mu > 0) or (SHORT and mu < 0) else -1
            alignment = sign * min(1.0, ts)

        Args:
            side: Position direction.
            mu: Trend slope from MarketState (positive = uptrend).
            ts: Trend strength from MarketState (|mu|/sigma).
            mu_threshold: Below this, trend is considered flat.

        Returns:
            Alignment score in [-1.0, +1.0].
        """
        if abs(mu) < mu_threshold:
            return 0.0

        mu_positive = mu > 0

        if side == "LONG":
            sign = 1.0 if mu_positive else -1.0
        else:  # SHORT
            sign = -1.0 if mu_positive else 1.0

        scaled_ts = min(1.0, ts)
        return sign * scaled_ts

    # ── Reversal Risk ────────────────────────────────────────────────────

    @staticmethod
    def compute_reversal_risk(
        regime_probs: Dict[str, float],
        side: Literal["LONG", "SHORT"],
        mu: float,
    ) -> float:
        """
        Probability that the market is about to reverse against this position.

        Heuristic:
        - For LONG:  reversal_risk ~ P(MR) when mu > 0, or P(TREND) when mu < 0
        - For SHORT: reversal_risk ~ P(MR) when mu < 0, or P(TREND) when mu > 0
        - High P(CHOP) adds uncertainty (moderate reversal risk)

        The idea: if we're LONG in an uptrend, mean-reversion is the reversal threat.
        If we're LONG in a downtrend, a strong trend continuation is the threat.

        Args:
            regime_probs: {"TREND": p, "MR": p, "CHOP": p}
            side: Position direction.
            mu: Trend slope (positive = uptrend).

        Returns:
            Reversal risk in [0.0, 1.0].
        """
        p_trend = regime_probs.get("TREND", 0.0)
        p_mr = regime_probs.get("MR", 0.0)
        p_chop = regime_probs.get("CHOP", 0.0)

        aligned = (side == "LONG" and mu > 0) or (side == "SHORT" and mu < 0)

        if aligned:
            # Position is with the trend. Reversal threat comes from mean-reversion
            risk = p_mr * 0.8 + p_chop * 0.3
        else:
            # Position is counter-trend. Trend continuation is the threat
            risk = p_trend * 0.8 + p_chop * 0.3

        return max(0.0, min(1.0, risk))

    # ── Chop Risk ────────────────────────────────────────────────────────

    @staticmethod
    def compute_chop_risk(
        regime_probs: Dict[str, float],
        sigma: float,
        ts: float,
        sigma_threshold: float = 0.025,
    ) -> float:
        """
        Risk that the market is in a choppy, directionless state.

        Chop damages positions through whipsaw (stop hunts, failed breakouts).
        High chop risk → consider wider stops or exits.

        Formula:
            base = P(CHOP)
            vol_boost = 0.2 if sigma > sigma_threshold and ts < 0.5
            chop_risk = base + vol_boost

        Args:
            regime_probs: {"TREND": p, "MR": p, "CHOP": p}
            sigma: Current volatility.
            ts: Trend strength.
            sigma_threshold: Above this vol + weak trend = extra chop signal.

        Returns:
            Chop risk in [0.0, 1.0].
        """
        base = regime_probs.get("CHOP", 0.0)

        vol_boost = 0.0
        if sigma > sigma_threshold and ts < 0.5:
            vol_boost = 0.2

        return max(0.0, min(1.0, base + vol_boost))

    # ── Aggregate ────────────────────────────────────────────────────────

    @classmethod
    def summarize_regime_state(
        cls,
        side: Literal["LONG", "SHORT"],
        regime_probs: Dict[str, float],
        mu: float,
        sigma: float,
        ts: float,
        prev_regime_probs: Optional[Dict[str, float]] = None,
        drift_threshold: float = 0.20,
    ) -> RegimeState:
        """
        Compute all regime-derived exit signals in one call.

        Args:
            side: Position direction.
            regime_probs: Current regime probabilities from MarketState.
            mu: Trend slope.
            sigma: Volatility.
            ts: Trend strength.
            prev_regime_probs: Previous cycle's regime_probs (for drift detection).
            drift_threshold: L1 threshold for drift.

        Returns:
            RegimeState with all signals.
        """
        # Dominant regime
        if regime_probs:
            dominant = max(regime_probs, key=regime_probs.get)
            confidence = regime_probs.get(dominant, 0.0)
        else:
            dominant = "UNKNOWN"
            confidence = 0.0

        # Map MarketState regime keys to standard labels
        label_map = {"TREND": "BULL" if mu > 0 else "BEAR", "MR": "RANGE", "CHOP": "VOLATILE"}
        regime_label = label_map.get(dominant, dominant)

        # Signals
        alignment = cls.compute_trend_alignment(side, mu, ts)
        reversal = cls.compute_reversal_risk(regime_probs, side, mu)
        chop = cls.compute_chop_risk(regime_probs, sigma, ts)
        mr_score = regime_probs.get("MR", 0.0)

        # Drift
        drift = None
        if prev_regime_probs is not None:
            drift = cls.detect_regime_drift(prev_regime_probs, regime_probs, drift_threshold)

        return RegimeState(
            regime_label=regime_label,
            regime_confidence=confidence,
            trend_alignment=alignment,
            reversal_risk=reversal,
            chop_risk=chop,
            mean_reversion_score=mr_score,
            drift=drift,
        )
