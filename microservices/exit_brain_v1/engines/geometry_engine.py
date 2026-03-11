"""
GeometryEngine — Pure math for position exit geometry.

All functions are stateless, side-effect-free, and deterministic.
No Redis. No network. No logging of side-effects.
Input: numbers. Output: numbers.

Used by: position_state_builder (enrichment), future belief_engine, future utility_engine.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import List, Literal, Optional


@dataclass(frozen=True)
class GeometryResult:
    """Aggregated geometry scores for a position."""

    mfe: float                       # Max favorable excursion (price distance)
    mae: float                       # Max adverse excursion (price distance)
    drawdown_from_peak: float        # Current drawdown from peak PnL (>=0)
    profit_protection_ratio: float   # current_pnl / peak_pnl  [0..1], 0 if peak<=0
    momentum_decay: float            # Rate of PnL erosion (negative = decaying)
    reward_to_risk_remaining: float  # Upside potential / downside exposure


class GeometryEngine:
    """
    Stateless calculator for position geometry metrics.

    All methods are classmethods or staticmethods for clarity.
    Instance exists only for future configuration injection.
    """

    # ── MFE ──────────────────────────────────────────────────────────────

    @staticmethod
    def compute_mfe(
        entry_price: float,
        peak_price: float,
        side: Literal["LONG", "SHORT"],
    ) -> float:
        """
        Max Favorable Excursion: largest move in favorable direction.

        For LONG: peak_price - entry_price (clamped >= 0)
        For SHORT: entry_price - peak_price (clamped >= 0)

        Args:
            entry_price: Trade entry price.
            peak_price: Best price seen during trade lifetime.
            side: Position direction.

        Returns:
            MFE as absolute price distance (always >= 0).
        """
        if entry_price <= 0:
            return 0.0

        if side == "LONG":
            return max(0.0, peak_price - entry_price)
        else:  # SHORT
            return max(0.0, entry_price - peak_price)

    # ── MAE ──────────────────────────────────────────────────────────────

    @staticmethod
    def compute_mae(
        entry_price: float,
        trough_price: float,
        side: Literal["LONG", "SHORT"],
    ) -> float:
        """
        Max Adverse Excursion: largest move against position direction.

        For LONG: entry_price - trough_price (clamped >= 0)
        For SHORT: trough_price - entry_price (clamped >= 0)

        Args:
            entry_price: Trade entry price.
            trough_price: Worst price seen during trade lifetime.
            side: Position direction.

        Returns:
            MAE as absolute price distance (always >= 0).
        """
        if entry_price <= 0:
            return 0.0

        if side == "LONG":
            return max(0.0, entry_price - trough_price)
        else:  # SHORT
            return max(0.0, trough_price - entry_price)

    # ── Drawdown from peak PnL ───────────────────────────────────────────

    @staticmethod
    def compute_drawdown_from_peak(
        current_pnl: float,
        peak_pnl: float,
    ) -> float:
        """
        How much PnL has the position given back from its peak.

        Returns absolute drawdown (>= 0).
        If peak_pnl <= 0, drawdown is 0 (never was profitable).

        Args:
            current_pnl: Current unrealized PnL.
            peak_pnl: Historical peak unrealized PnL.

        Returns:
            Drawdown amount (always >= 0).
        """
        if peak_pnl <= 0:
            return 0.0
        return max(0.0, peak_pnl - current_pnl)

    # ── Profit Protection Ratio ──────────────────────────────────────────

    @staticmethod
    def compute_profit_protection_ratio(
        current_pnl: float,
        peak_pnl: float,
    ) -> float:
        """
        Fraction of peak profit currently retained.

        = current_pnl / peak_pnl, clamped to [0, 1].
        Returns 0 if peak_pnl <= 0 (never profitable).

        Interpretation:
          1.0 = at peak profit
          0.5 = gave back 50% of max profit
          0.0 = all profit gone (or never profitable)

        Args:
            current_pnl: Current unrealized PnL.
            peak_pnl: Historical peak unrealized PnL.

        Returns:
            Ratio in [0.0, 1.0].
        """
        if peak_pnl <= 0:
            return 0.0
        ratio = current_pnl / peak_pnl
        return max(0.0, min(1.0, ratio))

    # ── Momentum Decay ───────────────────────────────────────────────────

    @staticmethod
    def compute_momentum_decay(
        pnl_history: List[float],
        window: int = 5,
    ) -> float:
        """
        Rate of PnL change over recent window (slope of linear fit).

        Positive = PnL improving. Negative = PnL decaying.
        Uses simple linear regression slope on the last `window` PnL samples.

        Args:
            pnl_history: Ordered PnL snapshots (oldest first).
            window: Number of recent samples to use.

        Returns:
            Slope (PnL units per sample). 0.0 if insufficient data.
        """
        if len(pnl_history) < 2:
            return 0.0

        recent = pnl_history[-window:] if len(pnl_history) >= window else pnl_history
        n = len(recent)
        if n < 2:
            return 0.0

        # Simple linear regression slope: Σ((x-x̄)(y-ȳ)) / Σ((x-x̄)²)
        x_mean = (n - 1) / 2.0
        y_mean = sum(recent) / n

        num = 0.0
        den = 0.0
        for i, y in enumerate(recent):
            dx = i - x_mean
            num += dx * (y - y_mean)
            den += dx * dx

        if den == 0:
            return 0.0

        return num / den

    # ── Reward-to-Risk Remaining ─────────────────────────────────────────

    @staticmethod
    def compute_reward_to_risk_remaining(
        current_price: float,
        entry_price: float,
        stop_price: float,
        target_price: float,
        side: Literal["LONG", "SHORT"],
    ) -> float:
        """
        Ratio of remaining upside to remaining downside.

        For LONG:
          upside   = target_price - current_price
          downside = current_price - stop_price

        For SHORT:
          upside   = current_price - target_price
          downside = stop_price - current_price

        Returns 0.0 if downside <= 0 (already at or past stop).
        Returns float('inf') capped at 100.0 if downside is near-zero but upside exists.

        Args:
            current_price: Current market price.
            entry_price: Entry price (for reference only).
            stop_price: Current stop-loss price.
            target_price: Current take-profit price.
            side: Position direction.

        Returns:
            Reward/risk ratio (>= 0). Capped at 100.0.
        """
        if current_price <= 0 or stop_price <= 0 or target_price <= 0:
            return 0.0

        if side == "LONG":
            upside = max(0.0, target_price - current_price)
            downside = max(0.0, current_price - stop_price)
        else:  # SHORT
            upside = max(0.0, current_price - target_price)
            downside = max(0.0, stop_price - current_price)

        if downside <= 0:
            return 0.0 if upside <= 0 else 100.0

        return min(100.0, upside / downside)

    # ── Aggregate ────────────────────────────────────────────────────────

    @classmethod
    def compute_all(
        cls,
        entry_price: float,
        current_price: float,
        peak_price: float,
        trough_price: float,
        side: Literal["LONG", "SHORT"],
        current_pnl: float,
        peak_pnl: float,
        pnl_history: Optional[List[float]] = None,
        stop_price: float = 0.0,
        target_price: float = 0.0,
    ) -> GeometryResult:
        """
        Compute all geometry metrics in one call.

        Args:
            entry_price: Position entry price.
            current_price: Current market price.
            peak_price: Best price for the position.
            trough_price: Worst price for the position.
            side: LONG or SHORT.
            current_pnl: Current unrealized PnL.
            peak_pnl: Peak unrealized PnL.
            pnl_history: Optional ordered PnL snapshots.
            stop_price: Current stop-loss price (0 = unknown).
            target_price: Current take-profit price (0 = unknown).

        Returns:
            GeometryResult with all metrics.
        """
        mfe = cls.compute_mfe(entry_price, peak_price, side)
        mae = cls.compute_mae(entry_price, trough_price, side)
        drawdown = cls.compute_drawdown_from_peak(current_pnl, peak_pnl)
        ppr = cls.compute_profit_protection_ratio(current_pnl, peak_pnl)
        momentum = cls.compute_momentum_decay(pnl_history or [])

        rtr = 0.0
        if stop_price > 0 and target_price > 0:
            rtr = cls.compute_reward_to_risk_remaining(
                current_price, entry_price, stop_price, target_price, side,
            )

        return GeometryResult(
            mfe=mfe,
            mae=mae,
            drawdown_from_peak=drawdown,
            profit_protection_ratio=ppr,
            momentum_decay=momentum,
            reward_to_risk_remaining=rtr,
        )
