"""decision_engine: deterministic formula-based exit decision (PATCH-1).

No LLM.  No network calls.  Purely functional given a PerceptionResult.

Rule priority (highest wins):
  1. Emergency drawdown stop   — R_net < DRAWDOWN_STOP_R   → FULL_CLOSE (EMERGENCY)
  2. Hard SL breach            — price through stop_loss   → FULL_CLOSE (EMERGENCY)
  3. Time stop                 — age >= max_hold_sec       → TIME_STOP_EXIT
  4. Partial harvest T1        — R_net >= r_effective_t1   → PARTIAL_CLOSE_25
  5. Trailing tighten          — giveback >= 50% at profit → TIGHTEN_TRAIL
  6. Move SL to break-even     — R_net >= r_effective_lock  → MOVE_TO_BREAKEVEN
  7. Hold                                                   → HOLD

All returned ExitDecision objects have dry_run=True (enforced by caller).
"""
from __future__ import annotations

import logging
from typing import Optional

from .models import ExitDecision, PerceptionResult, PositionSnapshot

_log = logging.getLogger("exit_management_agent.decision_engine")

# ── Action label constants ─────────────────────────────────────────────────────
HOLD = "HOLD"
MOVE_TO_BREAKEVEN = "MOVE_TO_BREAKEVEN"
TIGHTEN_TRAIL = "TIGHTEN_TRAIL"
PARTIAL_CLOSE_25 = "PARTIAL_CLOSE_25"
FULL_CLOSE = "FULL_CLOSE"
TIME_STOP_EXIT = "TIME_STOP_EXIT"

# ── Urgency label constants ────────────────────────────────────────────────────
URGENCY_LOW = "LOW"
URGENCY_MEDIUM = "MEDIUM"
URGENCY_HIGH = "HIGH"
URGENCY_EMERGENCY = "EMERGENCY"

# ── Rule thresholds ────────────────────────────────────────────────────────────
# R below this → emergency full_close regardless of SL position.
_DRAWDOWN_STOP_R: float = -1.5

# Giveback fraction at which TIGHTEN_TRAIL fires (if already in profit).
_GIVEBACK_TRAIL_THRESHOLD: float = 0.50

# break-even plus buffer: 0.2% above entry for LONG, 0.2% below for SHORT.
_BE_PLUS_PCT: float = 0.002

# Trailing stop distance from peak: 0.5%.
_TRAIL_DIST_PCT: float = 0.005


class DecisionEngine:
    """
    Stateless exit decision engine.

    Construct once; call decide() repeatedly with different PerceptionResults.
    """

    def __init__(self, max_hold_sec: float = 14400.0) -> None:
        self._max_hold_sec = max_hold_sec

    def decide(
        self,
        p: PerceptionResult,
        dry_run: bool = True,
    ) -> ExitDecision:
        """
        Evaluate a PerceptionResult and return the highest-priority ExitDecision.

        Args:
            p:       Computed observations for one position.
            dry_run: Must be True in PATCH-1 (enforced by audit.py as well).
        """
        snap = p.snapshot

        # ── Rule 1: Emergency drawdown ─────────────────────────────────────
        if p.R_net <= _DRAWDOWN_STOP_R:
            return _decision(
                snap=snap,
                action=FULL_CLOSE,
                reason=f"Drawdown stop: R_net={p.R_net:.2f} <= {_DRAWDOWN_STOP_R}R",
                urgency=URGENCY_EMERGENCY,
                r_net=p.R_net,
                confidence=0.95,
                suggested_qty_fraction=1.0,
                dry_run=dry_run,
            )

        # ── Rule 2: Hard SL breach ─────────────────────────────────────────
        if snap.stop_loss > 0.0 and p.distance_to_sl_pct < 0.0:
            return _decision(
                snap=snap,
                action=FULL_CLOSE,
                reason=(
                    f"SL breached: mark={snap.mark_price:.6f} "
                    f"SL={snap.stop_loss:.6f} "
                    f"dist={p.distance_to_sl_pct:.4%}"
                ),
                urgency=URGENCY_EMERGENCY,
                r_net=p.R_net,
                confidence=0.99,
                suggested_qty_fraction=1.0,
                dry_run=dry_run,
            )

        # ── Rule 3: Time stop ──────────────────────────────────────────────
        if p.age_sec >= self._max_hold_sec:
            return _decision(
                snap=snap,
                action=TIME_STOP_EXIT,
                reason=(
                    f"Max hold exceeded: age={p.age_sec / 3600:.1f}h "
                    f">= max={self._max_hold_sec / 3600:.1f}h "
                    f"(lower-bound age since agent start)"
                ),
                urgency=URGENCY_MEDIUM,
                r_net=p.R_net,
                confidence=0.80,
                suggested_qty_fraction=1.0,
                dry_run=dry_run,
            )

        # ── Rule 4: Partial harvest at T1 ──────────────────────────────────
        if p.R_net >= p.r_effective_t1:
            return _decision(
                snap=snap,
                action=PARTIAL_CLOSE_25,
                reason=(
                    f"Harvest T1: R_net={p.R_net:.2f} >= T1={p.r_effective_t1:.2f}R "
                    f"(leverage={snap.leverage:.0f}x scaled)"
                ),
                urgency=URGENCY_MEDIUM,
                r_net=p.R_net,
                confidence=0.75,
                suggested_qty_fraction=0.25,
                dry_run=dry_run,
            )

        # ── Rule 5: Tighten trailing stop ────────────────────────────────
        if (
            p.R_net >= p.r_effective_lock
            and p.giveback_pct >= _GIVEBACK_TRAIL_THRESHOLD
        ):
            suggested_sl = _compute_tighter_sl(snap, p.peak_price)
            return _decision(
                snap=snap,
                action=TIGHTEN_TRAIL,
                reason=(
                    f"Giveback: {p.giveback_pct:.0%} >= {_GIVEBACK_TRAIL_THRESHOLD:.0%} "
                    f"of peak profit at R={p.R_net:.2f}"
                ),
                urgency=URGENCY_MEDIUM,
                r_net=p.R_net,
                confidence=0.70,
                suggested_sl=suggested_sl,
                dry_run=dry_run,
            )

        # ── Rule 6: Move SL to break-even ─────────────────────────────────
        if p.R_net >= p.r_effective_lock and _sl_below_entry(snap):
            suggested_sl = _compute_breakeven_sl(snap)
            return _decision(
                snap=snap,
                action=MOVE_TO_BREAKEVEN,
                reason=(
                    f"Lock profit: R_net={p.R_net:.2f} >= lock={p.r_effective_lock:.2f} "
                    f"and SL not yet at break-even"
                ),
                urgency=URGENCY_LOW,
                r_net=p.R_net,
                confidence=0.65,
                suggested_sl=suggested_sl,
                dry_run=dry_run,
            )

        # ── Rule 7: Hold ────────────────────────────────────────────────────
        return _decision(
            snap=snap,
            action=HOLD,
            reason=f"No exit criteria met at R={p.R_net:.2f}",
            urgency=URGENCY_LOW,
            r_net=p.R_net,
            confidence=1.0,
            dry_run=dry_run,
        )


# ── Factory helper ─────────────────────────────────────────────────────────────


def _decision(
    snap: PositionSnapshot,
    action: str,
    reason: str,
    urgency: str,
    r_net: float,
    confidence: float,
    dry_run: bool,
    suggested_sl: Optional[float] = None,
    suggested_qty_fraction: Optional[float] = None,
) -> ExitDecision:
    return ExitDecision(
        snapshot=snap,
        action=action,
        reason=reason,
        urgency=urgency,
        R_net=r_net,
        confidence=confidence,
        suggested_sl=suggested_sl,
        suggested_qty_fraction=suggested_qty_fraction,
        dry_run=dry_run,
    )


# ── SL helpers ─────────────────────────────────────────────────────────────────


def _sl_below_entry(snap: PositionSnapshot) -> bool:
    """
    True if the current stop-loss has not yet been moved to break-even.

    LONG:  SL < entry  → below entry (not yet at BE)
    SHORT: SL > entry  → above entry (not yet at BE)
    Also True when stop_loss == 0.0 (no SL set).
    """
    if snap.stop_loss <= 0.0:
        return True
    if snap.is_long:
        return snap.stop_loss < snap.entry_price
    else:
        return snap.stop_loss > snap.entry_price


def _compute_breakeven_sl(snap: PositionSnapshot) -> float:
    """
    SL at entry + small buffer (_BE_PLUS_PCT = 0.2%) to cover fees.

    LONG:  entry * (1 + 0.002)
    SHORT: entry * (1 - 0.002)
    """
    if snap.is_long:
        return snap.entry_price * (1.0 + _BE_PLUS_PCT)
    else:
        return snap.entry_price * (1.0 - _BE_PLUS_PCT)


def _compute_tighter_sl(
    snap: PositionSnapshot,
    peak_price: float,
) -> Optional[float]:
    """
    Propose a tighter trailing SL at _TRAIL_DIST_PCT (0.5%) from peak.
    Only tightens; never loosens existing SL.

    LONG:  proposed = peak * (1 - 0.005); return max(proposed, existing_sl)
    SHORT: proposed = peak * (1 + 0.005); return min(proposed, existing_sl)
    """
    if snap.is_long:
        proposed = peak_price * (1.0 - _TRAIL_DIST_PCT)
        if snap.stop_loss > 0.0:
            return max(proposed, snap.stop_loss)
        return proposed
    else:
        proposed = peak_price * (1.0 + _TRAIL_DIST_PCT)
        if snap.stop_loss > 0.0:
            return min(proposed, snap.stop_loss)
        return proposed
