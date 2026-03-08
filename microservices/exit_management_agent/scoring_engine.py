"""scoring_engine: 5-dimension formula scoring for exit decisions (PATCH-7A).

Replaces the if-chain in decision_engine.py with a continuous scoring lattice.
Hard guards (emergency drawdown, SL breach, time stop) still bypass this engine
entirely — see scoring_guards.py.

Dimensions
----------
D1  d_r_loss       — Loss pressure relative to emergency drawdown threshold.
D2  d_r_gain       — Depth into the profit harvest zone.
D3  d_giveback     — Fraction of peak profit given back, zero-gated below BE.
D4  d_time         — Convex time-decay pressure toward max-hold.
D5  d_sl_proximity — Closeness to SL within a 5% buffer window.

Formulas
--------
D1 = clamp(-R_net / 1.5,  0, 1)
D2 = clamp((R_net - r_lock) / (r_t1 - r_lock),  0, 1)   [0 if r_t1 == r_lock]
D3 = giveback_pct * clamp(R_net / r_lock,  0, 1)         [0 if r_lock <= 0]
D4 = clamp(age_sec / max_hold_sec,  0, 1) ** 1.5
D5 = clamp(1 - dist_sl / 0.05,  0, 1) if SL set, else 0

exit_score = 0.30*D1 + 0.25*D2 + 0.20*D3 + 0.15*D4 + 0.10*D5

Decision map (highest matching row wins, ordered most significant first):
  exit_score >= 0.27 AND D1 >= 0.40  → FULL_CLOSE       / HIGH
  exit_score >= 0.22 AND D2 >= 0.80  → PARTIAL_CLOSE_25 / MEDIUM  (or FULL_CLOSE if past 2×T1)
  exit_score >= 0.12 AND D4 >= 0.85  → TIME_STOP_EXIT   / MEDIUM
  exit_score >= 0.08 AND D3 >= 0.60  → TIGHTEN_TRAIL    / MEDIUM
  exit_score >= 0.015 AND D2 >= 0.05 → MOVE_TO_BREAKEVEN/ LOW
  otherwise                           → HOLD             / LOW

Note: original design-doc thresholds (0.85/0.75/0.65/0.55/0.35) are unreachable
because D1 and D2 are semantically exclusive (a position cannot be in deep loss
AND at the profit target simultaneously).  Max achievable score per scenario ≈
0.55.  Thresholds above were calibrated to this achievable range.

Shadow mode (PATCH-7A initial deploy)
--------------------------------------
ScoringEngine runs alongside the legacy DecisionEngine every tick.  Its output
is attached to ExitDecision.score_state for audit visibility, but the legacy
action still drives maybe_publish().  Switch to formula mode by setting
EXIT_AGENT_SCORING_MODE=formula in config.
"""
from __future__ import annotations

import math
import logging
from typing import Optional

from .models import ExitScoreState, PerceptionResult

_log = logging.getLogger("exit_management_agent.scoring_engine")

# ── Tunable weights — sum must equal 1.0 ─────────────────────────────────────
_W_D1: float = 0.30  # loss pressure
_W_D2: float = 0.25  # gain signal
_W_D3: float = 0.20  # giveback
_W_D4: float = 0.15  # time decay
_W_D5: float = 0.10  # SL proximity

# ── Hard constants ────────────────────────────────────────────────────────────
_DRAWDOWN_STOP_R: float = -1.5          # same as decision_engine.py Rule 1
_SL_BUFFER_WIDTH: float = 0.05         # D5: 5% proximity window
_TIME_DECAY_EXPONENT: float = 1.5      # D4: convex weighting

# ── Score thresholds for decision map ────────────────────────────────────────
# With weights (0.30, 0.25, 0.20, 0.15, 0.10) a single dimension can contribute
# at most its own weight to the composite score.  D1 and D2 are semantically
# exclusive (position can't be in deep loss AND at profit target simultaneously),
# so the practically achievable maximum per scenario is roughly 0.50–0.60.
# Thresholds below are calibrated to these achievable ranges.
_THRESH_FULL_CLOSE_LOSS: float = 0.27     # Row 1: D1-dominant; 0.30*D1 alone can exceed this at R≤-0.90
_THRESH_FULL_CLOSE_LOSS_D1: float = 0.40
_THRESH_HARVEST: float = 0.22             # Row 2: D2-dominant; 0.25*D2 exceeds this at D2≥0.88
_THRESH_HARVEST_D2: float = 0.80
_THRESH_TIME_STOP: float = 0.12           # Row 3: D4-dominant; 0.15*D4 exceeds this at D4≥0.80
_THRESH_TIME_STOP_D4: float = 0.85
_THRESH_GIVEBACK: float = 0.08            # Row 4: D3-dominant; 0.20*D3 exceeds this at D3≥0.40
_THRESH_GIVEBACK_D3: float = 0.60
_THRESH_BE: float = 0.015                 # Row 5: early profit lock; any D2>0 contributes
_THRESH_BE_D2: float = 0.05

# ── Action/urgency constants ──────────────────────────────────────────────────
HOLD = "HOLD"
MOVE_TO_BREAKEVEN = "MOVE_TO_BREAKEVEN"
TIGHTEN_TRAIL = "TIGHTEN_TRAIL"
PARTIAL_CLOSE_25 = "PARTIAL_CLOSE_25"
FULL_CLOSE = "FULL_CLOSE"
TIME_STOP_EXIT = "TIME_STOP_EXIT"

URGENCY_LOW = "LOW"
URGENCY_MEDIUM = "MEDIUM"
URGENCY_HIGH = "HIGH"

# ── Formula-mode qty_fraction lookup ─────────────────────────────────────────
# Used by main.py when scoring_mode="formula" to construct ExitDecision with
# the correct suggested_qty_fraction for the formula action.  Values must NOT
# be inherited from the legacy DecisionEngine (C-1 forensic finding).
# TIGHTEN_TRAIL and MOVE_TO_BREAKEVEN modify the SL position rather than close
# a fraction; suggested_sl is intentionally None in PATCH-7A formula mode.
FORMULA_QTY_MAP: dict = {
    FULL_CLOSE:        1.0,
    TIME_STOP_EXIT:    1.0,
    PARTIAL_CLOSE_25:  0.25,
    TIGHTEN_TRAIL:     None,
    MOVE_TO_BREAKEVEN: None,
    HOLD:              None,
}


class ScoringEngine:
    """
    Stateless formula-based scoring engine.

    Construct once; call score() repeatedly with different PerceptionResults.
    max_hold_sec must match the value passed to DecisionEngine so D4 is
    consistent between both engines during shadow-mode comparison.
    """

    def __init__(self, max_hold_sec: float = 14400.0) -> None:
        self._max_hold_sec = max(max_hold_sec, 1.0)

    def score(self, p: PerceptionResult) -> ExitScoreState:
        """
        Compute ExitScoreState for one PerceptionResult.

        Never raises — invalid inputs produce score=0 and HOLD.
        """
        snap = p.snapshot

        # ── D1: loss pressure ─────────────────────────────────────────────────
        d1 = _clamp(-p.R_net / abs(_DRAWDOWN_STOP_R))

        # ── D2: gain signal ───────────────────────────────────────────────────
        d2 = _compute_d2(p.R_net, p.r_effective_lock, p.r_effective_t1)

        # ── D3: giveback pressure (zero-gated below break-even) ───────────────
        if p.r_effective_lock > 0.0:
            be_gate = _clamp(p.R_net / p.r_effective_lock)
        else:
            be_gate = 0.0
        d3 = p.giveback_pct * be_gate

        # ── D4: convex time pressure ──────────────────────────────────────────
        age_frac = _clamp(p.age_sec / self._max_hold_sec)
        d4 = age_frac ** _TIME_DECAY_EXPONENT

        # ── D5: SL proximity within 5% buffer ─────────────────────────────────
        if snap.stop_loss > 0.0:
            d5 = _clamp(1.0 - p.distance_to_sl_pct / _SL_BUFFER_WIDTH)
        else:
            d5 = 0.0

        # ── Composite score ───────────────────────────────────────────────────
        exit_score = (
            _W_D1 * d1
            + _W_D2 * d2
            + _W_D3 * d3
            + _W_D4 * d4
            + _W_D5 * d5
        )
        exit_score = _clamp(exit_score)

        # ── Decision map ──────────────────────────────────────────────────────
        action, urgency, reason = _apply_decision_map(
            exit_score=exit_score,
            d1=d1, d2=d2, d3=d3, d4=d4,
            R_net=p.R_net,
            r_effective_t1=p.r_effective_t1,
        )

        return ExitScoreState(
            # position context
            symbol=snap.symbol,
            side=snap.side,
            R_net=p.R_net,
            age_sec=p.age_sec,
            age_fraction=age_frac,
            giveback_pct=p.giveback_pct,
            distance_to_sl_pct=p.distance_to_sl_pct,
            peak_price=p.peak_price,
            mark_price=snap.mark_price,
            entry_price=snap.entry_price,
            leverage=snap.leverage,
            r_effective_t1=p.r_effective_t1,
            r_effective_lock=p.r_effective_lock,
            # dimension scores
            d_r_loss=d1,
            d_r_gain=d2,
            d_giveback=d3,
            d_time=d4,
            d_sl_proximity=d5,
            # composite
            exit_score=exit_score,
            # formula recommendation
            formula_action=action,
            formula_urgency=urgency,
            formula_confidence=exit_score,
            formula_reason=reason,
        )


# ── Pure helper functions (also directly importable in tests) ─────────────────

def compute_d1(R_net: float) -> float:
    """D1: loss pressure. clamp(-R_net / 1.5, 0, 1)."""
    return _clamp(-R_net / abs(_DRAWDOWN_STOP_R))


def compute_d2(R_net: float, r_lock: float, r_t1: float) -> float:
    """D2: gain signal. clamp((R_net - r_lock) / (r_t1 - r_lock), 0, 1)."""
    return _compute_d2(R_net, r_lock, r_t1)


def compute_d3(giveback_pct: float, R_net: float, r_lock: float) -> float:
    """D3: giveback pressure. giveback * clamp(R_net/r_lock, 0, 1)."""
    if r_lock <= 0.0:
        return 0.0
    return giveback_pct * _clamp(R_net / r_lock)


def compute_d4(age_sec: float, max_hold_sec: float) -> float:
    """D4: convex time pressure. clamp(age/max, 0, 1)^1.5."""
    max_hold = max(max_hold_sec, 1.0)
    return _clamp(age_sec / max_hold) ** _TIME_DECAY_EXPONENT


def compute_d5(distance_to_sl_pct: float, sl_set: bool) -> float:
    """D5: SL proximity. clamp(1 - dist/0.05, 0, 1) if SL set, else 0."""
    if not sl_set:
        return 0.0
    return _clamp(1.0 - distance_to_sl_pct / _SL_BUFFER_WIDTH)


def compute_exit_score(
    d1: float, d2: float, d3: float, d4: float, d5: float
) -> float:
    """Composite score: weighted sum clamped to [0, 1]."""
    return _clamp(_W_D1 * d1 + _W_D2 * d2 + _W_D3 * d3 + _W_D4 * d4 + _W_D5 * d5)


# ── Private helpers ───────────────────────────────────────────────────────────

def _clamp(v: float, lo: float = 0.0, hi: float = 1.0) -> float:
    """Clamp a float to [lo, hi]; return lo on NaN/inf."""
    try:
        if not math.isfinite(v):
            return lo
        return max(lo, min(hi, v))
    except (TypeError, ValueError):
        return lo


def _compute_d2(R_net: float, r_lock: float, r_t1: float) -> float:
    span = r_t1 - r_lock
    if span <= 0.0:
        return 0.0
    return _clamp((R_net - r_lock) / span)


def _apply_decision_map(
    exit_score: float,
    d1: float,
    d2: float,
    d3: float,
    d4: float,
    R_net: float,
    r_effective_t1: float,
) -> tuple:
    """
    Map (exit_score, dimension scores) → (action, urgency, reason).

    Returns (action: str, urgency: str, reason: str).
    """
    # Row 1: heavy loss pressure driving close
    if exit_score >= _THRESH_FULL_CLOSE_LOSS and d1 >= _THRESH_FULL_CLOSE_LOSS_D1:
        return (
            FULL_CLOSE,
            URGENCY_HIGH,
            f"Score={exit_score:.3f} D1={d1:.3f} — loss pressure close",
        )

    # Row 2: deep in profit zone
    if exit_score >= _THRESH_HARVEST and d2 >= _THRESH_HARVEST_D2:
        # Full close if R is well past T1 (2×); otherwise partial
        if r_effective_t1 > 0.0 and R_net >= 2.0 * r_effective_t1:
            return (
                FULL_CLOSE,
                URGENCY_MEDIUM,
                f"Score={exit_score:.3f} R={R_net:.2f} >= 2×T1={r_effective_t1:.2f} — full harvest",
            )
        return (
            PARTIAL_CLOSE_25,
            URGENCY_MEDIUM,
            f"Score={exit_score:.3f} D2={d2:.3f} — T1 harvest zone",
        )

    # Row 3: time pressure dominant
    if exit_score >= _THRESH_TIME_STOP and d4 >= _THRESH_TIME_STOP_D4:
        return (
            TIME_STOP_EXIT,
            URGENCY_MEDIUM,
            f"Score={exit_score:.3f} D4={d4:.3f} — time stop approaching",
        )

    # Row 4: giveback dominant
    if exit_score >= _THRESH_GIVEBACK and d3 >= _THRESH_GIVEBACK_D3:
        return (
            TIGHTEN_TRAIL,
            URGENCY_MEDIUM,
            f"Score={exit_score:.3f} D3={d3:.3f} — giveback at profit",
        )

    # Row 5: moderate gain, SL not yet at break-even
    if exit_score >= _THRESH_BE and d2 >= _THRESH_BE_D2:
        return (
            MOVE_TO_BREAKEVEN,
            URGENCY_LOW,
            f"Score={exit_score:.3f} D2={d2:.3f} — lock break-even",
        )

    # Default: hold
    return (
        HOLD,
        URGENCY_LOW,
        f"Score={exit_score:.3f} — no exit criteria met",
    )
