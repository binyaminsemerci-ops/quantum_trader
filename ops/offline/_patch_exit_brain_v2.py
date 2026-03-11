"""
PATCH-10: Exit Brain v2 — Reality-Aware, Thesis-Driven, Council-Voted Exit Organism.

Creates four new modules:
  pnl_geometry.py   — PnL shape tracker (velocity, acceleration, convexity, retention)
  thesis_engine.py  — Conviction budget + thesis health + alpha halflife decay
  kill_chain.py     — L0–L5 hierarchical kill chain (de-risk graduated response)
  council.py        — Multi-personality meta-vote deliberation layer

Patches existing files:
  models.py         — Add 16 new PerceptionResult fields (all defaulted, backward-compat)
  perception.py     — Integrate all four engines into compute()
  decision_engine.py — Add kill chain rules between time-stop and adaptive harvest

Apply: python3 /tmp/_patch_exit_brain_v2.py
"""
import os, re, sys

BASE = "/opt/quantum/microservices/exit_management_agent"

# ═══════════════════════════════════════════════════════════════════════════════
# HELPER
# ═══════════════════════════════════════════════════════════════════════════════

def read(name):
    with open(os.path.join(BASE, name), "r", encoding="utf-8") as f:
        return f.read()

def write(name, content):
    path = os.path.join(BASE, name)
    with open(path, "w", encoding="utf-8") as f:
        f.write(content)

def create(name, content):
    path = os.path.join(BASE, name)
    if os.path.exists(path):
        print(f"  [SKIP] {name} already exists - overwriting")
    with open(path, "w", encoding="utf-8") as f:
        f.write(content)
    print(f"  [CREATED] {name}")

def patch(name, old, new, required=True):
    content = read(name)
    if old not in content:
        if required:
            raise AssertionError(f"{name}: target string not found:\n{repr(old[:120])}")
        print(f"  [SKIP] {name}: optional target not found (already patched?)")
        return
    count = content.count(old)
    if count > 1:
        raise AssertionError(f"{name}: ambiguous match — found {count} occurrences")
    write(name, content.replace(old, new, 1))
    print(f"  [OK] {name}")

# ═══════════════════════════════════════════════════════════════════════════════
# MODULE 1: pnl_geometry.py
# ═══════════════════════════════════════════════════════════════════════════════

PNL_GEOMETRY = '''"""
pnl_geometry: PnL Shape Tracker (PATCH-10A).

Tracks per-symbol PnL time-series and computes:
  velocity     — dR/dt  (R per second, positive = gaining)
  acceleration — d²R/dt² (negative = decelerating)
  convexity    — 2nd-order polyfit coefficient  (negative = concave down = momentum dying)
  retention    — current_R / max_R_ever  (how much of the peak we have kept)
  noise_ratio  — std_dev / |mean_diff|  (high = chaotic / random payoff shape)

These signals detect:
  - Positions that peaked and are dying quietly (retention collapse)
  - Strong entry that has stalled and reversed (velocity → negative)
  - Payoff shape turning chaotic while still nominally profitable (noise_ratio spike)
  - Convex → concave shape change = "edge melting"
"""
from __future__ import annotations

import collections
import math
import time
from dataclasses import dataclass
from typing import Deque, Optional

_WINDOW_SEC: float = 120.0   # 2-minute rolling window
_MIN_POINTS: int   = 4        # minimum samples before computation is valid
_MAX_POINTS: int   = 40       # cap ring buffer size


@dataclass
class PnLGeometry:
    """Geometry snapshot for one symbol at one point in time."""
    valid:        bool  = False  # False until _MIN_POINTS accumulated
    velocity:     float = 0.0   # R/s
    acceleration: float = 0.0   # R/s²
    convexity:    float = 0.0   # quadratic fit 'a' coefficient
    retention:    float = 1.0   # [0,1] — 1.0 = holding all peak profit
    noise_ratio:  float = 0.0   # chaotic payoff indicator


class PnLGeometryTracker:
    """
    Stateful per-symbol PnL geometry tracker.
    Thread-safe as long as one asyncio task calls it (no locks needed in single-loop design).
    """

    def __init__(self) -> None:
        # symbol → deque of (monotonic_ts, r_net)
        self._history: dict[str, Deque[tuple[float, float]]] = {}
        self._max_r:   dict[str, float] = {}

    def update(self, symbol: str, r_net: float, ts: Optional[float] = None) -> None:
        """Record a new (timestamp, r_net) sample for symbol."""
        if ts is None:
            ts = time.monotonic()
        buf = self._history.setdefault(symbol, collections.deque(maxlen=_MAX_POINTS))
        buf.append((ts, r_net))
        # keep only the rolling window
        cutoff = ts - _WINDOW_SEC
        while buf and buf[0][0] < cutoff:
            buf.popleft()
        # track lifetime maximum R
        prev = self._max_r.get(symbol, r_net)
        self._max_r[symbol] = max(prev, r_net)

    def get(self, symbol: str) -> PnLGeometry:
        """Return geometry for the accumulated samples. Returns valid=False if too few points."""
        buf = self._history.get(symbol)
        if not buf or len(buf) < _MIN_POINTS:
            return PnLGeometry(valid=False)

        pts  = list(buf)
        t0   = pts[0][0]
        xs   = [p[0] - t0 for p in pts]
        ys   = [p[1]       for p in pts]
        n    = len(pts)

        # ── velocity: linear slope over the whole window ─────────────────────
        velocity, _ = _linear_slope(xs, ys)

        # ── acceleration: slope-of-slope between first and second half ────────
        mid = n // 2
        if mid >= 2:
            v_early, _ = _linear_slope(xs[:mid], ys[:mid])
            v_late,  _ = _linear_slope(xs[mid:], ys[mid:])
            dt           = max((xs[-1] - xs[0]) / 2.0, 1.0)
            acceleration = (v_late - v_early) / dt
        else:
            acceleration = 0.0

        # ── convexity: 2nd-order polynomial fit coefficient ───────────────────
        convexity = _quadratic_a(xs, ys)

        # ── retention: how much of peak profit we still hold ──────────────────
        max_r = self._max_r.get(symbol, ys[-1])
        if max_r > 0.0:
            retention = max(0.0, min(1.0, ys[-1] / max_r))
        else:
            retention = 1.0

        # ── noise ratio: chaotic payoff indicator ────────────────────────────
        diffs = [ys[i + 1] - ys[i] for i in range(len(ys) - 1)]
        if len(diffs) >= 2:
            mean_d  = sum(diffs) / len(diffs)
            var     = sum((d - mean_d) ** 2 for d in diffs) / len(diffs)
            std     = math.sqrt(var)
            noise_ratio = std / max(abs(mean_d), 1e-8)
        else:
            noise_ratio = 0.0

        return PnLGeometry(
            valid=True,
            velocity=round(velocity,     6),
            acceleration=round(acceleration, 6),
            convexity=round(convexity,    6),
            retention=round(retention,    4),
            noise_ratio=round(min(noise_ratio, 20.0), 3),
        )

    def forget(self, symbol: str) -> None:
        self._history.pop(symbol, None)
        self._max_r.pop(symbol, None)


# ── Maths helpers ──────────────────────────────────────────────────────────────

def _linear_slope(xs: list, ys: list) -> tuple:
    n   = len(xs)
    if n < 2:
        return 0.0, (ys[0] if ys else 0.0)
    sx  = sum(xs);  sy  = sum(ys)
    sxx = sum(x * x for x in xs)
    sxy = sum(x * y for x, y in zip(xs, ys))
    d   = n * sxx - sx * sx
    if abs(d) < 1e-12:
        return 0.0, sy / n
    slope     = (n * sxy - sx * sy) / d
    intercept = (sy - slope * sx)   / n
    return slope, intercept


def _quadratic_a(xs: list, ys: list) -> float:
    """Least-squares 2nd-order polynomial: return 'a' in y ≈ ax² + bx + c."""
    n = len(xs)
    if n < 3:
        return 0.0
    s0 = float(n)
    s1 = sum(xs);            s2 = sum(x**2 for x in xs)
    s3 = sum(x**3 for x in xs); s4 = sum(x**4 for x in xs)
    ty0 = sum(ys)
    ty1 = sum(x * y for x, y in zip(xs, ys))
    ty2 = sum(x * x * y for x, y in zip(xs, ys))
    try:
        det = _det3([[s4, s3, s2], [s3, s2, s1], [s2, s1, s0]])
        if abs(det) < 1e-20:
            return 0.0
        det_a = _det3([[ty2, s3, s2], [ty1, s2, s1], [ty0, s1, s0]])
        return det_a / det
    except Exception:
        return 0.0


def _det3(m: list) -> float:
    a, b, c = m[0], m[1], m[2]
    return (a[0] * (b[1]*c[2] - b[2]*c[1])
          - a[1] * (b[0]*c[2] - b[2]*c[0])
          + a[2] * (b[0]*c[1] - b[1]*c[0]))
'''

# ═══════════════════════════════════════════════════════════════════════════════
# MODULE 2: thesis_engine.py
# ═══════════════════════════════════════════════════════════════════════════════

THESIS_ENGINE = '''"""
thesis_engine: Trade Thesis Health + Conviction Budget + Alpha Halflife (PATCH-10B).

Each open position tracks:

  thesis_score       [0–1] — Is the original trade idea still supported by current data?
                              Built from: profitability, giveback tolerance, regime alignment.

  conviction_budget  [0–1] — Epistemic capital remaining.  Starts at 1.0 when position
                              first seen.  Drains on regime mutation, thesis degradation,
                              heavy giveback.  Passively recharges over time + on confirmations.

  alpha_remaining    [0–1] — Estimated remaining edge via exp(-age/halflife).
                              Halflife defaults to 4 hours, reduced in choppy regimes.

  regime_drift       [0–1] — How far the current regime has moved from the entry regime.

The key insight: a position can be "economically alive but epistemically dead."
Conviction budget = 0 means the original hypothesis has no remaining support — exit regardless
of PnL level.  This prevents "position inertia" — the most common cause of profit giveback.
"""
from __future__ import annotations

import logging
import math
import time
from dataclasses import dataclass, field
from typing import Optional

_log = logging.getLogger("exit_management_agent.thesis_engine")

# ── Conviction drain / recharge magnitudes ────────────────────────────────────
_DRAIN_REGIME_MUTATION:    float = 0.25
_DRAIN_THESIS_INVALIDATED: float = 0.35
_DRAIN_HIGH_GIVEBACK:      float = 0.12
_DRAIN_ADVERSE_FLOW:       float = 0.08
_RECHARGE_THESIS_CONFIRM:  float = 0.08
_RECHARGE_REGIME_STABLE:   float = 0.04
# Passive recharge: ~1.0 / 4 hours = very slow healing
_PASSIVE_RECHARGE_PER_SEC: float = 0.000069   # 0.25 / 3600

# Alpha halflives by regime (seconds)
_ALPHA_HALFLIFE: dict = {
    "TREND":    18000.0,   # 5 h — trends last
    "VOLATILE": 7200.0,    # 2 h — volatility windows are brief
    "NORMAL":   14400.0,   # 4 h — default
    "RANGE":    10800.0,   # 3 h
    "CHOP":     3600.0,    # 1 h — edge evaporates fastest in chop
}

# Regime ordering for distance calculation
_REGIME_RANK: dict = {"CHOP": 1, "RANGE": 2, "NORMAL": 3, "VOLATILE": 4, "TREND": 5}


@dataclass
class ThesisAssessment:
    thesis_score:      float  # [0–1]
    conviction_budget: float  # [0–1]
    alpha_remaining:   float  # [0–1]
    regime_drift:      float  # [0–1]
    reason:            str


@dataclass
class _SymbolState:
    conviction:    float = 1.0
    entry_regime:  str   = "NORMAL"
    prev_regime:   str   = "NORMAL"
    last_update:   float = field(default_factory=time.monotonic)


class ThesisEngine:
    """
    Stateful thesis + conviction tracker.  One instance lives inside PerceptionEngine.
    Call compute() every tick; call forget() when position closes.
    """

    def __init__(self) -> None:
        self._state: dict[str, _SymbolState] = {}

    def compute(
        self,
        symbol:       str,
        regime:       str,
        r_net:        float,
        r_lock:       float,
        giveback_pct: float,
        age_sec:      float,
    ) -> ThesisAssessment:
        now   = time.monotonic()
        state = self._state.setdefault(
            symbol,
            _SymbolState(entry_regime=regime, prev_regime=regime, last_update=now),
        )

        # ── Passive time recharge ─────────────────────────────────────────────
        dt = max(now - state.last_update, 0.0)
        state.conviction = min(1.0, state.conviction + _PASSIVE_RECHARGE_PER_SEC * dt)
        state.last_update = now

        # ── Regime mutation events ────────────────────────────────────────────
        regime_drift = _regime_distance(state.entry_regime, regime)

        if state.prev_regime != regime:
            # Regime changed THIS tick
            tick_drift = _regime_distance(state.prev_regime, regime)
            if tick_drift > 0.4:
                state.conviction = max(0.0, state.conviction - _DRAIN_REGIME_MUTATION)
            elif tick_drift > 0.0:
                state.conviction = max(0.0, state.conviction - _DRAIN_REGIME_MUTATION * 0.4)
        state.prev_regime = regime

        # ── Thesis health components ──────────────────────────────────────────
        components = []

        # Component A: profitability maintenance
        if r_net >= r_lock:
            components.append(1.0)
        elif r_net > 0.0:
            components.append(0.55)
        elif r_net > -0.5:
            components.append(0.25)
            state.conviction = max(0.0, state.conviction - _DRAIN_THESIS_INVALIDATED * 0.4)
        else:
            components.append(0.0)
            state.conviction = max(0.0, state.conviction - _DRAIN_THESIS_INVALIDATED * 0.7)

        # Component B: giveback tolerance
        if giveback_pct < 0.20:
            components.append(1.0)
        elif giveback_pct < 0.40:
            components.append(0.65)
        elif giveback_pct < 0.65:
            components.append(0.25)
            state.conviction = max(0.0, state.conviction - _DRAIN_HIGH_GIVEBACK)
        else:
            components.append(0.0)
            state.conviction = max(0.0, state.conviction - _DRAIN_THESIS_INVALIDATED * 0.5)

        # Component C: regime alignment with entry thesis
        align = _regime_alignment(state.entry_regime, regime)
        components.append(align)
        if align < 0.5:
            state.conviction = max(0.0, state.conviction - _DRAIN_ADVERSE_FLOW)

        thesis_score = sum(components) / len(components)

        # ── Conviction positive events ────────────────────────────────────────
        if r_net > r_lock and giveback_pct < 0.15:
            state.conviction = min(1.0, state.conviction + _RECHARGE_THESIS_CONFIRM)
        if state.entry_regime == regime:
            state.conviction = min(1.0, state.conviction + _RECHARGE_REGIME_STABLE)

        # ── Alpha halflife ────────────────────────────────────────────────────
        halflife = _ALPHA_HALFLIFE.get(state.entry_regime, 14400.0)
        alpha_remaining = math.exp(-age_sec / halflife)

        # ── Reason string ─────────────────────────────────────────────────────
        parts = []
        if regime_drift > 0.5:
            parts.append(f"RegimeDrift({state.entry_regime}→{regime} d={regime_drift:.1f})")
        if thesis_score < 0.5:
            parts.append(f"WeakThesis({thesis_score:.0%})")
        if state.conviction < 0.4:
            parts.append(f"LowConviction({state.conviction:.0%})")
        if alpha_remaining < 0.3:
            parts.append(f"AlphaDecay({alpha_remaining:.0%}@{age_sec/3600:.1f}h)")

        return ThesisAssessment(
            thesis_score=round(thesis_score, 3),
            conviction_budget=round(state.conviction, 3),
            alpha_remaining=round(alpha_remaining, 4),
            regime_drift=round(regime_drift, 3),
            reason=" | ".join(parts) if parts else "thesis_healthy",
        )

    def drain_conviction(self, symbol: str, amount: float, reason: str = "") -> None:
        """External signal (e.g. toxicity spike) can drain conviction directly."""
        state = self._state.get(symbol)
        if state:
            state.conviction = max(0.0, state.conviction - amount)
            _log.debug("%s conviction drain %.2f (%s) → %.3f", symbol, amount, reason, state.conviction)

    def forget(self, symbol: str) -> None:
        self._state.pop(symbol, None)


# ── Regime helpers ────────────────────────────────────────────────────────────

def _regime_distance(a: str, b: str) -> float:
    """[0–1] distance between two regimes on the CHOP→TREND scale."""
    ra = _REGIME_RANK.get(a.upper(), 3)
    rb = _REGIME_RANK.get(b.upper(), 3)
    return abs(ra - rb) / 4.0


def _regime_alignment(entry: str, current: str) -> float:
    """[0–1] how well current regime still supports a trade opened in entry regime."""
    if entry.upper() == current.upper():
        return 1.0
    dist = _regime_distance(entry, current)
    return max(0.0, 1.0 - dist * 1.6)
'''

# ═══════════════════════════════════════════════════════════════════════════════
# MODULE 3: kill_chain.py
# ═══════════════════════════════════════════════════════════════════════════════

KILL_CHAIN = '''"""
kill_chain: Hierarchical Exit Kill Chain L0–L5 (PATCH-10A).

Maps multiple continuous signals into a graduated pressure level.

Level   Name                Action              Description
─────   ─────────────────── ──────────────────  ──────────────────────────────────
L0      Nominal             HOLD                All signals healthy
L1      Soft Defense        TIGHTEN_TRAIL       Rising pressure, tighten leash
L2      Economic De-risk    PARTIAL_CLOSE_25    Reduce exposure, protect capital
L3      Tactical Unwind     PARTIAL_CLOSE_50    Aggressive de-risk, thesis weakening
L4      Thesis Death        FULL_CLOSE          Edge gone, environment hostile
L5      Quarantine Exit     FULL_CLOSE          Emergency, crowded invalidation

Kill chain is advisory — hard guards (drawdown stop / SL breach) still override.
Kill chain level is written into PerceptionResult and consumed by DecisionEngine.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass

_log = logging.getLogger("exit_management_agent.kill_chain")

_LEVEL_ACTIONS = {
    0: "HOLD",
    1: "TIGHTEN_TRAIL",
    2: "PARTIAL_CLOSE_25",
    3: "PARTIAL_CLOSE_50",
    4: "FULL_CLOSE",
    5: "FULL_CLOSE",
}

_LEVEL_QTY = {0: 0.0, 1: 0.0, 2: 0.25, 3: 0.50, 4: 1.0, 5: 1.0}
_LEVEL_CONF = {0: 0.40, 1: 0.55, 2: 0.65, 3: 0.75, 4: 0.88, 5: 0.95}


@dataclass
class KillChainAssessment:
    level:        int    # 0–5
    action:       str    # recommended action at this level
    qty_fraction: float  # 0.0 for non-close levels
    confidence:   float
    reason:       str


def evaluate(
    r_net:              float,
    r_lock:             float,
    r_t1:               float,
    giveback_pct:       float,
    age_sec:            float,
    pnl_velocity:       float = 0.0,
    pnl_convexity:      float = 0.0,
    pnl_retention:      float = 1.0,
    pnl_noise_ratio:    float = 0.0,
    pnl_geo_valid:      bool  = False,
    conviction_budget:  float = 1.0,
    thesis_score:       float = 1.0,
    alpha_remaining:    float = 1.0,
) -> KillChainAssessment:
    """
    Evaluate kill chain level from all available signals.
    Only fires >= L2 when position is in profit (r_net > 0) or r_net < -0.3 (loss protection).
    """
    level   = 0
    reasons: list[str] = []

    # ── Conviction collapse (epistemic death) ─────────────────────────────────
    if conviction_budget < 0.15:
        level = max(level, 5);  reasons.append(f"conviction={conviction_budget:.0%}")
    elif conviction_budget < 0.25:
        level = max(level, 4);  reasons.append(f"conviction={conviction_budget:.0%}")
    elif conviction_budget < 0.40:
        level = max(level, 3);  reasons.append(f"conv_low={conviction_budget:.0%}")
    elif conviction_budget < 0.60:
        level = max(level, 2);  reasons.append(f"conv_fade={conviction_budget:.0%}")
    elif conviction_budget < 0.75:
        level = max(level, 1);  reasons.append(f"conv_erode={conviction_budget:.0%}")

    # ── Thesis health collapse ────────────────────────────────────────────────
    if thesis_score < 0.15:
        level = max(level, 5);  reasons.append(f"thesis_dead={thesis_score:.0%}")
    elif thesis_score < 0.25:
        level = max(level, 4);  reasons.append(f"thesis_dying={thesis_score:.0%}")
    elif thesis_score < 0.40:
        level = max(level, 3);  reasons.append(f"thesis_weak={thesis_score:.0%}")
    elif thesis_score < 0.60:
        level = max(level, 2);  reasons.append(f"thesis_degrade={thesis_score:.0%}")

    # ── Alpha halflife exhaustion ─────────────────────────────────────────────
    if alpha_remaining < 0.08:
        level = max(level, 4);  reasons.append(f"alpha_exhausted={alpha_remaining:.0%}")
    elif alpha_remaining < 0.15:
        level = max(level, 3);  reasons.append(f"alpha_dying={alpha_remaining:.0%}")
    elif alpha_remaining < 0.25:
        level = max(level, 2);  reasons.append(f"alpha_decay={alpha_remaining:.0%}")
    elif alpha_remaining < 0.45:
        level = max(level, 1);  reasons.append(f"alpha_half={alpha_remaining:.0%}")

    # ── PnL geometry signals (only when valid and in profit) ─────────────────
    if pnl_geo_valid and r_net > 0.2:
        # Retention collapse
        if pnl_retention < 0.20:
            level = max(level, 4);  reasons.append(f"retention_critical={pnl_retention:.0%}")
        elif pnl_retention < 0.35:
            level = max(level, 3);  reasons.append(f"retention_collapse={pnl_retention:.0%}")
        elif pnl_retention < 0.50:
            level = max(level, 2);  reasons.append(f"retention_low={pnl_retention:.0%}")

        # Velocity turned negative while in meaningful profit
        if pnl_velocity < -0.0008 and r_net > 0.5:
            level = max(level, 3);  reasons.append(f"vel_negative={pnl_velocity:.5f}")
        elif pnl_velocity < -0.0003 and r_net > 0.3:
            level = max(level, 2);  reasons.append(f"vel_declining={pnl_velocity:.5f}")

        # Convexity flipped strongly negative (momentum melting)
        if pnl_convexity < -0.002:
            level = max(level, 3);  reasons.append(f"convex_collapse={pnl_convexity:.4f}")
        elif pnl_convexity < -0.0008:
            level = max(level, 2);  reasons.append(f"convex_negative={pnl_convexity:.4f}")
        elif pnl_convexity < -0.0002:
            level = max(level, 1);  reasons.append(f"convex_bending={pnl_convexity:.4f}")

        # Chaotic payoff shape
        if pnl_noise_ratio > 6.0:
            level = max(level, 2);  reasons.append(f"chaotic_pnl={pnl_noise_ratio:.1f}")
        elif pnl_noise_ratio > 3.5:
            level = max(level, 1);  reasons.append(f"noisy_pnl={pnl_noise_ratio:.1f}")

    # ── Giveback + profit guards ──────────────────────────────────────────────
    if r_net >= r_lock:  # only fire giveback signals when in profit
        if giveback_pct > 0.80:
            level = max(level, 4);  reasons.append(f"giveback_severe={giveback_pct:.0%}")
        elif giveback_pct > 0.65:
            level = max(level, 3);  reasons.append(f"giveback_heavy={giveback_pct:.0%}")
        elif giveback_pct > 0.50:
            level = max(level, 2);  reasons.append(f"giveback_high={giveback_pct:.0%}")

    # ── Time / stale position ─────────────────────────────────────────────────
    if age_sec > 28800 and r_net < r_t1 * 0.3:  # 8 h, less than 30% of T1
        level = max(level, 2);  reasons.append(f"stale_no_delivery={age_sec/3600:.1f}h")
    elif age_sec > 14400 and r_net < 0.1:          # 4 h, barely profitable
        level = max(level, 1);  reasons.append(f"stale_flat={age_sec/3600:.1f}h")

    # ── Safety floor: L >= 2 only fires when position has some profit context ─
    # (Prevents a fresh losing position from being de-risked via kill chain;
    #  that path goes through drawdown stop / SL breach instead.)
    if level >= 2 and r_net < 0.0 and r_net > -0.5:
        level = min(level, 1)   # downgrade to soft defence only below break-even

    reason_str = " | ".join(reasons) if reasons else "nominal"
    return KillChainAssessment(
        level=level,
        action=_LEVEL_ACTIONS[level],
        qty_fraction=_LEVEL_QTY[level],
        confidence=_LEVEL_CONF[level],
        reason=reason_str,
    )
'''

# ═══════════════════════════════════════════════════════════════════════════════
# MODULE 4: council.py
# ═══════════════════════════════════════════════════════════════════════════════

COUNCIL = '''"""
council: Multi-Personality Exit Deliberation Layer (PATCH-10C).

Six personalities deliberate on whether to exit, then a meta-voter decides.

  The Coward        — Capital protector.  Exits early on any profit threat.
  The Predator      — Trend worshipper.   Lets winners run if alpha is fresh.
  The Statistician  — Pure math.          Exits when edge < transaction cost.
  The Regime Priest — Environment judge.  Exits when regime is hostile to thesis.
  The Contrarian    — Anti-consensus.     Flips when everyone agrees too easily.
  The Obituarist    — Death-pattern detector.  Matches current state to trade obituaries.

Meta-voter: weighted consensus with a 3-vote veto rule for FULL_CLOSE.

The council is advisory — its recommendation is used as an additional signal in
DecisionEngine alongside kill_chain_level.  Council does NOT override hard guards.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field

_log = logging.getLogger("exit_management_agent.council")

# ── Action intensity scale (for weighted averaging) ───────────────────────────
_ACTION_SCORE: dict = {
    "HOLD":              0.00,
    "TIGHTEN_TRAIL":     0.10,
    "MOVE_TO_BREAKEVEN": 0.10,
    "PARTIAL_CLOSE_25":  0.35,
    "PARTIAL_CLOSE_50":  0.55,
    "PARTIAL_CLOSE_75":  0.75,
    "FULL_CLOSE":        1.00,
    "TIME_STOP_EXIT":    1.00,
}

_SCORE_TO_ACTION: list = [
    (0.85, "FULL_CLOSE"),
    (0.60, "PARTIAL_CLOSE_50"),
    (0.30, "PARTIAL_CLOSE_25"),
    (0.08, "TIGHTEN_TRAIL"),
    (0.00, "HOLD"),
]

# Personality weights — must sum to 1.0
_W = {
    "Coward":       0.15,
    "Predator":     0.15,
    "Statistician": 0.25,
    "RegimePriest": 0.20,
    "Contrarian":   0.10,
    "Obituarist":   0.15,
}


@dataclass
class Vote:
    personality: str
    action:      str
    confidence:  float
    weight:      float
    reason:      str


@dataclass
class CouncilDecision:
    action:          str
    confidence:      float
    consensus_score: float  # [0=chaos, 1=unanimous]
    reasoning:       str
    votes:           list = field(default_factory=list)


def deliberate(
    r_net:             float,
    r_lock:            float,
    r_t1:              float,
    giveback_pct:      float,
    age_sec:           float,
    thesis_score:      float = 1.0,
    conviction_budget: float = 1.0,
    alpha_remaining:   float = 1.0,
    regime_drift:      float = 0.0,
    pnl_velocity:      float = 0.0,
    pnl_convexity:     float = 0.0,
    pnl_retention:     float = 1.0,
    kill_chain_level:  int   = 0,
) -> CouncilDecision:
    """Run all council members and return meta-voted decision."""
    votes = [
        _coward(giveback_pct, pnl_velocity, pnl_convexity, pnl_retention, r_net, r_lock),
        _predator(r_net, r_t1, alpha_remaining, pnl_velocity),
        _statistician(alpha_remaining, conviction_budget, age_sec),
        _regime_priest(regime_drift, thesis_score),
        _contrarian(kill_chain_level, conviction_budget),
        _obituarist(giveback_pct, pnl_retention, thesis_score, conviction_budget, r_net, r_lock, age_sec),
    ]
    return _meta_vote(votes)


# ── Personality implementations ───────────────────────────────────────────────

def _coward(gb: float, vel: float, cvx: float, ret: float, r_net: float, r_lock: float) -> Vote:
    """Capital protector — exits early on profit threat signals."""
    score = 0.0
    reasons = []

    if r_net >= r_lock:        # only fire when we have profit to protect
        if gb > 0.50:          score += 0.45; reasons.append(f"gb={gb:.0%}")
        elif gb > 0.30:        score += 0.25; reasons.append(f"gb={gb:.0%}")
        if vel < -0.0005:      score += 0.30; reasons.append("vel-")
        elif vel < -0.0002:    score += 0.15; reasons.append("vel_slow-")
        if cvx < -0.001:       score += 0.20; reasons.append("cvx-")
        if ret < 0.40:         score += 0.35; reasons.append(f"ret={ret:.0%}")
        elif ret < 0.60:       score += 0.15; reasons.append(f"ret={ret:.0%}")

    return Vote("Coward", _s2a(min(score, 1.0)), min(score + 0.1, 0.88), _W["Coward"],
                " ".join(reasons) or "calm")


def _predator(r_net: float, r_t1: float, alpha: float, vel: float) -> Vote:
    """Trend worshipper — let winners run when alpha is fresh."""
    score = 0.0
    reasons = []

    if alpha < 0.30:       score += 0.45; reasons.append(f"alpha_dead={alpha:.0%}")
    elif alpha < 0.55:     score += 0.20; reasons.append(f"alpha_fade={alpha:.0%}")

    if r_net < r_t1 * 0.4 and r_net > 0.0:
        score += 0.25; reasons.append("under_t1")  # hasn't delivered yet — predator exits
    if vel < -0.001 and r_net > r_t1:
        score += 0.20; reasons.append("vel_neg_at_target")

    return Vote("Predator", _s2a(min(score, 1.0)), 0.60, _W["Predator"],
                " ".join(reasons) or "let_it_run")


def _statistician(alpha: float, conviction: float, age_sec: float) -> Vote:
    """Pure math — exits when expected edge < transaction cost proxy."""
    score = 0.0
    reasons = []

    score += (1.0 - alpha)       * 0.50;
    if alpha < 0.30:             reasons.append(f"alpha={alpha:.0%}")
    score += (1.0 - conviction)  * 0.40;
    if conviction < 0.50:        reasons.append(f"conv={conviction:.0%}")
    score += min(age_sec / 28800.0, 1.0) * 0.10  # 8 h = full time pressure

    return Vote("Statistician", _s2a(min(score, 1.0)), 0.72, _W["Statistician"],
                " ".join(reasons) or "edge_ok")


def _regime_priest(drift: float, thesis: float) -> Vote:
    """Environment judge — exits when regime is hostile to original thesis."""
    score = drift * 0.55 + (1.0 - thesis) * 0.45
    reasons = []
    if drift  > 0.5: reasons.append(f"drift={drift:.1f}")
    if thesis < 0.5: reasons.append(f"thesis={thesis:.0%}")
    return Vote("RegimePriest", _s2a(min(score, 1.0)), 0.68, _W["RegimePriest"],
                " ".join(reasons) or "regime_stable")


def _contrarian(kc_level: int, conviction: float) -> Vote:
    """Anti-consensus — challenges when the crowd is overconfident."""
    score = 0.0
    reasons = []

    # High kill chain + high conviction = suspicious groupthink → contrarian holds
    if kc_level >= 4 and conviction > 0.80:
        score = 0.00; reasons.append("anti-panic-hold")
    # Low kill chain + collapsed conviction = hidden decay → contrarian exits
    elif kc_level <= 1 and conviction < 0.20:
        score = 0.55; reasons.append("hidden_decay")
    else:
        score = kc_level * 0.12

    return Vote("Contrarian", _s2a(min(score, 1.0)), 0.38, _W["Contrarian"],
                " ".join(reasons) or "no_contrarian_signal")


def _obituarist(gb: float, ret: float, thesis: float, conv: float,
                r_net: float, r_lock: float, age_sec: float) -> Vote:
    """
    Pattern-matcher to classic trade death patterns:
    1. Slow Bleed    — gradual giveback + weak thesis + old age
    2. False Summit  — great gain then retention collapse
    3. Empty Tank    — conviction dead, barely profitable, very old
    4. Thesis Ghost  — thesis died but position lingers in profit
    """
    score = 0.0
    reasons = []

    # Pattern 1: Slow Bleed
    if gb > 0.40 and thesis < 0.45 and age_sec > 3600:
        score += 0.45; reasons.append("SlowBleed")

    # Pattern 2: False Summit
    if ret < 0.30 and r_net > r_lock * 0.5:
        score += 0.50; reasons.append("FalseSummit")

    # Pattern 3: Empty Tank
    if conv < 0.20 and r_net < 0.25 and age_sec > 7200:
        score += 0.55; reasons.append("EmptyTank")

    # Pattern 4: Thesis Ghost
    if thesis < 0.20 and r_net >= r_lock:
        score += 0.45; reasons.append("ThesisGhost")

    return Vote("Obituarist", _s2a(min(score, 1.0)), 0.65, _W["Obituarist"],
                " ".join(reasons) or "alive")


# ── Meta-voter ────────────────────────────────────────────────────────────────

def _meta_vote(votes: list[Vote]) -> CouncilDecision:
    total_w = sum(v.weight for v in votes)
    w_score = sum(_ACTION_SCORE.get(v.action, 0.0) * v.weight * v.confidence
                  for v in votes) / max(total_w, 1e-6)

    # Veto rule: 3+ high-confidence FULL_CLOSE votes → push to 0.88+
    full_close_veto = sum(1 for v in votes if v.action == "FULL_CLOSE" and v.confidence > 0.65)
    if full_close_veto >= 3:
        w_score = max(w_score, 0.88)

    action = _s2a(w_score)

    # Consensus: low variance = high agreement
    a_scores = [_ACTION_SCORE.get(v.action, 0.0) for v in votes]
    mean_s   = sum(a_scores) / len(a_scores)
    var      = sum((s - mean_s) ** 2 for s in a_scores) / len(a_scores)
    consensus = max(0.0, 1.0 - var * 5.0)

    non_hold = [f"{v.personality}→{v.action}({v.reason})"
                for v in votes if v.action != "HOLD"]
    reasoning = " || ".join(non_hold[:3]) if non_hold else "unanimous_hold"

    return CouncilDecision(
        action=action,
        confidence=round(min(w_score + 0.08, 0.95), 3),
        consensus_score=round(consensus, 3),
        reasoning=reasoning,
        votes=votes,
    )


def _s2a(score: float) -> str:
    for threshold, action in _SCORE_TO_ACTION:
        if score >= threshold:
            return action
    return "HOLD"
'''

# ═══════════════════════════════════════════════════════════════════════════════
# PATCH: models.py — add 16 new PerceptionResult fields
# ═══════════════════════════════════════════════════════════════════════════════

OLD_MODELS = (
    '    market_regime: str = "NORMAL"   # CHOP|RANGE|VOLATILE|TREND|NORMAL (PATCH-9A)\n'
    '    market_regime: str = "NORMAL"   # CHOP|RANGE|VOLATILE|TREND|NORMAL (PATCH-9A)'
)

NEW_MODELS = '''    market_regime: str = "NORMAL"   # CHOP|RANGE|VOLATILE|TREND|NORMAL (PATCH-9A)

    # ── PnL Geometry (PATCH-10A) ──────────────────────────────────────────────
    pnl_velocity:     float = 0.0   # dR/dt  — positive = gaining R
    pnl_acceleration: float = 0.0   # d²R/dt²
    pnl_convexity:    float = 0.0   # 2nd-order coeff — negative = concave down
    pnl_retention:    float = 1.0   # current_R / max_R_ever  [0–1]
    pnl_noise_ratio:  float = 0.0   # chaotic payoff indicator
    pnl_geo_valid:    bool  = False  # False until enough samples

    # ── Thesis + Conviction (PATCH-10B) ───────────────────────────────────────
    thesis_score:      float = 1.0  # [0–1] trade idea health
    conviction_budget: float = 1.0  # [0–1] epistemic capital remaining
    alpha_remaining:   float = 1.0  # [0–1] exp(-age/halflife) edge estimate
    regime_drift:      float = 0.0  # [0–1] regime distance from entry

    # ── Kill Chain (PATCH-10A) ────────────────────────────────────────────────
    kill_chain_level:  int  = 0     # 0–5 graduated exit pressure
    kill_chain_action: str  = "HOLD"
    kill_chain_reason: str  = ""

    # ── Council deliberation (PATCH-10C) ──────────────────────────────────────
    council_action:    str   = "HOLD"
    council_confidence:float = 0.0
    council_consensus: float = 1.0
    council_reasoning: str   = ""'''

# ═══════════════════════════════════════════════════════════════════════════════
# PATCH: perception.py — integrate new engines
# ═══════════════════════════════════════════════════════════════════════════════

OLD_PERC_IMPORTS = "from .models import PerceptionResult, PositionSnapshot"

NEW_PERC_IMPORTS = """from .models import PerceptionResult, PositionSnapshot
from .pnl_geometry import PnLGeometryTracker
from .thesis_engine import ThesisEngine
from .kill_chain import evaluate as kc_evaluate
from .council import deliberate as council_deliberate"""

OLD_PERC_INIT = "        self._peak_prices: dict = {}"

NEW_PERC_INIT = """        self._peak_prices: dict = {}
        # PATCH-10: Exit Brain v2 — stateful engines
        self._pnl_tracker  = PnLGeometryTracker()
        self._thesis_engine = ThesisEngine()"""

OLD_PERC_RETURN = """        return PerceptionResult(
            snapshot=snapshot,
            R_net=r_net,
            peak_price=peak_price,
            age_sec=age_sec,
            distance_to_sl_pct=dist_to_sl,
            giveback_pct=giveback,
            r_effective_t1=r_t1,
            r_effective_t2=r_t2,
            r_effective_t3=r_t3,
            r_effective_lock=r_lock,
            market_regime=regime,
        )"""

NEW_PERC_RETURN = """        # ── 8. PnL geometry (PATCH-10A) ─────────────────────────────────
        self._pnl_tracker.update(snapshot.symbol, r_net)
        geo = self._pnl_tracker.get(snapshot.symbol)

        # ── 9. Thesis + conviction (PATCH-10B) ───────────────────────────────
        thesis = self._thesis_engine.compute(
            symbol=snapshot.symbol,
            regime=regime,
            r_net=r_net,
            r_lock=r_lock,
            giveback_pct=giveback,
            age_sec=age_sec,
        )

        # ── 10. Kill chain (PATCH-10A) ────────────────────────────────────────
        kc = kc_evaluate(
            r_net=r_net,
            r_lock=r_lock,
            r_t1=r_t1,
            giveback_pct=giveback,
            age_sec=age_sec,
            pnl_velocity=geo.velocity,
            pnl_convexity=geo.convexity,
            pnl_retention=geo.retention,
            pnl_noise_ratio=geo.noise_ratio,
            pnl_geo_valid=geo.valid,
            conviction_budget=thesis.conviction_budget,
            thesis_score=thesis.thesis_score,
            alpha_remaining=thesis.alpha_remaining,
        )

        # ── 11. Council deliberation (PATCH-10C) ──────────────────────────────
        cd = council_deliberate(
            r_net=r_net,
            r_lock=r_lock,
            r_t1=r_t1,
            giveback_pct=giveback,
            age_sec=age_sec,
            thesis_score=thesis.thesis_score,
            conviction_budget=thesis.conviction_budget,
            alpha_remaining=thesis.alpha_remaining,
            regime_drift=thesis.regime_drift,
            pnl_velocity=geo.velocity,
            pnl_convexity=geo.convexity,
            pnl_retention=geo.retention,
            kill_chain_level=kc.level,
        )

        return PerceptionResult(
            snapshot=snapshot,
            R_net=r_net,
            peak_price=peak_price,
            age_sec=age_sec,
            distance_to_sl_pct=dist_to_sl,
            giveback_pct=giveback,
            r_effective_t1=r_t1,
            r_effective_t2=r_t2,
            r_effective_t3=r_t3,
            r_effective_lock=r_lock,
            market_regime=regime,
            # PnL Geometry
            pnl_velocity=geo.velocity,
            pnl_acceleration=geo.acceleration,
            pnl_convexity=geo.convexity,
            pnl_retention=geo.retention,
            pnl_noise_ratio=geo.noise_ratio,
            pnl_geo_valid=geo.valid,
            # Thesis
            thesis_score=thesis.thesis_score,
            conviction_budget=thesis.conviction_budget,
            alpha_remaining=thesis.alpha_remaining,
            regime_drift=thesis.regime_drift,
            # Kill chain
            kill_chain_level=kc.level,
            kill_chain_action=kc.action,
            kill_chain_reason=kc.reason,
            # Council
            council_action=cd.action,
            council_confidence=cd.confidence,
            council_consensus=cd.consensus_score,
            council_reasoning=cd.reasoning,
        )"""

# Also patch forget() to clear new engines
OLD_PERC_FORGET = '''    def forget(self, symbol: str) -> None:
        """Remove in-memory peak tracking for a symbol (call on position close)."""
        self._peak_prices.pop(symbol, None)'''

NEW_PERC_FORGET = '''    def forget(self, symbol: str) -> None:
        """Remove in-memory peak tracking for a symbol (call on position close)."""
        self._peak_prices.pop(symbol, None)
        self._pnl_tracker.forget(symbol)
        self._thesis_engine.forget(symbol)'''

# ═══════════════════════════════════════════════════════════════════════════════
# PATCH: decision_engine.py — add kill chain + council rules
# ═══════════════════════════════════════════════════════════════════════════════

OLD_DE_RULE4 = '''        # ── Rule 4: Adaptive harvest (PATCH-9A: regime-aware + giveback-triggered) ───'''

NEW_DE_RULE4 = '''        # ── Rule 3b: Kill chain L4/L5 — thesis death / quarantine ───────────────
        _kc_level = getattr(p, "kill_chain_level", 0)
        if _kc_level >= 4:
            _kc_reason = getattr(p, "kill_chain_reason", "kill_chain_L4")
            return _decision(
                snap=snap,
                action=FULL_CLOSE,
                reason=f"KillChain-L{_kc_level}: {_kc_reason}",
                urgency=URGENCY_HIGH,
                r_net=p.R_net,
                confidence=0.88,
                suggested_qty_fraction=1.0,
                dry_run=dry_run,
            )

        # ── Rule 3c: Kill chain L3 — tactical unwind (only in profit) ────────
        if _kc_level >= 3 and p.R_net >= p.r_effective_lock:
            _kc_reason = getattr(p, "kill_chain_reason", "kill_chain_L3")
            return _decision(
                snap=snap,
                action=PARTIAL_CLOSE_50,
                reason=f"KillChain-L3 tactical unwind: {_kc_reason}",
                urgency=URGENCY_MEDIUM,
                r_net=p.R_net,
                confidence=0.78,
                suggested_qty_fraction=0.50,
                dry_run=dry_run,
            )

        # ── Rule 3d: Kill chain L2 — economic de-risk (only in profit) ───────
        if _kc_level >= 2 and p.R_net >= p.r_effective_lock:
            _kc_reason = getattr(p, "kill_chain_reason", "kill_chain_L2")
            return _decision(
                snap=snap,
                action=PARTIAL_CLOSE_25,
                reason=f"KillChain-L2 de-risk: {_kc_reason}",
                urgency=URGENCY_MEDIUM,
                r_net=p.R_net,
                confidence=0.68,
                suggested_qty_fraction=0.25,
                dry_run=dry_run,
            )

        # ── Rule 3e: Council consensus exit (only in profit, L1+ kill chain) ──
        _council_action = getattr(p, "council_action", "HOLD")
        _council_conf   = getattr(p, "council_confidence", 0.0)
        _council_cons   = getattr(p, "council_consensus", 1.0)
        if (
            _council_action not in ("HOLD", "TIGHTEN_TRAIL", "MOVE_TO_BREAKEVEN")
            and _council_conf > 0.65
            and _council_cons > 0.55
            and _kc_level >= 1
            and p.R_net >= p.r_effective_lock
        ):
            _council_reason = getattr(p, "council_reasoning", "council_consensus")
            return _decision(
                snap=snap,
                action=_council_action,
                reason=f"Council({_council_cons:.0%}): {_council_reason}",
                urgency=URGENCY_MEDIUM,
                r_net=p.R_net,
                confidence=_council_conf,
                suggested_qty_fraction={
                    "PARTIAL_CLOSE_25": 0.25,
                    "PARTIAL_CLOSE_50": 0.50,
                    "PARTIAL_CLOSE_75": 0.75,
                    "FULL_CLOSE":       1.0,
                }.get(_council_action, 0.25),
                dry_run=dry_run,
            )

        # ── Rule 4: Adaptive harvest (PATCH-9A: regime-aware + giveback-triggered) ───'''

# ═══════════════════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════════════════

def main():
    print("\n=== PATCH-10: Exit Brain v2 ===\n")

    # 1. Create new modules
    create("pnl_geometry.py",  PNL_GEOMETRY)
    create("thesis_engine.py", THESIS_ENGINE)
    create("kill_chain.py",    KILL_CHAIN)
    create("council.py",       COUNCIL)

    # 2. Patch models.py
    patch("models.py", OLD_MODELS, NEW_MODELS)

    # 3. Patch perception.py — imports
    patch("perception.py", OLD_PERC_IMPORTS, NEW_PERC_IMPORTS)

    # 4. Patch perception.py — __init__
    patch("perception.py", OLD_PERC_INIT, NEW_PERC_INIT)

    # 5. Patch perception.py — return statement
    patch("perception.py", OLD_PERC_RETURN, NEW_PERC_RETURN)

    # 6. Patch perception.py — forget()
    patch("perception.py", OLD_PERC_FORGET, NEW_PERC_FORGET)

    # 7. Patch decision_engine.py — kill chain + council rules
    patch("decision_engine.py", OLD_DE_RULE4, NEW_DE_RULE4)

    print("\nAll PATCH-10 changes applied.")
    print("Next: restart quantum-exit-management-agent\n")


if __name__ == "__main__":
    main()
