"""Tests for decision_engine: one test class per rule, priority order.

All tests are synchronous (DecisionEngine.decide() is not async).
No Redis required.
"""
from __future__ import annotations

import pytest

from microservices.exit_management_agent.decision_engine import (
    FULL_CLOSE,
    HOLD,
    MOVE_TO_BREAKEVEN,
    PARTIAL_CLOSE_25,
    TIME_STOP_EXIT,
    TIGHTEN_TRAIL,
    DecisionEngine,
    _compute_breakeven_sl,
    _compute_tighter_sl,
    _sl_below_entry,
)
from microservices.exit_management_agent.models import PerceptionResult, PositionSnapshot


# ── Factories ─────────────────────────────────────────────────────────────────


def _snap(
    side: str = "LONG",
    entry_price: float = 30_000.0,
    mark_price: float = 31_500.0,
    stop_loss: float = 29_000.0,
    leverage: float = 1.0,
    quantity: float = 0.01,
) -> PositionSnapshot:
    return PositionSnapshot(
        symbol="BTCUSDT",
        side=side,
        quantity=quantity,
        entry_price=entry_price,
        mark_price=mark_price,
        leverage=leverage,
        stop_loss=stop_loss,
        take_profit=35_000.0,
        unrealized_pnl=(mark_price - entry_price) * quantity
        if side == "LONG"
        else (entry_price - mark_price) * quantity,
        entry_risk_usdt=abs(entry_price - stop_loss) * quantity,
        sync_timestamp=1_700_000_000.0,
    )


def _perc(
    snap: PositionSnapshot,
    R_net: float = 0.2,
    peak_price: float | None = None,
    age_sec: float = 100.0,
    distance_to_sl_pct: float = 0.05,
    giveback_pct: float = 0.0,
    r_effective_t1: float = 2.0,
    r_effective_lock: float = 1.5,
) -> PerceptionResult:
    return PerceptionResult(
        snapshot=snap,
        R_net=R_net,
        peak_price=peak_price or snap.mark_price,
        age_sec=age_sec,
        distance_to_sl_pct=distance_to_sl_pct,
        giveback_pct=giveback_pct,
        r_effective_t1=r_effective_t1,
        r_effective_lock=r_effective_lock,
    )


engine = DecisionEngine(max_hold_sec=14_400.0)


# ── Rule 1: Emergency drawdown ────────────────────────────────────────────────


class TestRule1EmergencyDrawdown:
    def test_r_below_minus_1_5_fires_full_close(self):
        dec = engine.decide(_perc(_snap(), R_net=-2.0))
        assert dec.action == FULL_CLOSE
        assert dec.urgency == "EMERGENCY"
        assert dec.suggested_qty_fraction == pytest.approx(1.0)

    def test_exactly_at_minus_1_5_fires(self):
        dec = engine.decide(_perc(_snap(), R_net=-1.5))
        assert dec.action == FULL_CLOSE

    def test_just_above_minus_1_5_does_not_fire(self):
        dec = engine.decide(_perc(_snap(), R_net=-1.4))
        assert dec.action != FULL_CLOSE

    def test_dry_run_is_true(self):
        dec = engine.decide(_perc(_snap(), R_net=-2.0))
        assert dec.dry_run is True


# ── Rule 2: Hard SL breach ────────────────────────────────────────────────────


class TestRule2HardSLBreach:
    def test_long_price_below_sl_fires(self):
        snap = _snap(entry_price=30_000.0, mark_price=28_500.0, stop_loss=29_000.0)
        # distance_to_sl = (28500-29000)/28500 < 0
        dist = (28_500.0 - 29_000.0) / 28_500.0
        dec = engine.decide(_perc(snap, R_net=-0.5, distance_to_sl_pct=dist))
        assert dec.action == FULL_CLOSE
        assert dec.urgency == "EMERGENCY"

    def test_short_price_above_sl_fires(self):
        snap = _snap(
            side="SHORT",
            entry_price=30_000.0,
            mark_price=31_500.0,
            stop_loss=31_000.0,
        )
        # SHORT: SL=31000, mark=31500  → dist=(31000-31500)/31500 < 0
        dist = (31_000.0 - 31_500.0) / 31_500.0
        dec = engine.decide(_perc(snap, R_net=-0.5, distance_to_sl_pct=dist))
        assert dec.action == FULL_CLOSE

    def test_sl_not_breached_does_not_fire(self):
        snap = _snap(entry_price=30_000.0, mark_price=30_500.0, stop_loss=29_000.0)
        dec = engine.decide(_perc(snap, R_net=0.1, distance_to_sl_pct=0.05))
        assert dec.action != FULL_CLOSE

    def test_no_sl_set_does_not_fire(self):
        snap = _snap(stop_loss=0.0, mark_price=28_000.0)
        dec = engine.decide(_perc(snap, R_net=-0.3, distance_to_sl_pct=0.0))
        assert dec.action != FULL_CLOSE


# ── Rule 3: Time stop ─────────────────────────────────────────────────────────


class TestRule3TimeStop:
    def test_age_exceeds_max_fires(self):
        snap = _snap()
        dec = engine.decide(_perc(snap, R_net=0.5, age_sec=14_401.0))
        assert dec.action == TIME_STOP_EXIT
        assert dec.urgency == "MEDIUM"

    def test_exactly_at_limit_fires(self):
        snap = _snap()
        dec = engine.decide(_perc(snap, R_net=0.5, age_sec=14_400.0))
        assert dec.action == TIME_STOP_EXIT

    def test_just_under_limit_does_not_fire(self):
        snap = _snap()
        dec = engine.decide(_perc(snap, R_net=0.5, age_sec=14_399.0))
        assert dec.action != TIME_STOP_EXIT


# ── Rule 4: Partial harvest ───────────────────────────────────────────────────


class TestRule4PartialHarvest:
    def test_r_at_t1_fires(self):
        snap = _snap()
        dec = engine.decide(_perc(snap, R_net=2.0, r_effective_t1=2.0))
        assert dec.action == PARTIAL_CLOSE_25
        assert dec.suggested_qty_fraction == pytest.approx(0.25)

    def test_r_above_t1_fires(self):
        snap = _snap()
        dec = engine.decide(_perc(snap, R_net=3.5, r_effective_t1=2.0))
        assert dec.action == PARTIAL_CLOSE_25

    def test_r_below_t1_does_not_fire(self):
        snap = _snap()
        dec = engine.decide(_perc(snap, R_net=1.9, r_effective_t1=2.0))
        assert dec.action != PARTIAL_CLOSE_25


# ── Rule 5: Tighten trail ─────────────────────────────────────────────────────


class TestRule5TightenTrail:
    def test_high_giveback_at_profit_fires(self):
        snap = _snap()
        dec = engine.decide(
            _perc(snap, R_net=1.6, r_effective_lock=1.5, giveback_pct=0.60)
        )
        assert dec.action == TIGHTEN_TRAIL
        assert dec.suggested_sl is not None

    def test_exactly_50pct_giveback_fires(self):
        snap = _snap()
        dec = engine.decide(
            _perc(snap, R_net=1.6, r_effective_lock=1.5, giveback_pct=0.50)
        )
        assert dec.action == TIGHTEN_TRAIL

    def test_giveback_below_threshold_does_not_fire(self):
        snap = _snap()
        dec = engine.decide(
            _perc(snap, R_net=1.6, r_effective_lock=1.5, giveback_pct=0.40)
        )
        assert dec.action != TIGHTEN_TRAIL

    def test_r_below_lock_does_not_fire_even_with_high_giveback(self):
        snap = _snap()
        dec = engine.decide(
            _perc(snap, R_net=1.0, r_effective_lock=1.5, giveback_pct=0.90)
        )
        assert dec.action != TIGHTEN_TRAIL


# ── Rule 6: Move to break-even ────────────────────────────────────────────────


class TestRule6MoveToBreakeven:
    def test_long_sl_below_entry_fires(self):
        snap = _snap(entry_price=30_000.0, stop_loss=29_000.0, mark_price=31_500.0)
        dec = engine.decide(
            _perc(snap, R_net=1.6, r_effective_lock=1.5, giveback_pct=0.1)
        )
        assert dec.action == MOVE_TO_BREAKEVEN
        assert dec.suggested_sl is not None
        # Suggested SL should be above entry for LONG
        assert dec.suggested_sl > snap.entry_price

    def test_short_sl_above_entry_fires(self):
        snap = _snap(
            side="SHORT",
            entry_price=30_000.0,
            stop_loss=31_000.0,
            mark_price=28_500.0,
        )
        dec = engine.decide(
            _perc(snap, R_net=1.6, r_effective_lock=1.5, giveback_pct=0.1)
        )
        assert dec.action == MOVE_TO_BREAKEVEN
        assert dec.suggested_sl < snap.entry_price

    def test_sl_already_above_entry_does_not_fire(self):
        snap = _snap(entry_price=30_000.0, stop_loss=30_100.0, mark_price=31_500.0)
        dec = engine.decide(
            _perc(snap, R_net=1.6, r_effective_lock=1.5, giveback_pct=0.1)
        )
        assert dec.action != MOVE_TO_BREAKEVEN


# ── Rule 7: Hold ──────────────────────────────────────────────────────────────


class TestRule7Hold:
    def test_no_criteria_returns_hold(self):
        snap = _snap()
        dec = engine.decide(_perc(snap, R_net=0.3))
        assert dec.action == HOLD

    def test_hold_is_not_actionable(self):
        snap = _snap()
        dec = engine.decide(_perc(snap, R_net=0.3))
        assert dec.is_actionable is False

    def test_hold_dry_run_true(self):
        snap = _snap()
        dec = engine.decide(_perc(snap, R_net=0.3))
        assert dec.dry_run is True


# ── SL helper functions ───────────────────────────────────────────────────────


class TestSLHelpers:
    def test_sl_below_entry_long(self):
        snap = _snap(entry_price=30_000.0, stop_loss=29_000.0)
        assert _sl_below_entry(snap) is True

    def test_sl_above_entry_long(self):
        snap = _snap(entry_price=30_000.0, stop_loss=30_100.0)
        assert _sl_below_entry(snap) is False

    def test_no_sl_long(self):
        snap = _snap(stop_loss=0.0)
        assert _sl_below_entry(snap) is True

    def test_sl_below_entry_short(self):
        # SHORT: SL above entry = not yet at BE → True
        snap = _snap(side="SHORT", entry_price=30_000.0, stop_loss=31_000.0)
        assert _sl_below_entry(snap) is True

    def test_sl_at_entry_short(self):
        snap = _snap(side="SHORT", entry_price=30_000.0, stop_loss=29_900.0)
        assert _sl_below_entry(snap) is False

    def test_compute_breakeven_sl_long(self):
        snap = _snap(entry_price=30_000.0)
        sl = _compute_breakeven_sl(snap)
        assert sl == pytest.approx(30_000.0 * 1.002)

    def test_compute_breakeven_sl_short(self):
        snap = _snap(side="SHORT", entry_price=30_000.0)
        sl = _compute_breakeven_sl(snap)
        assert sl == pytest.approx(30_000.0 * 0.998)

    def test_compute_tighter_sl_long_tightens(self):
        snap = _snap(entry_price=30_000.0, stop_loss=29_000.0)
        sl = _compute_tighter_sl(snap, peak_price=32_000.0)
        expected = 32_000.0 * 0.995
        assert sl == pytest.approx(expected)

    def test_compute_tighter_sl_never_loosens_long(self):
        # If existing SL is higher than proposed, keep existing.
        snap = _snap(entry_price=30_000.0, stop_loss=31_800.0)
        sl = _compute_tighter_sl(snap, peak_price=32_000.0)
        assert sl == pytest.approx(max(32_000.0 * 0.995, 31_800.0))

    def test_compute_tighter_sl_short_tightens(self):
        snap = _snap(
            side="SHORT", entry_price=30_000.0, stop_loss=31_000.0, mark_price=28_000.0
        )
        sl = _compute_tighter_sl(snap, peak_price=27_000.0)
        expected = 27_000.0 * 1.005
        assert sl == pytest.approx(expected)
