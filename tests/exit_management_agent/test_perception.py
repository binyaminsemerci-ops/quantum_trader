"""Tests for perception pure helper functions.

All functions under test are pure (no async, no Redis).
PerceptionEngine.compute() is tested via integration in test_heartbeat.py.
"""
from __future__ import annotations

import pytest

from microservices.exit_management_agent.models import PositionSnapshot
from microservices.exit_management_agent.perception import (
    _compute_distance_to_sl,
    _compute_giveback,
    _compute_r_net,
    _get_r_targets,
)


# ── Factory ───────────────────────────────────────────────────────────────────


def _snap(**kw) -> PositionSnapshot:
    base = dict(
        symbol="BTCUSDT",
        side="LONG",
        quantity=0.01,
        entry_price=30_000.0,
        mark_price=31_500.0,
        leverage=1.0,
        stop_loss=29_000.0,
        take_profit=0.0,
        unrealized_pnl=15.0,
        entry_risk_usdt=10.0,
        sync_timestamp=1_700_000_000.0,
    )
    base.update(kw)
    return PositionSnapshot(**base)


# ── _compute_r_net ────────────────────────────────────────────────────────────


class TestComputeRNet:
    def test_uses_entry_risk_usdt_when_available(self):
        # LONG: entry=30000, mark=31000, qty=0.01, risk_usdt=10
        # pnl = (31000-30000)*0.01 = 10 USD  → R = 10/10 = 1.0
        snap = _snap(entry_price=30_000.0, mark_price=31_000.0, entry_risk_usdt=10.0)
        assert _compute_r_net(snap, 31_000.0) == pytest.approx(1.0)

    def test_negative_r_long_drawdown(self):
        # LONG: entry=30000, mark=29500, qty=0.01, risk_usdt=10
        # pnl = (29500-30000)*0.01 = -5.0 USD  → R = -5.0/10 = -0.5
        snap = _snap(entry_price=30_000.0, entry_risk_usdt=10.0)
        r = _compute_r_net(snap, 29_500.0)
        assert r == pytest.approx(-0.5)

    def test_short_at_profit(self):
        # SHORT: entry=30000, mark=29000, qty=0.01, risk=10
        # pnl = (30000-29000)*0.01 = 10  → R = 1.0
        snap = _snap(
            side="SHORT",
            entry_price=30_000.0,
            mark_price=29_000.0,
            entry_risk_usdt=10.0,
        )
        assert _compute_r_net(snap, 29_000.0) == pytest.approx(1.0)

    def test_falls_back_to_sl_distance(self):
        # No entry_risk_usdt; use SL distance
        # LONG: entry=30000, SL=29000 → dist=1000 * 0.01 = 10 USD
        snap = _snap(
            entry_price=30_000.0, stop_loss=29_000.0, entry_risk_usdt=0.0
        )
        assert _compute_r_net(snap, 31_000.0) == pytest.approx(1.0)

    def test_returns_zero_when_nothing_computable(self):
        snap = _snap(entry_price=30_000.0, stop_loss=0.0, entry_risk_usdt=0.0, unrealized_pnl=0.0)
        assert _compute_r_net(snap, 30_000.0) == pytest.approx(0.0)


# ── _compute_distance_to_sl ───────────────────────────────────────────────────


class TestComputeDistanceToSL:
    def test_long_above_sl_positive(self):
        snap = _snap(stop_loss=29_000.0)
        d = _compute_distance_to_sl(snap, 31_500.0)
        assert d == pytest.approx((31_500.0 - 29_000.0) / 31_500.0)
        assert d > 0.0

    def test_long_at_sl_zero(self):
        snap = _snap(stop_loss=30_000.0)
        d = _compute_distance_to_sl(snap, 30_000.0)
        assert d == pytest.approx(0.0)

    def test_long_below_sl_negative(self):
        snap = _snap(stop_loss=29_000.0)
        d = _compute_distance_to_sl(snap, 28_500.0)
        assert d < 0.0

    def test_short_above_sl_positive(self):
        # SHORT: SL=31000, mark=29000 → (SL-mark)/mark = (31000-29000)/29000
        snap = _snap(side="SHORT", stop_loss=31_000.0)
        d = _compute_distance_to_sl(snap, 29_000.0)
        assert d == pytest.approx((31_000.0 - 29_000.0) / 29_000.0)
        assert d > 0.0

    def test_no_sl_returns_zero(self):
        snap = _snap(stop_loss=0.0)
        assert _compute_distance_to_sl(snap, 31_500.0) == pytest.approx(0.0)

    def test_zero_mark_price_returns_zero(self):
        snap = _snap(stop_loss=29_000.0)
        assert _compute_distance_to_sl(snap, 0.0) == pytest.approx(0.0)


# ── _compute_giveback ─────────────────────────────────────────────────────────


class TestComputeGiveback:
    def test_at_peak_no_giveback(self):
        snap = _snap(entry_price=30_000.0)
        assert _compute_giveback(snap, 31_500.0, 31_500.0) == pytest.approx(0.0)

    def test_fully_given_back(self):
        # entry=30000, peak=31000, mark=30000  → 100% giveback
        snap = _snap(entry_price=30_000.0)
        assert _compute_giveback(snap, 30_000.0, 31_000.0) == pytest.approx(1.0)

    def test_half_given_back(self):
        # entry=30000, peak=31000, mark=30500
        # peak_profit=10, current=5  → 50% giveback
        snap = _snap(entry_price=30_000.0)
        assert _compute_giveback(snap, 30_500.0, 31_000.0) == pytest.approx(0.5)

    def test_no_peak_profit_returns_zero(self):
        snap = _snap(entry_price=30_000.0)
        assert _compute_giveback(snap, 30_000.0, 30_000.0) == pytest.approx(0.0)

    def test_clipped_to_one_on_overshooting(self):
        # mark below entry for LONG → giveback > 1; must be clipped to 1.0
        snap = _snap(entry_price=30_000.0)
        g = _compute_giveback(snap, 29_000.0, 31_000.0)
        assert g == pytest.approx(1.0)

    def test_short_half_given_back(self):
        # SHORT: entry=30000, peak=29000 (lower is better for short)
        # mark=29500 → peak_profit=1000*0.01=10, current=500*0.01=5  → 50%
        snap = _snap(
            side="SHORT", entry_price=30_000.0, quantity=0.01
        )
        g = _compute_giveback(snap, 29_500.0, 29_000.0)
        assert g == pytest.approx(0.5)


# ── _get_r_targets ────────────────────────────────────────────────────────────


class TestGetRTargets:
    def test_leverage_1_base_targets(self):
        t1, lock = _get_r_targets(1.0)
        # At lev=1: T1=2.0/sqrt(1)=2.0, lock=1.5/sqrt(1)=1.5
        assert t1 == pytest.approx(2.0, rel=0.01)
        assert lock == pytest.approx(1.5, rel=0.01)

    def test_higher_leverage_reduces_targets(self):
        t1_1x, _ = _get_r_targets(1.0)
        t1_10x, _ = _get_r_targets(10.0)
        assert t1_10x < t1_1x

    def test_leverage_4_halves_t1(self):
        t1, _ = _get_r_targets(4.0)
        # sqrt(4)=2 → T1=2.0/2=1.0
        assert t1 == pytest.approx(1.0, rel=0.01)
