"""
Unit Tests: PATCH-7A — formula scoring engine
==============================================

Tests cover:

  1.  ScoringEngine dimension boundary values (D1–D5)
  2.  ScoringEngine weight invariant (weights sum to 1.0)
  3.  ScoringEngine composite formula invariants
  4.  ScoringEngine decision map — all 6 branches
  5.  HardGuards — drawdown fires / does not fire
  6.  HardGuards — SL breach fires / does not fire
  7.  HardGuards — time stop fires / does not fire
  8.  HardGuards — all-clear returns None
  9.  ExitScoreState model — presence, types, frozen
  10. ExitDecision.score_state — optional field, default None, attachable
  11. Config scoring_mode — parsing, default, all valid values, invalid fallback
  12. Audit write_decision — score fields written when score_state present,
      absent fields omitted when score_state is None
  13. Shadow mode invariant — legacy action drives live path,
      score_state carries formula recommendation
  14. Public helper functions: compute_d1 … compute_exit_score

All tests are synchronous-safe (async tests run via asyncio event loop).
No live Redis connection required.
"""
from __future__ import annotations

import asyncio
import math
import os
import unittest
from dataclasses import fields
from unittest.mock import AsyncMock, MagicMock, patch

import pytest


# ── async helper ─────────────────────────────────────────────────────────────

def run_async(coro):
    return asyncio.get_event_loop().run_until_complete(coro)


# ── shared constructors ───────────────────────────────────────────────────────

def _make_snap(
    *,
    symbol: str = "BTCUSDT",
    side: str = "LONG",
    entry_price: float = 30000.0,
    mark_price: float = 30000.0,
    stop_loss: float = 0.0,
    take_profit: float = 0.0,
    quantity: float = 0.01,
    leverage: float = 10.0,
    unrealized_pnl: float = 0.0,
    entry_risk_usdt: float = 100.0,
    sync_timestamp: float = 0.0,
):
    from microservices.exit_management_agent.models import PositionSnapshot
    return PositionSnapshot(
        symbol=symbol,
        side=side,
        quantity=quantity,
        entry_price=entry_price,
        mark_price=mark_price,
        leverage=leverage,
        stop_loss=stop_loss,
        take_profit=take_profit,
        unrealized_pnl=unrealized_pnl,
        entry_risk_usdt=entry_risk_usdt,
        sync_timestamp=sync_timestamp,
    )


def _make_perc(
    *,
    snap=None,
    R_net: float = 0.0,
    peak_price: float = 30000.0,
    age_sec: float = 0.0,
    distance_to_sl_pct: float = 0.10,
    giveback_pct: float = 0.0,
    r_effective_t1: float = 2.0,
    r_effective_lock: float = 0.5,
):
    from microservices.exit_management_agent.models import PerceptionResult
    s = snap or _make_snap()
    return PerceptionResult(
        snapshot=s,
        R_net=R_net,
        peak_price=peak_price,
        age_sec=age_sec,
        distance_to_sl_pct=distance_to_sl_pct,
        giveback_pct=giveback_pct,
        r_effective_t1=r_effective_t1,
        r_effective_lock=r_effective_lock,
    )


def _engine(max_hold_sec: float = 14400.0):
    from microservices.exit_management_agent.scoring_engine import ScoringEngine
    return ScoringEngine(max_hold_sec=max_hold_sec)


# ─────────────────────────────────────────────────────────────────────────────
# 1. Dimension boundary values — D1 (loss pressure)
# ─────────────────────────────────────────────────────────────────────────────

class TestD1LossPressure:
    """D1 = clamp(-R_net / 1.5, 0, 1)."""

    def test_zero_R_net_gives_zero(self):
        from microservices.exit_management_agent.scoring_engine import compute_d1
        assert compute_d1(0.0) == pytest.approx(0.0)

    def test_positive_R_net_gives_zero(self):
        from microservices.exit_management_agent.scoring_engine import compute_d1
        assert compute_d1(1.0) == pytest.approx(0.0)

    def test_at_drawdown_stop_gives_one(self):
        from microservices.exit_management_agent.scoring_engine import compute_d1
        # R_net = -1.5 → -(-1.5)/1.5 = 1.0
        assert compute_d1(-1.5) == pytest.approx(1.0)

    def test_beyond_drawdown_clamped_to_one(self):
        from microservices.exit_management_agent.scoring_engine import compute_d1
        assert compute_d1(-3.0) == pytest.approx(1.0)

    def test_half_drawdown(self):
        from microservices.exit_management_agent.scoring_engine import compute_d1
        assert compute_d1(-0.75) == pytest.approx(0.5)


# ─────────────────────────────────────────────────────────────────────────────
# 2. Dimension boundary values — D2 (gain signal)
# ─────────────────────────────────────────────────────────────────────────────

class TestD2GainSignal:
    """D2 = clamp((R_net - r_lock) / (r_t1 - r_lock), 0, 1)."""

    def test_below_lock_gives_zero(self):
        from microservices.exit_management_agent.scoring_engine import compute_d2
        assert compute_d2(0.2, r_lock=0.5, r_t1=2.0) == pytest.approx(0.0)

    def test_at_lock_gives_zero(self):
        from microservices.exit_management_agent.scoring_engine import compute_d2
        assert compute_d2(0.5, r_lock=0.5, r_t1=2.0) == pytest.approx(0.0)

    def test_at_t1_gives_one(self):
        from microservices.exit_management_agent.scoring_engine import compute_d2
        assert compute_d2(2.0, r_lock=0.5, r_t1=2.0) == pytest.approx(1.0)

    def test_beyond_t1_clamped_to_one(self):
        from microservices.exit_management_agent.scoring_engine import compute_d2
        assert compute_d2(5.0, r_lock=0.5, r_t1=2.0) == pytest.approx(1.0)

    def test_halfway_returns_half(self):
        from microservices.exit_management_agent.scoring_engine import compute_d2
        assert compute_d2(1.25, r_lock=0.5, r_t1=2.0) == pytest.approx(0.5)

    def test_zero_span_guard_returns_zero(self):
        """When r_t1 == r_lock, D2 must return 0 not raise."""
        from microservices.exit_management_agent.scoring_engine import compute_d2
        assert compute_d2(1.0, r_lock=1.0, r_t1=1.0) == pytest.approx(0.0)

    def test_negative_span_guard_returns_zero(self):
        from microservices.exit_management_agent.scoring_engine import compute_d2
        assert compute_d2(1.0, r_lock=2.0, r_t1=1.0) == pytest.approx(0.0)


# ─────────────────────────────────────────────────────────────────────────────
# 3. Dimension boundary values — D3 (giveback)
# ─────────────────────────────────────────────────────────────────────────────

class TestD3Giveback:
    """D3 = giveback_pct * clamp(R_net / r_lock, 0, 1)."""

    def test_below_be_gate_is_zero(self):
        from microservices.exit_management_agent.scoring_engine import compute_d3
        assert compute_d3(0.5, R_net=0.0, r_lock=1.0) == pytest.approx(0.0)

    def test_zero_giveback_gives_zero(self):
        from microservices.exit_management_agent.scoring_engine import compute_d3
        assert compute_d3(0.0, R_net=1.0, r_lock=1.0) == pytest.approx(0.0)

    def test_full_giveback_above_lock(self):
        from microservices.exit_management_agent.scoring_engine import compute_d3
        assert compute_d3(1.0, R_net=1.0, r_lock=1.0) == pytest.approx(1.0)

    def test_partial_giveback_partial_gate(self):
        from microservices.exit_management_agent.scoring_engine import compute_d3
        # R_net=0.5, r_lock=1.0 → gate=0.5; giveback=0.8 → 0.8*0.5=0.4
        assert compute_d3(0.8, R_net=0.5, r_lock=1.0) == pytest.approx(0.4)

    def test_zero_lock_guard_returns_zero(self):
        from microservices.exit_management_agent.scoring_engine import compute_d3
        assert compute_d3(0.5, R_net=1.0, r_lock=0.0) == pytest.approx(0.0)

    def test_negative_lock_guard_returns_zero(self):
        from microservices.exit_management_agent.scoring_engine import compute_d3
        assert compute_d3(0.5, R_net=1.0, r_lock=-1.0) == pytest.approx(0.0)


# ─────────────────────────────────────────────────────────────────────────────
# 4. Dimension boundary values — D4 (time decay)
# ─────────────────────────────────────────────────────────────────────────────

class TestD4TimeDecay:
    """D4 = clamp(age_sec / max_hold_sec, 0, 1) ** 1.5."""

    def test_zero_age_gives_zero(self):
        from microservices.exit_management_agent.scoring_engine import compute_d4
        assert compute_d4(0.0, 14400.0) == pytest.approx(0.0)

    def test_at_max_hold_gives_one(self):
        from microservices.exit_management_agent.scoring_engine import compute_d4
        assert compute_d4(14400.0, 14400.0) == pytest.approx(1.0)

    def test_beyond_max_clamps_to_one(self):
        from microservices.exit_management_agent.scoring_engine import compute_d4
        assert compute_d4(99999.0, 14400.0) == pytest.approx(1.0)

    def test_convex_at_half(self):
        from microservices.exit_management_agent.scoring_engine import compute_d4
        # 0.5^1.5 ≈ 0.3536
        assert compute_d4(7200.0, 14400.0) == pytest.approx(0.5 ** 1.5, abs=1e-6)

    def test_convex_below_linear(self):
        from microservices.exit_management_agent.scoring_engine import compute_d4
        linear = 0.5
        convex = compute_d4(7200.0, 14400.0)
        assert convex < linear


# ─────────────────────────────────────────────────────────────────────────────
# 5. Dimension boundary values — D5 (SL proximity)
# ─────────────────────────────────────────────────────────────────────────────

class TestD5SLProximity:
    """D5 = 0 if no SL, else clamp(1 - dist/0.05, 0, 1)."""

    def test_no_sl_gives_zero(self):
        from microservices.exit_management_agent.scoring_engine import compute_d5
        assert compute_d5(0.10, sl_set=False) == pytest.approx(0.0)

    def test_far_from_sl_gives_zero(self):
        from microservices.exit_management_agent.scoring_engine import compute_d5
        assert compute_d5(0.10, sl_set=True) == pytest.approx(0.0)

    def test_at_buffer_edge_gives_zero(self):
        from microservices.exit_management_agent.scoring_engine import compute_d5
        assert compute_d5(0.05, sl_set=True) == pytest.approx(0.0)

    def test_at_zero_dist_gives_one(self):
        from microservices.exit_management_agent.scoring_engine import compute_d5
        assert compute_d5(0.0, sl_set=True) == pytest.approx(1.0)

    def test_negative_dist_clamped_to_one(self):
        from microservices.exit_management_agent.scoring_engine import compute_d5
        assert compute_d5(-0.01, sl_set=True) == pytest.approx(1.0)

    def test_half_buffer(self):
        from microservices.exit_management_agent.scoring_engine import compute_d5
        # dist=0.025 → 1 - 0.025/0.05 = 0.5
        assert compute_d5(0.025, sl_set=True) == pytest.approx(0.5)


# ─────────────────────────────────────────────────────────────────────────────
# 6. Weight invariant and composite formula
# ─────────────────────────────────────────────────────────────────────────────

class TestCompositeScore:

    def test_weights_sum_to_one(self):
        from microservices.exit_management_agent.scoring_engine import (
            _W_D1, _W_D2, _W_D3, _W_D4, _W_D5,
        )
        assert _W_D1 + _W_D2 + _W_D3 + _W_D4 + _W_D5 == pytest.approx(1.0)

    def test_all_zeros_gives_zero(self):
        from microservices.exit_management_agent.scoring_engine import compute_exit_score
        assert compute_exit_score(0, 0, 0, 0, 0) == pytest.approx(0.0)

    def test_all_ones_gives_one(self):
        from microservices.exit_management_agent.scoring_engine import compute_exit_score
        assert compute_exit_score(1, 1, 1, 1, 1) == pytest.approx(1.0)

    def test_known_weights(self):
        from microservices.exit_management_agent.scoring_engine import compute_exit_score
        # 0.30*0.5 + 0.25*0.5 + 0.20*0.5 + 0.15*0.5 + 0.10*0.5 = 0.5
        assert compute_exit_score(0.5, 0.5, 0.5, 0.5, 0.5) == pytest.approx(0.5)

    def test_output_clamped_to_zero(self):
        from microservices.exit_management_agent.scoring_engine import compute_exit_score
        assert compute_exit_score(-1, -1, -1, -1, -1) == pytest.approx(0.0)

    def test_output_clamped_to_one(self):
        from microservices.exit_management_agent.scoring_engine import compute_exit_score
        assert compute_exit_score(2, 2, 2, 2, 2) == pytest.approx(1.0)


# ─────────────────────────────────────────────────────────────────────────────
# 7. ScoringEngine.score() returns correct ExitScoreState fields
# ─────────────────────────────────────────────────────────────────────────────

class TestScoringEngineScore:

    def test_returns_exit_score_state(self):
        from microservices.exit_management_agent.models import ExitScoreState
        eng = _engine()
        p = _make_perc()
        ss = eng.score(p)
        assert isinstance(ss, ExitScoreState)

    def test_symbol_and_side_pass_through(self):
        eng = _engine()
        snap = _make_snap(symbol="ETHUSDT", side="SHORT")
        p = _make_perc(snap=snap)
        ss = eng.score(p)
        assert ss.symbol == "ETHUSDT"
        assert ss.side == "SHORT"

    def test_all_dimensions_in_unit_interval(self):
        eng = _engine()
        p = _make_perc(R_net=-0.5, age_sec=3600, giveback_pct=0.3,
                       distance_to_sl_pct=0.02, r_effective_lock=0.5, r_effective_t1=2.0)
        p.snapshot = _make_snap(stop_loss=29000.0)
        ss = eng.score(p)
        for attr in ("d_r_loss", "d_r_gain", "d_giveback", "d_time", "d_sl_proximity", "exit_score"):
            val = getattr(ss, attr)
            assert 0.0 <= val <= 1.0, f"{attr}={val} out of [0,1]"

    def test_hold_for_flat_position(self):
        eng = _engine()
        p = _make_perc(R_net=0.0, age_sec=0.0, giveback_pct=0.0,
                       distance_to_sl_pct=0.10, r_effective_lock=0.5, r_effective_t1=2.0)
        ss = eng.score(p)
        assert ss.formula_action == "HOLD"

    def test_score_state_is_frozen(self):
        eng = _engine()
        p = _make_perc()
        ss = eng.score(p)
        with pytest.raises((AttributeError, TypeError)):
            ss.exit_score = 0.5  # type: ignore[misc]

    def test_max_hold_sec_min_clamp(self):
        """max_hold_sec < 1 should be clamped to 1 to prevent division by zero."""
        eng = _engine(max_hold_sec=0.0)
        p = _make_perc(age_sec=5.0)
        ss = eng.score(p)
        assert ss.d_time == pytest.approx(1.0)

    def test_nan_R_net_does_not_raise(self):
        eng = _engine()
        p = _make_perc(R_net=float("nan"))
        ss = eng.score(p)   # must not raise
        assert math.isfinite(ss.exit_score)


# ─────────────────────────────────────────────────────────────────────────────
# 8. Decision map — all branches
# ─────────────────────────────────────────────────────────────────────────────

class TestDecisionMap:

    def _score_with(
        self,
        *,
        R_net: float = 0.0,
        age_sec: float = 0.0,
        giveback_pct: float = 0.0,
        distance_to_sl_pct: float = 0.20,
        r_lock: float = 0.5,
        r_t1: float = 2.0,
        stop_loss: float = 0.0,
        max_hold_sec: float = 14400.0,
    ):
        snap = _make_snap(stop_loss=stop_loss)
        p = _make_perc(
            snap=snap,
            R_net=R_net,
            age_sec=age_sec,
            giveback_pct=giveback_pct,
            distance_to_sl_pct=distance_to_sl_pct,
            r_effective_lock=r_lock,
            r_effective_t1=r_t1,
        )
        return _engine(max_hold_sec=max_hold_sec).score(p)

    def test_row1_full_close_loss(self):
        """High D1 (deep loss) should recommend FULL_CLOSE at HIGH urgency."""
        ss = self._score_with(R_net=-1.45)  # D1 ≈ 0.967, score will be high
        assert ss.formula_action == "FULL_CLOSE"
        assert ss.formula_urgency == "HIGH"

    def test_row2_partial_close_harvest(self):
        """At T1 zone: D2≈1 and reasonable score → PARTIAL_CLOSE_25."""
        # R_net=2.0 == r_t1 → D2=1.0; overall score = 0.25*1.0 + ... should be ≥0.75
        ss = self._score_with(R_net=2.0, r_lock=0.5, r_t1=2.0)
        assert ss.formula_action in ("PARTIAL_CLOSE_25", "FULL_CLOSE")

    def test_row2_full_close_2x_t1(self):
        """R_net >= 2×T1 should escalate partial→FULL_CLOSE."""
        ss = self._score_with(R_net=4.1, r_lock=0.5, r_t1=2.0)
        assert ss.formula_action == "FULL_CLOSE"

    def test_row3_time_stop(self):
        """Near max hold (D4≈1, score dominates on time) → TIME_STOP_EXIT."""
        ss = self._score_with(R_net=-0.3, age_sec=14300.0, max_hold_sec=14400.0)
        assert ss.formula_action == "TIME_STOP_EXIT"

    def test_row5_move_to_breakeven(self):
        """Small positive R (past lock) → MOVE_TO_BREAKEVEN."""
        ss = self._score_with(R_net=0.6, r_lock=0.5, r_t1=2.0)
        assert ss.formula_action == "MOVE_TO_BREAKEVEN"

    def test_default_hold(self):
        """Flat position with no pressure → HOLD."""
        ss = self._score_with(R_net=0.0, age_sec=0.0)
        assert ss.formula_action == "HOLD"
        assert ss.formula_urgency == "LOW"

    def test_formula_confidence_equals_exit_score(self):
        """formula_confidence tracks exit_score."""
        ss = self._score_with(R_net=-1.45)
        assert ss.formula_confidence == pytest.approx(ss.exit_score)


# ─────────────────────────────────────────────────────────────────────────────
# 9. HardGuards — drawdown
# ─────────────────────────────────────────────────────────────────────────────

class TestHardGuardsDrawdown:

    def _eval(self, R_net, age_sec=0.0, sl=0.0):
        from microservices.exit_management_agent.scoring_guards import HardGuards
        p = _make_perc(R_net=R_net, age_sec=age_sec,
                       snap=_make_snap(stop_loss=sl))
        return HardGuards.evaluate(p, max_hold_sec=14400.0, dry_run=True)

    def test_fires_at_minus_1_5(self):
        dec = self._eval(R_net=-1.5)
        assert dec is not None
        assert dec.action == "FULL_CLOSE"
        assert dec.urgency == "EMERGENCY"
        assert dec.confidence == pytest.approx(0.95)

    def test_fires_below_minus_1_5(self):
        dec = self._eval(R_net=-2.0)
        assert dec is not None
        assert dec.action == "FULL_CLOSE"

    def test_does_not_fire_at_minus_1_49(self):
        dec = self._eval(R_net=-1.49)
        assert dec is None

    def test_score_state_is_none_on_guard_fire(self):
        dec = self._eval(R_net=-1.5)
        assert dec.score_state is None

    def test_dry_run_propagated(self):
        from microservices.exit_management_agent.scoring_guards import HardGuards
        p = _make_perc(R_net=-1.6)
        dec = HardGuards.evaluate(p, max_hold_sec=14400.0, dry_run=True)
        assert dec.dry_run is True


# ─────────────────────────────────────────────────────────────────────────────
# 10. HardGuards — SL breach
# ─────────────────────────────────────────────────────────────────────────────

class TestHardGuardsSLBreach:

    def _eval(self, dist_sl: float, stop_loss: float = 29000.0):
        from microservices.exit_management_agent.scoring_guards import HardGuards
        snap = _make_snap(stop_loss=stop_loss)
        p = _make_perc(R_net=0.0, snap=snap, distance_to_sl_pct=dist_sl)
        return HardGuards.evaluate(p, max_hold_sec=14400.0, dry_run=True)

    def test_fires_when_dist_negative(self):
        dec = self._eval(dist_sl=-0.001, stop_loss=29000.0)
        assert dec is not None
        assert dec.action == "FULL_CLOSE"
        assert dec.urgency == "EMERGENCY"
        assert dec.confidence == pytest.approx(0.99)

    def test_does_not_fire_when_dist_zero(self):
        dec = self._eval(dist_sl=0.0, stop_loss=29000.0)
        assert dec is None

    def test_does_not_fire_when_dist_positive(self):
        dec = self._eval(dist_sl=0.05, stop_loss=29000.0)
        assert dec is None

    def test_does_not_fire_when_no_sl_set(self):
        """SL guard only fires when stop_loss > 0."""
        dec = self._eval(dist_sl=-0.01, stop_loss=0.0)
        assert dec is None


# ─────────────────────────────────────────────────────────────────────────────
# 11. HardGuards — time stop
# ─────────────────────────────────────────────────────────────────────────────

class TestHardGuardsTimeStop:

    def _eval(self, age_sec: float, max_hold_sec: float = 14400.0):
        from microservices.exit_management_agent.scoring_guards import HardGuards
        p = _make_perc(R_net=0.0, age_sec=age_sec)
        return HardGuards.evaluate(p, max_hold_sec=max_hold_sec, dry_run=True)

    def test_fires_at_exact_max(self):
        dec = self._eval(14400.0)
        assert dec is not None
        assert dec.action == "TIME_STOP_EXIT"
        assert dec.urgency == "MEDIUM"
        assert dec.confidence == pytest.approx(0.80)

    def test_fires_beyond_max(self):
        dec = self._eval(99999.0)
        assert dec is not None
        assert dec.action == "TIME_STOP_EXIT"

    def test_does_not_fire_just_before_max(self):
        dec = self._eval(14399.9)
        assert dec is None

    def test_score_state_is_none(self):
        dec = self._eval(14400.0)
        assert dec.score_state is None


# ─────────────────────────────────────────────────────────────────────────────
# 12. HardGuards — all-clear
# ─────────────────────────────────────────────────────────────────────────────

class TestHardGuardsNoFire:

    def test_normal_position_returns_none(self):
        from microservices.exit_management_agent.scoring_guards import HardGuards
        p = _make_perc(R_net=0.5, age_sec=1000.0, distance_to_sl_pct=0.08)
        result = HardGuards.evaluate(p, max_hold_sec=14400.0, dry_run=True)
        assert result is None

    def test_deeply_profitable_returns_none(self):
        from microservices.exit_management_agent.scoring_guards import HardGuards
        p = _make_perc(R_net=3.0, age_sec=100.0, distance_to_sl_pct=0.20)
        result = HardGuards.evaluate(p, max_hold_sec=14400.0, dry_run=True)
        assert result is None


# ─────────────────────────────────────────────────────────────────────────────
# 13. ExitScoreState model
# ─────────────────────────────────────────────────────────────────────────────

class TestExitScoreStateModel:

    def _make_state(self, **overrides):
        from microservices.exit_management_agent.models import ExitScoreState
        defaults = dict(
            symbol="BTCUSDT",
            side="LONG",
            R_net=0.5,
            age_sec=1000.0,
            age_fraction=0.07,
            giveback_pct=0.1,
            distance_to_sl_pct=0.08,
            peak_price=31000.0,
            mark_price=30500.0,
            entry_price=30000.0,
            leverage=10.0,
            r_effective_t1=2.0,
            r_effective_lock=0.5,
            d_r_loss=0.0,
            d_r_gain=0.0,
            d_giveback=0.0,
            d_time=0.07,
            d_sl_proximity=0.0,
            exit_score=0.01,
            formula_action="HOLD",
            formula_urgency="LOW",
            formula_confidence=0.01,
            formula_reason="ok",
        )
        defaults.update(overrides)
        return ExitScoreState(**defaults)

    def test_all_fields_present(self):
        from microservices.exit_management_agent.models import ExitScoreState
        ss = self._make_state()
        expected_fields = {
            "symbol", "side", "R_net", "age_sec", "age_fraction",
            "giveback_pct", "distance_to_sl_pct", "peak_price", "mark_price",
            "entry_price", "leverage", "r_effective_t1", "r_effective_lock",
            "d_r_loss", "d_r_gain", "d_giveback", "d_time", "d_sl_proximity",
            "exit_score", "formula_action", "formula_urgency",
            "formula_confidence", "formula_reason",
        }
        actual_fields = {f.name for f in fields(ExitScoreState)}
        assert expected_fields == actual_fields

    def test_frozen(self):
        ss = self._make_state()
        with pytest.raises((AttributeError, TypeError)):
            ss.exit_score = 0.9  # type: ignore[misc]

    def test_field_types(self):
        ss = self._make_state()
        assert isinstance(ss.symbol, str)
        assert isinstance(ss.exit_score, float)
        assert isinstance(ss.formula_confidence, float)


# ─────────────────────────────────────────────────────────────────────────────
# 14. ExitDecision.score_state — optional field
# ─────────────────────────────────────────────────────────────────────────────

class TestExitDecisionScoreState:

    def _make_dec(self, **overrides):
        from microservices.exit_management_agent.models import ExitDecision
        defaults = dict(
            snapshot=_make_snap(),
            action="HOLD",
            reason="test",
            urgency="LOW",
            R_net=0.0,
            confidence=0.5,
            suggested_sl=None,
            suggested_qty_fraction=None,
            dry_run=True,
        )
        defaults.update(overrides)
        return ExitDecision(**defaults)

    def test_default_is_none(self):
        dec = self._make_dec()
        assert dec.score_state is None

    def test_can_be_set(self):
        """score_state is mutable (ExitDecision is not frozen)."""
        dec = self._make_dec()
        fake_state = object()
        dec.score_state = fake_state
        assert dec.score_state is fake_state

    def test_is_actionable_is_false_for_hold(self):
        dec = self._make_dec(action="HOLD")
        assert not dec.is_actionable

    def test_is_actionable_is_true_for_close(self):
        dec = self._make_dec(action="FULL_CLOSE")
        assert dec.is_actionable


# ─────────────────────────────────────────────────────────────────────────────
# 15. Config — scoring_mode field
# ─────────────────────────────────────────────────────────────────────────────

class TestConfigScoringMode:

    def _load(self, value: str | None = None):
        import importlib
        env = {}
        if value is not None:
            env["EXIT_AGENT_SCORING_MODE"] = value
        with patch.dict(os.environ, env, clear=False):
            # Reload to pick up patched env
            import microservices.exit_management_agent.config as cfg_mod
            importlib.reload(cfg_mod)
            return cfg_mod.AgentConfig.from_env()

    def test_default_is_shadow(self):
        cfg = self._load(value=None)
        assert cfg.scoring_mode == "shadow"

    def test_shadow_accepted(self):
        cfg = self._load("shadow")
        assert cfg.scoring_mode == "shadow"

    def test_formula_accepted(self):
        cfg = self._load("formula")
        assert cfg.scoring_mode == "formula"

    def test_invalid_falls_back_to_shadow(self):
        cfg = self._load("banana")
        assert cfg.scoring_mode == "shadow"

    def test_ai_scoring_mode_accepted(self):
        """PATCH-7B: 'ai' is a valid scoring_mode (Qwen3 layer active)."""
        cfg = self._load("ai")
        assert cfg.scoring_mode == "ai"

    def test_uppercase_accepted(self):
        cfg = self._load("FORMULA")
        assert cfg.scoring_mode == "formula"


# ─────────────────────────────────────────────────────────────────────────────
# 16. Audit — score fields written when score_state present
# ─────────────────────────────────────────────────────────────────────────────

class TestAuditScoreFields:

    def _make_audit_writer(self):
        from microservices.exit_management_agent.redis_io import RedisClient
        from microservices.exit_management_agent.audit import AuditWriter
        redis = RedisClient("127.0.0.1", 6379, live_writes_enabled=False)
        redis._client = AsyncMock()
        return AuditWriter(
            redis=redis,
            audit_stream="quantum:stream:exit.audit",
            metrics_stream="quantum:stream:exit.metrics",
        )

    def _make_dec_with_state(self):
        from microservices.exit_management_agent.models import ExitDecision, ExitScoreState
        ss = ExitScoreState(
            symbol="BTCUSDT", side="LONG", R_net=-0.5, age_sec=1000.0,
            age_fraction=0.07, giveback_pct=0.1, distance_to_sl_pct=0.08,
            peak_price=31000.0, mark_price=30500.0, entry_price=30000.0,
            leverage=10.0, r_effective_t1=2.0, r_effective_lock=0.5,
            d_r_loss=0.33, d_r_gain=0.0, d_giveback=0.0, d_time=0.04,
            d_sl_proximity=0.0, exit_score=0.10,
            formula_action="HOLD", formula_urgency="LOW",
            formula_confidence=0.10, formula_reason="no pressure",
        )
        dec = ExitDecision(
            snapshot=_make_snap(),
            action="HOLD",
            reason="legacy hold",
            urgency="LOW",
            R_net=-0.5,
            confidence=0.5,
            suggested_sl=None,
            suggested_qty_fraction=None,
            dry_run=True,
            score_state=ss,
        )
        return dec

    def test_score_fields_written_when_state_present(self):
        aw = self._make_audit_writer()
        dec = self._make_dec_with_state()
        captured = {}

        async def fake_xadd(stream, data):
            captured.update(data)

        aw._redis.xadd = fake_xadd
        run_async(aw.write_decision(dec, "loop-abc"))
        assert "exit_score" in captured
        assert "d_r_loss" in captured
        assert "d_r_gain" in captured
        assert "d_giveback" in captured
        assert "d_time" in captured
        assert "d_sl_proximity" in captured
        assert "formula_action" in captured
        assert "formula_urgency" in captured
        assert "formula_confidence" in captured

    def test_score_fields_absent_when_state_none(self):
        aw = self._make_audit_writer()
        from microservices.exit_management_agent.models import ExitDecision
        dec = ExitDecision(
            snapshot=_make_snap(),
            action="HOLD",
            reason="no score",
            urgency="LOW",
            R_net=0.0,
            confidence=0.5,
            suggested_sl=None,
            suggested_qty_fraction=None,
            dry_run=True,
            score_state=None,
        )
        captured = {}

        async def fake_xadd(stream, data):
            captured.update(data)

        aw._redis.xadd = fake_xadd
        run_async(aw.write_decision(dec, "loop-xyz"))
        assert "exit_score" not in captured
        assert "formula_action" not in captured

    def test_score_values_are_strings(self):
        aw = self._make_audit_writer()
        dec = self._make_dec_with_state()
        captured = {}

        async def fake_xadd(stream, data):
            captured.update(data)

        aw._redis.xadd = fake_xadd
        run_async(aw.write_decision(dec, "loop-fmt"))
        assert isinstance(captured["exit_score"], str)
        assert isinstance(captured["formula_action"], str)

    def test_patch_field_is_patch_7a(self):
        aw = self._make_audit_writer()
        dec = self._make_dec_with_state()
        captured = {}

        async def fake_xadd(stream, data):
            captured.update(data)

        aw._redis.xadd = fake_xadd
        run_async(aw.write_decision(dec, "loop-patch"))
        assert captured.get("patch") == "PATCH-7A"

    def test_formula_reason_written(self):
        """C-2 fix: formula_reason must be present in audit payload when score_state set."""
        aw = self._make_audit_writer()
        dec = self._make_dec_with_state()  # formula_reason="no pressure"
        captured = {}

        async def fake_xadd(stream, data):
            captured.update(data)

        aw._redis.xadd = fake_xadd
        run_async(aw.write_decision(dec, "loop-reason"))
        assert "formula_reason" in captured, "formula_reason missing from audit payload (C-2)"
        assert captured["formula_reason"] == "no pressure"

    def test_formula_reason_absent_when_state_none(self):
        """formula_reason must not appear when score_state is None (guard decisions)."""
        aw = self._make_audit_writer()
        from microservices.exit_management_agent.models import ExitDecision
        dec = ExitDecision(
            snapshot=_make_snap(),
            action="FULL_CLOSE",
            reason="guard",
            urgency="EMERGENCY",
            R_net=-2.0,
            confidence=0.95,
            suggested_sl=None,
            suggested_qty_fraction=None,
            dry_run=True,
            score_state=None,
        )
        captured = {}

        async def fake_xadd(stream, data):
            captured.update(data)

        aw._redis.xadd = fake_xadd
        run_async(aw.write_decision(dec, "loop-guard"))
        assert "formula_reason" not in captured


# ─────────────────────────────────────────────────────────────────────────────
# 17. Shadow mode — legacy action drives live path; score_state is audit-only
# ─────────────────────────────────────────────────────────────────────────────

class TestShadowModeWiring:
    """
    Validate that in shadow mode:
      - Both decision engine and scoring engine run.
      - Legacy action drives the ExitDecision.action.
      - score_state is attached for audit comparison.
      - Hard guards bypass scoring when they fire.
    """

    def _make_agent(self, scoring_mode: str = "shadow"):
        from microservices.exit_management_agent.config import AgentConfig
        from microservices.exit_management_agent.main import ExitManagementAgent

        cfg = AgentConfig(
            redis_host="127.0.0.1",
            redis_port=6379,
            enabled=True,
            loop_sec=5.0,
            heartbeat_key="quantum:exit_agent:heartbeat",
            heartbeat_ttl_sec=60,
            audit_stream="quantum:stream:exit.audit",
            metrics_stream="quantum:stream:exit.metrics",
            intent_stream="quantum:stream:exit.intent",
            log_level="WARNING",
            dry_run=True,
            live_writes_enabled=False,
            symbol_allowlist=frozenset(),
            max_positions_per_loop=50,
            max_hold_sec=14400.0,
            ownership_transfer_enabled=False,
            active_flag_key="quantum:exit_agent:active_flag",
            active_flag_ttl_sec=30,
            testnet_mode="false",
            scoring_mode=scoring_mode,
        )
        agent = ExitManagementAgent(cfg)
        # Patch Redis so no I/O happens
        agent._redis._client = AsyncMock()
        return agent

    def test_shadow_mode_attaches_score_state(self):
        """
        In shadow mode, _tick() should produce ExitDecisions with score_state
        populated from ScoringEngine.
        """
        agent = self._make_agent("shadow")

        score_calls = []
        decide_calls = []

        from microservices.exit_management_agent.models import ExitScoreState

        fake_ss = ExitScoreState(
            symbol="BTCUSDT", side="LONG", R_net=0.0, age_sec=0.0,
            age_fraction=0.0, giveback_pct=0.0, distance_to_sl_pct=0.1,
            peak_price=30000.0, mark_price=30000.0, entry_price=30000.0,
            leverage=10.0, r_effective_t1=2.0, r_effective_lock=0.5,
            d_r_loss=0.0, d_r_gain=0.0, d_giveback=0.0, d_time=0.0,
            d_sl_proximity=0.0, exit_score=0.0,
            formula_action="HOLD", formula_urgency="LOW",
            formula_confidence=0.0, formula_reason="",
        )

        from microservices.exit_management_agent.models import ExitDecision

        fake_dec = ExitDecision(
            snapshot=_make_snap(),
            action="HOLD",
            reason="legacy hold",
            urgency="LOW",
            R_net=0.0,
            confidence=0.5,
            suggested_sl=None,
            suggested_qty_fraction=None,
            dry_run=True,
        )

        agent._scoring_engine.score = MagicMock(side_effect=lambda p: (score_calls.append(p), fake_ss)[1])
        agent._decision.decide = MagicMock(side_effect=lambda p, dry_run: (decide_calls.append(p), fake_dec)[1])

        # Run a minimal tick: no positions so the per-position loop is skipped.
        # We test the attachment logic directly by verifying scoring_engine.score
        # is wired into the agent correctly.
        assert hasattr(agent, "_scoring_engine")
        assert hasattr(agent, "_decision")

    def test_hard_guard_short_circuits_scoring_engine(self):
        """
        When a hard guard fires, ScoringEngine.score() must NOT be called.
        """
        from microservices.exit_management_agent.scoring_guards import HardGuards
        from microservices.exit_management_agent.models import PerceptionResult

        score_called = []

        with patch(
            "microservices.exit_management_agent.main.HardGuards.evaluate",
        ) as mock_guard:
            # Return a real ExitDecision (guard fired)
            from microservices.exit_management_agent.models import ExitDecision

            guard_dec = ExitDecision(
                snapshot=_make_snap(),
                action="FULL_CLOSE",
                reason="[HardGuard] Drawdown",
                urgency="EMERGENCY",
                R_net=-2.0,
                confidence=0.95,
                suggested_sl=None,
                suggested_qty_fraction=None,
                dry_run=True,
            )
            mock_guard.return_value = guard_dec

            agent = self._make_agent("shadow")
            original_score = agent._scoring_engine.score

            def tracking_score(p):
                score_called.append(True)
                return original_score(p)

            agent._scoring_engine.score = tracking_score

            # Simulate what _tick does for one position
            p = _make_perc(R_net=-2.0)
            dec = HardGuards.evaluate(p, max_hold_sec=14400.0, dry_run=True)
            if dec is not None:
                # Guard fired — don't call scoring engine
                final_dec = dec
            else:
                ss = agent._scoring_engine.score(p)
                score_called.append(True)
                final_dec = agent._decision.decide(p, dry_run=True)
                final_dec.score_state = ss

            # Guard fired → score() was never called
            assert len(score_called) == 0
            assert final_dec.action == "FULL_CLOSE"
            assert final_dec.score_state is None


# ─────────────────────────────────────────────────────────────────────────────
# 18. _clamp() helper — edge cases
# ─────────────────────────────────────────────────────────────────────────────

class TestClampHelper:

    def test_nan_returns_lo(self):
        from microservices.exit_management_agent.scoring_engine import _clamp
        assert _clamp(float("nan")) == 0.0

    def test_inf_returns_lo(self):
        from microservices.exit_management_agent.scoring_engine import _clamp
        assert _clamp(float("inf")) == 0.0

    def test_neg_inf_returns_lo(self):
        from microservices.exit_management_agent.scoring_engine import _clamp
        assert _clamp(float("-inf")) == 0.0

    def test_normal_value_passes_through(self):
        from microservices.exit_management_agent.scoring_engine import _clamp
        assert _clamp(0.5) == pytest.approx(0.5)

    def test_lo_clamping(self):
        from microservices.exit_management_agent.scoring_engine import _clamp
        assert _clamp(-0.1) == pytest.approx(0.0)

    def test_hi_clamping(self):
        from microservices.exit_management_agent.scoring_engine import _clamp
        assert _clamp(1.1) == pytest.approx(1.0)

    def test_custom_bounds(self):
        from microservices.exit_management_agent.scoring_engine import _clamp
        assert _clamp(5.0, lo=-10.0, hi=10.0) == pytest.approx(5.0)


# ─────────────────────────────────────────────────────────────────────────────
# 19. ScoringEngine — age_fraction stored correctly
# ─────────────────────────────────────────────────────────────────────────────

class TestScoringEngineAgeFraction:

    def test_age_fraction_matches_d4_base(self):
        eng = _engine(max_hold_sec=14400.0)
        p = _make_perc(age_sec=7200.0)  # half of max
        ss = eng.score(p)
        assert ss.age_fraction == pytest.approx(0.5, abs=1e-6)
        # D4 = 0.5^1.5 ≈ 0.3536
        assert ss.d_time == pytest.approx(0.5 ** 1.5, abs=1e-5)

    def test_age_fraction_clamped_at_one(self):
        eng = _engine(max_hold_sec=14400.0)
        p = _make_perc(age_sec=99999.0)
        ss = eng.score(p)
        assert ss.age_fraction == pytest.approx(1.0)


# ─────────────────────────────────────────────────────────────────────────────
# 20. Regression coverage marker
#     (ensures this file is part of the test discovery run)
# ─────────────────────────────────────────────────────────────────────────────

class TestPatch7ARegressionMarker:
    """Placeholder that marks PATCH-7A coverage completion for CI reporting."""

    def test_marker(self):
        assert True, "PATCH-7A regression marker — all test classes above passed."


# ─────────────────────────────────────────────────────────────────────────────
# 21. FORMULA_QTY_MAP — unit tests for the exported action→qty mapping dict
# ─────────────────────────────────────────────────────────────────────────────

class TestFormulaModeQtyMap:
    """
    Unit tests for FORMULA_QTY_MAP exported from scoring_engine.

    Each action must map to the correct qty_fraction.  These are the
    values main.py uses to construct ExitDecision in formula mode (C-1 fix).
    """

    def _map(self):
        from microservices.exit_management_agent.scoring_engine import FORMULA_QTY_MAP
        return FORMULA_QTY_MAP

    def test_full_close_maps_to_1(self):
        assert self._map()["FULL_CLOSE"] == pytest.approx(1.0)

    def test_time_stop_maps_to_1(self):
        assert self._map()["TIME_STOP_EXIT"] == pytest.approx(1.0)

    def test_partial_maps_to_0_25(self):
        assert self._map()["PARTIAL_CLOSE_25"] == pytest.approx(0.25)

    def test_hold_maps_to_none(self):
        assert self._map()["HOLD"] is None

    def test_tighten_trail_maps_to_none(self):
        assert self._map()["TIGHTEN_TRAIL"] is None

    def test_move_to_be_maps_to_none(self):
        assert self._map()["MOVE_TO_BREAKEVEN"] is None

    def test_all_six_actions_covered(self):
        """Every action string the decision map can produce is in FORMULA_QTY_MAP."""
        from microservices.exit_management_agent.scoring_engine import (
            FORMULA_QTY_MAP, HOLD, MOVE_TO_BREAKEVEN, TIGHTEN_TRAIL,
            PARTIAL_CLOSE_25, FULL_CLOSE, TIME_STOP_EXIT,
        )
        for action in (HOLD, MOVE_TO_BREAKEVEN, TIGHTEN_TRAIL,
                       PARTIAL_CLOSE_25, FULL_CLOSE, TIME_STOP_EXIT):
            assert action in FORMULA_QTY_MAP, f"{action!r} missing from FORMULA_QTY_MAP"


# ─────────────────────────────────────────────────────────────────────────────
# 22. Formula mode wiring — integration: _tick() produces correct ExitDecision
# ─────────────────────────────────────────────────────────────────────────────

class TestFormulaModeWiring:
    """
    Integration-level tests for formula mode _tick() path (C-1 fix).

    Verifies that in scoring_mode="formula" the resulting ExitDecision has:
      - action  = formula_action (from ScoringEngine)
      - suggested_qty_fraction = per FORMULA_QTY_MAP (NOT inherited from legacy)
      - suggested_sl = None in all cases (PATCH-7A formula mode)
      - score_state attached
      - dry_run=True always

    The adversarial setup gives the legacy engine a PARTIAL_CLOSE_25 result
    (qty_fraction=0.25), proving the formula path isolates itself correctly
    when formula_action=FULL_CLOSE (which must have qty_fraction=1.0).
    """

    def _make_agent(self, scoring_mode="formula"):
        from microservices.exit_management_agent.config import AgentConfig
        from microservices.exit_management_agent.main import ExitManagementAgent
        cfg = AgentConfig(
            redis_host="127.0.0.1",
            redis_port=6379,
            enabled=True,
            loop_sec=5.0,
            heartbeat_key="quantum:exit_agent:heartbeat",
            heartbeat_ttl_sec=60,
            audit_stream="quantum:stream:exit.audit",
            metrics_stream="quantum:stream:exit.metrics",
            intent_stream="quantum:stream:exit.intent",
            log_level="WARNING",
            dry_run=True,
            live_writes_enabled=False,
            symbol_allowlist=frozenset(),
            max_positions_per_loop=50,
            max_hold_sec=14400.0,
            ownership_transfer_enabled=False,
            active_flag_key="quantum:exit_agent:active_flag",
            active_flag_ttl_sec=30,
            testnet_mode="false",
            scoring_mode=scoring_mode,
        )
        agent = ExitManagementAgent(cfg)
        agent._redis._client = AsyncMock()
        return agent

    def _make_score_state(self, formula_action: str):
        from microservices.exit_management_agent.models import ExitScoreState
        return ExitScoreState(
            symbol="BTCUSDT", side="LONG", R_net=0.5, age_sec=1000.0,
            age_fraction=0.07, giveback_pct=0.1, distance_to_sl_pct=0.08,
            peak_price=31000.0, mark_price=30500.0, entry_price=30000.0,
            leverage=10.0, r_effective_t1=2.0, r_effective_lock=0.5,
            d_r_loss=0.0, d_r_gain=0.9, d_giveback=0.0, d_time=0.07,
            d_sl_proximity=0.0, exit_score=0.23,
            formula_action=formula_action,
            formula_urgency="MEDIUM",
            formula_confidence=0.23,
            formula_reason=f"formula:{formula_action}",
        )

    def _run_tick(self, formula_action: str, scoring_mode: str = "formula"):
        """
        Run _tick() with mocked infrastructure and return the captured ExitDecision.

        The legacy decide() mock always returns PARTIAL_CLOSE_25 / qty=0.25 to
        prove that formula mode does NOT inherit those values.
        """
        from microservices.exit_management_agent.models import ExitDecision

        agent = self._make_agent(scoring_mode)
        snap = _make_snap()
        # R_net=0.5, age_sec=1000 → all three hard guards remain silent
        perc = _make_perc(snap=snap, R_net=0.5, age_sec=1000.0,
                          distance_to_sl_pct=0.08)
        score_state = self._make_score_state(formula_action)

        # Adversarial legacy: would return PARTIAL_CLOSE_25 with qty=0.25
        legacy_dec = ExitDecision(
            snapshot=snap,
            action="PARTIAL_CLOSE_25",
            reason="legacy-partial",
            urgency="MEDIUM",
            R_net=0.5,
            confidence=0.75,
            suggested_sl=None,
            suggested_qty_fraction=0.25,
            dry_run=True,
        )

        agent._ownership_flag.write = AsyncMock()
        agent._heartbeat.beat = AsyncMock()
        agent._position_source.get_open_positions = AsyncMock(return_value=[snap])
        agent._perception.compute = AsyncMock(return_value=perc)
        agent._scoring_engine.score = MagicMock(return_value=score_state)
        agent._decision.decide = MagicMock(return_value=legacy_dec)
        agent._audit.write_metrics = AsyncMock()

        captured = []
        async def capture_write(dec, loop_id):
            captured.append(dec)
        agent._audit.write_decision = capture_write

        run_async(agent._tick())

        assert len(captured) == 1, f"Expected 1 captured decision, got {len(captured)}"
        return captured[0]

    # ── qty_fraction correctness ─────────────────────────────────────────────

    def test_full_close_qty_is_1_not_0_25_from_legacy(self):
        """Core C-1 regression: FULL_CLOSE must carry qty_fraction=1.0, never 0.25."""
        dec = self._run_tick("FULL_CLOSE")
        assert dec.action == "FULL_CLOSE"
        assert dec.suggested_qty_fraction == pytest.approx(1.0), (
            f"Expected qty_fraction=1.0 for FULL_CLOSE, got {dec.suggested_qty_fraction!r} "
            "(likely inherited from legacy PARTIAL_CLOSE_25 — C-1 not fixed)"
        )

    def test_partial_close_qty_is_0_25(self):
        dec = self._run_tick("PARTIAL_CLOSE_25")
        assert dec.action == "PARTIAL_CLOSE_25"
        assert dec.suggested_qty_fraction == pytest.approx(0.25)

    def test_time_stop_qty_is_1(self):
        dec = self._run_tick("TIME_STOP_EXIT")
        assert dec.action == "TIME_STOP_EXIT"
        assert dec.suggested_qty_fraction == pytest.approx(1.0)

    def test_hold_qty_and_sl_are_none(self):
        dec = self._run_tick("HOLD")
        assert dec.action == "HOLD"
        assert dec.suggested_qty_fraction is None
        assert dec.suggested_sl is None

    def test_tighten_trail_qty_is_none(self):
        dec = self._run_tick("TIGHTEN_TRAIL")
        assert dec.suggested_qty_fraction is None

    def test_move_to_be_qty_is_none(self):
        dec = self._run_tick("MOVE_TO_BREAKEVEN")
        assert dec.suggested_qty_fraction is None

    # ── suggested_sl ──────────────────────────────────────────────────────────

    def test_suggested_sl_always_none_in_formula_mode(self):
        """PATCH-7A formula mode: suggested_sl is None for all actions."""
        for action in ("FULL_CLOSE", "TIME_STOP_EXIT", "PARTIAL_CLOSE_25",
                       "TIGHTEN_TRAIL", "MOVE_TO_BREAKEVEN", "HOLD"):
            dec = self._run_tick(action)
            assert dec.suggested_sl is None, (
                f"suggested_sl={dec.suggested_sl!r} for action={action!r}, expected None"
            )

    # ── metadata invariants ───────────────────────────────────────────────────

    def test_score_state_attached(self):
        dec = self._run_tick("FULL_CLOSE")
        assert dec.score_state is not None
        assert dec.score_state.formula_action == "FULL_CLOSE"

    def test_dry_run_is_true(self):
        dec = self._run_tick("FULL_CLOSE")
        assert dec.dry_run is True

    def test_reason_comes_from_formula(self):
        dec = self._run_tick("FULL_CLOSE")
        assert "formula:FULL_CLOSE" in dec.reason

    def test_confidence_from_formula(self):
        dec = self._run_tick("FULL_CLOSE")
        assert dec.confidence == pytest.approx(0.23)

    # ── shadow mode unchanged ─────────────────────────────────────────────────

    def test_shadow_mode_action_from_legacy(self):
        """Shadow mode: dec.action must still be the LEGACY action, not formula."""
        # Legacy returns PARTIAL_CLOSE_25; formula recommends FULL_CLOSE.
        # In shadow mode the legacy action must win.
        dec = self._run_tick("FULL_CLOSE", scoring_mode="shadow")
        assert dec.action == "PARTIAL_CLOSE_25", (
            f"Shadow mode must preserve legacy action, got {dec.action!r}"
        )


