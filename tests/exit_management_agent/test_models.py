"""Tests for models module.

PositionSnapshot is frozen; ExitDecision.is_actionable is tested here.
No Redis or network required.
"""
from __future__ import annotations

import pytest

from microservices.exit_management_agent.models import (
    ExitDecision,
    PerceptionResult,
    PositionSnapshot,
)


# ── Factories ─────────────────────────────────────────────────────────────────


def _snapshot(**kw) -> PositionSnapshot:
    base = dict(
        symbol="BTCUSDT",
        side="LONG",
        quantity=0.01,
        entry_price=30_000.0,
        mark_price=31_500.0,
        leverage=10.0,
        stop_loss=29_000.0,
        take_profit=35_000.0,
        unrealized_pnl=15.0,
        entry_risk_usdt=100.0,
        sync_timestamp=1_700_000_000.0,
    )
    base.update(kw)
    return PositionSnapshot(**base)


def _decision(snap: PositionSnapshot, action: str = "HOLD", **kw) -> ExitDecision:
    base = dict(
        snapshot=snap,
        action=action,
        reason="test",
        urgency="LOW",
        R_net=0.2,
        confidence=1.0,
        suggested_sl=None,
        suggested_qty_fraction=None,
        dry_run=True,
    )
    base.update(kw)
    return ExitDecision(**base)


# ── PositionSnapshot ──────────────────────────────────────────────────────────


class TestPositionSnapshotProperties:
    def test_is_long_for_long(self):
        assert _snapshot(side="LONG").is_long is True
        assert _snapshot(side="LONG").is_short is False

    def test_is_long_for_buy(self):
        assert _snapshot(side="BUY").is_long is True

    def test_is_short_for_short(self):
        assert _snapshot(side="SHORT").is_short is True
        assert _snapshot(side="SHORT").is_long is False

    def test_is_short_for_sell(self):
        assert _snapshot(side="SELL").is_short is True

    def test_frozen_raises_on_mutation(self):
        snap = _snapshot()
        with pytest.raises(Exception):
            snap.symbol = "ETHUSDT"  # type: ignore[misc]


# ── ExitDecision ──────────────────────────────────────────────────────────────


class TestExitDecisionIsActionable:
    def test_hold_is_not_actionable(self):
        assert _decision(_snapshot(), action="HOLD").is_actionable is False

    def test_full_close_is_actionable(self):
        assert _decision(_snapshot(), action="FULL_CLOSE").is_actionable is True

    def test_partial_close_is_actionable(self):
        assert _decision(_snapshot(), action="PARTIAL_CLOSE_25").is_actionable is True

    def test_move_to_be_is_actionable(self):
        assert _decision(_snapshot(), action="MOVE_TO_BREAKEVEN").is_actionable is True

    def test_time_stop_is_actionable(self):
        assert _decision(_snapshot(), action="TIME_STOP_EXIT").is_actionable is True

    def test_tighten_trail_is_actionable(self):
        assert _decision(_snapshot(), action="TIGHTEN_TRAIL").is_actionable is True


class TestExitDecisionDryRunField:
    def test_dry_run_true(self):
        assert _decision(_snapshot()).dry_run is True

    def test_dry_run_false_is_storable_but_check_in_audit(self):
        # dry_run=False is technically storable in the dataclass;
        # audit.py enforces the PATCH-1 constraint at write time.
        dec = _decision(_snapshot(), dry_run=False)
        assert dec.dry_run is False
