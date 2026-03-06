"""PATCH-2 kill-switch tests for AutonomousTrader exit-ownership suspension.

Coverage
--------
* _is_exit_ownership_suspended() — absent flag, present flag, Redis error (fail-open).
* _monitor_positions() — guard blocks evaluation when flag active; normal path
  proceeds when flag absent; fail-open on Redis error still proceeds normally.

Fakes
-----
_FakeRedis  — duck-typed fake implementing only the surface touched by the two
              methods under test (get + xadd).  Any unexpected call is omitted
              (AttributeError is an acceptable signal of test over-reach here).
_FakePositionTracker — returns a configurable list from get_all_positions().

These tests do NOT exercise entry scanning, RL sizing, or execution paths.
"""
from __future__ import annotations

import asyncio
import pytest
from unittest.mock import AsyncMock, MagicMock

from microservices.autonomous_trader.autonomous_trader import AutonomousTrader


# ── Fakes ──────────────────────────────────────────────────────────────────────

class _FakeRedis:
    """Minimal Redis fake — only get() and xadd() are implemented."""

    def __init__(self, flag_value: str | None = None, should_raise: bool = False):
        self._flag = flag_value
        self._raise = should_raise
        self.xadd_calls: list[tuple[str, dict]] = []

    async def get(self, _key: str) -> str | None:
        if self._raise:
            raise ConnectionError("test-induced Redis error")
        return self._flag

    async def xadd(self, stream: str, fields: dict) -> None:
        self.xadd_calls.append((stream, fields))


class _FakePositionTracker:
    def __init__(self, positions: list) -> None:
        self._positions = positions

    def get_all_positions(self) -> list:
        return self._positions


def _build_trader(
    flag_value: str | None = None,
    redis_raises: bool = False,
    positions: list | None = None,
) -> AutonomousTrader:
    """Bypass __init__ and inject only the attributes exercised by the methods
    under test.  All other components (PositionTracker, ExitManager, …) are
    either replaced or omitted intentionally — this is a unit test."""
    trader: AutonomousTrader = object.__new__(AutonomousTrader)
    trader.redis = _FakeRedis(flag_value=flag_value, should_raise=redis_raises)
    trader.position_tracker = _FakePositionTracker(positions or [])
    trader.exit_manager = AsyncMock()
    trader.exits_executed = 0
    return trader


def _run(coro):
    return asyncio.run(coro)


# ── TestIsExitOwnershipSuspended ───────────────────────────────────────────────

class TestIsExitOwnershipSuspended:
    """Unit tests for _is_exit_ownership_suspended() in isolation."""

    def test_absent_flag_returns_false(self):
        """Key does not exist (Redis returns None) → ownership is NOT suspended."""
        trader = _build_trader(flag_value=None)
        result = _run(trader._is_exit_ownership_suspended())
        assert result is False

    def test_present_flag_returns_true(self):
        """Key exists with a truthy value → ownership IS suspended."""
        trader = _build_trader(flag_value="1")
        result = _run(trader._is_exit_ownership_suspended())
        assert result is True

    def test_present_flag_any_nonempty_value_returns_true(self):
        """Key exists with a string value → still suspended (we don't inspect value)."""
        trader = _build_trader(flag_value="exit_management_agent")
        result = _run(trader._is_exit_ownership_suspended())
        assert result is True

    def test_redis_error_returns_false_fail_open(self):
        """Redis raises on get() → fail-open: return False so normal exit continues."""
        trader = _build_trader(redis_raises=True)
        result = _run(trader._is_exit_ownership_suspended())
        assert result is False


# ── TestMonitorPositionsKillSwitch ─────────────────────────────────────────────

class TestMonitorPositionsKillSwitch:
    """Integration tests for the PATCH-2 guard inside _monitor_positions()."""

    def _make_fake_position(self, symbol: str = "BTCUSDT") -> MagicMock:
        pos = MagicMock()
        pos.symbol = symbol
        pos.R_net = 0.5
        pos.pnl_usd = 10.0
        pos.entry_price = 40000.0
        pos.current_price = 40200.0
        return pos

    def test_flag_present_skips_exit_evaluation(self):
        """When kill-switch flag is active, evaluate_position must NOT be called."""
        fake_pos = self._make_fake_position()
        trader = _build_trader(flag_value="1", positions=[fake_pos])

        _run(trader._monitor_positions())

        trader.exit_manager.evaluate_position.assert_not_called()

    def test_flag_present_no_harvest_intent_written(self):
        """When kill-switch flag is active, zero xadd calls must reach Redis."""
        fake_pos = self._make_fake_position()
        trader = _build_trader(flag_value="1", positions=[fake_pos])

        _run(trader._monitor_positions())

        assert trader.redis.xadd_calls == [], (
            "harvest.intent must not be written while exit ownership is suspended"
        )

    def test_flag_present_exits_executed_counter_unchanged(self):
        """exits_executed counter stays at 0 — no exit was processed."""
        fake_pos = self._make_fake_position()
        trader = _build_trader(flag_value="1", positions=[fake_pos])

        _run(trader._monitor_positions())

        assert trader.exits_executed == 0

    def test_flag_absent_proceeds_to_evaluate_position(self):
        """When kill-switch flag is absent, evaluate_position MUST be called per position."""
        fake_pos = self._make_fake_position()
        # Decision that triggers no actual exit (HOLD) — keeps test isolated from
        # _execute_exit, which needs more position mock attributes.
        hold_decision = MagicMock()
        hold_decision.action = "HOLD"
        hold_decision.percentage = 0.0
        hold_decision.hold_score = 0.8
        hold_decision.exit_score = 0.2
        trader = _build_trader(flag_value=None, positions=[fake_pos])
        trader.exit_manager.evaluate_position = AsyncMock(return_value=hold_decision)

        _run(trader._monitor_positions())

        trader.exit_manager.evaluate_position.assert_called_once_with(fake_pos)

    def test_redis_error_fail_open_proceeds_to_evaluate(self):
        """Redis raises on the flag read → fail-open: proceed with normal exit evaluation."""
        fake_pos = self._make_fake_position()
        hold_decision = MagicMock()
        hold_decision.action = "HOLD"
        hold_decision.percentage = 0.0
        hold_decision.hold_score = 0.8
        hold_decision.exit_score = 0.2
        trader = _build_trader(redis_raises=True, positions=[fake_pos])
        trader.exit_manager.evaluate_position = AsyncMock(return_value=hold_decision)

        _run(trader._monitor_positions())

        trader.exit_manager.evaluate_position.assert_called_once_with(fake_pos)
