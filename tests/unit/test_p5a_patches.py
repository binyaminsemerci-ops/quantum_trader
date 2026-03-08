"""
Unit Tests: PATCH-5A — exit_management_agent live write path
=============================================================

Tests cover:
  1.  Config — live_writes_enabled flag wiring
  2.  RedisClient — write-guard behaviour in both modes
  3.  ExitIntentValidator — all 7 rules
  4.  IntentWriter — publish / no-publish paths
  5.  Forbidden stream protections still hold after PATCH-5A

All tests are synchronous-safe (async tests use pytest-asyncio via
asyncio.get_event_loop().run_until_complete or the pytest.mark.asyncio
decorator).  No live Redis connection is required — the RedisClient._client
attribute is mocked.
"""
from __future__ import annotations

import asyncio
import os
import time
import unittest
from typing import Optional
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

# ── helpers ───────────────────────────────────────────────────────────────────

def _make_snapshot(
    symbol: str = "BTCUSDT",
    side: str = "LONG",
    quantity: float = 0.01,
    entry_price: float = 50_000.0,
    mark_price: float = 52_000.0,
    leverage: float = 10.0,
    stop_loss: float = 48_000.0,
    take_profit: float = 55_000.0,
    unrealized_pnl: float = 20.0,
    entry_risk_usdt: float = 50.0,
    sync_timestamp: Optional[float] = None,
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
        sync_timestamp=sync_timestamp if sync_timestamp is not None else time.time(),
    )


def _make_decision(
    action: str = "FULL_CLOSE",
    urgency: str = "HIGH",
    confidence: float = 0.85,
    R_net: float = 1.5,
    reason: str = "test reason",
    suggested_sl: Optional[float] = None,
    suggested_qty_fraction: Optional[float] = None,
    dry_run: bool = True,
    snapshot=None,
):
    from microservices.exit_management_agent.models import ExitDecision

    snap = snapshot or _make_snapshot()
    return ExitDecision(
        snapshot=snap,
        action=action,
        urgency=urgency,
        confidence=confidence,
        R_net=R_net,
        reason=reason,
        suggested_sl=suggested_sl,
        suggested_qty_fraction=suggested_qty_fraction,
        dry_run=dry_run,
    )


def run_async(coro):
    """Run a coroutine synchronously for tests that don't use pytest-asyncio."""
    return asyncio.get_event_loop().run_until_complete(coro)


# ─────────────────────────────────────────────────────────────────────────────
# 1. CONFIG — live_writes_enabled
# ─────────────────────────────────────────────────────────────────────────────

class TestConfigPatch5A:
    """Config.from_env() correctly reads EXIT_AGENT_LIVE_WRITES_ENABLED."""

    def test_live_writes_disabled_by_default(self):
        """Without env var, live_writes_enabled must be False."""
        env = {k: v for k, v in os.environ.items()}
        env.pop("EXIT_AGENT_LIVE_WRITES_ENABLED", None)
        with patch.dict(os.environ, env, clear=True):
            from microservices.exit_management_agent.config import AgentConfig
            cfg = AgentConfig.from_env()
        assert cfg.live_writes_enabled is False

    def test_live_writes_enabled_when_env_true(self):
        """EXIT_AGENT_LIVE_WRITES_ENABLED=true sets live_writes_enabled=True."""
        with patch.dict(os.environ, {"EXIT_AGENT_LIVE_WRITES_ENABLED": "true"}):
            from microservices.exit_management_agent.config import AgentConfig
            cfg = AgentConfig.from_env()
        assert cfg.live_writes_enabled is True

    def test_live_writes_false_for_any_non_true_value(self):
        """Any value other than 'true' (case-insensitive) leaves flag False."""
        for val in ("false", "1", "yes", "enabled", "True_sort_of"):
            with patch.dict(os.environ, {"EXIT_AGENT_LIVE_WRITES_ENABLED": val}):
                from microservices.exit_management_agent.config import AgentConfig
                cfg = AgentConfig.from_env()
                # Only exact "true" (case-insensitive) enables it
                expected = val.lower() == "true"
                assert cfg.live_writes_enabled is expected, f"Failed for {val!r}"

    def test_dry_run_always_true_regardless_of_live_writes(self):
        """dry_run is always True even when live_writes_enabled=True."""
        with patch.dict(os.environ, {
            "EXIT_AGENT_LIVE_WRITES_ENABLED": "true",
            "EXIT_AGENT_DRY_RUN": "false",
        }):
            from microservices.exit_management_agent.config import AgentConfig
            cfg = AgentConfig.from_env()
        assert cfg.dry_run is True
        assert cfg.live_writes_enabled is True

    def test_intent_stream_default(self):
        """Default intent stream is quantum:stream:exit.intent."""
        env = {k: v for k, v in os.environ.items()}
        env.pop("EXIT_AGENT_INTENT_STREAM", None)
        with patch.dict(os.environ, env, clear=True):
            from microservices.exit_management_agent.config import AgentConfig
            cfg = AgentConfig.from_env()
        assert cfg.intent_stream == "quantum:stream:exit.intent"


# ─────────────────────────────────────────────────────────────────────────────
# 2. REDIS_IO — write guard
# ─────────────────────────────────────────────────────────────────────────────

class TestRedisClientWriteGuard:
    """RedisClient enforces write guard in both shadow and live modes."""

    def _make_client(self, live_writes_enabled: bool = False):
        from microservices.exit_management_agent.redis_io import RedisClient
        client = RedisClient("127.0.0.1", 6379, live_writes_enabled=live_writes_enabled)
        mock_redis = AsyncMock()
        client._client = mock_redis
        return client

    # ── Shadow mode (live_writes_enabled=False) ─────────────────────────────

    def test_shadow_audit_stream_allowed(self):
        """Exit.audit is allowed in shadow mode."""
        client = self._make_client(live_writes_enabled=False)
        run_async(client.xadd("quantum:stream:exit.audit", {"k": "v"}))
        client._client.xadd.assert_awaited_once()

    def test_shadow_metrics_stream_allowed(self):
        """Exit.metrics is allowed in shadow mode."""
        client = self._make_client(live_writes_enabled=False)
        run_async(client.xadd("quantum:stream:exit.metrics", {"k": "v"}))
        client._client.xadd.assert_awaited_once()

    def test_shadow_exit_intent_blocked(self):
        """exit.intent raises RuntimeError when live_writes_enabled=False."""
        client = self._make_client(live_writes_enabled=False)
        with pytest.raises(RuntimeError, match="live_writes_enabled=False"):
            run_async(client.xadd("quantum:stream:exit.intent", {"k": "v"}))
        client._client.xadd.assert_not_awaited()

    def test_shadow_trade_intent_forbidden(self):
        """trade.intent is always forbidden (unconditional)."""
        client = self._make_client(live_writes_enabled=False)
        with pytest.raises(RuntimeError, match="categorically forbidden"):
            run_async(client.xadd("quantum:stream:trade.intent", {"k": "v"}))
        client._client.xadd.assert_not_awaited()

    def test_shadow_apply_plan_forbidden(self):
        """apply.plan is always forbidden (unconditional)."""
        client = self._make_client(live_writes_enabled=False)
        with pytest.raises(RuntimeError, match="categorically forbidden"):
            run_async(client.xadd("quantum:stream:apply.plan", {"k": "v"}))
        client._client.xadd.assert_not_awaited()

    def test_shadow_harvest_intent_forbidden(self):
        """harvest.intent is always forbidden."""
        client = self._make_client(live_writes_enabled=False)
        with pytest.raises(RuntimeError, match="categorically forbidden"):
            run_async(client.xadd("quantum:stream:harvest.intent", {"k": "v"}))
        client._client.xadd.assert_not_awaited()

    def test_shadow_harvest_suggestions_forbidden(self):
        """harvest.suggestions is always forbidden."""
        client = self._make_client(live_writes_enabled=False)
        with pytest.raises(RuntimeError, match="categorically forbidden"):
            run_async(client.xadd("quantum:stream:harvest.suggestions", {"k": "v"}))
        client._client.xadd.assert_not_awaited()

    # ── Live mode (live_writes_enabled=True) ────────────────────────────────

    def test_live_exit_intent_allowed(self):
        """exit.intent is allowed when live_writes_enabled=True."""
        client = self._make_client(live_writes_enabled=True)
        run_async(client.xadd("quantum:stream:exit.intent", {"k": "v"}))
        client._client.xadd.assert_awaited_once()

    def test_live_audit_stream_still_allowed(self):
        """Audit stream remains allowed in live mode."""
        client = self._make_client(live_writes_enabled=True)
        run_async(client.xadd("quantum:stream:exit.audit", {"k": "v"}))
        client._client.xadd.assert_awaited_once()

    def test_live_trade_intent_still_forbidden(self):
        """trade.intent remains forbidden even when live_writes_enabled=True."""
        client = self._make_client(live_writes_enabled=True)
        with pytest.raises(RuntimeError, match="categorically forbidden"):
            run_async(client.xadd("quantum:stream:trade.intent", {"k": "v"}))
        client._client.xadd.assert_not_awaited()

    def test_live_apply_plan_still_forbidden(self):
        """apply.plan remains forbidden even when live_writes_enabled=True."""
        client = self._make_client(live_writes_enabled=True)
        with pytest.raises(RuntimeError, match="categorically forbidden"):
            run_async(client.xadd("quantum:stream:apply.plan", {"k": "v"}))
        client._client.xadd.assert_not_awaited()

    def test_unknown_stream_raises_regardless_of_mode(self):
        """Any stream not in the allowlist raises in both modes."""
        for live in (False, True):
            client = self._make_client(live_writes_enabled=live)
            with pytest.raises(RuntimeError):
                run_async(client.xadd("quantum:stream:some.random.stream", {"k": "v"}))

    def test_set_allowed_prefix(self):
        """SET to quantum:exit_agent: prefix is always permitted."""
        client = self._make_client(live_writes_enabled=False)
        client._client.set = AsyncMock()
        run_async(client.set_with_ttl("quantum:exit_agent:heartbeat", "1", 60))
        client._client.set.assert_awaited_once()

    def test_set_forbidden_prefix(self):
        """SET to any other prefix raises RuntimeError."""
        client = self._make_client(live_writes_enabled=False)
        with pytest.raises(RuntimeError, match="allowed namespace"):
            run_async(client.set_with_ttl("quantum:trade:something", "1", 60))


# ─────────────────────────────────────────────────────────────────────────────
# 3. VALIDATOR — all 7 rules
# ─────────────────────────────────────────────────────────────────────────────

class TestExitIntentValidator:
    """ExitIntentValidator enforces all 7 pre-publish checks."""

    @pytest.fixture(autouse=True)
    def validator(self):
        from microservices.exit_management_agent.validator import ExitIntentValidator
        self.v = ExitIntentValidator()

    def _valid_decision(self, **overrides):
        """Return a decision that passes all 7 checks, with optional overrides."""
        defaults = dict(
            action="FULL_CLOSE",
            urgency="HIGH",
            confidence=0.85,
            R_net=1.5,
            suggested_qty_fraction=None,
            snapshot=_make_snapshot(
                mark_price=52_000.0,
                quantity=0.01,
                sync_timestamp=time.time(),
            ),
        )
        defaults.update(overrides)
        return _make_decision(**defaults)

    # V1 — action whitelist
    def test_full_close_passes(self):
        result = self.v.validate(self._valid_decision(action="FULL_CLOSE"))
        assert result.passed

    def test_partial_close_25_passes(self):
        result = self.v.validate(self._valid_decision(action="PARTIAL_CLOSE_25"))
        assert result.passed

    def test_time_stop_exit_passes(self):
        result = self.v.validate(self._valid_decision(action="TIME_STOP_EXIT"))
        assert result.passed

    def test_hold_blocked(self):
        result = self.v.validate(self._valid_decision(action="HOLD"))
        assert result.failed
        assert result.rule == "action_whitelisted"

    def test_tighten_trail_blocked(self):
        result = self.v.validate(self._valid_decision(action="TIGHTEN_TRAIL"))
        assert result.failed
        assert result.rule == "action_whitelisted"

    def test_move_to_breakeven_blocked(self):
        result = self.v.validate(self._valid_decision(action="MOVE_TO_BREAKEVEN"))
        assert result.failed
        assert result.rule == "action_whitelisted"

    # V3 — urgency
    def test_low_urgency_blocked(self):
        result = self.v.validate(self._valid_decision(urgency="LOW"))
        assert result.failed
        assert result.rule == "urgency_sufficient"

    def test_medium_urgency_passes(self):
        result = self.v.validate(self._valid_decision(urgency="MEDIUM"))
        assert result.passed

    def test_emergency_urgency_passes(self):
        result = self.v.validate(self._valid_decision(urgency="EMERGENCY"))
        assert result.passed

    # V4 — confidence
    def test_low_confidence_blocked(self):
        result = self.v.validate(self._valid_decision(confidence=0.64))
        assert result.failed
        assert result.rule == "confidence_sufficient"

    def test_exact_min_confidence_passes(self):
        result = self.v.validate(self._valid_decision(confidence=0.65))
        assert result.passed

    def test_high_confidence_passes(self):
        result = self.v.validate(self._valid_decision(confidence=1.0))
        assert result.passed

    # V5 — data freshness
    def test_stale_data_blocked(self):
        stale_snap = _make_snapshot(sync_timestamp=time.time() - 31.0)
        result = self.v.validate(self._valid_decision(snapshot=stale_snap))
        assert result.failed
        assert result.rule == "data_fresh"

    def test_fresh_data_passes(self):
        fresh_snap = _make_snapshot(sync_timestamp=time.time() - 5.0)
        result = self.v.validate(self._valid_decision(snapshot=fresh_snap))
        assert result.passed

    def test_exactly_at_limit_passes(self):
        # 29.9s — just inside the 30s window
        snap = _make_snapshot(sync_timestamp=time.time() - 29.9)
        result = self.v.validate(self._valid_decision(snapshot=snap))
        assert result.passed

    # V6 — notional
    def test_below_min_notional_blocked(self):
        # mark_price=100, quantity=0.0001 → notional=0.01 USDT
        tiny_snap = _make_snapshot(
            mark_price=100.0, quantity=0.0001, sync_timestamp=time.time()
        )
        result = self.v.validate(self._valid_decision(snapshot=tiny_snap))
        assert result.failed
        assert result.rule == "notional_sufficient"

    def test_sufficient_notional_passes(self):
        # mark_price=50_000, quantity=0.01 → 500 USDT
        snap = _make_snapshot(mark_price=50_000.0, quantity=0.01, sync_timestamp=time.time())
        result = self.v.validate(self._valid_decision(snapshot=snap))
        assert result.passed

    # V7 — qty_fraction
    def test_invalid_qty_fraction_zero_blocked(self):
        result = self.v.validate(
            self._valid_decision(action="PARTIAL_CLOSE_25", suggested_qty_fraction=0.0)
        )
        assert result.failed
        assert result.rule == "qty_fraction_valid"

    def test_invalid_qty_fraction_negative_blocked(self):
        result = self.v.validate(
            self._valid_decision(action="PARTIAL_CLOSE_25", suggested_qty_fraction=-0.25)
        )
        assert result.failed
        assert result.rule == "qty_fraction_valid"

    def test_qty_fraction_over_one_blocked(self):
        result = self.v.validate(
            self._valid_decision(action="PARTIAL_CLOSE_25", suggested_qty_fraction=1.1)
        )
        assert result.failed
        assert result.rule == "qty_fraction_valid"

    def test_valid_qty_fraction_passes(self):
        result = self.v.validate(
            self._valid_decision(action="PARTIAL_CLOSE_25", suggested_qty_fraction=0.25)
        )
        assert result.passed

    def test_none_qty_fraction_passes_for_full_close(self):
        result = self.v.validate(
            self._valid_decision(action="FULL_CLOSE", suggested_qty_fraction=None)
        )
        assert result.passed


# ─────────────────────────────────────────────────────────────────────────────
# 4. INTENT WRITER — publish / no-publish paths
# ─────────────────────────────────────────────────────────────────────────────

class TestIntentWriter:
    """IntentWriter gating and serialisation."""

    def _make_writer(self, live_writes_enabled: bool, xadd_mock=None):
        from microservices.exit_management_agent.intent_writer import IntentWriter
        from microservices.exit_management_agent.redis_io import RedisClient

        redis = MagicMock(spec=RedisClient)
        redis.xadd = AsyncMock(return_value=None)
        if xadd_mock is not None:
            redis.xadd = xadd_mock

        return IntentWriter(
            redis=redis,
            live_writes_enabled=live_writes_enabled,
            intent_stream="quantum:stream:exit.intent",
        ), redis

    def _valid_actionable(self, action: str = "FULL_CLOSE") -> "ExitDecision":
        return _make_decision(
            action=action,
            urgency="HIGH",
            confidence=0.9,
            snapshot=_make_snapshot(sync_timestamp=time.time()),
        )

    # ── Flag off — always no-op ─────────────────────────────────────────────

    def test_flag_off_actionable_returns_false(self):
        writer, redis = self._make_writer(live_writes_enabled=False)
        dec = self._valid_actionable()
        result = run_async(writer.maybe_publish(dec, "loop001"))
        assert result is False
        redis.xadd.assert_not_awaited()

    def test_flag_off_hold_decision_returns_false(self):
        writer, redis = self._make_writer(live_writes_enabled=False)
        dec = _make_decision(action="HOLD")
        result = run_async(writer.maybe_publish(dec, "loop001"))
        assert result is False
        redis.xadd.assert_not_awaited()

    # ── Flag on — valid decisions ───────────────────────────────────────────

    def test_flag_on_valid_full_close_publishes(self):
        writer, redis = self._make_writer(live_writes_enabled=True)
        dec = self._valid_actionable(action="FULL_CLOSE")
        result = run_async(writer.maybe_publish(dec, "loop002"))
        assert result is True
        redis.xadd.assert_awaited_once()
        stream, fields = redis.xadd.call_args[0]
        assert stream == "quantum:stream:exit.intent"
        assert fields["action"] == "FULL_CLOSE"
        assert fields["source"] == "exit_management_agent"
        assert fields["patch"] == "PATCH-5A"

    def test_flag_on_valid_partial_close_publishes(self):
        writer, redis = self._make_writer(live_writes_enabled=True)
        dec = self._valid_actionable(action="PARTIAL_CLOSE_25")
        result = run_async(writer.maybe_publish(dec, "loop003"))
        assert result is True
        _, fields = redis.xadd.call_args[0]
        assert fields["action"] == "PARTIAL_CLOSE_25"
        assert fields["qty_fraction"] == "0.25"

    def test_flag_on_valid_time_stop_publishes(self):
        writer, redis = self._make_writer(live_writes_enabled=True)
        dec = self._valid_actionable(action="TIME_STOP_EXIT")
        result = run_async(writer.maybe_publish(dec, "loop004"))
        assert result is True
        _, fields = redis.xadd.call_args[0]
        assert fields["action"] == "TIME_STOP_EXIT"
        assert fields["qty_fraction"] == "1.0"

    # ── Flag on — invalid decisions blocked ────────────────────────────────

    def test_flag_on_invalid_action_blocked(self):
        writer, redis = self._make_writer(live_writes_enabled=True)
        dec = _make_decision(
            action="TIGHTEN_TRAIL",
            urgency="HIGH",
            confidence=0.9,
            snapshot=_make_snapshot(sync_timestamp=time.time()),
        )
        result = run_async(writer.maybe_publish(dec, "loop005"))
        assert result is False
        redis.xadd.assert_not_awaited()

    def test_flag_on_stale_data_blocked(self):
        writer, redis = self._make_writer(live_writes_enabled=True)
        dec = _make_decision(
            action="FULL_CLOSE",
            urgency="HIGH",
            confidence=0.9,
            snapshot=_make_snapshot(sync_timestamp=time.time() - 60.0),
        )
        result = run_async(writer.maybe_publish(dec, "loop006"))
        assert result is False
        redis.xadd.assert_not_awaited()

    def test_flag_on_low_confidence_blocked(self):
        writer, redis = self._make_writer(live_writes_enabled=True)
        dec = _make_decision(
            action="FULL_CLOSE",
            urgency="HIGH",
            confidence=0.50,
            snapshot=_make_snapshot(sync_timestamp=time.time()),
        )
        result = run_async(writer.maybe_publish(dec, "loop007"))
        assert result is False
        redis.xadd.assert_not_awaited()

    def test_flag_on_hold_decision_not_published(self):
        """HOLD is not_actionable — maybe_publish returns False without validation."""
        writer, redis = self._make_writer(live_writes_enabled=True)
        dec = _make_decision(action="HOLD")
        result = run_async(writer.maybe_publish(dec, "loop008"))
        assert result is False
        redis.xadd.assert_not_awaited()

    # ── Serialisation ──────────────────────────────────────────────────────

    def test_serialisation_contains_required_fields(self):
        from microservices.exit_management_agent.intent_writer import IntentWriter
        dec = self._valid_actionable()
        fields = IntentWriter._serialise(dec, "testloop")
        required = {
            "intent_id", "symbol", "action", "urgency", "side",
            "qty_fraction", "quantity", "entry_price", "mark_price",
            "R_net", "confidence", "reason", "loop_id",
            "source", "patch", "ts_epoch",
        }
        assert required.issubset(set(fields.keys()))
        assert fields["loop_id"] == "testloop"
        assert fields["source"] == "exit_management_agent"
        assert fields["patch"] == "PATCH-5A"

    def test_serialisation_all_values_are_strings(self):
        """All Redis stream field values must be strings."""
        from microservices.exit_management_agent.intent_writer import IntentWriter
        dec = self._valid_actionable()
        fields = IntentWriter._serialise(dec, "testloop")
        for key, val in fields.items():
            assert isinstance(val, str), f"Field {key!r} is not a string: {val!r}"

    def test_write_error_returns_false_does_not_raise(self):
        """If xadd raises, maybe_publish swallows the error and returns False."""
        from microservices.exit_management_agent.intent_writer import IntentWriter
        from microservices.exit_management_agent.redis_io import RedisClient

        redis = MagicMock(spec=RedisClient)
        redis.xadd = AsyncMock(side_effect=RuntimeError("boom"))
        writer = IntentWriter(redis, live_writes_enabled=True)
        dec = self._valid_actionable()
        result = run_async(writer.maybe_publish(dec, "loopERR"))
        assert result is False


# ─────────────────────────────────────────────────────────────────────────────
# 5. FORBIDDEN STREAM PROTECTIONS — regression after PATCH-5A
# ─────────────────────────────────────────────────────────────────────────────

class TestForbiddenStreamRegressions:
    """
    Ensure that PATCH-5A did not accidentally remove any PATCH-1 forbidden
    stream protections for trade.intent, apply.plan, harvest.intent,
    harvest.suggestions.  Both shadow and live modes are checked.
    """

    _PERMANENTLY_FORBIDDEN = [
        "quantum:stream:trade.intent",
        "quantum:stream:apply.plan",
        "quantum:stream:harvest.intent",
        "quantum:stream:harvest.suggestions",
    ]

    def _make_client(self, live: bool):
        from microservices.exit_management_agent.redis_io import RedisClient
        c = RedisClient("127.0.0.1", 6379, live_writes_enabled=live)
        c._client = AsyncMock()
        return c

    @pytest.mark.parametrize("stream", _PERMANENTLY_FORBIDDEN)
    def test_shadow_mode_forbidden(self, stream):
        c = self._make_client(live=False)
        with pytest.raises(RuntimeError, match="categorically forbidden"):
            run_async(c.xadd(stream, {"k": "v"}))

    @pytest.mark.parametrize("stream", _PERMANENTLY_FORBIDDEN)
    def test_live_mode_forbidden(self, stream):
        c = self._make_client(live=True)
        with pytest.raises(RuntimeError, match="categorically forbidden"):
            run_async(c.xadd(stream, {"k": "v"}))

    def test_exit_intent_forbidden_in_shadow_mode(self):
        """exit.intent becomes forbidden when live_writes_enabled=False."""
        c = self._make_client(live=False)
        with pytest.raises(RuntimeError):
            run_async(c.xadd("quantum:stream:exit.intent", {"k": "v"}))

    def test_exit_intent_NOT_in_permanent_forbidden_set(self):
        """After PATCH-5A, exit.intent must NOT appear in _FORBIDDEN_STREAMS."""
        from microservices.exit_management_agent.redis_io import _FORBIDDEN_STREAMS
        assert "quantum:stream:exit.intent" not in _FORBIDDEN_STREAMS

    def test_trade_intent_in_permanent_forbidden_set(self):
        """trade.intent must remain in _FORBIDDEN_STREAMS after PATCH-5A."""
        from microservices.exit_management_agent.redis_io import _FORBIDDEN_STREAMS
        assert "quantum:stream:trade.intent" in _FORBIDDEN_STREAMS

    def test_apply_plan_in_permanent_forbidden_set(self):
        """apply.plan must remain in _FORBIDDEN_STREAMS after PATCH-5A."""
        from microservices.exit_management_agent.redis_io import _FORBIDDEN_STREAMS
        assert "quantum:stream:apply.plan" in _FORBIDDEN_STREAMS
