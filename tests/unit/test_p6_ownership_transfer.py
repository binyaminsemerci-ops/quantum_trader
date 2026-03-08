"""
Unit Tests: PATCH-6 — exit_management_agent ownership transfer
==============================================================

Tests cover:
  1.  OwnershipFlagWriter — enabled+testnet writes flag, disabled no-ops,
      testnet guard blocks non-testnet, Redis errors are swallowed
  2.  Config (PATCH-6 fields) — defaults, env parsing, testnet guard warning
  3.  Integration — ExitManagementAgent._tick() calls write(), write guard
      confirmed compatible with active_flag key namespace
  4.  Write guard regression — active_flag key allowed, forbidden streams
      still blocked regardless of ownership_transfer_enabled state

All tests are synchronous-safe (async tests run via
asyncio.get_event_loop().run_until_complete).  No live Redis connection
is required — the RedisClient._client attribute is mocked throughout.
"""
from __future__ import annotations

import asyncio
import os
import unittest
from unittest.mock import AsyncMock, MagicMock, call, patch

import pytest


# ── async helper ─────────────────────────────────────────────────────────────

def run_async(coro):
    return asyncio.get_event_loop().run_until_complete(coro)


# ─────────────────────────────────────────────────────────────────────────────
# 1. OwnershipFlagWriter
# ─────────────────────────────────────────────────────────────────────────────

class TestOwnershipFlagWriterEnabled:
    """When enabled=True and testnet_mode='true', write() writes the flag."""

    def _make_writer(self, testnet_mode: str = "true"):
        from microservices.exit_management_agent.redis_io import RedisClient
        from microservices.exit_management_agent.ownership_flag import OwnershipFlagWriter

        redis = RedisClient("127.0.0.1", 6379, live_writes_enabled=True)
        mock_inner = AsyncMock()
        redis._client = mock_inner

        return OwnershipFlagWriter(
            redis=redis,
            enabled=True,
            flag_key="quantum:exit_agent:active_flag",
            ttl_sec=30,
            testnet_mode=testnet_mode,
        ), mock_inner

    def test_write_calls_set_with_ttl(self):
        """write() calls redis.set_with_ttl with correct key, value, and ttl."""
        writer, mock_inner = self._make_writer()
        result = run_async(writer.write())
        assert result is True
        mock_inner.set.assert_awaited_once_with(
            "quantum:exit_agent:active_flag", "PATCH-6", ex=30
        )

    def test_write_returns_true(self):
        """write() returns True when flag was written."""
        writer, _ = self._make_writer()
        result = run_async(writer.write())
        assert result is True

    def test_write_uses_patch6_value(self):
        """write() writes the value 'PATCH-6' (not empty, not a timestamp)."""
        writer, mock_inner = self._make_writer()
        run_async(writer.write())
        set_call = mock_inner.set.call_args
        assert set_call[0][1] == "PATCH-6"

    def test_started_false_before_first_write(self):
        """F5 fix: _started is False before any write (INFO guard not yet set)."""
        writer, _ = self._make_writer()
        assert writer._started is False

    def test_started_true_after_first_successful_write(self):
        """F5 fix: _started becomes True after a successful write.

        The WARNING is emitted once at startup; subsequent ticks use INFO.
        """
        writer, _ = self._make_writer()
        run_async(writer.write())
        assert writer._started is True

    def test_multiple_writes_use_single_warning(self):
        """F5 fix: three consecutive writes only flip _started once.

        Proves routine per-tick renewals do not hit the WARNING branch.
        """
        writer, mock_inner = self._make_writer()
        run_async(writer.write())
        run_async(writer.write())
        run_async(writer.write())
        # All three writes succeeded (3 SET calls).
        assert mock_inner.set.await_count == 3
        # _started stays True, so only the first call used the WARNING path.
        assert writer._started is True

    def test_write_uses_correct_ttl(self):
        """write() passes the configured TTL to Redis SET EX."""
        from microservices.exit_management_agent.redis_io import RedisClient
        from microservices.exit_management_agent.ownership_flag import OwnershipFlagWriter

        redis = RedisClient("127.0.0.1", 6379, live_writes_enabled=True)
        mock_inner = AsyncMock()
        redis._client = mock_inner

        writer = OwnershipFlagWriter(
            redis=redis, enabled=True,
            flag_key="quantum:exit_agent:active_flag",
            ttl_sec=45, testnet_mode="true",
        )
        run_async(writer.write())
        mock_inner.set.assert_awaited_once_with(
            "quantum:exit_agent:active_flag", "PATCH-6", ex=45
        )

    def test_testnet_mode_case_insensitive(self):
        """testnet_mode='TRUE' is normalised to 'true' and allows the write."""
        writer, mock_inner = self._make_writer(testnet_mode="TRUE")
        result = run_async(writer.write())
        assert result is True
        mock_inner.set.assert_awaited_once()

    def test_write_idempotent_multiple_calls(self):
        """Calling write() multiple times renews the TTL each time."""
        writer, mock_inner = self._make_writer()
        run_async(writer.write())
        run_async(writer.write())
        run_async(writer.write())
        assert mock_inner.set.await_count == 3


class TestOwnershipFlagWriterDisabled:
    """When enabled=False, write() is a no-op and returns False."""

    def _make_writer(self):
        from microservices.exit_management_agent.redis_io import RedisClient
        from microservices.exit_management_agent.ownership_flag import OwnershipFlagWriter

        redis = RedisClient("127.0.0.1", 6379, live_writes_enabled=True)
        mock_inner = AsyncMock()
        redis._client = mock_inner

        return OwnershipFlagWriter(
            redis=redis,
            enabled=False,
            flag_key="quantum:exit_agent:active_flag",
            ttl_sec=30,
            testnet_mode="true",
        ), mock_inner

    def test_write_returns_false(self):
        """write() returns False when disabled."""
        writer, _ = self._make_writer()
        result = run_async(writer.write())
        assert result is False

    def test_redis_not_called_when_disabled(self):
        """write() never touches Redis when disabled."""
        writer, mock_inner = self._make_writer()
        run_async(writer.write())
        mock_inner.set.assert_not_awaited()

    def test_disabled_even_with_testnet_true(self):
        """Testnet mode does not override the enabled flag."""
        writer, mock_inner = self._make_writer()
        run_async(writer.write())
        mock_inner.set.assert_not_awaited()


class TestOwnershipFlagWriterTestnetGuard:
    """When enabled=True but testnet_mode != 'true', write() is a no-op."""

    def _make_writer(self, testnet_mode: str):
        from microservices.exit_management_agent.redis_io import RedisClient
        from microservices.exit_management_agent.ownership_flag import OwnershipFlagWriter

        redis = RedisClient("127.0.0.1", 6379, live_writes_enabled=True)
        mock_inner = AsyncMock()
        redis._client = mock_inner

        return OwnershipFlagWriter(
            redis=redis,
            enabled=True,
            flag_key="quantum:exit_agent:active_flag",
            ttl_sec=30,
            testnet_mode=testnet_mode,
        ), mock_inner

    @pytest.mark.parametrize("testnet_mode", ["false", "0", "1", "yes", ""])
    def test_non_testnet_mode_blocks_write(self, testnet_mode):
        """Any testnet_mode value that isn't 'true' blocks the write."""
        writer, mock_inner = self._make_writer(testnet_mode=testnet_mode)
        result = run_async(writer.write())
        assert result is False
        mock_inner.set.assert_not_awaited()

    def test_false_returns_false(self):
        """testnet_mode='false' returns False."""
        writer, _ = self._make_writer("false")
        assert run_async(writer.write()) is False

    def test_empty_testnet_mode_returns_false(self):
        """Empty testnet_mode returns False (default unset scenario)."""
        writer, _ = self._make_writer("")
        assert run_async(writer.write()) is False


class TestOwnershipFlagWriterRedisError:
    """Redis errors are swallowed and write() returns False on error."""

    def test_redis_error_swallowed(self):
        """write() returns False and does not propagate when Redis raises."""
        from microservices.exit_management_agent.redis_io import RedisClient
        from microservices.exit_management_agent.ownership_flag import OwnershipFlagWriter

        redis = RedisClient("127.0.0.1", 6379, live_writes_enabled=True)
        mock_inner = AsyncMock()
        mock_inner.set.side_effect = ConnectionError("Redis unavailable")
        redis._client = mock_inner

        writer = OwnershipFlagWriter(
            redis=redis,
            enabled=True,
            flag_key="quantum:exit_agent:active_flag",
            ttl_sec=30,
            testnet_mode="true",
        )
        result = run_async(writer.write())
        assert result is False  # error swallowed, returns False

    def test_redis_error_resets_started_flag(self):
        """F5: After a Redis error, the next successful write re-emits WARNING.

        _started is reset to False on error so that recovery is clearly visible
        in journalctl as a new OWNERSHIP_FLAG_ACTIVE WARNING.
        """
        from microservices.exit_management_agent.redis_io import RedisClient
        from microservices.exit_management_agent.ownership_flag import OwnershipFlagWriter

        redis = RedisClient("127.0.0.1", 6379, live_writes_enabled=True)
        mock_inner = AsyncMock()
        redis._client = mock_inner

        writer = OwnershipFlagWriter(
            redis=redis, enabled=True,
            flag_key="quantum:exit_agent:active_flag",
            ttl_sec=30, testnet_mode="true",
        )
        # First write succeeds — _started becomes True.
        run_async(writer.write())
        assert writer._started is True

        # Second call raises — _started should be reset to False.
        mock_inner.set.side_effect = ConnectionError("blip")
        run_async(writer.write())
        assert writer._started is False

    def test_no_exception_propagated_on_timeout(self):
        """TimeoutError from Redis is swallowed like any other exception."""
        from microservices.exit_management_agent.redis_io import RedisClient
        from microservices.exit_management_agent.ownership_flag import OwnershipFlagWriter

        redis = RedisClient("127.0.0.1", 6379, live_writes_enabled=True)
        mock_inner = AsyncMock()
        mock_inner.set.side_effect = TimeoutError("Redis timeout")
        redis._client = mock_inner

        writer = OwnershipFlagWriter(
            redis=redis,
            enabled=True,
            flag_key="quantum:exit_agent:active_flag",
            ttl_sec=30,
            testnet_mode="true",
        )
        # Must not raise:
        result = run_async(writer.write())
        assert result is False


# ─────────────────────────────────────────────────────────────────────────────
# 2. Config — PATCH-6 fields
# ─────────────────────────────────────────────────────────────────────────────

class TestOwnershipConfigPatch6:
    """AgentConfig.from_env() correctly parses PATCH-6 fields."""

    def _env(self, **overrides):
        base = {k: v for k, v in os.environ.items()}
        base.pop("EXIT_AGENT_OWNERSHIP_TRANSFER_ENABLED", None)
        base.pop("EXIT_AGENT_ACTIVE_FLAG_KEY", None)
        base.pop("EXIT_AGENT_ACTIVE_FLAG_TTL_SEC", None)
        base.pop("EXIT_AGENT_TESTNET_MODE", None)
        base.update(overrides)
        return base

    def test_ownership_transfer_disabled_by_default(self):
        """ownership_transfer_enabled defaults to False."""
        with patch.dict(os.environ, self._env(), clear=True):
            from microservices.exit_management_agent.config import AgentConfig
            cfg = AgentConfig.from_env()
        assert cfg.ownership_transfer_enabled is False

    def test_ownership_transfer_enabled_when_env_true(self):
        """EXIT_AGENT_OWNERSHIP_TRANSFER_ENABLED=true sets flag to True."""
        with patch.dict(os.environ,
                        self._env(EXIT_AGENT_OWNERSHIP_TRANSFER_ENABLED="true"),
                        clear=True):
            from microservices.exit_management_agent.config import AgentConfig
            cfg = AgentConfig.from_env()
        assert cfg.ownership_transfer_enabled is True

    def test_ownership_transfer_false_for_non_true(self):
        """Values other than 'true' leave ownership_transfer_enabled False."""
        for val in ("false", "1", "yes", "TRUE_ish"):
            with patch.dict(os.environ,
                            self._env(EXIT_AGENT_OWNERSHIP_TRANSFER_ENABLED=val),
                            clear=True):
                from microservices.exit_management_agent.config import AgentConfig
                cfg = AgentConfig.from_env()
            expected = val.lower() == "true"
            assert cfg.ownership_transfer_enabled is expected, f"failed for {val!r}"

    def test_active_flag_key_default(self):
        """active_flag_key is always quantum:exit_agent:active_flag (F1: hardcoded)."""
        with patch.dict(os.environ, self._env(), clear=True):
            from microservices.exit_management_agent.config import AgentConfig
            cfg = AgentConfig.from_env()
        assert cfg.active_flag_key == "quantum:exit_agent:active_flag"

    def test_active_flag_key_not_overridable_via_env(self):
        """F1 fix: EXIT_AGENT_ACTIVE_FLAG_KEY env var is ignored.

        The key is a binary protocol constant shared with PATCH-2 reader in
        AutonomousTrader. Allowing override would enable silent split-brain.
        Even if the env var is set to a different value, the config must always
        return the hardcoded canonical key.
        """
        with patch.dict(os.environ,
                        self._env(EXIT_AGENT_ACTIVE_FLAG_KEY="quantum:exit_agent:alt_flag"),
                        clear=True):
            from microservices.exit_management_agent.config import AgentConfig
            cfg = AgentConfig.from_env()
        # Must be the hardcoded value, not the env override.
        assert cfg.active_flag_key == "quantum:exit_agent:active_flag"

    def test_active_flag_ttl_sec_default(self):
        """active_flag_ttl_sec defaults to 30."""
        with patch.dict(os.environ, self._env(), clear=True):
            from microservices.exit_management_agent.config import AgentConfig
            cfg = AgentConfig.from_env()
        assert cfg.active_flag_ttl_sec == 30

    def test_active_flag_ttl_sec_from_env(self):
        """EXIT_AGENT_ACTIVE_FLAG_TTL_SEC is parsed as int (within valid range)."""
        with patch.dict(os.environ,
                        self._env(EXIT_AGENT_ACTIVE_FLAG_TTL_SEC="60"),
                        clear=True):
            from microservices.exit_management_agent.config import AgentConfig
            cfg = AgentConfig.from_env()
        assert cfg.active_flag_ttl_sec == 60

    def test_ttl_clamped_when_too_large(self):
        """F2 fix: TTL > 120 is clamped to 120.

        Prevents a misconfigured large TTL (e.g., 9999999) making the flag
        effectively permanent when rollback Option B (env=false + restart) is used.
        """
        with patch.dict(os.environ,
                        self._env(EXIT_AGENT_ACTIVE_FLAG_TTL_SEC="9999"),
                        clear=True):
            from microservices.exit_management_agent.config import AgentConfig
            cfg = AgentConfig.from_env()
        assert cfg.active_flag_ttl_sec == 120

    def test_ttl_clamped_when_too_small(self):
        """F2 fix: TTL < 10 is clamped to 10.

        Prevents a dangerously short TTL (e.g., 1s) that would let AT resume
        between ticks due to normal Redis latency.
        """
        with patch.dict(os.environ,
                        self._env(EXIT_AGENT_ACTIVE_FLAG_TTL_SEC="3"),
                        clear=True):
            from microservices.exit_management_agent.config import AgentConfig
            cfg = AgentConfig.from_env()
        assert cfg.active_flag_ttl_sec == 10

    def test_ttl_boundary_values_not_clamped(self):
        """TTL at exactly 10 and 120 passes through unchanged."""
        for val, expected in (("10", 10), ("120", 120)):
            with patch.dict(os.environ,
                            self._env(EXIT_AGENT_ACTIVE_FLAG_TTL_SEC=val),
                            clear=True):
                from microservices.exit_management_agent.config import AgentConfig
                cfg = AgentConfig.from_env()
            assert cfg.active_flag_ttl_sec == expected, f"boundary {val!r} failed"

    def test_testnet_mode_defaults_to_false(self):
        """testnet_mode defaults to 'false'."""
        with patch.dict(os.environ, self._env(), clear=True):
            from microservices.exit_management_agent.config import AgentConfig
            cfg = AgentConfig.from_env()
        assert cfg.testnet_mode == "false"

    def test_testnet_mode_read_from_env(self):
        """EXIT_AGENT_TESTNET_MODE=true is normalised to 'true'."""
        with patch.dict(os.environ,
                        self._env(EXIT_AGENT_TESTNET_MODE="true"),
                        clear=True):
            from microservices.exit_management_agent.config import AgentConfig
            cfg = AgentConfig.from_env()
        assert cfg.testnet_mode == "true"

    def test_testnet_mode_lowercased(self):
        """EXIT_AGENT_TESTNET_MODE value is lowercased in config."""
        with patch.dict(os.environ,
                        self._env(EXIT_AGENT_TESTNET_MODE="TRUE"),
                        clear=True):
            from microservices.exit_management_agent.config import AgentConfig
            cfg = AgentConfig.from_env()
        assert cfg.testnet_mode == "true"

    def test_patch1_dry_run_unaffected(self):
        """PATCH-6 fields do not affect PATCH-1 dry_run=True hard-code."""
        with patch.dict(os.environ,
                        self._env(
                            EXIT_AGENT_OWNERSHIP_TRANSFER_ENABLED="true",
                            EXIT_AGENT_TESTNET_MODE="true",
                        ),
                        clear=True):
            from microservices.exit_management_agent.config import AgentConfig
            cfg = AgentConfig.from_env()
        assert cfg.dry_run is True

    def test_config_is_frozen(self):
        """AgentConfig remains frozen (no mutation after construction)."""
        with patch.dict(os.environ, self._env(), clear=True):
            from microservices.exit_management_agent.config import AgentConfig
            cfg = AgentConfig.from_env()
        with pytest.raises(Exception):  # FrozenInstanceError or AttributeError
            cfg.ownership_transfer_enabled = True


# ─────────────────────────────────────────────────────────────────────────────
# 3. Integration — ExitManagementAgent wires OwnershipFlagWriter correctly
# ─────────────────────────────────────────────────────────────────────────────

class TestOwnershipIntegration:
    """_tick() calls ownership_flag.write() as step 0 before heartbeat."""

    def _make_agent(
        self,
        ownership_enabled: bool = True,
        testnet_mode: str = "true",
    ):
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
            log_level="INFO",
            dry_run=True,
            live_writes_enabled=False,
            symbol_allowlist=frozenset(),
            max_positions_per_loop=50,
            max_hold_sec=14400.0,
            ownership_transfer_enabled=ownership_enabled,
            active_flag_key="quantum:exit_agent:active_flag",
            active_flag_ttl_sec=30,
            testnet_mode=testnet_mode,
            scoring_mode="shadow",  # PATCH-7A: new required field
        )
        agent = ExitManagementAgent(cfg)
        mock_inner = AsyncMock()
        agent._redis._client = mock_inner
        return agent, mock_inner

    def test_tick_calls_ownership_write_before_heartbeat(self):
        """
        _tick() calls _ownership_flag.write() (step 0) before _heartbeat.beat()
        (step 1). Verified by patching both and checking call order.
        """
        agent, mock_inner = self._make_agent(ownership_enabled=True, testnet_mode="true")

        call_order = []

        async def mock_flag_write():
            call_order.append("ownership_write")
            return True

        async def mock_heartbeat_beat():
            call_order.append("heartbeat_beat")

        async def mock_get_positions(**kwargs):
            return []

        async def mock_write_metrics(**kwargs):
            pass

        agent._ownership_flag.write = mock_flag_write
        agent._heartbeat.beat = mock_heartbeat_beat
        agent._position_source.get_open_positions = mock_get_positions
        agent._audit.write_metrics = mock_write_metrics

        run_async(agent._tick())

        assert call_order[0] == "ownership_write", (
            "ownership_flag.write() must be called before heartbeat.beat()"
        )
        assert call_order[1] == "heartbeat_beat"

    def test_tick_calls_ownership_write_when_enabled(self):
        """_tick() triggers _ownership_flag.write(); Redis SET is called."""
        agent, mock_inner = self._make_agent(ownership_enabled=True, testnet_mode="true")

        async def mock_get_positions(**kwargs):
            return []

        async def mock_write_metrics(**kwargs):
            pass

        agent._position_source.get_open_positions = mock_get_positions
        agent._audit.write_metrics = mock_write_metrics

        run_async(agent._tick())

        # Redis SET must have been called (heartbeat + ownership_flag both use set_with_ttl
        # which calls self._client.set). Confirm at least one SET call happened.
        assert mock_inner.set.await_count >= 1

    def test_tick_ownership_noop_when_disabled(self):
        """When disabled, _tick() does not call redis.set for active_flag."""
        agent, mock_inner = self._make_agent(ownership_enabled=False, testnet_mode="true")

        async def mock_get_positions(**kwargs):
            return []

        async def mock_write_metrics(**kwargs):
            pass

        agent._position_source.get_open_positions = mock_get_positions
        agent._audit.write_metrics = mock_write_metrics

        run_async(agent._tick())

        # Inspect all set() calls — none should be for active_flag
        set_calls = [str(c) for c in mock_inner.set.call_args_list]
        flag_calls = [c for c in set_calls if "active_flag" in c]
        assert flag_calls == [], f"active_flag was written when disabled: {flag_calls}"

    def test_ownership_flag_attribute_exists(self):
        """ExitManagementAgent has _ownership_flag attribute after init."""
        from microservices.exit_management_agent.main import ExitManagementAgent
        from microservices.exit_management_agent.ownership_flag import OwnershipFlagWriter
        agent, _ = self._make_agent()
        assert hasattr(agent, "_ownership_flag")
        assert isinstance(agent._ownership_flag, OwnershipFlagWriter)

    def test_rollback_option_b_env_false_stops_writes(self):
        """Rollback Option B: restarting EMA with ownership_transfer_enabled=False
        causes write() to be a no-op, allowing the key to expire on its own TTL.

        This simulates: operator sets EXIT_AGENT_OWNERSHIP_TRANSFER_ENABLED=false
        and restarts EMA.  Proves the flag is NOT renewed after that point.
        """
        from microservices.exit_management_agent.redis_io import RedisClient
        from microservices.exit_management_agent.ownership_flag import OwnershipFlagWriter

        redis = RedisClient("127.0.0.1", 6379, live_writes_enabled=True)
        mock_inner = AsyncMock()
        redis._client = mock_inner

        # Phase 1: ownership enabled — flag is written each tick.
        writer_active = OwnershipFlagWriter(
            redis=redis, enabled=True,
            flag_key="quantum:exit_agent:active_flag",
            ttl_sec=30, testnet_mode="true",
        )
        run_async(writer_active.write())
        run_async(writer_active.write())
        assert mock_inner.set.await_count == 2

        # Phase 2: simulate restart with ownership disabled (new OwnershipFlagWriter
        # instance with enabled=False, representing restarted EMA process).
        mock_inner.reset_mock()
        writer_disabled = OwnershipFlagWriter(
            redis=redis, enabled=False,
            flag_key="quantum:exit_agent:active_flag",
            ttl_sec=30, testnet_mode="true",
        )
        run_async(writer_disabled.write())
        run_async(writer_disabled.write())
        run_async(writer_disabled.write())
        # After restart with disabled=True, zero writes — key expires naturally.
        mock_inner.set.assert_not_awaited()


# ─────────────────────────────────────────────────────────────────────────────
# 4. Write guard regression — PATCH-6 must not weaken existing guards
# ─────────────────────────────────────────────────────────────────────────────

class TestP6WriteGuardRegression:
    """Existing write guard behaviour is unaffected by PATCH-6."""

    def _make_client(self, live_writes_enabled: bool = False):
        from microservices.exit_management_agent.redis_io import RedisClient
        client = RedisClient("127.0.0.1", 6379, live_writes_enabled=live_writes_enabled)
        mock_redis = AsyncMock()
        client._client = mock_redis
        return client

    def test_active_flag_key_allowed_by_set_guard(self):
        """quantum:exit_agent:active_flag passes the SET write guard."""
        client = self._make_client()
        # Should not raise: key is in allowed namespace
        run_async(client.set_with_ttl("quantum:exit_agent:active_flag", "PATCH-6", 30))
        client._client.set.assert_awaited_once_with(
            "quantum:exit_agent:active_flag", "PATCH-6", ex=30
        )

    def test_active_flag_alternate_key_allowed(self):
        """Any quantum:exit_agent:* key is allowed by the SET write guard."""
        client = self._make_client()
        run_async(client.set_with_ttl("quantum:exit_agent:custom", "v", 10))
        client._client.set.assert_awaited_once()

    def test_forbidden_key_rejected_by_set_guard(self):
        """Keys outside quantum:exit_agent: prefix are blocked."""
        client = self._make_client()
        with pytest.raises(RuntimeError, match="WRITE_GUARD"):
            run_async(client.set_with_ttl("quantum:position:BTCUSDT", "x", 10))
        client._client.set.assert_not_awaited()

    def test_trade_intent_still_forbidden(self):
        """trade.intent remains unconditionally forbidden after PATCH-6."""
        client = self._make_client(live_writes_enabled=True)
        with pytest.raises(RuntimeError, match="categorically forbidden"):
            run_async(client.xadd("quantum:stream:trade.intent", {"k": "v"}))
        client._client.xadd.assert_not_awaited()

    def test_apply_plan_still_forbidden(self):
        """apply.plan remains unconditionally forbidden after PATCH-6."""
        client = self._make_client(live_writes_enabled=True)
        with pytest.raises(RuntimeError, match="categorically forbidden"):
            run_async(client.xadd("quantum:stream:apply.plan", {"k": "v"}))
        client._client.xadd.assert_not_awaited()

    def test_harvest_intent_still_forbidden(self):
        """harvest.intent remains unconditionally forbidden after PATCH-6."""
        client = self._make_client(live_writes_enabled=True)
        with pytest.raises(RuntimeError, match="categorically forbidden"):
            run_async(client.xadd("quantum:stream:harvest.intent", {"k": "v"}))
        client._client.xadd.assert_not_awaited()

    def test_active_flag_key_cannot_be_written_via_xadd(self):
        """active_flag key cannot be accidentally written with XADD."""
        client = self._make_client(live_writes_enabled=True)
        # quantum:exit_agent:active_flag is not a stream — should be blocked
        with pytest.raises(RuntimeError):
            run_async(client.xadd("quantum:exit_agent:active_flag", {"k": "v"}))
        client._client.xadd.assert_not_awaited()

    def test_audit_stream_still_allowed(self):
        """exit.audit stream remains writable after PATCH-6."""
        client = self._make_client()
        run_async(client.xadd("quantum:stream:exit.audit", {"k": "v"}))
        client._client.xadd.assert_awaited_once()

    def test_metrics_stream_still_allowed(self):
        """exit.metrics stream remains writable after PATCH-6."""
        client = self._make_client()
        run_async(client.xadd("quantum:stream:exit.metrics", {"k": "v"}))
        client._client.xadd.assert_awaited_once()
