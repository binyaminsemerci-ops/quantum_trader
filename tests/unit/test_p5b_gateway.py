"""
Unit Tests: PATCH-5B — exit_intent_gateway
==========================================

Tests cover:
  1.  Config — testnet required hard-abort, enabled default false
  2.  Models — IntentMessage parsing, order_side mapping, qty computation
  3.  RedisIO — write guard (trade.intent allowed, apply.plan forbidden)
  4.  Validator — all 9 rules (V1–V9), dedup/cooldown side effects
  5.  Gateway main — valid forwarded, each rejection type, format verification
  6.  Forbidden stream regressions — apply.plan never reachable

All tests are synchronous-safe (async tests run via asyncio.run or
asyncio.get_event_loop().run_until_complete).  No live Redis connection
is required — the internal Redis client is mocked throughout.
"""
from __future__ import annotations

import asyncio
import json
import os
import time
import unittest
from unittest.mock import AsyncMock, MagicMock, patch

import pytest


# ── async helper ─────────────────────────────────────────────────────────────

def run_async(coro):
    return asyncio.get_event_loop().run_until_complete(coro)


# ── helpers ──────────────────────────────────────────────────────────────────

def _base_fields(
    intent_id: str = "abc123",
    symbol: str = "BTCUSDT",
    action: str = "FULL_CLOSE",
    urgency: str = "HIGH",
    side: str = "LONG",
    qty_fraction: str = "1.0",
    quantity: str = "0.01",
    entry_price: str = "50000.0",
    mark_price: str = "52000.0",
    confidence: str = "0.85",
    reason: str = "test",
    loop_id: str = "loop1",
    source: str = "exit_management_agent",
    patch_field: str = "PATCH-5A",
    ts_epoch: str = None,
    R_net: str = "1.5",
) -> dict:
    return {
        "intent_id": intent_id,
        "symbol": symbol,
        "action": action,
        "urgency": urgency,
        "side": side,
        "qty_fraction": qty_fraction,
        "quantity": quantity,
        "entry_price": entry_price,
        "mark_price": mark_price,
        "confidence": confidence,
        "reason": reason,
        "loop_id": loop_id,
        "source": source,
        "patch": patch_field,
        "ts_epoch": ts_epoch if ts_epoch is not None else str(time.time()),
        "R_net": R_net,
    }


def _make_config(
    testnet_mode: str = "true",
    enabled: bool = True,
    stale_sec: int = 60,
    dedup_ttl_sec: int = 300,
    cooldown_sec: int = 90,
    rate_limit: int = 100,
):
    """Create a GatewayConfig directly, bypassing from_env."""
    from microservices.exit_intent_gateway.config import GatewayConfig
    return GatewayConfig(
        redis_host="127.0.0.1",
        redis_port=6379,
        testnet_mode=testnet_mode,
        enabled=enabled,
        intent_stream="quantum:stream:exit.intent",
        trade_stream="quantum:stream:trade.intent",
        rejected_stream="quantum:stream:exit.intent.rejected",
        group="exit-intent-gateway",
        consumer="gateway-1",
        stale_sec=stale_sec,
        dedup_ttl_sec=dedup_ttl_sec,
        cooldown_sec=cooldown_sec,
        rate_limit=rate_limit,
        log_level="INFO",
    )


def _make_mock_redis(
    lockdown: bool = False,
    dedup_acquired: bool = True,
    cooldown_acquired: bool = True,
):
    """Create a mock GatewayRedisClient with controllable behavior."""
    from microservices.exit_intent_gateway.redis_io import (
        GatewayRedisClient,
        _ALLOWED_WRITE_STREAMS,
        _FORBIDDEN_STREAMS,
    )
    mock = MagicMock(spec=GatewayRedisClient)
    mock.get_lockdown = AsyncMock(return_value=lockdown)
    mock.set_nx_with_ttl = AsyncMock(side_effect=[dedup_acquired, cooldown_acquired])
    mock.xadd = AsyncMock(return_value="1234567890000-0")
    mock.xack = AsyncMock(return_value=1)
    mock.xreadgroup = AsyncMock(return_value=[])
    mock._assert_stream_writable = GatewayRedisClient._assert_stream_writable.__get__(mock)
    return mock


# ─────────────────────────────────────────────────────────────────────────────
# 1. CONFIG
# ─────────────────────────────────────────────────────────────────────────────

class TestGatewayConfig:

    def test_testnet_required_raises_when_not_true(self):
        """from_env raises RuntimeError if TESTNET_MODE != 'true'."""
        from microservices.exit_intent_gateway.config import GatewayConfig
        for bad_val in ("false", "0", "no", "True_maybe", ""):
            with patch.dict(os.environ, {"TESTNET_MODE": bad_val}, clear=False):
                with pytest.raises(RuntimeError, match="TESTNET_MODE"):
                    GatewayConfig.from_env()

    def test_testnet_required_passes_with_true(self):
        """from_env succeeds when TESTNET_MODE=true."""
        from microservices.exit_intent_gateway.config import GatewayConfig
        env_patch = {"TESTNET_MODE": "true", "EXIT_GATEWAY_ENABLED": "false"}
        with patch.dict(os.environ, env_patch, clear=False):
            cfg = GatewayConfig.from_env()
        assert cfg.testnet_mode == "true"

    def test_enabled_default_false(self):
        """EXIT_GATEWAY_ENABLED defaults to false."""
        from microservices.exit_intent_gateway.config import GatewayConfig
        env = {k: v for k, v in os.environ.items()}
        env.pop("EXIT_GATEWAY_ENABLED", None)
        env["TESTNET_MODE"] = "true"
        with patch.dict(os.environ, env, clear=True):
            cfg = GatewayConfig.from_env()
        assert cfg.enabled is False

    def test_enabled_true_when_set(self):
        """EXIT_GATEWAY_ENABLED=true sets enabled=True."""
        from microservices.exit_intent_gateway.config import GatewayConfig
        env_patch = {"TESTNET_MODE": "true", "EXIT_GATEWAY_ENABLED": "true"}
        with patch.dict(os.environ, env_patch, clear=False):
            cfg = GatewayConfig.from_env()
        assert cfg.enabled is True

    def test_defaults(self):
        """Stream and threshold defaults are correct."""
        from microservices.exit_intent_gateway.config import GatewayConfig
        env = {"TESTNET_MODE": "true"}
        keys_to_remove = [
            "EXIT_GATEWAY_ENABLED", "EXIT_GATEWAY_INTENT_STREAM",
            "EXIT_GATEWAY_TRADE_STREAM", "EXIT_GATEWAY_REJECTED_STREAM",
            "EXIT_GATEWAY_GROUP", "EXIT_GATEWAY_CONSUMER",
            "EXIT_GATEWAY_STALE_SEC", "EXIT_GATEWAY_DEDUP_TTL_SEC",
            "EXIT_GATEWAY_COOLDOWN_SEC", "EXIT_GATEWAY_RATE_LIMIT",
        ]
        clean_env = {k: v for k, v in os.environ.items() if k not in keys_to_remove}
        clean_env["TESTNET_MODE"] = "true"
        with patch.dict(os.environ, clean_env, clear=True):
            cfg = GatewayConfig.from_env()
        assert cfg.intent_stream == "quantum:stream:exit.intent"
        assert cfg.trade_stream == "quantum:stream:trade.intent"
        assert cfg.rejected_stream == "quantum:stream:exit.intent.rejected"
        assert cfg.stale_sec == 60
        assert cfg.dedup_ttl_sec == 300
        assert cfg.cooldown_sec == 90
        assert cfg.rate_limit == 10
        assert cfg.enabled is False


# ─────────────────────────────────────────────────────────────────────────────
# 2. MODELS
# ─────────────────────────────────────────────────────────────────────────────

class TestIntentMessage:

    def test_parse_valid_fields(self):
        """from_redis_fields returns IntentMessage with correct types."""
        from microservices.exit_intent_gateway.models import IntentMessage
        ts = time.time()
        fields = _base_fields(ts_epoch=str(ts))
        msg = IntentMessage.from_redis_fields("1234-0", fields)
        assert msg.symbol == "BTCUSDT"
        assert msg.action == "FULL_CLOSE"
        assert msg.side == "LONG"
        assert msg.qty_fraction == 1.0
        assert msg.quantity == 0.01
        assert msg.confidence == 0.85
        assert msg.ts_epoch == ts
        assert msg.stream_id == "1234-0"

    def test_symbol_uppercased(self):
        """Symbol is coerced to uppercase."""
        from microservices.exit_intent_gateway.models import IntentMessage
        fields = _base_fields(symbol="btcusdt")
        msg = IntentMessage.from_redis_fields("1-0", fields)
        assert msg.symbol == "BTCUSDT"

    def test_missing_required_field_raises(self):
        """Missing mandatory field raises ValueError."""
        from microservices.exit_intent_gateway.models import IntentMessage
        for key in ("intent_id", "symbol", "action", "qty_fraction", "quantity", "ts_epoch"):
            fields = _base_fields()
            del fields[key]
            with pytest.raises(ValueError, match=key):
                IntentMessage.from_redis_fields("1-0", fields)

    def test_order_side_long_to_sell(self):
        """LONG position close maps to SELL order."""
        from microservices.exit_intent_gateway.models import IntentMessage
        msg = IntentMessage.from_redis_fields("1-0", _base_fields(side="LONG"))
        assert msg.order_side == "SELL"

    def test_order_side_short_to_buy(self):
        """SHORT position close maps to BUY order."""
        from microservices.exit_intent_gateway.models import IntentMessage
        msg = IntentMessage.from_redis_fields("1-0", _base_fields(side="SHORT"))
        assert msg.order_side == "BUY"

    def test_order_side_unknown_raises(self):
        """Invalid side raises ValueError on order_side access."""
        from microservices.exit_intent_gateway.models import IntentMessage
        fields = _base_fields(side="FLAT")
        msg = IntentMessage.from_redis_fields("1-0", fields)
        with pytest.raises(ValueError, match="side"):
            _ = msg.order_side

    def test_computed_qty_full(self):
        """computed_qty = qty_fraction * quantity."""
        from microservices.exit_intent_gateway.models import IntentMessage
        msg = IntentMessage.from_redis_fields(
            "1-0", _base_fields(qty_fraction="1.0", quantity="0.05")
        )
        assert abs(msg.computed_qty - 0.05) < 1e-10

    def test_computed_qty_partial(self):
        """Partial close: qty_fraction=0.25 * quantity=0.04 = 0.01."""
        from microservices.exit_intent_gateway.models import IntentMessage
        msg = IntentMessage.from_redis_fields(
            "1-0", _base_fields(qty_fraction="0.25", quantity="0.04")
        )
        assert abs(msg.computed_qty - 0.01) < 1e-10

    def test_trade_intent_payload_format(self):
        """to_trade_intent_payload produces correct JSON for intent_bridge."""
        from microservices.exit_intent_gateway.models import IntentMessage
        fields = _base_fields(side="LONG", qty_fraction="1.0", quantity="0.01")
        msg = IntentMessage.from_redis_fields("1-0", fields)
        payload_str = msg.to_trade_intent_payload(patch="PATCH-5B")
        payload = json.loads(payload_str)
        assert payload["symbol"] == "BTCUSDT"
        assert payload["side"] == "SELL"     # LONG close → SELL
        assert payload["qty"] == 0.01
        assert payload["type"] == "MARKET"
        assert payload["reduceOnly"] is True
        assert payload["source"] == "exit_management_agent"
        assert payload["patch"] == "PATCH-5B"

    def test_trade_intent_payload_short_close(self):
        """SHORT close serialises as BUY order."""
        from microservices.exit_intent_gateway.models import IntentMessage
        fields = _base_fields(side="SHORT", qty_fraction="1.0", quantity="0.02")
        msg = IntentMessage.from_redis_fields("1-0", fields)
        payload = json.loads(msg.to_trade_intent_payload())
        assert payload["side"] == "BUY"
        assert payload["qty"] == 0.02
        assert payload["reduceOnly"] is True

    def test_rate_limit_state_allows_until_limit(self):
        """RateLimitState permits up to limit then blocks."""
        from microservices.exit_intent_gateway.models import RateLimitState
        rl = RateLimitState(window_sec=60.0)
        for _ in range(5):
            assert rl.check_and_record(5) is True
        assert rl.check_and_record(5) is False

    def test_rate_limit_state_resets_after_window(self):
        """Timestamps outside the window are pruned."""
        from microservices.exit_intent_gateway.models import RateLimitState
        rl = RateLimitState(window_sec=1.0)
        for _ in range(3):
            rl.check_and_record(3)
        # All 3 slots consumed: next should block
        assert rl.check_and_record(3) is False
        # Force expire
        rl._timestamps = [t - 2.0 for t in rl._timestamps]
        # Now all expired; should allow again
        assert rl.check_and_record(3) is True


# ─────────────────────────────────────────────────────────────────────────────
# 3. REDIS_IO — write guard
# ─────────────────────────────────────────────────────────────────────────────

class TestGatewayRedisIOWriteGuard:

    def _make_client(self):
        from microservices.exit_intent_gateway.redis_io import GatewayRedisClient
        client = GatewayRedisClient("127.0.0.1", 6379)
        client._client = AsyncMock()
        client._client.xadd = AsyncMock(return_value="1-0")
        client._client.set = AsyncMock(return_value=True)
        return client

    def test_trade_intent_allowed(self):
        """trade.intent is in the allowed write set — no error raised."""
        client = self._make_client()
        # Should not raise
        client._assert_stream_writable("quantum:stream:trade.intent")

    def test_rejected_stream_allowed(self):
        """exit.intent.rejected is in the allowed write set."""
        client = self._make_client()
        client._assert_stream_writable("quantum:stream:exit.intent.rejected")

    def test_apply_plan_forbidden(self):
        """apply.plan is unconditionally forbidden — raises RuntimeError."""
        client = self._make_client()
        with pytest.raises(RuntimeError, match="FORBIDDEN_STREAM_WRITE"):
            client._assert_stream_writable("quantum:stream:apply.plan")

    def test_exit_audit_forbidden(self):
        """exit.audit belongs to exit_management_agent, not gateway."""
        client = self._make_client()
        with pytest.raises(RuntimeError, match="FORBIDDEN_STREAM_WRITE"):
            client._assert_stream_writable("quantum:stream:exit.audit")

    def test_harvest_intent_forbidden(self):
        """harvest.intent is unconditionally forbidden."""
        client = self._make_client()
        with pytest.raises(RuntimeError, match="FORBIDDEN_STREAM_WRITE"):
            client._assert_stream_writable("quantum:stream:harvest.intent")

    def test_unknown_stream_forbidden(self):
        """Unknown stream not in allowed set raises RuntimeError."""
        client = self._make_client()
        with pytest.raises(RuntimeError, match="UNKNOWN_STREAM_WRITE"):
            client._assert_stream_writable("quantum:stream:some.other.stream")

    def test_set_nx_allowed_key(self):
        """SET NX succeeds for keys with quantum:exit_gw: prefix."""
        client = self._make_client()
        client._assert_set_key_allowed("quantum:exit_gw:dedup:abc123")
        client._assert_set_key_allowed("quantum:exit_gw:cooldown:BTCUSDT")

    def test_set_nx_forbidden_key_prefix(self):
        """SET NX raises for keys without the allowed prefix."""
        client = self._make_client()
        with pytest.raises(RuntimeError, match="FORBIDDEN_SET_KEY"):
            client._assert_set_key_allowed("quantum:position:BTCUSDT")
        with pytest.raises(RuntimeError, match="FORBIDDEN_SET_KEY"):
            client._assert_set_key_allowed("quantum:lockdown")

    def test_xadd_calls_assert_then_client(self):
        """xadd() calls the guard and then the underlying client."""
        client = self._make_client()
        run_async(client.xadd("quantum:stream:trade.intent", {"k": "v"}))
        client._client.xadd.assert_called_once()

    def test_xadd_apply_plan_raises_before_network(self):
        """xadd to apply.plan raises RuntimeError without touching network."""
        client = self._make_client()
        with pytest.raises(RuntimeError, match="FORBIDDEN_STREAM_WRITE"):
            run_async(client.xadd("quantum:stream:apply.plan", {"k": "v"}))
        client._client.xadd.assert_not_called()

    def test_set_nx_with_ttl_returns_true_when_acquired(self):
        """set_nx_with_ttl returns True when Redis SET NX succeeds."""
        client = self._make_client()
        client._client.set = AsyncMock(return_value=True)
        result = run_async(client.set_nx_with_ttl("quantum:exit_gw:dedup:xyz", 300))
        assert result is True

    def test_set_nx_with_ttl_returns_false_when_key_exists(self):
        """set_nx_with_ttl returns False when key already exists (SET NX returns None)."""
        client = self._make_client()
        client._client.set = AsyncMock(return_value=None)
        result = run_async(client.set_nx_with_ttl("quantum:exit_gw:dedup:xyz", 300))
        assert result is False


# ─────────────────────────────────────────────────────────────────────────────
# 4. VALIDATOR
# ─────────────────────────────────────────────────────────────────────────────

class TestGatewayValidator:

    def _make_validator(self, config=None, mock_redis=None):
        from microservices.exit_intent_gateway.validator import GatewayValidator
        cfg = config or _make_config()
        redis = mock_redis or _make_mock_redis()
        return GatewayValidator(cfg, redis)

    def _make_msg(self, **kwargs):
        from microservices.exit_intent_gateway.models import IntentMessage
        fields = _base_fields(**kwargs)
        return IntentMessage.from_redis_fields("1-0", fields)

    def test_valid_intent_passes(self):
        """A well-formed intent with good data passes all 9 checks."""
        validator = self._make_validator()
        msg = self._make_msg()
        result = run_async(validator.validate(msg))
        assert result.passed is True
        assert result.rule == "PASS"

    def test_v1_fails_when_testnet_not_true(self):
        """V1: testnet_mode != 'true' → V1_TESTNET_REQUIRED."""
        cfg = _make_config(testnet_mode="false")
        validator = self._make_validator(config=cfg)
        msg = self._make_msg()
        result = run_async(validator.validate(msg))
        assert result.passed is False
        assert result.rule == "V1_TESTNET_REQUIRED"

    def test_v2_fails_when_disabled(self):
        """V2: enabled=False → V2_GATEWAY_DISABLED."""
        cfg = _make_config(enabled=False)
        validator = self._make_validator(config=cfg)
        msg = self._make_msg()
        result = run_async(validator.validate(msg))
        assert result.passed is False
        assert result.rule == "V2_GATEWAY_DISABLED"

    def test_v3_fails_on_lockdown(self):
        """V3: lockdown key present → V3_LOCKDOWN."""
        mock_redis = _make_mock_redis(lockdown=True)
        validator = self._make_validator(mock_redis=mock_redis)
        msg = self._make_msg()
        result = run_async(validator.validate(msg))
        assert result.passed is False
        assert result.rule == "V3_LOCKDOWN"

    def test_v4_fails_stale_intent(self):
        """V4: ts_epoch too old → V4_STALE."""
        old_ts = str(time.time() - 120.0)  # 2 minutes old
        validator = self._make_validator(config=_make_config(stale_sec=60))
        msg = self._make_msg(ts_epoch=old_ts)
        result = run_async(validator.validate(msg))
        assert result.passed is False
        assert result.rule == "V4_STALE"

    def test_v4_passes_fresh_intent(self):
        """V4: ts_epoch within stale_sec → passes V4."""
        fresh_ts = str(time.time() - 5.0)  # 5 seconds old
        validator = self._make_validator(config=_make_config(stale_sec=60))
        msg = self._make_msg(ts_epoch=fresh_ts)
        result = run_async(validator.validate(msg))
        assert result.passed is True

    def test_v5_fails_on_duplicate(self):
        """V5: dedup key already exists → V5_DEDUP."""
        mock_redis = _make_mock_redis(dedup_acquired=False)
        validator = self._make_validator(mock_redis=mock_redis)
        msg = self._make_msg()
        result = run_async(validator.validate(msg))
        assert result.passed is False
        assert result.rule == "V5_DEDUP"

    def test_v5_sets_dedup_key_on_pass(self):
        """V5: when dedup passes, SET NX is called with correct key and TTL."""
        cfg = _make_config(dedup_ttl_sec=300)
        mock_redis = _make_mock_redis()
        validator = self._make_validator(config=cfg, mock_redis=mock_redis)
        msg = self._make_msg(intent_id="unique_id_42")
        result = run_async(validator.validate(msg))
        # First call should be for dedup key
        first_call = mock_redis.set_nx_with_ttl.call_args_list[0]
        assert "dedup" in first_call.args[0]
        assert "unique_id_42" in first_call.args[0]
        assert first_call.args[1] == 300

    def test_v6_fails_on_cooldown(self):
        """V6: symbol in cooldown → V6_COOLDOWN."""
        mock_redis = _make_mock_redis(dedup_acquired=True, cooldown_acquired=False)
        validator = self._make_validator(mock_redis=mock_redis)
        msg = self._make_msg()
        result = run_async(validator.validate(msg))
        assert result.passed is False
        assert result.rule == "V6_COOLDOWN"

    def test_v6_sets_cooldown_key_on_pass(self):
        """V6: when cooldown passes, SET NX is called with symbol key."""
        cfg = _make_config(cooldown_sec=90)
        mock_redis = _make_mock_redis()
        validator = self._make_validator(config=cfg, mock_redis=mock_redis)
        msg = self._make_msg(symbol="ETHUSDT")
        result = run_async(validator.validate(msg))
        second_call = mock_redis.set_nx_with_ttl.call_args_list[1]
        assert "cooldown" in second_call.args[0]
        assert "ETHUSDT" in second_call.args[0]
        assert second_call.args[1] == 90

    def test_v7_fails_invalid_action(self):
        """V7: action not in whitelist → V7_ACTION."""
        validator = self._make_validator()
        msg = self._make_msg(action="HOLD")
        result = run_async(validator.validate(msg))
        assert result.passed is False
        assert result.rule == "V7_ACTION"

    def test_v7_passes_all_whitelisted_actions(self):
        """V7: FULL_CLOSE, PARTIAL_CLOSE_25, TIME_STOP_EXIT all pass."""
        for action in ("FULL_CLOSE", "PARTIAL_CLOSE_25", "TIME_STOP_EXIT"):
            mock_redis = _make_mock_redis()
            validator = self._make_validator(mock_redis=mock_redis)
            msg = self._make_msg(action=action)
            result = run_async(validator.validate(msg))
            assert result.passed is True, f"Expected PASS for action={action!r}"

    def test_v8_fails_wrong_source(self):
        """V8: source != 'exit_management_agent' → V8_SOURCE."""
        validator = self._make_validator()
        msg = self._make_msg(source="autonomous_trader")
        result = run_async(validator.validate(msg))
        assert result.passed is False
        assert result.rule == "V8_SOURCE"

    def test_v9_fails_on_rate_limit(self):
        """V9: rate limit exhausted → V9_RATE_LIMIT."""
        cfg = _make_config(rate_limit=2)
        validator = self._make_validator(config=cfg)
        for i in range(2):
            mock_redis = _make_mock_redis()
            validator._redis = mock_redis
            msg = self._make_msg(intent_id=f"id_{i}", symbol=f"SYM{i}USDT")
            r = run_async(validator.validate(msg))
            assert r.passed is True
        # Third call should hit rate limit
        mock_redis = _make_mock_redis()
        validator._redis = mock_redis
        msg = self._make_msg(intent_id="id_2", symbol="SYM2USDT")
        result = run_async(validator.validate(msg))
        assert result.passed is False
        assert result.rule == "V9_RATE_LIMIT"

    def test_checks_short_circuit_at_first_failure(self):
        """When V3 fails (lockdown), Redis SET NX is never called."""
        mock_redis = _make_mock_redis(lockdown=True)
        validator = self._make_validator(mock_redis=mock_redis)
        msg = self._make_msg()
        run_async(validator.validate(msg))
        mock_redis.set_nx_with_ttl.assert_not_called()

    def test_v1_v2_prevent_redis_calls(self):
        """V1/V2 failures prevent any Redis read (no lockdown check etc.)."""
        mock_redis = _make_mock_redis()
        cfg = _make_config(testnet_mode="false")
        validator = self._make_validator(config=cfg, mock_redis=mock_redis)
        msg = self._make_msg()
        run_async(validator.validate(msg))
        mock_redis.get_lockdown.assert_not_called()
        mock_redis.set_nx_with_ttl.assert_not_called()


# ─────────────────────────────────────────────────────────────────────────────
# 5. GATEWAY MAIN — message processing
# ─────────────────────────────────────────────────────────────────────────────

class TestGatewayMessageProcessing:

    def _make_gateway(self, config=None, mock_redis=None):
        from microservices.exit_intent_gateway.main import ExitIntentGateway
        cfg = config or _make_config()
        gw = ExitIntentGateway(cfg)
        gw._redis = mock_redis or _make_mock_redis()
        # Rebind validator to use the same mock redis
        from microservices.exit_intent_gateway.validator import GatewayValidator
        gw._validator = GatewayValidator(cfg, gw._redis)
        return gw

    def test_valid_intent_forwarded_to_trade_intent(self):
        """A valid intent produces an XADD to trade.intent."""
        mock_redis = _make_mock_redis()
        gw = self._make_gateway(mock_redis=mock_redis)
        fields = _base_fields()
        run_async(gw._process_message("1-0", fields))
        # xadd should have been called for trade.intent
        calls = mock_redis.xadd.call_args_list
        streams_written = [c.args[0] for c in calls]
        assert "quantum:stream:trade.intent" in streams_written

    def test_valid_intent_never_writes_apply_plan(self):
        """apply.plan is never written regardless of intent validity."""
        mock_redis = _make_mock_redis()
        gw = self._make_gateway(mock_redis=mock_redis)
        fields = _base_fields()
        run_async(gw._process_message("1-0", fields))
        calls = mock_redis.xadd.call_args_list
        streams_written = [c.args[0] for c in calls]
        assert "quantum:stream:apply.plan" not in streams_written

    def test_trade_intent_payload_structure(self):
        """trade.intent message has event_type + payload JSON string."""
        mock_redis = _make_mock_redis()
        gw = self._make_gateway(mock_redis=mock_redis)
        fields = _base_fields(side="LONG", qty_fraction="1.0", quantity="0.01")
        run_async(gw._process_message("1-0", fields))
        trade_call = next(
            c for c in mock_redis.xadd.call_args_list
            if c.args[0] == "quantum:stream:trade.intent"
        )
        msg_fields = trade_call.args[1]
        assert msg_fields["event_type"] == "trade.intent"
        payload = json.loads(msg_fields["payload"])
        assert payload["symbol"] == "BTCUSDT"
        assert payload["side"] == "SELL"
        assert payload["reduceOnly"] is True
        assert payload["type"] == "MARKET"
        assert "qty" in payload

    def test_rejected_intent_goes_to_rejected_stream(self):
        """A lockdown-rejected intent is written to exit.intent.rejected."""
        mock_redis = _make_mock_redis(lockdown=True)
        gw = self._make_gateway(mock_redis=mock_redis)
        fields = _base_fields()
        run_async(gw._process_message("1-0", fields))
        calls = mock_redis.xadd.call_args_list
        streams_written = [c.args[0] for c in calls]
        assert "quantum:stream:exit.intent.rejected" in streams_written
        assert "quantum:stream:trade.intent" not in streams_written

    def test_message_always_acked(self):
        """Message is ACK'd regardless of validation outcome."""
        mock_redis = _make_mock_redis(lockdown=True)
        gw = self._make_gateway(mock_redis=mock_redis)
        run_async(gw._process_message("1-0", _base_fields()))
        mock_redis.xack.assert_called_once()

    def test_parse_error_acked_and_rejected(self):
        """Malformed message with missing fields is ACK'd and logged to rejected stream."""
        mock_redis = _make_mock_redis()
        gw = self._make_gateway(mock_redis=mock_redis)
        bad_fields = {"symbol": "BTCUSDT"}  # missing most fields
        run_async(gw._process_message("bad-0", bad_fields))
        mock_redis.xack.assert_called_once()
        calls = mock_redis.xadd.call_args_list
        streams_written = [c.args[0] for c in calls]
        assert "quantum:stream:exit.intent.rejected" in streams_written
        assert "quantum:stream:trade.intent" not in streams_written

    def test_gateway_disabled_rejects_without_forwarding(self):
        """When enabled=False, no intent is forwarded to trade.intent."""
        cfg = _make_config(enabled=False)
        mock_redis = _make_mock_redis()
        gw = self._make_gateway(config=cfg, mock_redis=mock_redis)
        run_async(gw._process_message("1-0", _base_fields()))
        calls = mock_redis.xadd.call_args_list
        streams_written = [c.args[0] for c in calls]
        assert "quantum:stream:trade.intent" not in streams_written

    def test_counters_increment_on_forward(self):
        """_total_forwarded increments on successful forward."""
        mock_redis = _make_mock_redis()
        gw = self._make_gateway(mock_redis=mock_redis)
        assert gw._total_forwarded == 0
        run_async(gw._process_message("1-0", _base_fields()))
        assert gw._total_forwarded == 1
        assert gw._total_received == 1

    def test_counters_increment_on_reject(self):
        """_total_rejected increments on each rejected message."""
        mock_redis = _make_mock_redis(lockdown=True)
        gw = self._make_gateway(mock_redis=mock_redis)
        run_async(gw._process_message("1-0", _base_fields()))
        assert gw._total_rejected == 1
        assert gw._total_forwarded == 0


# ─────────────────────────────────────────────────────────────────────────────
# 6. FORBIDDEN STREAM REGRESSIONS
# ─────────────────────────────────────────────────────────────────────────────

class TestForbiddenStreamRegressions:
    """Defence-in-depth: verify critical streams are permanently blocked."""

    def _get_client(self):
        from microservices.exit_intent_gateway.redis_io import GatewayRedisClient
        client = GatewayRedisClient("127.0.0.1", 6379)
        client._client = AsyncMock()
        return client

    def test_apply_plan_xadd_raises(self):
        """apply.plan is UNCONDITIONALLY forbidden for XADD."""
        client = self._get_client()
        with pytest.raises(RuntimeError, match="FORBIDDEN_STREAM_WRITE"):
            run_async(client.xadd("quantum:stream:apply.plan", {"k": "v"}))

    def test_apply_plan_not_in_allowed_write_streams(self):
        """apply.plan must not be in _ALLOWED_WRITE_STREAMS constant."""
        from microservices.exit_intent_gateway.redis_io import _ALLOWED_WRITE_STREAMS
        assert "quantum:stream:apply.plan" not in _ALLOWED_WRITE_STREAMS

    def test_apply_plan_in_forbidden_streams(self):
        """apply.plan must be in _FORBIDDEN_STREAMS constant."""
        from microservices.exit_intent_gateway.redis_io import _FORBIDDEN_STREAMS
        assert "quantum:stream:apply.plan" in _FORBIDDEN_STREAMS

    def test_trade_intent_in_allowed_write_streams(self):
        """trade.intent must be in _ALLOWED_WRITE_STREAMS constant."""
        from microservices.exit_intent_gateway.redis_io import _ALLOWED_WRITE_STREAMS
        assert "quantum:stream:trade.intent" in _ALLOWED_WRITE_STREAMS

    def test_rejected_stream_in_allowed_write_streams(self):
        """exit.intent.rejected must be in _ALLOWED_WRITE_STREAMS constant."""
        from microservices.exit_intent_gateway.redis_io import _ALLOWED_WRITE_STREAMS
        assert "quantum:stream:exit.intent.rejected" in _ALLOWED_WRITE_STREAMS

    def test_harvest_intent_xadd_raises(self):
        """harvest.intent is forbidden for XADD."""
        client = self._get_client()
        with pytest.raises(RuntimeError):
            run_async(client.xadd("quantum:stream:harvest.intent", {"k": "v"}))

    def test_exit_intent_source_not_writable_by_gateway(self):
        """Gateway cannot write to exit.intent (source stream — read-only for gateway)."""
        client = self._get_client()
        with pytest.raises(RuntimeError):
            run_async(client.xadd("quantum:stream:exit.intent", {"k": "v"}))

    def test_gateway_process_message_cannot_reach_apply_plan(self):
        """End-to-end: _process_message with a valid intent cannot reach apply.plan."""
        from microservices.exit_intent_gateway.main import ExitIntentGateway
        cfg = _make_config()
        gw = ExitIntentGateway(cfg)

        # Real GatewayRedisClient but with mocked _client underneath.
        from microservices.exit_intent_gateway.redis_io import GatewayRedisClient
        real_redis = GatewayRedisClient("127.0.0.1", 6379)
        real_redis._client = AsyncMock()
        real_redis._client.exists = AsyncMock(return_value=0)   # no lockdown
        real_redis._client.set = AsyncMock(return_value=True)   # dedup+cooldown acquire
        real_redis._client.xadd = AsyncMock(return_value="1-0")
        real_redis._client.xack = AsyncMock(return_value=1)

        gw._redis = real_redis
        from microservices.exit_intent_gateway.validator import GatewayValidator
        gw._validator = GatewayValidator(cfg, real_redis)

        run_async(gw._process_message("1-0", _base_fields()))

        # Check that xadd was only called for allowed streams
        for call in real_redis._client.xadd.call_args_list:
            stream = call.args[0]
            assert stream != "quantum:stream:apply.plan", (
                f"Forbidden stream apply.plan was written: {call}"
            )
