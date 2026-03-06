"""Tests for heartbeat module and redis_io write-guards.

Uses a minimal fake Redis client so tests run without a real Redis server.
"""
from __future__ import annotations

import asyncio
from unittest.mock import AsyncMock

import pytest
import pytest_asyncio

from microservices.exit_management_agent.heartbeat import HeartbeatWriter
from microservices.exit_management_agent.redis_io import RedisClient


# ── Minimal fake Redis client ─────────────────────────────────────────────────


class _FakeRedis:
    """Records set_with_ttl calls; optionally raises on demand."""

    def __init__(self, should_raise: bool = False):
        self.calls: list = []
        self.should_raise = should_raise

    async def set_with_ttl(self, key: str, value: str, ttl_sec: int) -> None:
        if self.should_raise:
            raise RuntimeError("Simulated Redis failure")
        self.calls.append((key, value, ttl_sec))


# ── HeartbeatWriter ───────────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_heartbeat_writes_correct_key_and_ttl():
    fake = _FakeRedis()
    hb = HeartbeatWriter(fake, "quantum:exit_agent:heartbeat", 60)
    await hb.beat()
    assert len(fake.calls) == 1
    key, value, ttl = fake.calls[0]
    assert key == "quantum:exit_agent:heartbeat"
    assert ttl == 60
    assert value.isdigit()
    assert int(value) > 0


@pytest.mark.asyncio
async def test_heartbeat_updates_last_beat():
    fake = _FakeRedis()
    hb = HeartbeatWriter(fake, "quantum:exit_agent:heartbeat", 60)
    assert hb.last_beat == 0.0
    await hb.beat()
    assert hb.last_beat > 0.0


@pytest.mark.asyncio
async def test_heartbeat_does_not_raise_on_redis_error():
    fake = _FakeRedis(should_raise=True)
    hb = HeartbeatWriter(fake, "quantum:exit_agent:heartbeat", 60)
    # Must NOT raise; HeartbeatWriter swallows errors.
    await hb.beat()
    assert hb.last_beat == 0.0  # No successful beat.


@pytest.mark.asyncio
async def test_heartbeat_multiple_beats_update_last_beat():
    fake = _FakeRedis()
    hb = HeartbeatWriter(fake, "quantum:exit_agent:heartbeat", 60)
    await hb.beat()
    t1 = hb.last_beat
    # Tiny sleep so the timestamp can advance.
    await asyncio.sleep(0.01)
    await hb.beat()
    assert hb.last_beat >= t1


# ── RedisClient write-guard (no network needed) ───────────────────────────────


def _run(coro):
    """Helper: run a coroutine synchronously in the default event loop."""
    return asyncio.get_event_loop().run_until_complete(coro)


class TestRedisWriteGuard:
    def test_xadd_rejects_forbidden_stream(self):
        client = RedisClient("127.0.0.1", 6379)
        with pytest.raises(RuntimeError, match="WRITE_GUARD"):
            _run(client.xadd("quantum:stream:trade.intent", {"k": "v"}))

    def test_xadd_rejects_apply_plan(self):
        client = RedisClient("127.0.0.1", 6379)
        with pytest.raises(RuntimeError, match="WRITE_GUARD"):
            _run(client.xadd("quantum:stream:apply.plan", {"k": "v"}))

    def test_xadd_rejects_harvest_intent(self):
        client = RedisClient("127.0.0.1", 6379)
        with pytest.raises(RuntimeError, match="WRITE_GUARD"):
            _run(client.xadd("quantum:stream:harvest.intent", {"k": "v"}))

    def test_xadd_rejects_unknown_stream(self):
        client = RedisClient("127.0.0.1", 6379)
        with pytest.raises(RuntimeError, match="WRITE_GUARD"):
            _run(client.xadd("quantum:stream:some.other.stream", {"k": "v"}))

    def test_set_with_ttl_rejects_wrong_namespace(self):
        client = RedisClient("127.0.0.1", 6379)
        with pytest.raises(RuntimeError, match="WRITE_GUARD"):
            _run(client.set_with_ttl("quantum:kill", "1", 60))

    def test_set_with_ttl_rejects_position_key(self):
        client = RedisClient("127.0.0.1", 6379)
        with pytest.raises(RuntimeError, match="WRITE_GUARD"):
            _run(client.set_with_ttl("quantum:position:BTCUSDT", "bad", 60))

    def test_set_with_ttl_rejects_arbitrary_key(self):
        client = RedisClient("127.0.0.1", 6379)
        with pytest.raises(RuntimeError, match="WRITE_GUARD"):
            _run(client.set_with_ttl("somekey", "value", 10))


# ── Positive path: allowed writes must reach the underlying client ─────────────


class TestRedisWriteGuardPositive:
    """Prove that the write guard does NOT block the two permitted write paths."""

    @pytest.mark.asyncio
    async def test_xadd_accepts_exit_audit_stream(self):
        """quantum:stream:exit.audit is the primary audit target and must be allowed."""
        client = RedisClient("127.0.0.1", 6379)
        client._client = AsyncMock()
        await client.xadd("quantum:stream:exit.audit", {"k": "v"})
        client._client.xadd.assert_called_once_with(
            "quantum:stream:exit.audit", {"k": "v"}
        )

    @pytest.mark.asyncio
    async def test_xadd_accepts_exit_metrics_stream(self):
        """quantum:stream:exit.metrics is the metrics target and must be allowed."""
        client = RedisClient("127.0.0.1", 6379)
        client._client = AsyncMock()
        await client.xadd("quantum:stream:exit.metrics", {"n": "5"})
        client._client.xadd.assert_called_once_with(
            "quantum:stream:exit.metrics", {"n": "5"}
        )

    @pytest.mark.asyncio
    async def test_set_with_ttl_accepts_exit_agent_prefix(self):
        """Any key under quantum:exit_agent: must be accepted (heartbeat lives here)."""
        client = RedisClient("127.0.0.1", 6379)
        client._client = AsyncMock()
        await client.set_with_ttl("quantum:exit_agent:heartbeat", "123456", 60)
        client._client.set.assert_called_once_with(
            "quantum:exit_agent:heartbeat", "123456", ex=60
        )
