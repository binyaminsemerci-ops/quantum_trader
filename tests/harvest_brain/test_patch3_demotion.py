"""PATCH-3 demotion tests for HarvestBrain StreamPublisher.

Coverage
--------
* _publish_live() — blocked (routed to shadow) when HARVEST_LIVE_WRITES_ENABLED
  is absent or any non-"true" value; passes through when explicitly "true".
* _publish_shadow() — still works normally regardless of the demotion guard.
* publish() dispatch — dry_run=True routes to shadow directly (unaffected path);
  dry_run=False hits the demotion guard when HARVEST_LIVE_WRITES_ENABLED != true.

Fakes
-----
_RecordingRedis  — records xadd calls and zadd calls; no network.
_SilentConfig    — minimal Config-shaped object with only the stream names
                   accessed by StreamPublisher.  Avoids importing Config which
                   would trigger full env-parse at import time.

These tests do NOT exercise HarvestPolicy, DedupManager, or the main event loop.
"""
from __future__ import annotations

import json
import os
import pytest

from microservices.harvest_brain.harvest_brain import HarvestIntent, StreamPublisher


# ── Fakes ──────────────────────────────────────────────────────────────────────

class _RecordingRedis:
    """Records xadd and zadd calls; raises on any unexpected method."""

    def __init__(self) -> None:
        self.xadd_calls: list[tuple[str, object]] = []
        self.zadd_calls: list[tuple[str, dict]] = []

    def xadd(self, stream: str, fields) -> str:
        self.xadd_calls.append((stream, fields))
        return "1-0"

    def zadd(self, key: str, mapping: dict) -> None:
        self.zadd_calls.append((key, mapping))

    def zremrangebyrank(self, *_) -> None:
        pass

    def hset(self, *_args, **_kwargs) -> None:
        pass

    def expire(self, *_args) -> None:
        pass


class _SilentConfig:
    """Minimal config slice used by StreamPublisher."""

    stream_harvest_suggestions = "quantum:stream:harvest.suggestions"
    stream_apply_plan = "quantum:stream:apply.plan"


def _make_intent(dry_run: bool = False) -> HarvestIntent:
    return HarvestIntent(
        intent_type="HARVEST_PARTIAL",
        symbol="BTCUSDT",
        side="SELL",
        qty=0.01,
        reason="test",
        r_level=1.0,
        unrealized_pnl=50.0,
        correlation_id="test:corr:001",
        trace_id="test:trace:001",
        dry_run=dry_run,
    )


def _make_publisher() -> tuple[StreamPublisher, _RecordingRedis]:
    redis = _RecordingRedis()
    config = _SilentConfig()
    pub = StreamPublisher.__new__(StreamPublisher)
    pub.redis = redis
    pub.config = config
    return pub, redis


# ── TestLiveWriteBlocked ────────────────────────────────────────────────────────

class TestLiveWriteBlocked:
    """_publish_live() must be blocked when HARVEST_LIVE_WRITES_ENABLED is absent or falsy."""

    def test_no_env_var_blocks_apply_plan_write(self, monkeypatch):
        """Env var absent → apply.plan xadd must NOT be called."""
        monkeypatch.delenv("HARVEST_LIVE_WRITES_ENABLED", raising=False)
        pub, redis = _make_publisher()
        intent = _make_intent(dry_run=False)

        result = pub._publish_live(intent)

        apply_plan_streams = [s for s, _ in redis.xadd_calls if s == "quantum:stream:apply.plan"]
        assert apply_plan_streams == [], "apply.plan must not be written when demotion guard is active"

    def test_no_env_var_routes_to_shadow_stream(self, monkeypatch):
        """Env var absent → intent is re-routed to harvest.suggestions instead."""
        monkeypatch.delenv("HARVEST_LIVE_WRITES_ENABLED", raising=False)
        pub, redis = _make_publisher()
        intent = _make_intent(dry_run=False)

        pub._publish_live(intent)

        shadow_streams = [s for s, _ in redis.xadd_calls if s == "quantum:stream:harvest.suggestions"]
        assert len(shadow_streams) == 1, "shadow stream must receive exactly one entry"

    def test_false_string_blocks_apply_plan_write(self, monkeypatch):
        """HARVEST_LIVE_WRITES_ENABLED=false → still blocked."""
        monkeypatch.setenv("HARVEST_LIVE_WRITES_ENABLED", "false")
        pub, redis = _make_publisher()
        intent = _make_intent(dry_run=False)

        pub._publish_live(intent)

        apply_plan_streams = [s for s, _ in redis.xadd_calls if s == "quantum:stream:apply.plan"]
        assert apply_plan_streams == []

    def test_arbitrary_value_blocks_apply_plan_write(self, monkeypatch):
        """Any value other than 'true' (case-insensitive) → still blocked."""
        monkeypatch.setenv("HARVEST_LIVE_WRITES_ENABLED", "yes")
        pub, redis = _make_publisher()
        intent = _make_intent(dry_run=False)

        pub._publish_live(intent)

        apply_plan_streams = [s for s, _ in redis.xadd_calls if s == "quantum:stream:apply.plan"]
        assert apply_plan_streams == []

    def test_no_env_var_returns_shadow_result(self, monkeypatch):
        """Guard returns the return value of _publish_shadow() (True on success)."""
        monkeypatch.delenv("HARVEST_LIVE_WRITES_ENABLED", raising=False)
        pub, redis = _make_publisher()
        intent = _make_intent(dry_run=False)

        result = pub._publish_live(intent)

        assert result is True


# ── TestLiveWriteAllowed ────────────────────────────────────────────────────────

class TestLiveWriteAllowed:
    """_publish_live() must reach apply.plan when HARVEST_LIVE_WRITES_ENABLED=true."""

    def test_true_value_reaches_apply_plan(self, monkeypatch):
        """HARVEST_LIVE_WRITES_ENABLED=true → apply.plan xadd IS called."""
        monkeypatch.setenv("HARVEST_LIVE_WRITES_ENABLED", "true")
        pub, redis = _make_publisher()
        intent = _make_intent(dry_run=False)

        pub._publish_live(intent)

        apply_plan_streams = [s for s, _ in redis.xadd_calls if s == "quantum:stream:apply.plan"]
        assert len(apply_plan_streams) == 1

    def test_true_uppercase_reaches_apply_plan(self, monkeypatch):
        """HARVEST_LIVE_WRITES_ENABLED=TRUE (uppercase) → also allowed."""
        monkeypatch.setenv("HARVEST_LIVE_WRITES_ENABLED", "TRUE")
        pub, redis = _make_publisher()
        intent = _make_intent(dry_run=False)

        pub._publish_live(intent)

        apply_plan_streams = [s for s, _ in redis.xadd_calls if s == "quantum:stream:apply.plan"]
        assert len(apply_plan_streams) == 1


# ── TestShadowPathUnaffected ────────────────────────────────────────────────────

class TestShadowPathUnaffected:
    """_publish_shadow() must be completely unaffected by PATCH-3."""

    def test_shadow_writes_to_suggestions_stream(self, monkeypatch):
        """Shadow publishes to harvest.suggestions regardless of env var."""
        monkeypatch.delenv("HARVEST_LIVE_WRITES_ENABLED", raising=False)
        pub, redis = _make_publisher()
        intent = _make_intent(dry_run=True)

        result = pub._publish_shadow(intent)

        shadow_streams = [s for s, _ in redis.xadd_calls if s == "quantum:stream:harvest.suggestions"]
        assert len(shadow_streams) == 1
        assert result is True

    def test_shadow_never_writes_to_apply_plan(self, monkeypatch):
        """Shadow path must never touch apply.plan in any env state."""
        monkeypatch.setenv("HARVEST_LIVE_WRITES_ENABLED", "true")  # even when live is enabled
        pub, redis = _make_publisher()
        intent = _make_intent(dry_run=True)

        pub._publish_shadow(intent)

        apply_plan_streams = [s for s, _ in redis.xadd_calls if s == "quantum:stream:apply.plan"]
        assert apply_plan_streams == []


# ── TestPublishDispatch ─────────────────────────────────────────────────────────

class TestPublishDispatch:
    """publish() top-level dispatch — PATCH-3 guard applies on the live path."""

    def test_dry_run_true_does_not_reach_live(self, monkeypatch):
        """publish(dry_run=True intent) must not call _publish_live at all.
        Guard never needs to fire — we confirm apply.plan is never written."""
        monkeypatch.setenv("HARVEST_LIVE_WRITES_ENABLED", "true")  # guard disabled
        pub, redis = _make_publisher()
        intent = _make_intent(dry_run=True)

        pub.publish(intent)

        apply_plan_streams = [s for s, _ in redis.xadd_calls if s == "quantum:stream:apply.plan"]
        assert apply_plan_streams == [], "dry_run=True must never reach apply.plan"

    def test_dry_run_false_blocked_without_env(self, monkeypatch):
        """publish(dry_run=False intent) → _publish_live → guard fires → shadow only."""
        monkeypatch.delenv("HARVEST_LIVE_WRITES_ENABLED", raising=False)
        pub, redis = _make_publisher()
        intent = _make_intent(dry_run=False)

        pub.publish(intent)

        apply_plan_streams = [s for s, _ in redis.xadd_calls if s == "quantum:stream:apply.plan"]
        shadow_streams = [s for s, _ in redis.xadd_calls if s == "quantum:stream:harvest.suggestions"]
        assert apply_plan_streams == []
        assert len(shadow_streams) == 1
