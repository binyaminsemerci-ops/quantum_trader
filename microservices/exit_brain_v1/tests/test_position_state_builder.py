"""
Integration tests for PositionStateBuilder.

Uses a mock Redis client to simulate real data sources.
Tests the assembly logic, fail-closed behavior, and flag generation.
"""

import time
import json
import pytest
from unittest.mock import MagicMock, patch

from microservices.exit_brain_v1.services.position_state_builder import PositionStateBuilder


class FakeRedis:
    """Minimal Redis mock for PositionStateBuilder tests."""

    def __init__(self):
        self._hashes = {}
        self._strings = {}
        self._streams = {}

    def hgetall(self, key):
        return self._hashes.get(key, {})

    def hget(self, key, field):
        h = self._hashes.get(key, {})
        return h.get(field)

    def get(self, key):
        return self._strings.get(key)

    def xrevrange(self, stream, count=1):
        entries = self._streams.get(stream, [])
        return entries[-count:]

    def scan(self, cursor=0, match="*", count=100):
        """Simulate SCAN. Returns all matching keys in one pass."""
        matching = [k for k in self._hashes if self._match(k, match)]
        return (0, matching)

    def _match(self, key, pattern):
        # Simple glob for quantum:state:positions:*
        if pattern.endswith("*"):
            prefix = pattern[:-1]
            return key.startswith(prefix)
        return key == pattern

    # ── Helpers for test setup ───────────────────────────────────────────

    def set_hash(self, key, data):
        self._hashes[key] = data

    def set_string(self, key, value):
        self._strings[key] = value

    def set_stream(self, stream, entries):
        """entries: list of (id, {field: value})"""
        self._streams[stream] = entries


def _make_snapshot(symbol="BTCUSDT", amt=0.01, side="LONG",
                   entry_price=50000.0, mark_price=51000.0,
                   unrealized_pnl=10.0, leverage=10, ts_epoch=None):
    if ts_epoch is None:
        ts_epoch = int(time.time())
    return {
        "position_amt": str(amt),
        "side": side,
        "entry_price": str(entry_price),
        "mark_price": str(mark_price),
        "unrealized_pnl": str(unrealized_pnl),
        "leverage": str(leverage),
        "ts_epoch": str(ts_epoch),
        "source": "binance_testnet",
    }


class TestBuildHappyPath:
    """Test successful state building."""

    def test_builds_state_from_snapshot(self):
        redis = FakeRedis()
        redis.set_hash("quantum:state:positions:BTCUSDT", _make_snapshot())

        builder = PositionStateBuilder(redis)
        state = builder.build("BTCUSDT")

        assert state is not None
        assert state.symbol == "BTCUSDT"
        assert state.side == "LONG"
        assert state.entry_price == 50000.0
        assert state.current_price == 51000.0
        assert state.quantity == pytest.approx(0.01)
        assert state.shadow_only is True

    def test_unrealized_pnl_pct_long(self):
        redis = FakeRedis()
        redis.set_hash("quantum:state:positions:BTCUSDT",
                        _make_snapshot(entry_price=50000.0, mark_price=51000.0))

        builder = PositionStateBuilder(redis)
        state = builder.build("BTCUSDT")

        expected_pct = ((51000 - 50000) / 50000) * 100  # 2.0%
        assert state.unrealized_pnl_pct == pytest.approx(expected_pct)

    def test_unrealized_pnl_pct_short(self):
        redis = FakeRedis()
        redis.set_hash("quantum:state:positions:ETHUSDT",
                        _make_snapshot(symbol="ETHUSDT", amt=-0.1, side="SHORT",
                                       entry_price=3000.0, mark_price=2900.0))

        builder = PositionStateBuilder(redis)
        state = builder.build("ETHUSDT")

        expected_pct = ((3000 - 2900) / 3000) * 100  # 3.33%
        assert state.unrealized_pnl_pct == pytest.approx(expected_pct, rel=0.01)


class TestFailClosed:
    """Test that missing required data returns None."""

    def test_no_snapshot_returns_none(self):
        redis = FakeRedis()
        builder = PositionStateBuilder(redis)
        assert builder.build("BTCUSDT") is None

    def test_zero_position_amt_returns_none(self):
        redis = FakeRedis()
        redis.set_hash("quantum:state:positions:BTCUSDT",
                        _make_snapshot(amt=0.0))
        builder = PositionStateBuilder(redis)
        assert builder.build("BTCUSDT") is None

    def test_invalid_side_returns_none(self):
        redis = FakeRedis()
        redis.set_hash("quantum:state:positions:BTCUSDT",
                        _make_snapshot(side="NONE"))
        builder = PositionStateBuilder(redis)
        assert builder.build("BTCUSDT") is None

    def test_zero_entry_price_returns_none(self):
        redis = FakeRedis()
        redis.set_hash("quantum:state:positions:BTCUSDT",
                        _make_snapshot(entry_price=0.0))
        builder = PositionStateBuilder(redis)
        assert builder.build("BTCUSDT") is None


class TestDataQualityFlags:
    """Test that missing optional data produces correct flags."""

    def test_stale_snapshot_flagged(self):
        redis = FakeRedis()
        old_ts = int(time.time()) - 60  # 60 seconds ago
        redis.set_hash("quantum:state:positions:BTCUSDT",
                        _make_snapshot(ts_epoch=old_ts))

        builder = PositionStateBuilder(redis)
        state = builder.build("BTCUSDT")

        assert state is not None
        assert "STALE_SNAPSHOT" in state.data_quality_flags

    def test_missing_marketstate_flagged(self):
        redis = FakeRedis()
        redis.set_hash("quantum:state:positions:BTCUSDT", _make_snapshot())

        builder = PositionStateBuilder(redis)
        state = builder.build("BTCUSDT")

        assert "MISSING_MARKETSTATE" in state.data_quality_flags

    def test_missing_regime_flagged(self):
        redis = FakeRedis()
        redis.set_hash("quantum:state:positions:BTCUSDT", _make_snapshot())

        builder = PositionStateBuilder(redis)
        state = builder.build("BTCUSDT")

        assert "MISSING_REGIME" in state.data_quality_flags

    def test_missing_atr_flagged(self):
        redis = FakeRedis()
        redis.set_hash("quantum:state:positions:BTCUSDT", _make_snapshot())

        builder = PositionStateBuilder(redis)
        state = builder.build("BTCUSDT")

        assert "MISSING_ATR" in state.data_quality_flags

    def test_mark_price_fallback_flagged(self):
        redis = FakeRedis()
        redis.set_hash("quantum:state:positions:BTCUSDT",
                        _make_snapshot(mark_price=0.0))

        builder = PositionStateBuilder(redis)
        state = builder.build("BTCUSDT")

        assert "MARK_PRICE_FALLBACK" in state.data_quality_flags
        assert state.current_price == state.entry_price


class TestWithMarketState:
    """Test MarketState enrichment."""

    def test_reads_marketstate_hash(self):
        redis = FakeRedis()
        redis.set_hash("quantum:state:positions:BTCUSDT", _make_snapshot())
        redis.set_hash("quantum:marketstate:BTCUSDT", {
            "sigma": "0.025",
            "mu": "0.001",
            "ts": str(time.time()),
            "regime_probs": json.dumps({"TREND": 0.6, "MR": 0.2, "CHOP": 0.2}),
        })

        builder = PositionStateBuilder(redis)
        state = builder.build("BTCUSDT")

        assert state.volatility_short == pytest.approx(0.025)
        assert state.trend_signal == pytest.approx(0.001)
        assert "MISSING_MARKETSTATE" not in state.data_quality_flags


class TestWithMetaRegime:
    """Test meta-regime enrichment."""

    def test_reads_meta_regime_stream(self):
        redis = FakeRedis()
        redis.set_hash("quantum:state:positions:BTCUSDT", _make_snapshot())
        redis.set_stream("quantum:stream:meta.regime", [
            ("1-0", {
                "regime": "BULL",
                "confidence": "0.85",
                "timestamp": str(time.time()),
            }),
        ])

        builder = PositionStateBuilder(redis)
        state = builder.build("BTCUSDT")

        assert state.regime_label == "BULL"
        assert state.regime_confidence == pytest.approx(0.85)
        assert "MISSING_REGIME" not in state.data_quality_flags


class TestDiscoverOpenPositions:
    """Test position discovery."""

    def test_finds_non_zero_positions(self):
        redis = FakeRedis()
        redis.set_hash("quantum:state:positions:BTCUSDT",
                        {"position_amt": "0.01"})
        redis.set_hash("quantum:state:positions:ETHUSDT",
                        {"position_amt": "0.0"})
        redis.set_hash("quantum:state:positions:SOLUSDT",
                        {"position_amt": "-0.5"})

        builder = PositionStateBuilder(redis)
        symbols = builder.discover_open_positions()

        assert "BTCUSDT" in symbols
        assert "SOLUSDT" in symbols
        assert "ETHUSDT" not in symbols


class TestBuildAll:
    """Test build_all method."""

    def test_builds_multiple_skips_failures(self):
        redis = FakeRedis()
        redis.set_hash("quantum:state:positions:BTCUSDT", _make_snapshot())
        # ETHUSDT has no snapshot → will fail

        builder = PositionStateBuilder(redis)
        states = builder.build_all(["BTCUSDT", "ETHUSDT"])

        assert len(states) == 1
        assert states[0].symbol == "BTCUSDT"
