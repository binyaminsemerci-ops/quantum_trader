"""
Tests for ShadowPublisher.

Verifies:
- Correct stream names used
- Forbidden streams blocked
- Non-shadow streams blocked
- Data formatting
"""

import time
import json
import pytest
from unittest.mock import MagicMock

from microservices.exit_brain_v1.publishers.shadow_publisher import (
    ShadowPublisher,
    _FORBIDDEN_STREAMS,
)
from microservices.exit_brain_v1.models.position_exit_state import PositionExitState
from microservices.exit_brain_v1.engines.geometry_engine import GeometryResult
from microservices.exit_brain_v1.engines.regime_drift_engine import RegimeState


def _make_valid_state():
    return PositionExitState(
        position_id="BTCUSDT_LONG",
        symbol="BTCUSDT",
        side="LONG",
        status="OPEN",
        entry_price=50000.0,
        current_price=51000.0,
        quantity=0.01,
        notional=510.0,
        unrealized_pnl=10.0,
        unrealized_pnl_pct=2.0,
        open_timestamp=time.time() - 300,
        source_timestamps={"p33_snapshot": time.time()},
        data_quality_flags=[],
        shadow_only=True,
    )


class TestPublishState:
    """Test publish_state."""

    def test_writes_to_correct_stream(self):
        redis = MagicMock()
        redis.xadd.return_value = b"1-0"

        pub = ShadowPublisher(redis)
        entry_id = pub.publish_state(_make_valid_state())

        assert entry_id == "1-0"
        redis.xadd.assert_called_once()
        call_args = redis.xadd.call_args
        assert call_args[0][0] == "quantum:stream:exit.state.shadow"

    def test_flattens_nested_fields(self):
        redis = MagicMock()
        redis.xadd.return_value = b"1-0"

        pub = ShadowPublisher(redis)
        pub.publish_state(_make_valid_state())

        fields = redis.xadd.call_args[0][1]
        # source_timestamps should be JSON string
        assert isinstance(fields["source_timestamps"], str)
        json.loads(fields["source_timestamps"])  # Should not raise
        # data_quality_flags should be JSON string
        assert isinstance(fields["data_quality_flags"], str)

    def test_none_values_converted(self):
        redis = MagicMock()
        redis.xadd.return_value = b"1-0"

        pub = ShadowPublisher(redis)
        pub.publish_state(_make_valid_state())

        fields = redis.xadd.call_args[0][1]
        # None values should be empty string, not "None"
        for key, val in fields.items():
            assert val != "None" or key not in ("momentum_score", "atr")


class TestPublishGeometry:
    """Test publish_geometry."""

    def test_writes_geometry_to_correct_stream(self):
        redis = MagicMock()
        redis.xadd.return_value = b"2-0"

        pub = ShadowPublisher(redis)
        result = GeometryResult(
            mfe=10.0, mae=3.0, drawdown_from_peak=2.0,
            profit_protection_ratio=0.8, momentum_decay=-0.5,
            reward_to_risk_remaining=2.5,
        )
        entry_id = pub.publish_geometry("BTCUSDT", "LONG", result)

        assert entry_id == "2-0"
        call_args = redis.xadd.call_args
        assert call_args[0][0] == "quantum:stream:exit.geometry.shadow"


class TestPublishRegime:
    """Test publish_regime."""

    def test_writes_regime_to_correct_stream(self):
        redis = MagicMock()
        redis.xadd.return_value = b"3-0"

        pub = ShadowPublisher(redis)
        regime = RegimeState(
            regime_label="BULL", regime_confidence=0.8,
            trend_alignment=0.6, reversal_risk=0.2,
            chop_risk=0.1, mean_reversion_score=0.2,
            drift=None,
        )
        entry_id = pub.publish_regime(regime)

        assert entry_id == "3-0"
        call_args = redis.xadd.call_args
        assert call_args[0][0] == "quantum:stream:exit.regime.shadow"


class TestForbiddenStreams:
    """Test that forbidden streams are blocked."""

    def test_all_forbidden_streams_blocked(self):
        redis = MagicMock()
        pub = ShadowPublisher(redis)

        for stream in _FORBIDDEN_STREAMS:
            result = pub._xadd(stream, {"test": "data"})
            assert result is None
            redis.xadd.assert_not_called()

    def test_non_shadow_stream_blocked(self):
        redis = MagicMock()
        pub = ShadowPublisher(redis)

        result = pub._xadd("quantum:stream:exit.state", {"test": "data"})
        assert result is None
        redis.xadd.assert_not_called()

    def test_shadow_stream_allowed(self):
        redis = MagicMock()
        redis.xadd.return_value = b"1-0"
        pub = ShadowPublisher(redis)

        result = pub._xadd("quantum:stream:test.shadow", {"test": "data"})
        assert result == "1-0"


class TestXaddFailure:
    """Test graceful failure on Redis errors."""

    def test_redis_error_returns_none(self):
        redis = MagicMock()
        redis.xadd.side_effect = Exception("connection lost")

        pub = ShadowPublisher(redis)
        result = pub._xadd("quantum:stream:exit.state.shadow", {"test": "data"})
        assert result is None
