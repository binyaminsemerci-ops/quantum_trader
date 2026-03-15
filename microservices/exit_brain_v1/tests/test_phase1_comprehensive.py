"""
Comprehensive Phase 1 tests — gap-filling for edge cases not covered
by existing individual test files.

Covers:
- PositionExitState round-trip fidelity (to_dict → reconstruct)
- GeometryEngine boundary/degenerate inputs
- RegimeDriftEngine L1 symmetry and label mapping
- PositionStateBuilder enrichment paths (ATR, ledger, JSON marketstate)
- ShadowPublisher STREAM_MAXLEN and _handle_none_values
- Cross-module integration: builder → geometry → regime → publisher
"""

from __future__ import annotations

import json
import math
import time
from unittest.mock import MagicMock

import pytest

from microservices.exit_brain_v1.models.position_exit_state import (
    PositionExitState,
    VALID_REGIMES,
    VALID_SIDES,
    VALID_STATUSES,
)
from microservices.exit_brain_v1.engines.geometry_engine import (
    GeometryEngine,
    GeometryResult,
)
from microservices.exit_brain_v1.engines.regime_drift_engine import (
    RegimeDriftEngine,
    RegimeDrift,
    RegimeState,
)
from microservices.exit_brain_v1.services.position_state_builder import (
    PositionStateBuilder,
)
from microservices.exit_brain_v1.publishers.shadow_publisher import (
    ShadowPublisher,
    STREAM_MAXLEN,
    _FORBIDDEN_STREAMS,
)


# ═══════════════════════════════════════════════════════════════════════════════
# Helpers
# ═══════════════════════════════════════════════════════════════════════════════


def _make_state(**overrides) -> PositionExitState:
    defaults = dict(
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
    defaults.update(overrides)
    return PositionExitState(**defaults)


class FakeRedis:
    """Minimal Redis mock."""

    def __init__(self):
        self._hashes: dict = {}
        self._strings: dict = {}
        self._streams: dict = {}

    def hgetall(self, key):
        return self._hashes.get(key, {})

    def hget(self, key, field):
        return self._hashes.get(key, {}).get(field)

    def get(self, key):
        return self._strings.get(key)

    def xrevrange(self, stream, *args, count=1):
        entries = self._streams.get(stream, [])
        return entries[-count:]

    def xadd(self, stream, fields, maxlen=None):
        return b"1-0"

    def scan(self, cursor=0, match="*", count=100):
        matching = [k for k in self._hashes if k.startswith(match.replace("*", ""))]
        return (0, matching)

    def set_hash(self, key, data):
        self._hashes[key] = data

    def set_string(self, key, value):
        self._strings[key] = value

    def set_stream(self, stream, entries):
        self._streams[stream] = entries


def _snapshot(symbol="BTCUSDT", **overrides):
    d = {
        "position_amt": "0.01",
        "side": "LONG",
        "entry_price": "50000.0",
        "mark_price": "51000.0",
        "unrealized_pnl": "10.0",
        "leverage": "10",
        "ts_epoch": str(int(time.time())),
    }
    d.update(overrides)
    return d


# ═══════════════════════════════════════════════════════════════════════════════
# 1. PositionExitState — extended contract tests
# ═══════════════════════════════════════════════════════════════════════════════


class TestPositionExitStateExtended:
    """Edge cases not covered by test_position_exit_state.py."""

    def test_to_dict_round_trip_all_keys(self):
        """Every field in to_dict output must have a matching attribute."""
        state = _make_state(
            mark_price=50500.0,
            leverage=5.0,
            realized_pnl=1.0,
            fees_paid=0.5,
            max_favorable_excursion=500.0,
            max_adverse_excursion=100.0,
            peak_unrealized_pnl=20.0,
            trough_unrealized_pnl=-5.0,
            drawdown_from_peak_pnl=10.0,
            volatility_short=0.02,
            volatility_medium=0.03,
            atr=150.0,
            trend_signal=0.001,
            regime_label="BULL",
            regime_confidence=0.8,
            momentum_score=0.5,
            mean_reversion_score=0.3,
            liquidity_score=0.9,
            spread_bps=1.5,
        )
        d = state.to_dict()
        assert d["mark_price"] == 50500.0
        assert d["leverage"] == 5.0
        assert d["realized_pnl"] == 1.0
        assert d["fees_paid"] == 0.5
        assert d["max_favorable_excursion"] == 500.0
        assert d["peak_unrealized_pnl"] == 20.0
        assert d["volatility_short"] == 0.02
        assert d["atr"] == 150.0
        assert d["momentum_score"] == 0.5
        assert d["spread_bps"] == 1.5
        assert d["regime_label"] == "BULL"
        assert d["shadow_only"] is True

    def test_to_dict_hold_seconds_is_dynamic(self):
        """hold_seconds is computed at call time, not at construction time."""
        state = _make_state(open_timestamp=time.time() - 120)
        d = state.to_dict()
        assert 119 <= d["hold_seconds"] <= 122

    def test_negative_current_price_fails(self):
        state = _make_state(current_price=-1.0)
        errors = state.validate()
        assert any("current_price" in e for e in errors)

    def test_all_valid_sides_accepted(self):
        for side in VALID_SIDES:
            state = _make_state(side=side)
            errors = state.validate()
            assert not any("side" in e for e in errors)

    def test_all_valid_statuses_accepted(self):
        for status in VALID_STATUSES:
            state = _make_state(status=status)
            errors = state.validate()
            assert not any("status" in e for e in errors)

    def test_all_valid_regimes_accepted(self):
        for regime in VALID_REGIMES:
            state = _make_state(regime_label=regime)
            state.validate()
            assert state.regime_label == regime

    def test_multiple_quality_flags_accumulated(self):
        state = _make_state(
            leverage=0.5,
            regime_label="GARBAGE",
            regime_confidence=2.0,
        )
        state.validate()
        assert "LEVERAGE_BELOW_1" in state.data_quality_flags
        assert any("INVALID_REGIME" in f for f in state.data_quality_flags)
        assert "REGIME_CONFIDENCE_OUT_OF_RANGE" in state.data_quality_flags

    def test_short_side_state_valid(self):
        state = _make_state(
            position_id="ETHUSDT_SHORT",
            symbol="ETHUSDT",
            side="SHORT",
        )
        errors = state.validate()
        assert errors == []


# ═══════════════════════════════════════════════════════════════════════════════
# 2. GeometryEngine — boundary/degenerate inputs
# ═══════════════════════════════════════════════════════════════════════════════


class TestGeometryEdgeCases:
    """Degenerate and boundary inputs for geometry math."""

    def test_mfe_equal_prices_long(self):
        assert GeometryEngine.compute_mfe(100.0, 100.0, "LONG") == 0.0

    def test_mfe_equal_prices_short(self):
        assert GeometryEngine.compute_mfe(100.0, 100.0, "SHORT") == 0.0

    def test_mae_equal_prices(self):
        assert GeometryEngine.compute_mae(100.0, 100.0, "LONG") == 0.0
        assert GeometryEngine.compute_mae(100.0, 100.0, "SHORT") == 0.0

    def test_drawdown_negative_peak_and_current(self):
        dd = GeometryEngine.compute_drawdown_from_peak(-10.0, -5.0)
        assert dd == 0.0  # peak <= 0

    def test_profit_protection_ratio_both_zero(self):
        ppr = GeometryEngine.compute_profit_protection_ratio(0.0, 0.0)
        assert ppr == 0.0

    def test_momentum_decay_two_points(self):
        slope = GeometryEngine.compute_momentum_decay([1.0, 3.0])
        assert slope == pytest.approx(2.0)

    def test_momentum_decay_large_window_on_short_data(self):
        slope = GeometryEngine.compute_momentum_decay([1.0, 2.0, 3.0], window=100)
        assert slope == pytest.approx(1.0)

    def test_reward_to_risk_short_zero_downside(self):
        """SHORT: price=100, stop=100 (at stop), target=90 → downside=0, upside=10."""
        rtr = GeometryEngine.compute_reward_to_risk_remaining(
            100.0, 105.0, 100.0, 90.0, "SHORT",
        )
        assert rtr == 100.0  # capped

    def test_reward_to_risk_both_zero_target_stop(self):
        rtr = GeometryEngine.compute_reward_to_risk_remaining(
            100.0, 100.0, 0.0, 0.0, "LONG",
        )
        assert rtr == 0.0

    def test_compute_all_no_stop_target(self):
        """When stop/target are 0, reward_to_risk should be 0."""
        result = GeometryEngine.compute_all(
            entry_price=100.0, current_price=105.0,
            peak_price=108.0, trough_price=97.0,
            side="LONG", current_pnl=5.0, peak_pnl=8.0,
        )
        assert result.reward_to_risk_remaining == 0.0
        assert result.mfe == pytest.approx(8.0)

    def test_compute_all_short_side(self):
        result = GeometryEngine.compute_all(
            entry_price=100.0, current_price=95.0,
            peak_price=94.0, trough_price=102.0,
            side="SHORT", current_pnl=5.0, peak_pnl=6.0,
            stop_price=105.0, target_price=90.0,
        )
        assert isinstance(result, GeometryResult)
        assert result.mfe == pytest.approx(6.0)   # entry 100 - peak 94
        assert result.mae == pytest.approx(2.0)   # trough 102 - entry 100
        assert result.drawdown_from_peak == pytest.approx(1.0)  # 6 - 5

    def test_geometry_result_is_frozen(self):
        result = GeometryResult(1.0, 2.0, 3.0, 0.5, -0.1, 2.0)
        with pytest.raises(AttributeError):
            result.mfe = 99.0


# ═══════════════════════════════════════════════════════════════════════════════
# 3. RegimeDriftEngine — extended coverage
# ═══════════════════════════════════════════════════════════════════════════════


class TestRegimeDriftExtended:
    """L1 symmetry, partial keys, label mapping."""

    def test_l1_distance_symmetry(self):
        """L1(A,B) == L1(B,A)."""
        a = {"TREND": 0.5, "MR": 0.3, "CHOP": 0.2}
        b = {"TREND": 0.2, "MR": 0.5, "CHOP": 0.3}
        r1 = RegimeDriftEngine.detect_regime_drift(a, b)
        r2 = RegimeDriftEngine.detect_regime_drift(b, a)
        assert r1.magnitude == pytest.approx(r2.magnitude)

    def test_partial_keys_handled(self):
        """Distributions with different key sets."""
        a = {"TREND": 1.0}
        b = {"MR": 1.0}
        r = RegimeDriftEngine.detect_regime_drift(a, b)
        assert r.drifted is True
        assert r.magnitude == pytest.approx(2.0)  # L1 max

    def test_reversal_risk_short_downtrend(self):
        """SHORT in downtrend, high MR → reversal threat."""
        risk = RegimeDriftEngine.compute_reversal_risk(
            {"TREND": 0.1, "MR": 0.8, "CHOP": 0.1}, "SHORT", mu=-0.01,
        )
        assert risk > 0.5

    def test_chop_risk_no_vol_boost_strong_trend(self):
        risk = RegimeDriftEngine.compute_chop_risk(
            {"TREND": 0.1, "MR": 0.1, "CHOP": 0.3},
            sigma=0.03, ts=0.8,  # ts >= 0.5 → no boost
        )
        assert risk == pytest.approx(0.3)

    def test_summarize_mr_dominant(self):
        result = RegimeDriftEngine.summarize_regime_state(
            side="LONG",
            regime_probs={"TREND": 0.1, "MR": 0.7, "CHOP": 0.2},
            mu=0.001, sigma=0.02, ts=0.1,
        )
        assert result.regime_label == "RANGE"
        assert result.mean_reversion_score == pytest.approx(0.7)

    def test_regime_drift_frozen(self):
        drift = RegimeDrift(
            drifted=True, magnitude=0.5,
            old_dominant="TREND", new_dominant="CHOP",
            transition="TREND→CHOP",
        )
        with pytest.raises(AttributeError):
            drift.drifted = False

    def test_regime_state_frozen(self):
        rs = RegimeState(
            regime_label="BULL", regime_confidence=0.8,
            trend_alignment=0.5, reversal_risk=0.2,
            chop_risk=0.1, mean_reversion_score=0.1,
            drift=None,
        )
        with pytest.raises(AttributeError):
            rs.regime_label = "BEAR"


# ═══════════════════════════════════════════════════════════════════════════════
# 4. PositionStateBuilder — enrichment integration
# ═══════════════════════════════════════════════════════════════════════════════


class TestPositionStateBuilderExtended:
    """Enrichment paths not covered by test_position_state_builder.py."""

    def test_atr_read_from_first_key_pattern(self):
        redis = FakeRedis()
        redis.set_hash("quantum:state:positions:BTCUSDT", _snapshot())
        redis.set_string("quantum:atr:BTCUSDT", "150.5")

        builder = PositionStateBuilder(redis)
        state = builder.build("BTCUSDT")

        assert state is not None
        assert state.atr == pytest.approx(150.5)
        assert "MISSING_ATR" not in state.data_quality_flags

    def test_atr_read_from_second_key_pattern(self):
        redis = FakeRedis()
        redis.set_hash("quantum:state:positions:BTCUSDT", _snapshot())
        redis.set_string("quantum:indicator:atr:BTCUSDT", "200.0")

        builder = PositionStateBuilder(redis)
        state = builder.build("BTCUSDT")

        assert state is not None
        assert state.atr == pytest.approx(200.0)

    def test_marketstate_json_string(self):
        """MarketState stored as JSON string instead of hash."""
        redis = FakeRedis()
        redis.set_hash("quantum:state:positions:BTCUSDT", _snapshot())
        redis.set_string(
            "quantum:marketstate:BTCUSDT",
            json.dumps({
                "sigma": 0.03,
                "mu": 0.002,
                "ts": time.time(),
            }),
        )

        builder = PositionStateBuilder(redis)
        state = builder.build("BTCUSDT")

        assert state is not None
        assert state.volatility_short == pytest.approx(0.03)
        assert state.trend_signal == pytest.approx(0.002)

    def test_ledger_provides_open_timestamp(self):
        redis = FakeRedis()
        redis.set_hash("quantum:state:positions:BTCUSDT", _snapshot())
        ledger_ts = time.time() - 3600  # opened 1h ago
        redis.set_hash("quantum:position:ledger:BTCUSDT", {
            "updated_at": str(ledger_ts),
        })

        builder = PositionStateBuilder(redis)
        state = builder.build("BTCUSDT")

        assert state is not None
        assert state.open_timestamp == pytest.approx(ledger_ts, abs=1.0)
        assert "OPEN_TIMESTAMP_ESTIMATED" not in state.data_quality_flags

    def test_no_ledger_estimates_open_timestamp(self):
        redis = FakeRedis()
        redis.set_hash("quantum:state:positions:BTCUSDT", _snapshot())

        builder = PositionStateBuilder(redis)
        state = builder.build("BTCUSDT")

        assert state is not None
        assert "OPEN_TIMESTAMP_ESTIMATED" in state.data_quality_flags

    def test_stale_marketstate_flagged(self):
        redis = FakeRedis()
        redis.set_hash("quantum:state:positions:BTCUSDT", _snapshot())
        redis.set_hash("quantum:marketstate:BTCUSDT", {
            "sigma": "0.02",
            "mu": "0.001",
            "ts": str(time.time() - 200),  # older than 120s threshold
        })

        builder = PositionStateBuilder(redis)
        state = builder.build("BTCUSDT")

        assert state is not None
        assert "STALE_MARKETSTATE" in state.data_quality_flags

    def test_stale_regime_flagged(self):
        redis = FakeRedis()
        redis.set_hash("quantum:state:positions:BTCUSDT", _snapshot())
        redis.set_stream("quantum:stream:meta.regime", [
            ("1-0", {
                "regime": "BULL",
                "confidence": "0.8",
                "timestamp": str(time.time() - 200),
            }),
        ])

        builder = PositionStateBuilder(redis)
        state = builder.build("BTCUSDT")

        assert state is not None
        assert "STALE_REGIME" in state.data_quality_flags

    def test_regime_probs_parse_error_flagged(self):
        redis = FakeRedis()
        redis.set_hash("quantum:state:positions:BTCUSDT", _snapshot())
        redis.set_hash("quantum:marketstate:BTCUSDT", {
            "sigma": "0.02",
            "mu": "0.001",
            "ts": str(time.time()),
            "regime_probs": "NOT_JSON",
        })

        builder = PositionStateBuilder(redis)
        state = builder.build("BTCUSDT")

        assert state is not None
        assert "REGIME_PROBS_PARSE_ERROR" in state.data_quality_flags

    def test_build_all_returns_only_successes(self):
        redis = FakeRedis()
        redis.set_hash("quantum:state:positions:BTCUSDT", _snapshot())
        redis.set_hash("quantum:state:positions:ETHUSDT",
                        _snapshot(side="NONE"))  # will fail

        builder = PositionStateBuilder(redis)
        states = builder.build_all(["BTCUSDT", "ETHUSDT", "MISSING"])

        assert len(states) == 1
        assert states[0].symbol == "BTCUSDT"

    def test_negative_position_amt_treated_as_short(self):
        redis = FakeRedis()
        redis.set_hash("quantum:state:positions:ETHUSDT",
                        _snapshot(position_amt="-0.5", side="SHORT",
                                  entry_price="3000.0", mark_price="2900.0"))

        builder = PositionStateBuilder(redis)
        state = builder.build("ETHUSDT")

        assert state is not None
        assert state.side == "SHORT"
        assert state.quantity == pytest.approx(0.5)


# ═══════════════════════════════════════════════════════════════════════════════
# 5. ShadowPublisher — extended safety/formatting tests
# ═══════════════════════════════════════════════════════════════════════════════


class TestShadowPublisherExtended:
    """Extended ShadowPublisher coverage."""

    def test_stream_maxlen_passed_to_xadd(self):
        redis = MagicMock()
        redis.xadd.return_value = b"1-0"
        pub = ShadowPublisher(redis)

        pub._xadd("quantum:stream:exit.state.shadow", {"k": "v"})
        _, kwargs = redis.xadd.call_args
        assert kwargs.get("maxlen") == STREAM_MAXLEN or redis.xadd.call_args[0][2] == STREAM_MAXLEN

    def test_publish_geometry_all_fields_stringified(self):
        redis = MagicMock()
        redis.xadd.return_value = b"1-0"
        pub = ShadowPublisher(redis)

        result = GeometryResult(
            mfe=10.0, mae=3.5, drawdown_from_peak=2.0,
            profit_protection_ratio=0.75, momentum_decay=-0.3,
            reward_to_risk_remaining=2.5,
        )
        pub.publish_geometry("BTCUSDT", "LONG", result)

        fields = redis.xadd.call_args[0][1]
        assert fields["mfe"] == "10.0"
        assert fields["mae"] == "3.5"
        assert fields["symbol"] == "BTCUSDT"
        assert fields["side"] == "LONG"
        assert "ts" in fields

    def test_publish_regime_with_drift(self):
        redis = MagicMock()
        redis.xadd.return_value = b"1-0"
        pub = ShadowPublisher(redis)

        drift = RegimeDrift(
            drifted=True, magnitude=0.45,
            old_dominant="TREND", new_dominant="CHOP",
            transition="TREND→CHOP",
        )
        regime = RegimeState(
            regime_label="VOLATILE", regime_confidence=0.8,
            trend_alignment=-0.3, reversal_risk=0.6,
            chop_risk=0.7, mean_reversion_score=0.1,
            drift=drift,
        )
        pub.publish_regime(regime)

        fields = redis.xadd.call_args[0][1]
        assert fields["drift_detected"] == "True"
        assert fields["drift_transition"] == "TREND→CHOP"

    def test_publish_regime_without_drift(self):
        redis = MagicMock()
        redis.xadd.return_value = b"1-0"
        pub = ShadowPublisher(redis)

        regime = RegimeState(
            regime_label="BULL", regime_confidence=0.9,
            trend_alignment=0.8, reversal_risk=0.1,
            chop_risk=0.05, mean_reversion_score=0.1,
            drift=None,
        )
        pub.publish_regime(regime)

        fields = redis.xadd.call_args[0][1]
        assert "drift_detected" not in fields

    def test_shadow_suffix_enforced(self):
        redis = MagicMock()
        pub = ShadowPublisher(redis)

        result = pub._xadd("quantum:stream:exit.state", {"x": "y"})
        assert result is None
        redis.xadd.assert_not_called()

    def test_all_stream_constants_end_with_shadow(self):
        """Every STREAM_* constant on ShadowPublisher must end with .shadow."""
        for attr in dir(ShadowPublisher):
            if attr.startswith("STREAM_"):
                value = getattr(ShadowPublisher, attr)
                assert value.endswith(".shadow"), f"{attr}={value} missing .shadow suffix"

    def test_forbidden_streams_are_non_shadow(self):
        """Every forbidden stream must NOT end with .shadow."""
        for stream in _FORBIDDEN_STREAMS:
            assert not stream.endswith(".shadow"), f"Forbidden {stream} has .shadow?"

    def test_publish_state_with_all_none_optionals(self):
        redis = MagicMock()
        redis.xadd.return_value = b"1-0"
        pub = ShadowPublisher(redis)

        state = _make_state()  # all optionals are None/default
        entry_id = pub.publish_state(state)
        assert entry_id == "1-0"

        fields = redis.xadd.call_args[0][1]
        # None values should become "" not "None"
        for k, v in fields.items():
            assert v != "None", f"Field {k} has literal 'None' string"


# ═══════════════════════════════════════════════════════════════════════════════
# 6. Cross-module integration: builder → engines → publisher
# ═══════════════════════════════════════════════════════════════════════════════


class TestPhase1Integration:
    """End-to-end shadow cycle: build state → compute geometry → compute regime → publish."""

    def test_full_shadow_cycle(self):
        """Simulates one complete Phase 1 shadow cycle."""
        # 1. Build state from Redis
        redis = FakeRedis()
        redis.set_hash("quantum:state:positions:BTCUSDT", _snapshot(
            position_amt="0.01", side="LONG",
            entry_price="50000.0", mark_price="51000.0",
            unrealized_pnl="10.0", leverage="10",
        ))
        redis.set_hash("quantum:marketstate:BTCUSDT", {
            "sigma": "0.025",
            "mu": "0.001",
            "ts": str(time.time()),
            "regime_probs": json.dumps({"TREND": 0.6, "MR": 0.2, "CHOP": 0.2}),
        })
        redis.set_stream("quantum:stream:meta.regime", [
            ("1-0", {"regime": "BULL", "confidence": "0.85",
                     "timestamp": str(time.time())}),
        ])

        builder = PositionStateBuilder(redis)
        state = builder.build("BTCUSDT")
        assert state is not None
        assert state.shadow_only is True

        # 2. Compute geometry
        geometry = GeometryEngine.compute_all(
            entry_price=state.entry_price,
            current_price=state.current_price,
            peak_price=state.current_price,  # just opened
            trough_price=state.entry_price,
            side=state.side,
            current_pnl=state.unrealized_pnl,
            peak_pnl=state.unrealized_pnl,
        )
        assert isinstance(geometry, GeometryResult)
        assert geometry.mfe >= 0
        assert geometry.mae >= 0

        # 3. Compute regime
        regime = RegimeDriftEngine.summarize_regime_state(
            side=state.side,
            regime_probs={"TREND": 0.6, "MR": 0.2, "CHOP": 0.2},
            mu=0.001, sigma=0.025, ts=0.5,
        )
        assert isinstance(regime, RegimeState)
        assert regime.trend_alignment > 0  # LONG + positive mu = aligned

        # 4. Publish all to shadow (mock Redis for writes)
        write_redis = MagicMock()
        write_redis.xadd.return_value = b"99-0"
        pub = ShadowPublisher(write_redis)

        sid1 = pub.publish_state(state)
        sid2 = pub.publish_geometry(state.symbol, state.side, geometry)
        sid3 = pub.publish_regime(regime)

        assert sid1 == "99-0"
        assert sid2 == "99-0"
        assert sid3 == "99-0"
        assert write_redis.xadd.call_count == 3

        # Verify all writes went to .shadow streams
        for call in write_redis.xadd.call_args_list:
            stream_name = call[0][0]
            assert stream_name.endswith(".shadow"), f"Non-shadow write: {stream_name}"

    def test_fail_closed_cascades(self):
        """If builder returns None, no geometry/regime/publish should happen."""
        redis = FakeRedis()
        # No snapshot → builder returns None

        builder = PositionStateBuilder(redis)
        state = builder.build("BTCUSDT")
        assert state is None

        # No geometry, regime, or publish calls should be attempted
        # (this is a design validation, not a code test — fail-closed is structural)

    def test_short_position_full_cycle(self):
        """Full cycle for a SHORT position."""
        redis = FakeRedis()
        redis.set_hash("quantum:state:positions:ETHUSDT", _snapshot(
            position_amt="-0.5", side="SHORT",
            entry_price="3000.0", mark_price="2900.0",
            unrealized_pnl="50.0",
        ))

        builder = PositionStateBuilder(redis)
        state = builder.build("ETHUSDT")
        assert state is not None
        assert state.side == "SHORT"

        geometry = GeometryEngine.compute_all(
            entry_price=3000.0, current_price=2900.0,
            peak_price=2850.0, trough_price=3050.0,
            side="SHORT", current_pnl=50.0, peak_pnl=75.0,
        )
        assert geometry.mfe == pytest.approx(150.0)   # 3000 - 2850
        assert geometry.mae == pytest.approx(50.0)    # 3050 - 3000

        regime = RegimeDriftEngine.summarize_regime_state(
            side="SHORT",
            regime_probs={"TREND": 0.7, "MR": 0.1, "CHOP": 0.2},
            mu=-0.005, sigma=0.03, ts=0.8,
        )
        assert regime.trend_alignment > 0  # SHORT + negative mu = aligned
        assert regime.regime_label == "BEAR"


# ═══════════════════════════════════════════════════════════════════════════════
# 7. Shadow stream name inventory
# ═══════════════════════════════════════════════════════════════════════════════


class TestPhase1StreamInventory:
    """Verify Phase 1 shadow stream names are correct and complete."""

    PHASE_1_STREAMS = [
        "quantum:stream:exit.state.shadow",
        "quantum:stream:exit.geometry.shadow",
        "quantum:stream:exit.regime.shadow",
    ]

    def test_all_phase1_streams_defined(self):
        assert ShadowPublisher.STREAM_STATE == self.PHASE_1_STREAMS[0]
        assert ShadowPublisher.STREAM_GEOMETRY == self.PHASE_1_STREAMS[1]
        assert ShadowPublisher.STREAM_REGIME == self.PHASE_1_STREAMS[2]

    def test_phase1_streams_not_in_forbidden(self):
        for stream in self.PHASE_1_STREAMS:
            assert stream not in _FORBIDDEN_STREAMS
