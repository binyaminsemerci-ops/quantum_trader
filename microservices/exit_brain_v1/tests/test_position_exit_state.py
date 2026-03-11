"""
Contract tests for PositionExitState.

Validates:
- Required field enforcement
- Default values
- Validation rules
- Fail-closed behavior
- shadow_only enforcement
- Serialization round-trip
"""

import time
import pytest
from microservices.exit_brain_v1.models.position_exit_state import (
    PositionExitState,
    VALID_REGIMES,
    VALID_SIDES,
    VALID_STATUSES,
)


def _make_valid_state(**overrides) -> PositionExitState:
    """Factory for a valid PositionExitState with reasonable defaults."""
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


class TestRequiredFields:
    """Required fields must cause validation errors when invalid."""

    def test_empty_position_id_fails(self):
        state = _make_valid_state(position_id="")
        errors = state.validate()
        assert any("position_id" in e for e in errors)

    def test_empty_symbol_fails(self):
        state = _make_valid_state(symbol="")
        errors = state.validate()
        assert any("symbol" in e for e in errors)

    def test_invalid_side_fails(self):
        state = _make_valid_state(side="UP")
        errors = state.validate()
        assert any("side" in e for e in errors)

    def test_invalid_status_fails(self):
        state = _make_valid_state(status="DEAD")
        errors = state.validate()
        assert any("status" in e for e in errors)

    def test_zero_entry_price_fails(self):
        state = _make_valid_state(entry_price=0.0)
        errors = state.validate()
        assert any("entry_price" in e for e in errors)

    def test_negative_quantity_fails(self):
        state = _make_valid_state(quantity=-1.0)
        errors = state.validate()
        assert any("quantity" in e for e in errors)

    def test_zero_notional_fails(self):
        state = _make_valid_state(notional=0.0)
        errors = state.validate()
        assert any("notional" in e for e in errors)

    def test_zero_open_timestamp_fails(self):
        state = _make_valid_state(open_timestamp=0.0)
        errors = state.validate()
        assert any("open_timestamp" in e for e in errors)


class TestDefaults:
    """Optional fields have correct defaults."""

    def test_mark_price_default(self):
        state = _make_valid_state()
        assert state.mark_price == 0.0

    def test_leverage_default(self):
        state = _make_valid_state()
        assert state.leverage == 1.0

    def test_regime_label_default(self):
        state = _make_valid_state()
        assert state.regime_label == "UNKNOWN"

    def test_regime_confidence_default(self):
        state = _make_valid_state()
        assert state.regime_confidence == 0.0

    def test_shadow_only_default(self):
        state = _make_valid_state()
        assert state.shadow_only is True

    def test_optional_scores_default_none(self):
        state = _make_valid_state()
        assert state.momentum_score is None
        assert state.mean_reversion_score is None
        assert state.liquidity_score is None
        assert state.spread_bps is None


class TestShadowEnforcement:
    """shadow_only must be True in Phase 1."""

    def test_shadow_false_fails_validation(self):
        state = _make_valid_state(shadow_only=False)
        errors = state.validate()
        assert any("shadow_only" in e for e in errors)

    def test_shadow_true_passes(self):
        state = _make_valid_state(shadow_only=True)
        errors = state.validate()
        assert len(errors) == 0


class TestValidationRules:
    """Edge case validation."""

    def test_invalid_regime_label_corrected(self):
        state = _make_valid_state(regime_label="GARBAGE")
        state.validate()
        assert state.regime_label == "UNKNOWN"
        assert any("INVALID_REGIME" in f for f in state.data_quality_flags)

    def test_regime_confidence_clamped(self):
        state = _make_valid_state(regime_confidence=1.5)
        state.validate()
        assert state.regime_confidence == 1.0

    def test_leverage_below_1_flagged(self):
        state = _make_valid_state(leverage=0.5)
        state.validate()
        assert "LEVERAGE_BELOW_1" in state.data_quality_flags

    def test_valid_state_no_errors(self):
        state = _make_valid_state()
        errors = state.validate()
        assert errors == []


class TestComputedProperties:
    """Computed properties work correctly."""

    def test_hold_seconds_positive(self):
        state = _make_valid_state(open_timestamp=time.time() - 60)
        assert 59 <= state.hold_seconds <= 62  # Allow small timing variance

    def test_feature_freshness_seconds(self):
        old_ts = time.time() - 100
        state = _make_valid_state(source_timestamps={"snap": old_ts})
        assert state.feature_freshness_seconds >= 99

    def test_feature_freshness_no_sources(self):
        state = _make_valid_state(source_timestamps={})
        assert state.feature_freshness_seconds == float("inf")


class TestSerialization:
    """to_dict round-trip."""

    def test_to_dict_contains_all_required_keys(self):
        state = _make_valid_state()
        d = state.to_dict()
        required_keys = {
            "position_id", "symbol", "side", "status",
            "entry_price", "current_price", "quantity", "notional",
            "unrealized_pnl", "unrealized_pnl_pct",
            "open_timestamp", "source_timestamps",
            "data_quality_flags", "shadow_only",
        }
        for key in required_keys:
            assert key in d, f"Missing key: {key}"

    def test_to_dict_values_match(self):
        state = _make_valid_state(entry_price=42000.0)
        d = state.to_dict()
        assert d["entry_price"] == 42000.0
        assert d["shadow_only"] is True

    def test_to_dict_optional_none_preserved(self):
        state = _make_valid_state()
        d = state.to_dict()
        assert d["momentum_score"] is None
        assert d["atr"] is None
