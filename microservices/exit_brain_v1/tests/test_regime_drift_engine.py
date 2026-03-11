"""
Unit tests for RegimeDriftEngine.

Pure math — no mocks needed.
"""

import pytest
from microservices.exit_brain_v1.engines.regime_drift_engine import (
    RegimeDriftEngine,
    RegimeDrift,
    RegimeState,
)


class TestDetectRegimeDrift:
    """Regime drift detection tests."""

    def test_no_change_no_drift(self):
        probs = {"TREND": 0.6, "MR": 0.2, "CHOP": 0.2}
        result = RegimeDriftEngine.detect_regime_drift(probs, probs)
        assert result.drifted is False
        assert result.magnitude == pytest.approx(0.0)

    def test_large_change_triggers_drift(self):
        prev = {"TREND": 0.8, "MR": 0.1, "CHOP": 0.1}
        curr = {"TREND": 0.1, "MR": 0.1, "CHOP": 0.8}
        result = RegimeDriftEngine.detect_regime_drift(prev, curr)
        assert result.drifted is True
        assert result.magnitude > 0.2
        assert result.old_dominant == "TREND"
        assert result.new_dominant == "CHOP"
        assert result.transition == "TREND→CHOP"

    def test_small_change_no_drift(self):
        prev = {"TREND": 0.5, "MR": 0.3, "CHOP": 0.2}
        curr = {"TREND": 0.48, "MR": 0.32, "CHOP": 0.20}
        result = RegimeDriftEngine.detect_regime_drift(prev, curr, threshold=0.20)
        assert result.drifted is False

    def test_empty_probs(self):
        result = RegimeDriftEngine.detect_regime_drift({}, {})
        assert result.drifted is False
        assert result.old_dominant == "UNKNOWN"
        assert result.new_dominant == "UNKNOWN"

    def test_custom_threshold(self):
        prev = {"TREND": 0.5, "MR": 0.3, "CHOP": 0.2}
        curr = {"TREND": 0.4, "MR": 0.4, "CHOP": 0.2}
        # L1 = |0.1| + |0.1| + 0 = 0.2
        result_exact = RegimeDriftEngine.detect_regime_drift(prev, curr, threshold=0.20)
        assert result_exact.drifted is True

        result_high = RegimeDriftEngine.detect_regime_drift(prev, curr, threshold=0.25)
        assert result_high.drifted is False


class TestTrendAlignment:
    """Trend alignment tests."""

    def test_long_uptrend_positive(self):
        # LONG + positive mu → aligned → positive
        alignment = RegimeDriftEngine.compute_trend_alignment("LONG", mu=0.01, ts=0.8)
        assert alignment > 0
        assert alignment <= 1.0

    def test_long_downtrend_negative(self):
        # LONG + negative mu → counter-trend → negative
        alignment = RegimeDriftEngine.compute_trend_alignment("LONG", mu=-0.01, ts=0.8)
        assert alignment < 0

    def test_short_downtrend_positive(self):
        # SHORT + negative mu → aligned → positive
        alignment = RegimeDriftEngine.compute_trend_alignment("SHORT", mu=-0.01, ts=0.8)
        assert alignment > 0

    def test_short_uptrend_negative(self):
        # SHORT + positive mu → counter-trend → negative
        alignment = RegimeDriftEngine.compute_trend_alignment("SHORT", mu=0.01, ts=0.8)
        assert alignment < 0

    def test_flat_trend_zero(self):
        alignment = RegimeDriftEngine.compute_trend_alignment("LONG", mu=0.0, ts=0.8)
        assert alignment == 0.0

    def test_weak_trend_small_alignment(self):
        alignment = RegimeDriftEngine.compute_trend_alignment("LONG", mu=0.01, ts=0.1)
        assert 0 < alignment <= 0.1

    def test_strong_trend_capped_at_1(self):
        alignment = RegimeDriftEngine.compute_trend_alignment("LONG", mu=0.01, ts=5.0)
        assert alignment == pytest.approx(1.0)


class TestReversalRisk:
    """Reversal risk tests."""

    def test_aligned_mr_regime_high_risk(self):
        # LONG in uptrend, high MR probability → high reversal risk
        risk = RegimeDriftEngine.compute_reversal_risk(
            {"TREND": 0.1, "MR": 0.8, "CHOP": 0.1}, "LONG", mu=0.01,
        )
        assert risk > 0.5

    def test_counter_trend_high_trend_prob(self):
        # LONG in downtrend, high TREND probability → trend continues against us
        risk = RegimeDriftEngine.compute_reversal_risk(
            {"TREND": 0.8, "MR": 0.1, "CHOP": 0.1}, "LONG", mu=-0.01,
        )
        assert risk > 0.5

    def test_low_risk_scenario(self):
        # LONG in uptrend, high TREND probability → trend continues with us
        risk = RegimeDriftEngine.compute_reversal_risk(
            {"TREND": 0.8, "MR": 0.1, "CHOP": 0.1}, "LONG", mu=0.01,
        )
        assert risk < 0.3

    def test_clamped_0_to_1(self):
        risk = RegimeDriftEngine.compute_reversal_risk(
            {"TREND": 0.0, "MR": 1.0, "CHOP": 1.0}, "LONG", mu=0.01,
        )
        assert 0.0 <= risk <= 1.0


class TestChopRisk:
    """Chop risk tests."""

    def test_high_chop_prob(self):
        risk = RegimeDriftEngine.compute_chop_risk(
            {"TREND": 0.1, "MR": 0.1, "CHOP": 0.8},
            sigma=0.02, ts=0.3,
        )
        assert risk >= 0.8

    def test_low_chop_prob(self):
        risk = RegimeDriftEngine.compute_chop_risk(
            {"TREND": 0.8, "MR": 0.1, "CHOP": 0.1},
            sigma=0.01, ts=2.0,
        )
        assert risk <= 0.2

    def test_vol_boost_applied(self):
        # High sigma + low ts → vol_boost adds 0.2
        base_risk = RegimeDriftEngine.compute_chop_risk(
            {"TREND": 0.1, "MR": 0.1, "CHOP": 0.3},
            sigma=0.01, ts=2.0,  # No boost
        )
        boosted_risk = RegimeDriftEngine.compute_chop_risk(
            {"TREND": 0.1, "MR": 0.1, "CHOP": 0.3},
            sigma=0.03, ts=0.3,  # Boost
        )
        assert boosted_risk > base_risk

    def test_clamped_to_1(self):
        risk = RegimeDriftEngine.compute_chop_risk(
            {"TREND": 0.0, "MR": 0.0, "CHOP": 1.0},
            sigma=0.05, ts=0.1,
        )
        assert risk <= 1.0


class TestSummarizeRegimeState:
    """Aggregate summarize_regime_state tests."""

    def test_returns_regime_state(self):
        result = RegimeDriftEngine.summarize_regime_state(
            side="LONG",
            regime_probs={"TREND": 0.6, "MR": 0.2, "CHOP": 0.2},
            mu=0.01, sigma=0.02, ts=0.5,
        )
        assert isinstance(result, RegimeState)
        assert result.regime_label in ("BULL", "BEAR", "RANGE", "VOLATILE", "TREND", "MR", "CHOP")
        assert 0.0 <= result.regime_confidence <= 1.0
        assert -1.0 <= result.trend_alignment <= 1.0
        assert 0.0 <= result.reversal_risk <= 1.0
        assert 0.0 <= result.chop_risk <= 1.0
        assert result.drift is None  # No prev probs supplied

    def test_with_drift_detection(self):
        prev = {"TREND": 0.8, "MR": 0.1, "CHOP": 0.1}
        curr = {"TREND": 0.1, "MR": 0.1, "CHOP": 0.8}
        result = RegimeDriftEngine.summarize_regime_state(
            side="LONG", regime_probs=curr,
            mu=0.001, sigma=0.02, ts=0.05,
            prev_regime_probs=prev,
        )
        assert result.drift is not None
        assert result.drift.drifted is True

    def test_uptrend_label_bull(self):
        result = RegimeDriftEngine.summarize_regime_state(
            side="LONG",
            regime_probs={"TREND": 0.7, "MR": 0.2, "CHOP": 0.1},
            mu=0.01, sigma=0.02, ts=0.5,
        )
        assert result.regime_label == "BULL"

    def test_downtrend_label_bear(self):
        result = RegimeDriftEngine.summarize_regime_state(
            side="SHORT",
            regime_probs={"TREND": 0.7, "MR": 0.2, "CHOP": 0.1},
            mu=-0.01, sigma=0.02, ts=0.5,
        )
        assert result.regime_label == "BEAR"

    def test_empty_probs_unknown(self):
        result = RegimeDriftEngine.summarize_regime_state(
            side="LONG", regime_probs={},
            mu=0.0, sigma=0.01, ts=0.0,
        )
        assert result.regime_label == "UNKNOWN"
        assert result.regime_confidence == 0.0
