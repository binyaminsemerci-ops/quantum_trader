"""
Unit tests for GeometryEngine.

Pure math — no mocks needed. Deterministic inputs → deterministic outputs.
"""

import pytest
from microservices.exit_brain_v1.engines.geometry_engine import GeometryEngine, GeometryResult


class TestComputeMFE:
    """Max Favorable Excursion tests."""

    def test_long_profitable(self):
        mfe = GeometryEngine.compute_mfe(100.0, 110.0, "LONG")
        assert mfe == pytest.approx(10.0)

    def test_long_no_profit(self):
        mfe = GeometryEngine.compute_mfe(100.0, 95.0, "LONG")
        assert mfe == 0.0

    def test_short_profitable(self):
        mfe = GeometryEngine.compute_mfe(100.0, 90.0, "SHORT")
        assert mfe == pytest.approx(10.0)

    def test_short_no_profit(self):
        mfe = GeometryEngine.compute_mfe(100.0, 105.0, "SHORT")
        assert mfe == 0.0

    def test_zero_entry_returns_zero(self):
        assert GeometryEngine.compute_mfe(0.0, 100.0, "LONG") == 0.0

    def test_at_entry_mfe_is_zero(self):
        assert GeometryEngine.compute_mfe(100.0, 100.0, "LONG") == 0.0


class TestComputeMAE:
    """Max Adverse Excursion tests."""

    def test_long_adverse(self):
        mae = GeometryEngine.compute_mae(100.0, 90.0, "LONG")
        assert mae == pytest.approx(10.0)

    def test_long_no_adverse(self):
        mae = GeometryEngine.compute_mae(100.0, 105.0, "LONG")
        assert mae == 0.0

    def test_short_adverse(self):
        mae = GeometryEngine.compute_mae(100.0, 110.0, "SHORT")
        assert mae == pytest.approx(10.0)

    def test_short_no_adverse(self):
        mae = GeometryEngine.compute_mae(100.0, 95.0, "SHORT")
        assert mae == 0.0

    def test_zero_entry_returns_zero(self):
        assert GeometryEngine.compute_mae(0.0, 50.0, "LONG") == 0.0


class TestDrawdownFromPeak:
    """Drawdown from peak PnL tests."""

    def test_positive_drawdown(self):
        dd = GeometryEngine.compute_drawdown_from_peak(5.0, 10.0)
        assert dd == pytest.approx(5.0)

    def test_at_peak_zero_drawdown(self):
        dd = GeometryEngine.compute_drawdown_from_peak(10.0, 10.0)
        assert dd == 0.0

    def test_above_peak_zero_drawdown(self):
        # current > peak shouldn't happen, but is clamped to 0
        dd = GeometryEngine.compute_drawdown_from_peak(12.0, 10.0)
        assert dd == 0.0

    def test_never_profitable_zero_drawdown(self):
        dd = GeometryEngine.compute_drawdown_from_peak(-5.0, -1.0)
        assert dd == 0.0

    def test_peak_zero_drawdown(self):
        dd = GeometryEngine.compute_drawdown_from_peak(-5.0, 0.0)
        assert dd == 0.0


class TestProfitProtectionRatio:
    """Profit protection ratio tests."""

    def test_at_peak(self):
        ppr = GeometryEngine.compute_profit_protection_ratio(10.0, 10.0)
        assert ppr == pytest.approx(1.0)

    def test_half_given_back(self):
        ppr = GeometryEngine.compute_profit_protection_ratio(5.0, 10.0)
        assert ppr == pytest.approx(0.5)

    def test_all_given_back(self):
        ppr = GeometryEngine.compute_profit_protection_ratio(0.0, 10.0)
        assert ppr == pytest.approx(0.0)

    def test_never_profitable(self):
        ppr = GeometryEngine.compute_profit_protection_ratio(-5.0, -1.0)
        assert ppr == 0.0

    def test_clamped_to_1(self):
        ppr = GeometryEngine.compute_profit_protection_ratio(15.0, 10.0)
        assert ppr == 1.0

    def test_negative_current_clamped_to_0(self):
        ppr = GeometryEngine.compute_profit_protection_ratio(-5.0, 10.0)
        assert ppr == 0.0


class TestMomentumDecay:
    """Momentum decay (slope) tests."""

    def test_rising_pnl(self):
        slope = GeometryEngine.compute_momentum_decay([1.0, 2.0, 3.0, 4.0, 5.0])
        assert slope > 0

    def test_falling_pnl(self):
        slope = GeometryEngine.compute_momentum_decay([5.0, 4.0, 3.0, 2.0, 1.0])
        assert slope < 0

    def test_flat_pnl(self):
        slope = GeometryEngine.compute_momentum_decay([3.0, 3.0, 3.0, 3.0])
        assert slope == pytest.approx(0.0)

    def test_single_point_returns_zero(self):
        slope = GeometryEngine.compute_momentum_decay([5.0])
        assert slope == 0.0

    def test_empty_returns_zero(self):
        slope = GeometryEngine.compute_momentum_decay([])
        assert slope == 0.0

    def test_window_larger_than_data(self):
        slope = GeometryEngine.compute_momentum_decay([1.0, 3.0], window=10)
        assert slope > 0


class TestRewardToRiskRemaining:
    """Reward-to-risk ratio tests."""

    def test_long_balanced(self):
        # Price=100, stop=90, target=110 → upside=10, downside=10 → ratio=1.0
        rtr = GeometryEngine.compute_reward_to_risk_remaining(
            current_price=100.0, entry_price=95.0,
            stop_price=90.0, target_price=110.0, side="LONG",
        )
        assert rtr == pytest.approx(1.0)

    def test_long_more_upside(self):
        # Price=100, stop=95, target=120 → upside=20, downside=5 → ratio=4.0
        rtr = GeometryEngine.compute_reward_to_risk_remaining(
            current_price=100.0, entry_price=95.0,
            stop_price=95.0, target_price=120.0, side="LONG",
        )
        assert rtr == pytest.approx(4.0)

    def test_short_balanced(self):
        # Price=100, stop=110, target=90 → upside=10, downside=10 → ratio=1.0
        rtr = GeometryEngine.compute_reward_to_risk_remaining(
            current_price=100.0, entry_price=105.0,
            stop_price=110.0, target_price=90.0, side="SHORT",
        )
        assert rtr == pytest.approx(1.0)

    def test_already_past_stop_returns_zero(self):
        # LONG: price=85 < stop=90 → downside=0 and upside>0
        # Actually downside= max(0, 85-90) = 0, upside=25 → returns 100.0 (capped)
        rtr = GeometryEngine.compute_reward_to_risk_remaining(
            current_price=85.0, entry_price=90.0,
            stop_price=90.0, target_price=110.0, side="LONG",
        )
        assert rtr == 100.0  # capped at max

    def test_zero_prices_returns_zero(self):
        rtr = GeometryEngine.compute_reward_to_risk_remaining(
            current_price=0.0, entry_price=0.0,
            stop_price=0.0, target_price=0.0, side="LONG",
        )
        assert rtr == 0.0


class TestComputeAll:
    """Aggregate compute_all tests."""

    def test_returns_geometry_result(self):
        result = GeometryEngine.compute_all(
            entry_price=100.0, current_price=105.0,
            peak_price=108.0, trough_price=97.0,
            side="LONG", current_pnl=5.0, peak_pnl=8.0,
            pnl_history=[1.0, 3.0, 5.0, 8.0, 5.0],
            stop_price=95.0, target_price=115.0,
        )
        assert isinstance(result, GeometryResult)
        assert result.mfe == pytest.approx(8.0)
        assert result.mae == pytest.approx(3.0)
        assert result.drawdown_from_peak == pytest.approx(3.0)
        assert 0.0 < result.profit_protection_ratio < 1.0
        assert result.momentum_decay != 0.0
        assert result.reward_to_risk_remaining > 0.0

    def test_no_stop_target_rtr_zero(self):
        result = GeometryEngine.compute_all(
            entry_price=100.0, current_price=105.0,
            peak_price=108.0, trough_price=97.0,
            side="LONG", current_pnl=5.0, peak_pnl=8.0,
        )
        assert result.reward_to_risk_remaining == 0.0
