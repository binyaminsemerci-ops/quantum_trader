#!/usr/bin/env python3
"""
Test suite for P1 Risk Kernel: Stops/Trailing Proposal Engine

Validates:
- LONG/SHORT direction correctness
- Monotonic SL tightening (never loosens)
- Peak/trough trailing logic
- Regime-weighted stop distances
- TP extension on strong trends
- Edge cases (zero sigma, missing fields)
"""

import pytest
from ai_engine.risk_kernel_stops import (
    compute_proposal,
    PositionSnapshot,
    DEFAULT_THETA_RISK,
)


def test_output_contract():
    """Proposal dict has required keys"""
    market_state = {
        "sigma": 0.01,
        "mu": 0.001,
        "ts": 0.5,
        "regime_probs": {"trend": 0.6, "mr": 0.2, "chop": 0.2}
    }
    position = PositionSnapshot(
        symbol="BTCUSDT",
        side="LONG",
        entry_price=100.0,
        current_price=105.0,
        peak_price=106.0,
        trough_price=99.0,
        age_sec=300.0,
        unrealized_pnl=500.0,
    )
    
    proposal = compute_proposal("BTCUSDT", market_state, position)
    
    assert "proposed_sl" in proposal
    assert "proposed_tp" in proposal
    assert "reason_codes" in proposal
    assert "audit" in proposal
    assert "meta" in proposal
    assert isinstance(proposal["reason_codes"], list)
    assert proposal["meta"]["symbol"] == "BTCUSDT"
    assert proposal["meta"]["side"] == "LONG"


def test_long_sl_below_tp_above():
    """LONG: SL must be below current price, TP above"""
    market_state = {
        "sigma": 0.01,
        "mu": 0.001,
        "ts": 0.3,
        "regime_probs": {"trend": 0.5, "mr": 0.3, "chop": 0.2}
    }
    position = PositionSnapshot(
        symbol="BTCUSDT",
        side="LONG",
        entry_price=100.0,
        current_price=105.0,
        peak_price=106.0,
        trough_price=99.0,
        age_sec=300.0,
        unrealized_pnl=500.0,
    )
    
    proposal = compute_proposal("BTCUSDT", market_state, position)
    
    assert proposal["proposed_sl"] < position.current_price, "LONG SL must be below current"
    assert proposal["proposed_tp"] > position.current_price, "LONG TP must be above current"


def test_short_sl_above_tp_below():
    """SHORT: SL must be above current price, TP below"""
    market_state = {
        "sigma": 0.01,
        "mu": -0.001,
        "ts": 0.3,
        "regime_probs": {"trend": 0.5, "mr": 0.3, "chop": 0.2}
    }
    position = PositionSnapshot(
        symbol="BTCUSDT",
        side="SHORT",
        entry_price=100.0,
        current_price=95.0,
        peak_price=101.0,  # peak for SHORT is highest
        trough_price=94.0,  # trough for SHORT is lowest
        age_sec=300.0,
        unrealized_pnl=500.0,
    )
    
    proposal = compute_proposal("BTCUSDT", market_state, position)
    
    assert proposal["proposed_sl"] > position.current_price, "SHORT SL must be above current"
    assert proposal["proposed_tp"] < position.current_price, "SHORT TP must be below current"


def test_monotonic_sl_long_never_loosens():
    """LONG: Monotonic SL enforcement - never loosens"""
    market_state = {
        "sigma": 0.005,  # low volatility → tighter stops
        "mu": 0.001,
        "ts": 0.2,
        "regime_probs": {"trend": 0.3, "mr": 0.5, "chop": 0.2}
    }
    
    # First proposal without existing SL
    position1 = PositionSnapshot(
        symbol="BTCUSDT",
        side="LONG",
        entry_price=100.0,
        current_price=110.0,
        peak_price=112.0,
        trough_price=99.0,
        age_sec=300.0,
        unrealized_pnl=1000.0,
        current_sl=None,
    )
    proposal1 = compute_proposal("BTCUSDT", market_state, position1)
    first_sl = proposal1["proposed_sl"]
    
    # Second proposal with existing SL, price moved up but sigma dropped
    # Should NOT loosen SL (must stay at or tighten from first_sl)
    position2 = PositionSnapshot(
        symbol="BTCUSDT",
        side="LONG",
        entry_price=100.0,
        current_price=108.0,  # price dropped slightly
        peak_price=112.0,
        trough_price=99.0,
        age_sec=400.0,
        unrealized_pnl=800.0,
        current_sl=first_sl,
    )
    proposal2 = compute_proposal("BTCUSDT", market_state, position2)
    
    assert proposal2["proposed_sl"] >= first_sl - 1e-6, "LONG SL must not loosen"


def test_monotonic_sl_short_never_loosens():
    """SHORT: Monotonic SL enforcement - never loosens"""
    market_state = {
        "sigma": 0.005,
        "mu": -0.001,
        "ts": 0.2,
        "regime_probs": {"trend": 0.3, "mr": 0.5, "chop": 0.2}
    }
    
    # First proposal
    position1 = PositionSnapshot(
        symbol="BTCUSDT",
        side="SHORT",
        entry_price=100.0,
        current_price=90.0,
        peak_price=101.0,
        trough_price=88.0,
        age_sec=300.0,
        unrealized_pnl=1000.0,
        current_sl=None,
    )
    proposal1 = compute_proposal("BTCUSDT", market_state, position1)
    first_sl = proposal1["proposed_sl"]
    
    # Second proposal with price moved higher
    position2 = PositionSnapshot(
        symbol="BTCUSDT",
        side="SHORT",
        entry_price=100.0,
        current_price=92.0,
        peak_price=101.0,
        trough_price=88.0,
        age_sec=400.0,
        unrealized_pnl=800.0,
        current_sl=first_sl,
    )
    proposal2 = compute_proposal("BTCUSDT", market_state, position2)
    
    assert proposal2["proposed_sl"] <= first_sl + 1e-6, "SHORT SL must not loosen (must decrease or stay)"


def test_trailing_activates_long():
    """LONG: Trailing SL activates when peak creates tighter stop"""
    market_state = {
        "sigma": 0.01,
        "mu": 0.002,
        "ts": 0.4,
        "regime_probs": {"trend": 0.7, "mr": 0.1, "chop": 0.2}
    }
    
    # Price at peak, trailing should be active
    position = PositionSnapshot(
        symbol="BTCUSDT",
        side="LONG",
        entry_price=100.0,
        current_price=115.0,  # current = peak
        peak_price=115.0,
        trough_price=99.0,
        age_sec=600.0,
        unrealized_pnl=1500.0,
    )
    
    proposal = compute_proposal("BTCUSDT", market_state, position)
    
    # Trail SL should be close to peak
    trail_gap_pct = proposal["audit"]["intermediates"]["trail_gap_pct"]
    expected_trail_sl = position.peak_price * (1.0 - trail_gap_pct)
    
    assert abs(proposal["proposed_sl"] - expected_trail_sl) < 0.1, "Trailing should dominate"
    assert "trail_active" in proposal["reason_codes"] or proposal["proposed_sl"] > position.current_price * 0.98


def test_trailing_activates_short():
    """SHORT: Trailing SL activates based on trough"""
    market_state = {
        "sigma": 0.01,
        "mu": -0.002,
        "ts": 0.4,
        "regime_probs": {"trend": 0.7, "mr": 0.1, "chop": 0.2}
    }
    
    # Price at trough, trailing should be active
    position = PositionSnapshot(
        symbol="BTCUSDT",
        side="SHORT",
        entry_price=100.0,
        current_price=85.0,  # current = trough
        peak_price=101.0,
        trough_price=85.0,
        age_sec=600.0,
        unrealized_pnl=1500.0,
    )
    
    proposal = compute_proposal("BTCUSDT", market_state, position)
    
    # Trail SL should be close to trough
    trail_gap_pct = proposal["audit"]["intermediates"]["trail_gap_pct"]
    expected_trail_sl = position.trough_price * (1.0 + trail_gap_pct)
    
    assert abs(proposal["proposed_sl"] - expected_trail_sl) < 0.1, "Trailing should dominate"


def test_regime_weighted_stops():
    """Stop distances vary by regime weights"""
    position = PositionSnapshot(
        symbol="BTCUSDT",
        side="LONG",
        entry_price=100.0,
        current_price=105.0,
        peak_price=106.0,
        trough_price=99.0,
        age_sec=300.0,
        unrealized_pnl=500.0,
    )
    
    # High trend regime → wider stops (k_sl trend = 1.5)
    ms_trend = {
        "sigma": 0.01,
        "mu": 0.002,
        "ts": 0.6,
        "regime_probs": {"trend": 0.8, "mr": 0.1, "chop": 0.1}
    }
    proposal_trend = compute_proposal("BTCUSDT", ms_trend, position)
    
    # High MR regime → tighter stops (k_sl mr = 0.8)
    ms_mr = {
        "sigma": 0.01,
        "mu": 0.0,
        "ts": 0.1,
        "regime_probs": {"trend": 0.1, "mr": 0.8, "chop": 0.1}
    }
    proposal_mr = compute_proposal("BTCUSDT", ms_mr, position)
    
    # Trend regime should have wider stop distance
    dist_trend = proposal_trend["audit"]["intermediates"]["stop_dist_pct"]
    dist_mr = proposal_mr["audit"]["intermediates"]["stop_dist_pct"]
    
    assert dist_trend > dist_mr, "Trend regime should have wider stops than MR"


def test_tp_extension_on_strong_trend():
    """TP extends when trend prob high and TS strong"""
    position = PositionSnapshot(
        symbol="BTCUSDT",
        side="LONG",
        entry_price=100.0,
        current_price=105.0,
        peak_price=106.0,
        trough_price=99.0,
        age_sec=300.0,
        unrealized_pnl=500.0,
    )
    
    # Weak trend
    ms_weak = {
        "sigma": 0.01,
        "mu": 0.001,
        "ts": 0.05,
        "regime_probs": {"trend": 0.2, "mr": 0.5, "chop": 0.3}
    }
    proposal_weak = compute_proposal("BTCUSDT", ms_weak, position)
    
    # Strong trend
    ms_strong = {
        "sigma": 0.01,
        "mu": 0.003,
        "ts": 0.7,
        "regime_probs": {"trend": 0.8, "mr": 0.1, "chop": 0.1}
    }
    proposal_strong = compute_proposal("BTCUSDT", ms_strong, position)
    
    ext_weak = proposal_weak["audit"]["intermediates"]["tp_extension_factor"]
    ext_strong = proposal_strong["audit"]["intermediates"]["tp_extension_factor"]
    
    assert ext_strong > ext_weak, "Strong trend should have larger TP extension"
    assert "tp_extended" in proposal_strong["reason_codes"]


def test_zero_sigma_uses_min_pct():
    """Zero volatility falls back to minimum pct floors"""
    market_state = {
        "sigma": 0.0,
        "mu": 0.0,
        "ts": 0.0,
        "regime_probs": {"trend": 0.3, "mr": 0.4, "chop": 0.3}
    }
    position = PositionSnapshot(
        symbol="BTCUSDT",
        side="LONG",
        entry_price=100.0,
        current_price=105.0,
        peak_price=106.0,
        trough_price=99.0,
        age_sec=300.0,
        unrealized_pnl=500.0,
    )
    
    proposal = compute_proposal("BTCUSDT", market_state, position)
    
    # Should use sl_min_pct
    stop_dist = proposal["audit"]["intermediates"]["stop_dist_pct"]
    assert stop_dist == DEFAULT_THETA_RISK["sl_min_pct"], "Zero sigma should use min floor"


def test_custom_theta_override():
    """Custom theta overrides defaults"""
    market_state = {
        "sigma": 0.01,
        "mu": 0.001,
        "ts": 0.3,
        "regime_probs": {"trend": 0.5, "mr": 0.3, "chop": 0.2}
    }
    position = PositionSnapshot(
        symbol="BTCUSDT",
        side="LONG",
        entry_price=100.0,
        current_price=105.0,
        peak_price=106.0,
        trough_price=99.0,
        age_sec=300.0,
        unrealized_pnl=500.0,
    )
    
    custom_theta = {
        **DEFAULT_THETA_RISK,
        "sl_min_pct": 0.02,  # custom 2% min
        "k_sl": {"trend": 3.0, "mr": 3.0, "chop": 3.0},  # aggressive stops
    }
    
    proposal = compute_proposal("BTCUSDT", market_state, position, theta=custom_theta)
    
    # Should reflect custom theta
    stop_dist = proposal["audit"]["intermediates"]["stop_dist_pct"]
    assert stop_dist >= 0.02, "Should respect custom sl_min_pct"


def test_symbol_mismatch_raises():
    """Symbol mismatch between args and position raises error"""
    market_state = {
        "sigma": 0.01,
        "mu": 0.001,
        "ts": 0.3,
        "regime_probs": {"trend": 0.5, "mr": 0.3, "chop": 0.2}
    }
    position = PositionSnapshot(
        symbol="ETHUSDT",
        side="LONG",
        entry_price=100.0,
        current_price=105.0,
        peak_price=106.0,
        trough_price=99.0,
        age_sec=300.0,
        unrealized_pnl=500.0,
    )
    
    with pytest.raises(ValueError, match="Symbol mismatch"):
        compute_proposal("BTCUSDT", market_state, position)


def test_invalid_side_raises():
    """Invalid side raises error"""
    market_state = {
        "sigma": 0.01,
        "mu": 0.001,
        "ts": 0.3,
        "regime_probs": {"trend": 0.5, "mr": 0.3, "chop": 0.2}
    }
    position = PositionSnapshot(
        symbol="BTCUSDT",
        side="INVALID",
        entry_price=100.0,
        current_price=105.0,
        peak_price=106.0,
        trough_price=99.0,
        age_sec=300.0,
        unrealized_pnl=500.0,
    )
    
    with pytest.raises(ValueError, match="Invalid side"):
        compute_proposal("BTCUSDT", market_state, position)
