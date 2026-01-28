"""
Test suite for P2 Harvest Kernel (calc-only)

Validates:
- R_net triggers correct harvest actions
- Profit lock SL monotonic tightening
- Kill score triggers full close proposal
- Fallback behavior with missing inputs
- NO trading side-effects
"""

import pytest
import math
from ai_engine.risk_kernel_harvest import (
    compute_harvest_proposal,
    compute_tranche_weights,
    compute_risk_unit,
    compute_R_net,
    determine_harvest_action,
    compute_profit_lock_sl,
    compute_kill_score,
    HarvestTheta,
    PositionSnapshot,
    MarketState,
    P1Proposal,
)


# ============================================================================
# FIXTURES
# ============================================================================

@pytest.fixture
def theta():
    """Default harvest theta"""
    return HarvestTheta(
        fallback_stop_pct=0.02,
        cost_bps=10.0,
        T1_R=2.0,
        T2_R=4.0,
        T3_R=6.0,
        lock_R=1.5,
        be_plus_pct=0.002,
        trend_min=0.3,
        sigma_ref=0.01,
        ts_ref=0.3,
        max_age_sec=86400.0,
        kill_threshold=0.6,
    )


@pytest.fixture
def base_position():
    """Base LONG position with profit"""
    return PositionSnapshot(
        symbol="BTCUSDT",
        side="LONG",
        entry_price=100.0,
        current_price=105.0,
        peak_price=106.0,
        trough_price=99.0,
        age_sec=3600.0,
        unrealized_pnl=5.0,
        current_sl=99.0,
        current_tp=110.0,
    )


@pytest.fixture
def base_market():
    """Trending market state"""
    return MarketState(
        sigma=0.01,
        ts=0.35,
        p_trend=0.5,
        p_mr=0.2,
        p_chop=0.3,
    )


@pytest.fixture
def p1_proposal():
    """P1 proposal with stop_dist_pct"""
    return P1Proposal(stop_dist_pct=0.02, proposed_sl=99.0)


# ============================================================================
# TEST: R_NET TRIGGERS
# ============================================================================

def test_R_net_trigger_none(base_position, base_market, p1_proposal, theta):
    """R_net < T1 → NONE"""
    # Set unrealized_pnl to give R_net < 2.0
    pos = base_position
    pos.unrealized_pnl = 1.0  # Low profit
    
    result = compute_harvest_proposal(pos, base_market, p1_proposal, theta)
    
    assert result["harvest_action"] == "NONE"
    assert result["R_net"] < theta.T1_R


def test_R_net_trigger_partial_25(base_position, base_market, p1_proposal, theta):
    """T1 <= R_net < T2 → PARTIAL_25"""
    pos = base_position
    # risk_unit = 100 * 0.02 = 2.0
    # cost_est = 100 * 0.001 = 0.1
    # R_net = (unrealized_pnl - 0.1) / 2.0
    # For R_net = 2.5: unrealized_pnl = 2.5*2 + 0.1 = 5.1
    pos.unrealized_pnl = 5.1
    
    result = compute_harvest_proposal(pos, base_market, p1_proposal, theta)
    
    assert result["harvest_action"] == "PARTIAL_25"
    assert theta.T1_R <= result["R_net"] < theta.T2_R


def test_R_net_trigger_partial_50(base_position, base_market, p1_proposal, theta):
    """T2 <= R_net < T3 → PARTIAL_50"""
    pos = base_position
    # For R_net = 4.5: unrealized_pnl = 4.5*2 + 0.1 = 9.1
    pos.unrealized_pnl = 9.1
    
    result = compute_harvest_proposal(pos, base_market, p1_proposal, theta)
    
    assert result["harvest_action"] == "PARTIAL_50"
    assert theta.T2_R <= result["R_net"] < theta.T3_R


def test_R_net_trigger_partial_75(base_position, base_market, p1_proposal, theta):
    """R_net >= T3 → PARTIAL_75"""
    pos = base_position
    # For R_net = 7.0: unrealized_pnl = 7*2 + 0.1 = 14.1
    pos.unrealized_pnl = 14.1
    
    result = compute_harvest_proposal(pos, base_market, p1_proposal, theta)
    
    assert result["harvest_action"] == "PARTIAL_75"
    assert result["R_net"] >= theta.T3_R


# ============================================================================
# TEST: PROFIT LOCK SL (MONOTONIC TIGHTENING)
# ============================================================================

def test_profit_lock_long_tightening(base_position, base_market, p1_proposal, theta):
    """LONG: new_sl > current_sl (tightening)"""
    pos = base_position
    pos.unrealized_pnl = 4.0  # R_net > lock_R
    pos.current_sl = 99.0
    
    result = compute_harvest_proposal(pos, base_market, p1_proposal, theta)
    
    # BE+ = 100 * (1 + 0.002) = 100.2
    assert result["new_sl_proposed"] is not None
    assert result["new_sl_proposed"] >= 100.2
    assert result["new_sl_proposed"] > pos.current_sl
    assert "profit_lock" in result["reason_codes"]


def test_profit_lock_long_no_loosen(base_position, base_market, p1_proposal, theta):
    """LONG: new_sl <= current_sl → None (no proposal)"""
    pos = base_position
    pos.unrealized_pnl = 4.0
    pos.current_sl = 101.0  # Already above BE+
    
    result = compute_harvest_proposal(pos, base_market, p1_proposal, theta)
    
    # new_sl would be max(101, 100.2) = 101, same as current → None
    assert result["new_sl_proposed"] is None


def test_profit_lock_short_tightening(base_position, base_market, p1_proposal, theta):
    """SHORT: new_sl < current_sl (tightening)"""
    pos = base_position
    pos.side = "SHORT"
    pos.entry_price = 100.0
    pos.current_price = 95.0
    pos.unrealized_pnl = 5.0
    pos.current_sl = 101.0  # Above entry
    
    result = compute_harvest_proposal(pos, base_market, p1_proposal, theta)
    
    # BE+ = 100 * (1 - 0.002) = 99.8
    assert result["new_sl_proposed"] is not None
    assert result["new_sl_proposed"] <= 99.8
    assert result["new_sl_proposed"] < pos.current_sl
    assert "profit_lock" in result["reason_codes"]


def test_profit_lock_short_no_loosen(base_position, base_market, p1_proposal, theta):
    """SHORT: new_sl >= current_sl → None (no proposal)"""
    pos = base_position
    pos.side = "SHORT"
    pos.entry_price = 100.0
    pos.current_price = 95.0
    pos.unrealized_pnl = 5.0
    pos.current_sl = 99.0  # Already below BE+
    
    result = compute_harvest_proposal(pos, base_market, p1_proposal, theta)
    
    # new_sl would be min(99, 99.8) = 99, same as current → None
    assert result["new_sl_proposed"] is None


# ============================================================================
# TEST: KILL SCORE
# ============================================================================

def test_kill_score_regime_flip(base_position, p1_proposal, theta):
    """Regime flip: p_trend < trend_min and p_chop+p_mr > 0.5 → high K"""
    market = MarketState(
        sigma=0.01,
        ts=0.35,
        p_trend=0.1,  # Below trend_min=0.3
        p_mr=0.3,
        p_chop=0.6,  # High chop
    )
    
    result = compute_harvest_proposal(base_position, market, p1_proposal, theta)
    
    assert result["kill_score"] > 0.5  # Should be elevated
    assert result["audit"]["k_components"]["regime_flip"] == 1.0


def test_kill_score_sigma_spike(base_position, p1_proposal, theta):
    """Sigma spike: sigma >> sigma_ref → high K"""
    market = MarketState(
        sigma=0.03,  # 3x reference (0.01)
        ts=0.35,
        p_trend=0.5,
        p_mr=0.2,
        p_chop=0.3,
    )
    
    result = compute_harvest_proposal(base_position, market, p1_proposal, theta)
    
    assert result["audit"]["k_components"]["sigma_spike"] > 0  # Should detect spike
    assert result["kill_score"] > 0.3


def test_kill_score_ts_drop(base_position, p1_proposal, theta):
    """TS drop: ts << ts_ref → high K"""
    market = MarketState(
        sigma=0.01,
        ts=0.1,  # Below ts_ref=0.3
        p_trend=0.5,
        p_mr=0.2,
        p_chop=0.3,
    )
    
    result = compute_harvest_proposal(base_position, market, p1_proposal, theta)
    
    assert result["audit"]["k_components"]["ts_drop"] > 0
    assert result["kill_score"] > 0.3


def test_kill_score_age_penalty(base_market, p1_proposal, theta):
    """Age penalty: old position → high K"""
    pos = PositionSnapshot(
        symbol="BTCUSDT",
        side="LONG",
        entry_price=100.0,
        current_price=105.0,
        peak_price=106.0,
        trough_price=99.0,
        age_sec=100000.0,  # > max_age_sec=86400
        unrealized_pnl=5.0,
        current_sl=99.0,
    )
    
    result = compute_harvest_proposal(pos, base_market, p1_proposal, theta)
    
    assert result["audit"]["k_components"]["age_penalty"] >= 1.0  # Capped at 1.0
    assert result["kill_score"] > 0.3


def test_kill_score_full_close_proposal(base_position, p1_proposal):
    """K >= kill_threshold → FULL_CLOSE_PROPOSED"""
    # Create conditions for high kill score
    market = MarketState(
        sigma=0.03,  # High sigma
        ts=0.1,  # Low TS
        p_trend=0.1,  # Low trend
        p_mr=0.3,
        p_chop=0.6,  # High chop
    )
    
    pos = base_position
    pos.age_sec = 100000.0  # Old position
    
    theta = HarvestTheta(kill_threshold=0.5)  # Lower threshold for test
    
    result = compute_harvest_proposal(pos, market, p1_proposal, theta)
    
    assert result["kill_score"] >= theta.kill_threshold
    assert result["harvest_action"] == "FULL_CLOSE_PROPOSED"
    assert "kill_score_triggered" in result["reason_codes"]


# ============================================================================
# TEST: FALLBACK BEHAVIOR
# ============================================================================

def test_fallback_no_p1_proposal(base_position, base_market, theta):
    """No P1 proposal → use fallback_stop_pct"""
    result = compute_harvest_proposal(base_position, base_market, p1_proposal=None, theta=theta)
    
    # risk_unit should use fallback_stop_pct=0.02
    expected_risk_unit = base_position.entry_price * theta.fallback_stop_pct
    assert abs(result["risk_unit"] - expected_risk_unit) < 1e-6


def test_fallback_no_current_sl(base_market, p1_proposal, theta):
    """No current_sl → profit lock proposes from scratch"""
    pos = PositionSnapshot(
        symbol="BTCUSDT",
        side="LONG",
        entry_price=100.0,
        current_price=105.0,
        peak_price=106.0,
        trough_price=99.0,
        age_sec=3600.0,
        unrealized_pnl=4.0,
        current_sl=None,  # No existing SL
    )
    
    result = compute_harvest_proposal(pos, base_market, p1_proposal, theta)
    
    # Should propose BE+ = 100.2
    assert result["new_sl_proposed"] is not None
    assert abs(result["new_sl_proposed"] - 100.2) < 1e-6


def test_fallback_no_theta(base_position, base_market, p1_proposal):
    """No theta → use defaults"""
    result = compute_harvest_proposal(base_position, base_market, p1_proposal, theta=None)
    
    # Should use HarvestTheta() defaults
    assert result["harvest_action"] in ["NONE", "PARTIAL_25", "PARTIAL_50", "PARTIAL_75", "FULL_CLOSE_PROPOSED"]
    assert "risk_unit" in result
    assert "kill_score" in result


# ============================================================================
# TEST: TRANCHE WEIGHTS
# ============================================================================

def test_tranche_weights_sum_to_one(base_market, theta):
    """Tranche weights sum to 1.0"""
    weights = compute_tranche_weights(theta, base_market)
    
    assert len(weights) == 3
    assert abs(sum(weights) - 1.0) < 1e-6


def test_tranche_weights_uniform_fallback(base_market):
    """u1=u2=u3=0 → uniform weights"""
    theta = HarvestTheta(u1=0.0, u2=0.0, u3=0.0)
    weights = compute_tranche_weights(theta, base_market)
    
    # Should be approximately [1/3, 1/3, 1/3]
    for w in weights:
        assert abs(w - 1/3) < 0.01


# ============================================================================
# TEST: NO TRADING SIDE-EFFECTS
# ============================================================================

def test_no_trading_side_effects(base_position, base_market, p1_proposal, theta):
    """Verify output is pure dict, no side-effects"""
    # Run twice with same inputs
    result1 = compute_harvest_proposal(base_position, base_market, p1_proposal, theta)
    result2 = compute_harvest_proposal(base_position, base_market, p1_proposal, theta)
    
    # Results should be identical (deterministic)
    assert result1["harvest_action"] == result2["harvest_action"]
    assert result1["new_sl_proposed"] == result2["new_sl_proposed"]
    assert result1["R_net"] == result2["R_net"]
    assert result1["kill_score"] == result2["kill_score"]
    
    # Verify output is a dict
    assert isinstance(result1, dict)
    assert "harvest_action" in result1
    assert "new_sl_proposed" in result1
    assert "R_net" in result1
    assert "kill_score" in result1
    assert "reason_codes" in result1
    assert "audit" in result1


# ============================================================================
# TEST: REASON CODES
# ============================================================================

def test_reason_codes_harvest(base_position, base_market, p1_proposal, theta):
    """Harvest action → reason_code"""
    pos = base_position
    pos.unrealized_pnl = 9.1  # R_net ~ 4.5 → PARTIAL_50
    
    result = compute_harvest_proposal(pos, base_market, p1_proposal, theta)
    
    assert "harvest_partial_50" in result["reason_codes"]


def test_reason_codes_profit_lock(base_position, base_market, p1_proposal, theta):
    """Profit lock → reason_code"""
    pos = base_position
    pos.unrealized_pnl = 4.0
    pos.current_sl = 99.0
    
    result = compute_harvest_proposal(pos, base_market, p1_proposal, theta)
    
    if result["new_sl_proposed"] is not None:
        assert "profit_lock" in result["reason_codes"]


def test_reason_codes_kill_score(base_position, p1_proposal):
    """Kill score triggered → multiple reason_codes"""
    market = MarketState(
        sigma=0.03,
        ts=0.1,
        p_trend=0.1,
        p_mr=0.3,
        p_chop=0.6,
    )
    
    pos = base_position
    pos.age_sec = 100000.0
    
    theta = HarvestTheta(kill_threshold=0.5)
    
    result = compute_harvest_proposal(pos, market, p1_proposal, theta)
    
    assert "kill_score_triggered" in result["reason_codes"]
    # May also have regime_flip, sigma_spike, ts_drop, age_penalty


# ============================================================================
# TEST: AUDIT TRAIL
# ============================================================================

def test_audit_trail_complete(base_position, base_market, p1_proposal, theta):
    """Audit contains all inputs and intermediates"""
    result = compute_harvest_proposal(base_position, base_market, p1_proposal, theta)
    
    audit = result["audit"]
    
    # Inputs
    assert "position" in audit
    assert "market_state" in audit
    assert "p1_proposal" in audit
    assert "theta" in audit
    
    # Intermediates
    assert "tranche_weights" in audit
    assert "k_components" in audit
    
    # K components
    k_comp = audit["k_components"]
    assert "regime_flip" in k_comp
    assert "sigma_spike" in k_comp
    assert "ts_drop" in k_comp
    assert "age_penalty" in k_comp


# ============================================================================
# RUN TESTS
# ============================================================================

if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
