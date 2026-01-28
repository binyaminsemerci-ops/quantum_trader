#!/usr/bin/env python3
"""
Replay scenarios for P2 Harvest Kernel

Synthetic scenarios to demonstrate:
- Trend vs chop regimes
- Increasing PNL → higher tranches
- Kill score triggers
- Profit lock SL tightening
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ai_engine.risk_kernel_harvest import (
    compute_harvest_proposal,
    HarvestTheta,
    PositionSnapshot,
    MarketState,
    P1Proposal,
)


def print_separator(title: str):
    """Print section separator"""
    print(f"\n{'='*70}")
    print(f"  {title}")
    print('='*70)


def print_result(result: dict, scenario: str):
    """Print harvest proposal result"""
    print(f"\nScenario: {scenario}")
    print(f"  Harvest Action:   {result['harvest_action']}")
    print(f"  New SL Proposed:  {result['new_sl_proposed']}")
    print(f"  R_net:            {result['R_net']:.2f}R")
    print(f"  Risk Unit:        ${result['risk_unit']:.4f}")
    print(f"  Cost Est:         ${result['cost_est']:.4f}")
    print(f"  Kill Score:       {result['kill_score']:.3f}")
    print(f"  Reason Codes:     {', '.join(result['reason_codes']) if result['reason_codes'] else 'none'}")
    
    # K components
    k_comp = result['audit']['k_components']
    print(f"  K Components:")
    print(f"    Regime Flip:    {k_comp['regime_flip']:.3f}")
    print(f"    Sigma Spike:    {k_comp['sigma_spike']:.3f}")
    print(f"    TS Drop:        {k_comp['ts_drop']:.3f}")
    print(f"    Age Penalty:    {k_comp['age_penalty']:.3f}")


def scenario_1_increasing_pnl():
    """Scenario 1: Increasing PNL triggers higher tranches"""
    print_separator("SCENARIO 1: Increasing PNL (Trend Regime)")
    
    theta = HarvestTheta()
    p1 = P1Proposal(stop_dist_pct=0.02)
    
    # Trending market
    market = MarketState(
        sigma=0.01,
        ts=0.4,
        p_trend=0.6,
        p_mr=0.2,
        p_chop=0.2,
    )
    
    # Test increasing PNL levels
    pnl_levels = [
        (1.0, "Low profit (R_net ~ 0.45)"),
        (5.0, "Medium profit (R_net ~ 2.45)"),
        (9.0, "High profit (R_net ~ 4.45)"),
        (14.0, "Very high profit (R_net ~ 6.95)"),
    ]
    
    for pnl, desc in pnl_levels:
        pos = PositionSnapshot(
            symbol="BTCUSDT",
            side="LONG",
            entry_price=100.0,
            current_price=100.0 + pnl,
            peak_price=100.0 + pnl + 1.0,
            trough_price=99.0,
            age_sec=3600.0,
            unrealized_pnl=pnl,
            current_sl=99.0,
        )
        
        result = compute_harvest_proposal(pos, market, p1, theta)
        print_result(result, desc)


def scenario_2_regime_flip():
    """Scenario 2: Regime flip → high kill score"""
    print_separator("SCENARIO 2: Regime Flip (Trend → Chop)")
    
    theta = HarvestTheta()
    p1 = P1Proposal(stop_dist_pct=0.02)
    
    pos = PositionSnapshot(
        symbol="BTCUSDT",
        side="LONG",
        entry_price=100.0,
        current_price=103.0,
        peak_price=105.0,
        trough_price=99.0,
        age_sec=3600.0,
        unrealized_pnl=3.0,
        current_sl=99.0,
    )
    
    # Before: Trending
    market_before = MarketState(
        sigma=0.01,
        ts=0.4,
        p_trend=0.6,
        p_mr=0.2,
        p_chop=0.2,
    )
    
    result_before = compute_harvest_proposal(pos, market_before, p1, theta)
    print_result(result_before, "Before: Trend intact")
    
    # After: Chop
    market_after = MarketState(
        sigma=0.01,
        ts=0.4,
        p_trend=0.1,  # Dropped below trend_min
        p_mr=0.3,
        p_chop=0.6,  # High chop
    )
    
    result_after = compute_harvest_proposal(pos, market_after, p1, theta)
    print_result(result_after, "After: Regime flip to chop")


def scenario_3_volatility_spike():
    """Scenario 3: Volatility spike → elevated kill score"""
    print_separator("SCENARIO 3: Volatility Spike")
    
    theta = HarvestTheta()
    p1 = P1Proposal(stop_dist_pct=0.02)
    
    pos = PositionSnapshot(
        symbol="BTCUSDT",
        side="LONG",
        entry_price=100.0,
        current_price=103.0,
        peak_price=105.0,
        trough_price=99.0,
        age_sec=3600.0,
        unrealized_pnl=3.0,
        current_sl=99.0,
    )
    
    # Normal volatility
    market_normal = MarketState(
        sigma=0.01,
        ts=0.4,
        p_trend=0.5,
        p_mr=0.2,
        p_chop=0.3,
    )
    
    result_normal = compute_harvest_proposal(pos, market_normal, p1, theta)
    print_result(result_normal, "Normal volatility (σ=0.01)")
    
    # High volatility
    market_spike = MarketState(
        sigma=0.03,  # 3x reference
        ts=0.4,
        p_trend=0.5,
        p_mr=0.2,
        p_chop=0.3,
    )
    
    result_spike = compute_harvest_proposal(pos, market_spike, p1, theta)
    print_result(result_spike, "Volatility spike (σ=0.03)")


def scenario_4_ts_collapse():
    """Scenario 4: Trend strength collapse"""
    print_separator("SCENARIO 4: Trend Strength Collapse")
    
    theta = HarvestTheta()
    p1 = P1Proposal(stop_dist_pct=0.02)
    
    pos = PositionSnapshot(
        symbol="BTCUSDT",
        side="LONG",
        entry_price=100.0,
        current_price=103.0,
        peak_price=105.0,
        trough_price=99.0,
        age_sec=3600.0,
        unrealized_pnl=3.0,
        current_sl=99.0,
    )
    
    # Strong trend
    market_strong = MarketState(
        sigma=0.01,
        ts=0.5,
        p_trend=0.6,
        p_mr=0.2,
        p_chop=0.2,
    )
    
    result_strong = compute_harvest_proposal(pos, market_strong, p1, theta)
    print_result(result_strong, "Strong trend (TS=0.5)")
    
    # Weak trend
    market_weak = MarketState(
        sigma=0.01,
        ts=0.1,  # Dropped below ts_ref
        p_trend=0.6,
        p_mr=0.2,
        p_chop=0.2,
    )
    
    result_weak = compute_harvest_proposal(pos, market_weak, p1, theta)
    print_result(result_weak, "Weak trend (TS=0.1)")


def scenario_5_aging_position():
    """Scenario 5: Aging position → kill score increase"""
    print_separator("SCENARIO 5: Aging Position")
    
    theta = HarvestTheta()
    p1 = P1Proposal(stop_dist_pct=0.02)
    
    market = MarketState(
        sigma=0.01,
        ts=0.4,
        p_trend=0.5,
        p_mr=0.2,
        p_chop=0.3,
    )
    
    # Young position
    pos_young = PositionSnapshot(
        symbol="BTCUSDT",
        side="LONG",
        entry_price=100.0,
        current_price=103.0,
        peak_price=105.0,
        trough_price=99.0,
        age_sec=3600.0,  # 1 hour
        unrealized_pnl=3.0,
        current_sl=99.0,
    )
    
    result_young = compute_harvest_proposal(pos_young, market, p1, theta)
    print_result(result_young, "Young position (1 hour)")
    
    # Old position
    pos_old = PositionSnapshot(
        symbol="BTCUSDT",
        side="LONG",
        entry_price=100.0,
        current_price=103.0,
        peak_price=105.0,
        trough_price=99.0,
        age_sec=100000.0,  # > 24 hours
        unrealized_pnl=3.0,
        current_sl=99.0,
    )
    
    result_old = compute_harvest_proposal(pos_old, market, p1, theta)
    print_result(result_old, "Old position (>24 hours)")


def scenario_6_perfect_storm():
    """Scenario 6: Perfect storm → FULL_CLOSE_PROPOSED"""
    print_separator("SCENARIO 6: Perfect Storm (All Kill Factors)")
    
    theta = HarvestTheta(kill_threshold=0.5)  # Lower threshold
    p1 = P1Proposal(stop_dist_pct=0.02)
    
    # Everything bad at once
    market = MarketState(
        sigma=0.03,  # High volatility
        ts=0.1,  # Low trend strength
        p_trend=0.1,  # Low trend prob
        p_mr=0.3,
        p_chop=0.6,  # High chop
    )
    
    pos = PositionSnapshot(
        symbol="BTCUSDT",
        side="LONG",
        entry_price=100.0,
        current_price=102.0,
        peak_price=105.0,
        trough_price=99.0,
        age_sec=100000.0,  # Very old
        unrealized_pnl=2.0,
        current_sl=99.0,
    )
    
    result = compute_harvest_proposal(pos, market, p1, theta)
    print_result(result, "Perfect storm (all factors)")


def scenario_7_profit_lock():
    """Scenario 7: Profit lock SL tightening"""
    print_separator("SCENARIO 7: Profit Lock SL Tightening")
    
    theta = HarvestTheta(lock_R=1.5, be_plus_pct=0.002)
    p1 = P1Proposal(stop_dist_pct=0.02)
    
    market = MarketState(
        sigma=0.01,
        ts=0.4,
        p_trend=0.6,
        p_mr=0.2,
        p_chop=0.2,
    )
    
    # LONG with profit
    pos_long = PositionSnapshot(
        symbol="BTCUSDT",
        side="LONG",
        entry_price=100.0,
        current_price=104.0,
        peak_price=105.0,
        trough_price=99.0,
        age_sec=3600.0,
        unrealized_pnl=4.0,
        current_sl=99.0,  # Below BE
    )
    
    result_long = compute_harvest_proposal(pos_long, market, p1, theta)
    print_result(result_long, "LONG profit lock (SL 99.0 → 100.2)")
    
    # SHORT with profit
    pos_short = PositionSnapshot(
        symbol="BTCUSDT",
        side="SHORT",
        entry_price=100.0,
        current_price=96.0,
        peak_price=101.0,
        trough_price=95.0,
        age_sec=3600.0,
        unrealized_pnl=4.0,
        current_sl=101.0,  # Above BE
    )
    
    result_short = compute_harvest_proposal(pos_short, market, p1, theta)
    print_result(result_short, "SHORT profit lock (SL 101.0 → 99.8)")


def scenario_8_no_p1_fallback():
    """Scenario 8: No P1 proposal (fallback behavior)"""
    print_separator("SCENARIO 8: No P1 Proposal (Fallback)")
    
    theta = HarvestTheta()
    
    market = MarketState(
        sigma=0.01,
        ts=0.4,
        p_trend=0.5,
        p_mr=0.2,
        p_chop=0.3,
    )
    
    pos = PositionSnapshot(
        symbol="BTCUSDT",
        side="LONG",
        entry_price=100.0,
        current_price=105.0,
        peak_price=106.0,
        trough_price=99.0,
        age_sec=3600.0,
        unrealized_pnl=5.0,
        current_sl=99.0,
    )
    
    result = compute_harvest_proposal(pos, market, p1_proposal=None, theta=theta)
    print_result(result, "No P1 proposal (uses fallback_stop_pct)")


def main():
    """Run all replay scenarios"""
    print("\n" + "="*70)
    print("  P2 HARVEST KERNEL REPLAY")
    print("  Calc-only profit harvesting proposals")
    print("="*70)
    
    scenario_1_increasing_pnl()
    scenario_2_regime_flip()
    scenario_3_volatility_spike()
    scenario_4_ts_collapse()
    scenario_5_aging_position()
    scenario_6_perfect_storm()
    scenario_7_profit_lock()
    scenario_8_no_p1_fallback()
    
    print("\n" + "="*70)
    print("  REPLAY COMPLETE ✅")
    print("  All scenarios executed (calc-only, no trading)")
    print("="*70 + "\n")


if __name__ == "__main__":
    main()
