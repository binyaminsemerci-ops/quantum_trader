#!/usr/bin/env python3
"""
P1 Risk Kernel Replay/Demo Harness

Demonstrates stop/TP proposal engine across synthetic position scenarios:
- LONG trending up
- LONG mean-reverting
- SHORT trending down
- Monotonic SL tightening demo
- Trailing activation demo
"""

import sys
import os
import argparse

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ai_engine.risk_kernel_stops import (
    compute_proposal,
    format_proposal,
    PositionSnapshot,
    DEFAULT_THETA_RISK,
)


def generate_synthetic_market_state(regime: str):
    """Generate synthetic MarketState for demo purposes"""
    if regime == "trend":
        return {
            "sigma": 0.015,
            "mu": 0.003,
            "ts": 0.6,
            "regime_probs": {"trend": 0.7, "mr": 0.1, "chop": 0.2},
            "features": {"dp": 0.65, "vr": 1.3}
        }
    elif regime == "mean_revert":
        return {
            "sigma": 0.008,
            "mu": 0.0001,
            "ts": 0.08,
            "regime_probs": {"trend": 0.1, "mr": 0.7, "chop": 0.2},
            "features": {"dp": 0.48, "vr": 0.6}
        }
    elif regime == "chop":
        return {
            "sigma": 0.012,
            "mu": 0.0,
            "ts": 0.05,
            "regime_probs": {"trend": 0.2, "mr": 0.2, "chop": 0.6},
            "features": {"dp": 0.50, "vr": 0.9}
        }
    else:
        raise ValueError(f"Unknown regime: {regime}")


def demo_long_trending():
    """Demo: LONG position in trending market"""
    print("\n" + "=" * 80)
    print("DEMO 1: LONG Position - Trending Market")
    print("=" * 80)
    
    market_state = generate_synthetic_market_state("trend")
    position = PositionSnapshot(
        symbol="BTCUSDT",
        side="LONG",
        entry_price=100.0,
        current_price=115.0,
        peak_price=118.0,
        trough_price=99.0,
        age_sec=1200.0,
        unrealized_pnl=1500.0,
        current_sl=110.0,  # existing SL
        current_tp=125.0,
    )
    
    proposal = compute_proposal("BTCUSDT", market_state, position)
    print(format_proposal(proposal))
    print(f"\nKey Insights:")
    print(f"  - Stop dist %: {proposal['audit']['intermediates']['stop_dist_pct']:.4f}")
    print(f"  - TP dist %: {proposal['audit']['intermediates']['tp_dist_pct']:.4f}")
    print(f"  - Trail gap %: {proposal['audit']['intermediates']['trail_gap_pct']:.4f}")
    print(f"  - TP extension factor: {proposal['audit']['intermediates']['tp_extension_factor']:.3f}")


def demo_long_mean_revert():
    """Demo: LONG position in mean-reverting market"""
    print("\n" + "=" * 80)
    print("DEMO 2: LONG Position - Mean-Reverting Market")
    print("=" * 80)
    
    market_state = generate_synthetic_market_state("mean_revert")
    position = PositionSnapshot(
        symbol="ETHUSDT",
        side="LONG",
        entry_price=100.0,
        current_price=102.5,
        peak_price=103.0,
        trough_price=99.5,
        age_sec=600.0,
        unrealized_pnl=250.0,
        current_sl=100.5,
    )
    
    proposal = compute_proposal("ETHUSDT", market_state, position)
    print(format_proposal(proposal))
    print(f"\nKey Insights:")
    print(f"  - Tighter stops in MR regime (k_sl_mr=0.8 vs k_sl_trend=1.5)")
    print(f"  - Stop dist %: {proposal['audit']['intermediates']['stop_dist_pct']:.4f}")
    print(f"  - TP dist %: {proposal['audit']['intermediates']['tp_dist_pct']:.4f}")


def demo_short_trending():
    """Demo: SHORT position in trending down market"""
    print("\n" + "=" * 80)
    print("DEMO 3: SHORT Position - Trending Down Market")
    print("=" * 80)
    
    market_state = generate_synthetic_market_state("trend")
    position = PositionSnapshot(
        symbol="SOLUSDT",
        side="SHORT",
        entry_price=100.0,
        current_price=85.0,
        peak_price=101.0,  # peak for SHORT is highest
        trough_price=84.0,  # trough is lowest
        age_sec=900.0,
        unrealized_pnl=1500.0,
        current_sl=90.0,
    )
    
    proposal = compute_proposal("SOLUSDT", market_state, position)
    print(format_proposal(proposal))
    print(f"\nKey Insights:")
    print(f"  - SHORT: SL above current, TP below")
    print(f"  - Trailing based on trough (lowest price)")
    print(f"  - Raw SL: ${proposal['audit']['intermediates']['raw_sl']:.4f}")
    print(f"  - Trail SL: ${proposal['audit']['intermediates']['trail_sl']:.4f}")


def demo_monotonic_tightening():
    """Demo: Monotonic SL tightening across multiple updates"""
    print("\n" + "=" * 80)
    print("DEMO 4: Monotonic SL Tightening (LONG)")
    print("=" * 80)
    
    market_state = generate_synthetic_market_state("chop")
    
    # Scenario: Price climbs, then drops slightly
    scenarios = [
        {"price": 105.0, "peak": 105.0, "existing_sl": None, "label": "Initial"},
        {"price": 110.0, "peak": 110.0, "existing_sl": 103.0, "label": "Price up"},
        {"price": 108.0, "peak": 110.0, "existing_sl": 106.0, "label": "Price drops (SL should hold)"},
    ]
    
    for i, scenario in enumerate(scenarios, 1):
        print(f"\n--- Update {i}: {scenario['label']} ---")
        position = PositionSnapshot(
            symbol="BTCUSDT",
            side="LONG",
            entry_price=100.0,
            current_price=scenario["price"],
            peak_price=scenario["peak"],
            trough_price=99.0,
            age_sec=300.0 * i,
            unrealized_pnl=(scenario["price"] - 100.0) * 10,
            current_sl=scenario["existing_sl"],
        )
        
        proposal = compute_proposal("BTCUSDT", market_state, position)
        print(f"  Current Price: ${position.current_price:.2f}  Peak: ${position.peak_price:.2f}")
        print(f"  Existing SL: ${position.current_sl if position.current_sl else 'None'}")
        print(f"  Proposed SL: ${proposal['proposed_sl']:.4f}")
        print(f"  Reasons: {', '.join(proposal['reason_codes'])}")
        
        if i > 1:
            prev_sl = scenarios[i-2]["existing_sl"] or 0.0
            if proposal["proposed_sl"] < prev_sl - 1e-6:
                print(f"  ❌ ERROR: SL loosened from ${prev_sl:.4f} to ${proposal['proposed_sl']:.4f}")
            else:
                print(f"  ✅ Monotonic: SL did not loosen")


def demo_trailing_activation():
    """Demo: Trailing SL activation on profitable position"""
    print("\n" + "=" * 80)
    print("DEMO 5: Trailing SL Activation (LONG)")
    print("=" * 80)
    
    market_state = generate_synthetic_market_state("trend")
    
    # Position at peak
    position = PositionSnapshot(
        symbol="BTCUSDT",
        side="LONG",
        entry_price=100.0,
        current_price=120.0,
        peak_price=120.0,
        trough_price=99.0,
        age_sec=1800.0,
        unrealized_pnl=2000.0,
    )
    
    proposal = compute_proposal("BTCUSDT", market_state, position)
    print(format_proposal(proposal))
    
    raw_sl = proposal["audit"]["intermediates"]["raw_sl"]
    trail_sl = proposal["audit"]["intermediates"]["trail_sl"]
    
    print(f"\n  Raw SL (from current): ${raw_sl:.4f}")
    print(f"  Trail SL (from peak): ${trail_sl:.4f}")
    print(f"  Final SL: ${proposal['proposed_sl']:.4f}")
    
    if trail_sl > raw_sl + 1e-6:
        print(f"  ✅ Trailing is ACTIVE (tighter than raw)")
    else:
        print(f"  ℹ️  Raw stop is tighter (trailing not yet active)")


def main():
    parser = argparse.ArgumentParser(description="P1 Risk Kernel Replay Harness")
    parser.add_argument("--synthetic", action="store_true", help="Run all synthetic demos")
    parser.add_argument("--regime", choices=["trend", "mean_revert", "chop"], help="Single regime demo")
    
    args = parser.parse_args()
    
    if args.synthetic:
        demo_long_trending()
        demo_long_mean_revert()
        demo_short_trending()
        demo_monotonic_tightening()
        demo_trailing_activation()
        
        print("\n" + "=" * 80)
        print("✅ All demos complete")
        print("=" * 80)
        
    elif args.regime:
        market_state = generate_synthetic_market_state(args.regime)
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
        print(format_proposal(proposal))
    else:
        parser.print_help()
        sys.exit(1)


if __name__ == "__main__":
    main()
