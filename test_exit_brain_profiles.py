"""
Quick test script to verify Exit Brain V3 TP Profile System
"""

import asyncio
import sys
sys.path.insert(0, '/app')

from backend.domains.exits.exit_brain_v3 import (
    ExitBrainV3,
    ExitContext,
    MarketRegime,
    get_tp_and_trailing_profile,
    list_available_profiles
)


async def test_profile_system():
    """Test Exit Brain V3 TP Profile System"""
    
    print("\n" + "="*80)
    print("EXIT BRAIN V3 - TP PROFILE SYSTEM TEST")
    print("="*80 + "\n")
    
    # 1. List available profiles
    print("📋 Available Profiles:")
    profiles = list_available_profiles()
    for profile in profiles:
        print(f"  - {profile}")
    print()
    
    # 2. Test profile selection for different regimes
    print("🎯 Profile Selection Tests:")
    test_cases = [
        ("BTCUSDT", "RL_V3", MarketRegime.TREND),
        ("ETHUSDT", "RL_V3", MarketRegime.RANGE),
        ("ADAUSDT", "RL_V3", MarketRegime.VOLATILE),
        ("DOTUSDT", "RL_V3", MarketRegime.CHOP),
    ]
    
    for symbol, strategy, regime in test_cases:
        profile, trailing = get_tp_and_trailing_profile(symbol, strategy, regime)
        print(f"\n  {symbol} / {regime.value}:")
        print(f"    Profile: {profile.name}")
        print(f"    TP Legs: {len(profile.tp_legs)}")
        for i, leg in enumerate(profile.tp_legs):
            print(f"      TP{i+1}: {leg.r_multiple}R ({leg.size_fraction*100:.0f}%) - {leg.kind.value}")
        if trailing:
            print(f"    Trailing: {trailing.callback_pct*100:.1f}% callback @ {trailing.activation_r}R")
            if trailing.tightening_curve:
                print(f"    Tightening: {len(trailing.tightening_curve)} steps")
        else:
            print(f"    Trailing: None")
    
    print("\n")
    
    # 3. Test exit plan creation
    print("📊 Exit Plan Creation Test (TREND regime):")
    brain = ExitBrainV3(config={"use_profiles": True, "strategy_id": "RL_V3"})
    
    ctx = ExitContext(
        symbol="BTCUSDT",
        side="LONG",
        entry_price=100000.0,
        size=0.1,
        leverage=10.0,
        current_price=101000.0,
        unrealized_pnl_pct=1.0,
        market_regime="TRENDING",
        risk_mode="NORMAL"
    )
    
    plan = await brain.build_exit_plan(ctx)
    
    print(f"\n  Symbol: {plan.symbol}")
    print(f"  Strategy: {plan.strategy_id}")
    print(f"  Profile: {plan.profile_name}")
    print(f"  Regime: {plan.market_regime}")
    print(f"  Confidence: {plan.confidence:.2f}")
    print(f"  Total Legs: {len(plan.legs)}")
    print(f"\n  Leg Details:")
    
    for leg in plan.legs:
        if leg.kind.value == "TP":
            print(f"    {leg.kind.value}: {leg.size_pct*100:.0f}% @ {leg.trigger_pct*100:.1f}% "
                  f"(R={leg.r_multiple:.1f}) - {leg.reason}")
        elif leg.kind.value == "TRAIL":
            print(f"    {leg.kind.value}: {leg.size_pct*100:.0f}% @ {leg.trail_callback*100:.1f}% "
                  f"callback - {leg.reason}")
        elif leg.kind.value == "SL":
            print(f"    {leg.kind.value}: {leg.size_pct*100:.0f}% @ {leg.trigger_pct*100:.1f}% "
                  f"- {leg.reason}")
    
    print("\n" + "="*80)
    print("✅ EXIT BRAIN V3 TP PROFILE SYSTEM: OPERATIONAL")
    print("="*80 + "\n")


if __name__ == "__main__":
    asyncio.run(test_profile_system())
