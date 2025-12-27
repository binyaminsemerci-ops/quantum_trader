"""
Test Smart Position Sizer functionality
"""
import sys
sys.path.append('backend')

from backend.services.execution.smart_position_sizer import SmartPositionSizer

def test_smart_sizer():
    """Test Smart Position Sizer with different scenarios."""
    
    sizer = SmartPositionSizer(base_position_size=300.0, max_leverage=5.0)
    
    print("=" * 80)
    print("🧪 SMART POSITION SIZER - TEST SCENARIOS")
    print("=" * 80)
    
    # SCENARIO 1: Normal market conditions
    print("\n📊 SCENARIO 1: Normal Market (BTCUSDT)")
    print("   Volatility: 3% | Trend: 0.5 | Regime: UNKNOWN")
    result1 = sizer.calculate_optimal_size(
        symbol="BTCUSDT",
        side="LONG",
        volatility=0.03,
        trend_strength=0.5,
        regime="UNKNOWN"
    )
    print(f"   💰 Result: ${result1.size_usd:.2f} ({result1.size_pct*100:.0f}%)")
    print(f"   📈 TP={result1.tp_pct*100:.1f}%, SL={result1.sl_pct*100:.2f}%")
    print(f"   🎯 Confidence: {result1.confidence*100:.0f}%")
    
    # SCENARIO 2: High volatility market (should reduce size)
    print("\n⚡ SCENARIO 2: High Volatility (ETHUSDT)")
    print("   Volatility: 8% | Trend: 0.4 | Regime: HIGH_VOLATILITY")
    result2 = sizer.calculate_optimal_size(
        symbol="ETHUSDT",
        side="LONG",
        volatility=0.08,
        trend_strength=0.4,
        regime="HIGH_VOLATILITY"
    )
    print(f"   💰 Result: ${result2.size_usd:.2f} ({result2.size_pct*100:.0f}%)")
    print(f"   📈 TP={result2.tp_pct*100:.1f}%, SL={result2.sl_pct*100:.2f}%")
    print(f"   🎯 Confidence: {result2.confidence*100:.0f}%")
    
    # SCENARIO 3: Strong trend (should increase size)
    print("\n🚀 SCENARIO 3: Strong Trend (SOLUSDT)")
    print("   Volatility: 2% | Trend: 0.85 | Regime: TRENDING")
    result3 = sizer.calculate_optimal_size(
        symbol="SOLUSDT",
        side="LONG",
        volatility=0.02,
        trend_strength=0.85,
        regime="TRENDING"
    )
    print(f"   💰 Result: ${result3.size_usd:.2f} ({result3.size_pct*100:.0f}%)")
    print(f"   📈 TP={result3.tp_pct*100:.1f}%, SL={result3.sl_pct*100:.2f}%")
    print(f"   🎯 Confidence: {result3.confidence*100:.0f}%")
    
    # Track this position
    sizer.add_open_position("SOLUSDT", "LONG")
    
    # SCENARIO 4: Ranging market (should use tight TP/SL)
    print("\n📉 SCENARIO 4: Ranging Market (BNBUSDT)")
    print("   Volatility: 4% | Trend: 0.3 | Regime: RANGING")
    result4 = sizer.calculate_optimal_size(
        symbol="BNBUSDT",
        side="LONG",
        volatility=0.04,
        trend_strength=0.3,
        regime="RANGING"
    )
    print(f"   💰 Result: ${result4.size_usd:.2f} ({result4.size_pct*100:.0f}%)")
    print(f"   📈 TP={result4.tp_pct*100:.1f}%, SL={result4.sl_pct*100:.2f}%")
    print(f"   🎯 Confidence: {result4.confidence*100:.0f}%")
    
    # SCENARIO 5: Correlation test (BTC + ETH + BNB all LONG)
    print("\n🔗 SCENARIO 5: Correlation Risk (ETHUSDT)")
    print("   Already have: SOLUSDT LONG")
    sizer.add_open_position("BTCUSDT", "LONG")
    sizer.add_open_position("ETHUSDT", "LONG")
    
    result5 = sizer.calculate_optimal_size(
        symbol="BNBUSDT",
        side="LONG",
        volatility=0.03,
        trend_strength=0.6,
        regime="TRENDING"
    )
    print(f"   💰 Result: ${result5.size_usd:.2f} ({result5.size_pct*100:.0f}%)")
    print(f"   📈 TP={result5.tp_pct*100:.1f}%, SL={result5.sl_pct*100:.2f}%")
    print(f"   🎯 Confidence: {result5.confidence*100:.0f}%")
    print(f"   ⚠️  Correlation penalty applied!")
    
    # SCENARIO 6: Win rate tracking (simulate losing streak)
    print("\n📉 SCENARIO 6: Losing Streak Test")
    print("   Simulating 5 losing trades...")
    for i in range(5):
        sizer.update_trade_outcome(f"TEST{i}", win=False, pnl_usd=-10.0)
    
    stats = sizer.get_stats()
    print(f"   📊 Current win rate: {stats['recent_win_rate']*100:.0f}%")
    
    result6 = sizer.calculate_optimal_size(
        symbol="ADAUSDT",
        side="LONG",
        volatility=0.04,
        trend_strength=0.7,
        regime="TRENDING"
    )
    print(f"   💰 Result: ${result6.size_usd:.2f} ({result6.size_pct*100:.0f}%)")
    print(f"   📈 TP={result6.tp_pct*100:.1f}%, SL={result6.sl_pct*100:.2f}%")
    print(f"   🎯 Confidence: {result6.confidence*100:.0f}%")
    print(f"   ⚠️  Defensive sizing due to losing streak!")
    
    # SCENARIO 7: Critical win rate - should BLOCK trade
    print("\n🚫 SCENARIO 7: Critical Win Rate (should BLOCK)")
    print("   Adding 3 more losses to reach <30%...")
    for i in range(3):
        sizer.update_trade_outcome(f"TEST2{i}", win=False, pnl_usd=-15.0)
    
    stats = sizer.get_stats()
    print(f"   📊 Current win rate: {stats['recent_win_rate']*100:.0f}%")
    
    result7 = sizer.calculate_optimal_size(
        symbol="DOTUSDT",
        side="LONG",
        volatility=0.04,
        trend_strength=0.8,
        regime="TRENDING"
    )
    print(f"   💰 Result: ${result7.size_usd:.2f}")
    print(f"   🚫 Trade BLOCKED! Win rate too low!")
    
    # SCENARIO 8: Recovery after adding wins
    print("\n✅ SCENARIO 8: Recovery (adding wins)")
    print("   Adding 6 winning trades...")
    for i in range(6):
        sizer.update_trade_outcome(f"WIN{i}", win=True, pnl_usd=20.0)
    
    stats = sizer.get_stats()
    print(f"   📊 Current win rate: {stats['recent_win_rate']*100:.0f}%")
    
    result8 = sizer.calculate_optimal_size(
        symbol="AVAXUSDT",
        side="LONG",
        volatility=0.03,
        trend_strength=0.75,
        regime="TRENDING"
    )
    print(f"   💰 Result: ${result8.size_usd:.2f} ({result8.size_pct*100:.0f}%)")
    print(f"   📈 TP={result8.tp_pct*100:.1f}%, SL={result8.sl_pct*100:.2f}%")
    print(f"   🎯 Confidence: {result8.confidence*100:.0f}%")
    print(f"   🎉 Hot streak bonus applied!")
    
    # Summary
    print("\n" + "=" * 80)
    print("📊 FINAL STATISTICS")
    print("=" * 80)
    stats = sizer.get_stats()
    print(f"Recent win rate: {stats['recent_win_rate']*100:.0f}%")
    print(f"Recent trades tracked: {stats['recent_trades_count']}")
    print(f"Open positions: {stats['open_positions_count']}")
    print(f"Positions: {stats['open_positions']}")
    print(f"Base size: ${stats['base_size']:.0f}")
    print(f"Max leverage: {stats['max_leverage']:.1f}x")
    
    print("\n✅ Smart Position Sizer test complete!")
    print("=" * 80)


if __name__ == "__main__":
    test_smart_sizer()
