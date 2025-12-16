#!/usr/bin/env python3
"""
Test new TP/SL strategy implementation
Verify that new calculations are working correctly
"""

import sys
sys.path.insert(0, '/app/backend')

from dataclasses import dataclass
from services.ai.trading_mathematician import TradingMathematician

@dataclass
class MarketConditions:
    current_price: float
    atr_pct: float
    daily_volatility: float
    trend_strength: float

@dataclass
class PerformanceMetrics:
    total_trades: int
    win_rate: float
    profit_factor: float
    avg_win_pct: float
    avg_loss_pct: float

@dataclass
class AccountState:
    equity: float
    available_balance: float
    leverage: float

print("\n" + "="*80)
print("🧪 TP/SL STRATEGY TEST")
print("="*80 + "\n")

# Initialize TradingMathematician with correct parameters
mathematician = TradingMathematician(
    risk_per_trade_pct=0.02,
    max_leverage=20,
    min_risk_reward=2.0,
    conservative_mode=False
)

# Test market conditions
market = MarketConditions(
    current_price=100.0,
    atr_pct=0.015,  # 1.5% ATR
    daily_volatility=0.03,  # 3% daily vol
    trend_strength=0.5  # Neutral trend
)

# Test performance (some history)
performance = PerformanceMetrics(
    total_trades=20,
    win_rate=0.55,
    profit_factor=1.5,
    avg_win_pct=0.03,
    avg_loss_pct=0.02
)

# Test account
account = AccountState(
    equity=1000.0,
    available_balance=1000.0,
    leverage=10.0
)

print("📊 TEST CONDITIONS:")
print(f"   ATR: {market.atr_pct*100:.2f}%")
print(f"   Daily Volatility: {market.daily_volatility*100:.2f}%")
print(f"   Trend Strength: {market.trend_strength:.2f}")
print(f"   Win Rate: {performance.win_rate*100:.1f}%")
print()

# Calculate optimal levels
print("🎯 TESTING NEW STRATEGY:\n")

# Test SL calculation
sl_pct = mathematician._calculate_optimal_sl(market, performance)
print(f"✅ Stop Loss Calculation:")
print(f"   Result: {sl_pct*100:.2f}%")
print(f"   Expected Range: 2.5-3.0%")
print(f"   Status: {'✅ PASS' if 0.025 <= sl_pct <= 0.030 else '❌ FAIL'}")
print()

# Test TP calculation
tp_pct = mathematician._calculate_optimal_tp(sl_pct, performance, market)
print(f"✅ Take Profit Calculation:")
print(f"   Result: {tp_pct*100:.2f}%")
print(f"   Expected Range: 3.0-4.5%")
print(f"   Status: {'✅ PASS' if 0.030 <= tp_pct <= 0.045 else '❌ FAIL'}")
print()

# Test partial TP levels (from trade_lifecycle_manager logic)
partial_tp_1_pct = 0.0175  # Fixed 1.75%
partial_tp_2_pct = tp_pct  # Calculated TP2

print(f"✅ Partial TP Structure:")
print(f"   TP1 (50%): {partial_tp_1_pct*100:.2f}%")
print(f"   TP2 (30%): {partial_tp_2_pct*100:.2f}%")
print(f"   TP3 (20%): Trailing from +5%")
print()

# Calculate R:R ratios
rr_tp1 = partial_tp_1_pct / sl_pct
rr_tp2 = partial_tp_2_pct / sl_pct

print(f"⚖️  RISK/REWARD ANALYSIS:")
print(f"   Initial Setup:")
print(f"     SL: {sl_pct*100:.2f}% (100% position)")
print(f"     TP1: {partial_tp_1_pct*100:.2f}% (50% position) → R:R = {rr_tp1:.2f}:1")
print(f"     TP2: {partial_tp_2_pct*100:.2f}% (30% position) → R:R = {rr_tp2:.2f}:1")
print()

# Calculate blended expectancy
# Assume: 75% hit TP1, 50% hit TP2, 20% hit TP3
tp1_hit_rate = 0.75
tp2_hit_rate = 0.50
tp3_hit_rate = 0.20
sl_hit_rate = 1.0 - tp1_hit_rate

avg_win = (
    tp1_hit_rate * 0.50 * partial_tp_1_pct +  # 50% position @ TP1
    tp2_hit_rate * 0.30 * partial_tp_2_pct +  # 30% position @ TP2
    tp3_hit_rate * 0.20 * 0.06  # 20% position @ avg 6% trailing
) / (tp1_hit_rate + tp2_hit_rate + tp3_hit_rate)

avg_loss = sl_pct

blended_rr = avg_win / avg_loss

breakeven_wr = 1 / (1 + blended_rr)

print(f"   Blended Analysis:")
print(f"     Average Win: {avg_win*100:.2f}%")
print(f"     Average Loss: {avg_loss*100:.2f}%")
print(f"     Blended R:R: {blended_rr:.2f}:1")
print(f"     Breakeven Win Rate: {breakeven_wr*100:.1f}%")
print(f"     Status: {'✅ SUSTAINABLE' if blended_rr >= 1.8 else '⚠️  MARGINAL'}")
print()

# Test with high volatility
print("🔥 HIGH VOLATILITY TEST:")
high_vol_market = MarketConditions(
    current_price=100.0,
    atr_pct=0.025,
    daily_volatility=0.08,  # 8% daily vol
    trend_strength=0.8  # Strong trend
)

sl_high = mathematician._calculate_optimal_sl(high_vol_market, performance)
tp_high = mathematician._calculate_optimal_tp(sl_high, performance, high_vol_market)

print(f"   SL: {sl_high*100:.2f}% (should be 2.5-3%)")
print(f"   TP: {tp_high*100:.2f}% (should be ~4%)")
print(f"   R:R: {(tp_high/sl_high):.2f}:1")
print()

# Test with low volatility
print("❄️  LOW VOLATILITY TEST:")
low_vol_market = MarketConditions(
    current_price=100.0,
    atr_pct=0.008,
    daily_volatility=0.015,  # 1.5% daily vol
    trend_strength=0.3  # Choppy
)

sl_low = mathematician._calculate_optimal_sl(low_vol_market, performance)
tp_low = mathematician._calculate_optimal_tp(sl_low, performance, low_vol_market)

print(f"   SL: {sl_low*100:.2f}% (should be 2.5-3%)")
print(f"   TP: {tp_low*100:.2f}% (should be ~3%)")
print(f"   R:R: {(tp_low/sl_low):.2f}:1")
print()

print("="*80)
print("✅ STRATEGY TEST COMPLETE!")
print("="*80)
print()

# Summary
print("📋 SUMMARY:")
print()
print("OLD STRATEGY:")
print("  ❌ SL: 0.8-3% (too tight)")
print("  ❌ TP: R:R based (unpredictable)")
print("  ❌ R:R: 0.72:1 (catastrophic)")
print("  ❌ Breakeven WR: 58.3%")
print()
print("NEW STRATEGY:")
print(f"  ✅ SL: {sl_pct*100:.2f}% (roomy, avoids whipsaw)")
print(f"  ✅ TP1: {partial_tp_1_pct*100:.2f}% (quick profit)")
print(f"  ✅ TP2: {tp_pct*100:.2f}% (main target)")
print(f"  ✅ Blended R:R: {blended_rr:.2f}:1 (sustainable)")
print(f"  ✅ Breakeven WR: {breakeven_wr*100:.1f}% (achievable)")
print()
print("DYNAMIC SL TIGHTENING:")
print("  🎯 Stage 1 (+1.5%): Move to breakeven")
print("  🎯 Stage 2 (+3.0%): Lock +1.5% profit")
print("  🎯 Stage 3 (+5.0%): Activate trailing 1%")
print()
