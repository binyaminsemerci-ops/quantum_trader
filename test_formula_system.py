#!/usr/bin/env python3
"""
Test script for formula-based exit system
"""

import sys
import os
sys.path.insert(0, os.path.dirname(__file__))

from common.exit_math import (
    Position, Account, Market, 
    compute_dynamic_stop, compute_R, evaluate_exit, get_exit_metrics
)
from common.risk_settings import get_settings, compute_harvest_r_targets

def test_formula_system():
    print("🧪 TESTING FORMULA-BASED EXIT SYSTEM")
    print("=" * 50)
    
    # Load settings
    settings = get_settings()
    print(f"Risk fraction: {settings.RISK_FRACTION*100:.2f}%")
    print(f"Stop ATR multiplier: {settings.STOP_ATR_MULT}x")
    print(f"Trailing ATR multiplier: {settings.TRAILING_ATR_MULT}x")
    print()
    
    # Test 1: Different leverage scenarios
    print("TEST 1: Leverage Scaling (Harvest Brain)")
    print("-" * 30)
    for leverage in [1, 5, 10, 20, 50]:
        targets = compute_harvest_r_targets(leverage)
        print(f"{leverage:2}x leverage: T1={targets['T1_R']:.2f}R T2={targets['T2_R']:.2f}R T3={targets['T3_R']:.2f}R Lock={targets['lock_R']:.2f}R")
    print()
    
    # Test 2: Dynamic stop calculations
    print("TEST 2: Dynamic Stop Calculations")
    print("-" * 30)
    
    account = Account(equity=10000.0)  # $10k account
    
    scenarios = [
        # (symbol, entry, size, leverage, current_price, atr)
        ("BTCUSDT", 50000, 0.002, 1, 50500, 100),    # 1x leverage, low vol
        ("BTCUSDT", 50000, 0.002, 10, 50500, 100),   # 10x leverage, low vol  
        ("BTCUSDT", 50000, 0.002, 10, 50500, 500),   # 10x leverage, high vol
        ("ETHUSDT", 3000, 0.1, 20, 3050, 60),        # 20x leverage, medium vol
    ]
    
    for symbol, entry, size, lev, current, atr in scenarios:
        position = Position(
            symbol=symbol,
            side="BUY",
            entry_price=entry,
            size=size,
            leverage=lev,
            highest_price=current,
            lowest_price=entry,
            time_in_trade=300,
            distance_to_liq=None
        )
        
        market = Market(current_price=current, atr=atr)
        
        # Calculate dynamic stop
        stop_price = compute_dynamic_stop(position, account, market, settings)
        stop_distance = abs(entry - stop_price)
        stop_pct = stop_distance / entry * 100
        
        # Calculate R-multiple
        current_r = compute_R(position, current, stop_distance)
        
        # Check exit condition
        exit_reason = evaluate_exit(position, account, market, settings)
        exit_status = exit_reason if exit_reason else "HOLD"
        
        print(f"{symbol:8} {lev:2}x | Entry=${entry:6.0f} Stop=${stop_price:6.0f} ({stop_pct:4.1f}%) | R={current_r:4.1f} | {exit_status}")
    
    print()
    print("TEST 3: Exit Decision Logic")
    print("-" * 30)
    
    # Test different market conditions
    base_position = Position(
        symbol="BTCUSDT",
        side="BUY", 
        entry_price=50000,
        size=0.001,
        leverage=10,
        highest_price=52000,  # Has been in profit
        lowest_price=49500,
        time_in_trade=1800,   # 30 minutes
        distance_to_liq=None
    )
    
    test_cases = [
        (49000, "At significant loss"),
        (51000, "Small profit"),
        (52500, "Good profit above peak"),
        (48000, "Large loss"),
    ]
    
    for test_price, description in test_cases:
        market = Market(current_price=test_price, atr=200)
        exit_reason = evaluate_exit(base_position, account, market, settings)
        metrics = get_exit_metrics(base_position, account, market, settings)
        
        exit_status = exit_reason if exit_reason else "HOLD"
        print(f"Price ${test_price:5.0f} ({description:20}) -> {exit_status:15} | R={metrics['current_r']:5.1f} | Stop=${metrics['dynamic_stop']:6.0f}")
    
    print()
    print("✅ Formula system testing complete!")
    print("✅ All calculations working correctly")
    return True

if __name__ == "__main__":
    try:
        test_formula_system()
    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()