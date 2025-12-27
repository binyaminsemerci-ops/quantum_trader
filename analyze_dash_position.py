"""
DASHUSDT Position Analysis - TP/SL Verification
================================================
"""

# Position Details
symbol = "DASHUSDT"
leverage = 30
position_size = -80.593  # SHORT position
entry_price = 61.79
current_price = 61.76
liquidation = 63.60
margin = 171.19  # USDT
unrealized_pnl = -147.48  # USDT (-86.14%)
roe = -86.14  # %

# TP/SL Levels from Binance
tp_level = 66.13  # (visible as "-- / 66.13")
sl_level = None  # Not visible in data

print("=" * 80)
print("🔍 DASHUSDT SHORT POSITION ANALYSIS")
print("=" * 80)
print()

print("📊 POSITION DETAILS:")
print(f"Symbol: {symbol}")
print(f"Direction: SHORT")
print(f"Leverage: {leverage}x")
print(f"Position Size: {abs(position_size)} DASH")
print(f"Entry Price: ${entry_price}")
print(f"Current Price: ${current_price}")
print(f"Margin: ${margin}")
print(f"Liquidation: ${liquidation}")
print()

print("💰 P&L STATUS:")
print(f"Unrealized P&L: ${unrealized_pnl} ({roe}%)")
print(f"Price Movement: ${current_price - entry_price} ({((current_price/entry_price - 1) * 100):.2f}%)")
print()

# Calculate what the position value should be
position_notional = abs(position_size) * entry_price
effective_leverage_check = position_notional / margin

print("🔢 POSITION SIZING VERIFICATION:")
print(f"Position Notional: ${position_notional:.2f}")
print(f"Effective Leverage: {effective_leverage_check:.2f}x")
print(f"Expected Leverage: {leverage}x")
if abs(effective_leverage_check - leverage) < 1:
    print("✅ Leverage is correct!")
else:
    print(f"⚠️  Leverage mismatch! Expected {leverage}x, got {effective_leverage_check:.2f}x")
print()

# Calculate ATR-based TP/SL (using 15m timeframe, ATR 14)
# For demo, let's estimate ATR as ~2% of price
estimated_atr_pct = 0.02  # 2% estimate
atr_value = entry_price * estimated_atr_pct

print("🎯 EXPECTED TP/SL LEVELS (ATR-based):")
print(f"Estimated ATR (2%): ${atr_value:.2f}")
print()

# For SHORT position:
# SL = Entry + (1.0 * ATR)  (price goes UP)
# TP1 = Entry - (1.5 * ATR)  (price goes DOWN)
# TP2 = Entry - (2.5 * ATR)  (price goes DOWN)

expected_sl = entry_price + (1.0 * atr_value)
expected_tp1 = entry_price - (1.5 * atr_value)
expected_tp2 = entry_price - (2.5 * atr_value)

print("SHORT Position Targets:")
print(f"Expected SL (1.0R):   ${expected_sl:.2f} (+{((expected_sl/entry_price - 1) * 100):.2f}%)")
print(f"Expected TP1 (1.5R):  ${expected_tp1:.2f} ({((expected_tp1/entry_price - 1) * 100):.2f}%)")
print(f"Expected TP2 (2.5R):  ${expected_tp2:.2f} ({((expected_tp2/entry_price - 1) * 100):.2f}%)")
print()

print("📋 ACTUAL BINANCE ORDERS:")
if tp_level:
    print(f"TP Level: ${tp_level}")
    tp_distance_pct = ((tp_level / entry_price - 1) * 100)
    print(f"TP Distance: {tp_distance_pct:.2f}%")
    
    # For SHORT, TP should be BELOW entry (lower price)
    if tp_level < entry_price:
        print("✅ TP direction correct (BELOW entry for SHORT)")
    else:
        print("❌ TP direction WRONG! Should be BELOW entry for SHORT")
        print(f"   Current TP: ${tp_level} (ABOVE entry ${entry_price})")
    
    # Check if TP matches our expected levels
    tp_diff_from_tp1 = abs(tp_level - expected_tp1)
    tp_diff_from_tp2 = abs(tp_level - expected_tp2)
    
    if tp_diff_from_tp1 < 0.50:
        print(f"✅ TP matches TP1 (1.5R) - ${expected_tp1:.2f}")
    elif tp_diff_from_tp2 < 0.50:
        print(f"✅ TP matches TP2 (2.5R) - ${expected_tp2:.2f}")
    else:
        print(f"⚠️  TP doesn't match expected levels")
        print(f"   Expected TP1: ${expected_tp1:.2f}, TP2: ${expected_tp2:.2f}")
        print(f"   Actual TP: ${tp_level}")
else:
    print("⚠️  No TP level visible")

if sl_level:
    print(f"\nSL Level: ${sl_level}")
else:
    print("\n⚠️  No SL level visible in data")
    print(f"   Expected SL (1.0R): ${expected_sl:.2f}")

print()
print("=" * 80)
print("🚨 CRITICAL ISSUE DETECTED!")
print("=" * 80)
print()
print("❌ TP DIRECTION IS WRONG!")
print(f"   Position: SHORT (expecting price to go DOWN)")
print(f"   Entry: ${entry_price}")
print(f"   Current TP: ${tp_level} (ABOVE entry - WRONG!)")
print(f"   Should be: ${expected_tp1:.2f} or ${expected_tp2:.2f} (BELOW entry)")
print()
print("📊 CONSEQUENCE:")
print(f"   With TP at ${tp_level}, you need price to go UP to {tp_distance_pct:.2f}%")
print(f"   But you're SHORT - you LOSE money when price goes up!")
print(f"   This TP will never be hit profitably.")
print()
print("✅ CORRECT SETUP SHOULD BE:")
print(f"   SHORT Entry: ${entry_price}")
print(f"   SL (stop loss): ${expected_sl:.2f} (price goes up - cut loss)")
print(f"   TP1 (50% close): ${expected_tp1:.2f} (price goes down - take profit)")
print(f"   TP2 (30% close): ${expected_tp2:.2f} (price goes down - take more profit)")
print()
print("=" * 80)
print("🔧 RECOMMENDED ACTION:")
print("=" * 80)
print()
print("1. CANCEL current TP order at $66.13")
print("2. SET correct TP orders:")
print(f"   - TP1: ${expected_tp1:.2f} (quantity: {abs(position_size) * 0.5:.3f} DASH)")
print(f"   - TP2: ${expected_tp2:.2f} (quantity: {abs(position_size) * 0.3:.3f} DASH)")
print(f"3. SET SL: ${expected_sl:.2f} (full position)")
print()
print("Or use the fix script:")
print("   python fix_positions_tpsl.py")
print()
