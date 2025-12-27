"""
Fix DASHUSDT Position - Correct TP/SL Direction
==============================================
"""
import os
from binance.client import Client

# Initialize Binance client
api_key = os.getenv("BINANCE_API_KEY")
api_secret = os.getenv("BINANCE_API_SECRET")
client = Client(api_key, api_secret)

symbol = "DASHUSDT"
entry_price = 61.79
position_size = 80.593  # Full position size
atr_value = entry_price * 0.02  # 2% ATR estimate

# Calculate correct levels for SHORT
sl_price = round(entry_price + (1.0 * atr_value), 2)  # Above entry
tp1_price = round(entry_price - (1.5 * atr_value), 2)  # Below entry
tp2_price = round(entry_price - (2.5 * atr_value), 2)  # Below entry

tp1_qty = round(position_size * 0.5, 3)  # 50%
tp2_qty = round(position_size * 0.3, 3)  # 30%

print("=" * 80)
print("🔧 FIXING DASHUSDT SHORT POSITION")
print("=" * 80)
print()
print(f"Symbol: {symbol}")
print(f"Position: SHORT {position_size} DASH")
print(f"Entry: ${entry_price}")
print()
print("📋 CORRECTED ORDERS:")
print(f"1. Cancel wrong TP at $66.13")
print(f"2. Set SL: ${sl_price} (STOP_MARKET, full position)")
print(f"3. Set TP1: ${tp1_price} (TAKE_PROFIT_MARKET, {tp1_qty} DASH - 50%)")
print(f"4. Set TP2: ${tp2_price} (TAKE_PROFIT_MARKET, {tp2_qty} DASH - 30%)")
print()
print("=" * 80)
print("⚠️  MANUAL EXECUTION REQUIRED")
print("=" * 80)
print()
print("Gå til Binance Futures og:")
print()
print("1. CANCEL eksisterende TP order ($66.13)")
print()
print("2. SET STOP LOSS:")
print(f"   Type: STOP MARKET")
print(f"   Side: BUY (to close SHORT)")
print(f"   Stop Price: ${sl_price}")
print(f"   Quantity: {position_size} DASH")
print(f"   Reduce Only: YES")
print()
print("3. SET TAKE PROFIT 1:")
print(f"   Type: TAKE PROFIT MARKET")
print(f"   Side: BUY (to close SHORT)")
print(f"   Stop Price: ${tp1_price}")
print(f"   Quantity: {tp1_qty} DASH")
print(f"   Reduce Only: YES")
print()
print("4. SET TAKE PROFIT 2:")
print(f"   Type: TAKE PROFIT MARKET")
print(f"   Side: BUY (to close SHORT)")
print(f"   Stop Price: ${tp2_price}")
print(f"   Quantity: {tp2_qty} DASH")
print(f"   Reduce Only: YES")
print()
print("=" * 80)
print("✅ AFTER FIX:")
print("=" * 80)
print(f"Entry: ${entry_price}")
print(f"SL:    ${sl_price} (+{((sl_price/entry_price - 1) * 100):.1f}%) - Max loss ~2%")
print(f"TP1:   ${tp1_price} ({((tp1_price/entry_price - 1) * 100):.1f}%) - Take 50% profit")
print(f"TP2:   ${tp2_price} ({((tp2_price/entry_price - 1) * 100):.1f}%) - Take 30% profit")
print()
print("R:R Ratio: 1:1.5 (TP1) and 1:2.5 (TP2) ✅")
print()

# Uncomment below to execute automatically (CAREFUL!)
# print("\n🤖 AUTO-EXECUTION (commented out for safety)")
# print("Uncomment the code below to execute automatically\n")

"""
try:
    # 1. Cancel existing TP orders
    print("Cancelling all existing orders...")
    client.futures_cancel_all_open_orders(symbol=symbol)
    print("✅ Cancelled\n")
    
    # 2. Place Stop Loss
    print(f"Placing SL at ${sl_price}...")
    sl_order = client.futures_create_order(
        symbol=symbol,
        side='BUY',
        type='STOP_MARKET',
        stopPrice=sl_price,
        quantity=position_size,
        reduceOnly=True
    )
    print(f"✅ SL placed: {sl_order['orderId']}\n")
    
    # 3. Place TP1
    print(f"Placing TP1 at ${tp1_price}...")
    tp1_order = client.futures_create_order(
        symbol=symbol,
        side='BUY',
        type='TAKE_PROFIT_MARKET',
        stopPrice=tp1_price,
        quantity=tp1_qty,
        reduceOnly=True
    )
    print(f"✅ TP1 placed: {tp1_order['orderId']}\n")
    
    # 4. Place TP2
    print(f"Placing TP2 at ${tp2_price}...")
    tp2_order = client.futures_create_order(
        symbol=symbol,
        side='BUY',
        type='TAKE_PROFIT_MARKET',
        stopPrice=tp2_price,
        quantity=tp2_qty,
        reduceOnly=True
    )
    print(f"✅ TP2 placed: {tp2_order['orderId']}\n")
    
    print("=" * 80)
    print("✅ ALL ORDERS PLACED SUCCESSFULLY!")
    print("=" * 80)
    
except Exception as e:
    print(f"❌ Error: {e}")
"""
