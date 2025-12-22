#!/usr/bin/env python3
"""Test Binance position mode and find the actual issue"""

from binance.client import Client
import os

def main():
    client = Client(
        os.getenv('BINANCE_API_KEY'),
        os.getenv('BINANCE_API_SECRET'),
        testnet=True
    )
    
    print("=" * 60)
    print("BINANCE POSITION MODE DIAGNOSTIC")
    print("=" * 60)
    
    # 1. Check position mode setting
    try:
        mode = client.futures_get_position_mode()
        print(f"\n1. Position Mode Setting:")
        print(f"   dualSidePosition: {mode.get('dualSidePosition')}")
        if mode.get('dualSidePosition'):
            print("   → HEDGE MODE (supports LONG/SHORT positionSide)")
        else:
            print("   → ONE-WAY MODE (does NOT support positionSide parameter)")
    except Exception as e:
        print(f"   ERROR: {e}")
    
    # 2. Check actual positions
    try:
        positions = client.futures_position_information()
        active = [p for p in positions if float(p['positionAmt']) != 0]
        print(f"\n2. Active Positions: {len(active)}")
        for p in active[:5]:
            print(f"   {p['symbol']:10s} | amt={p['positionAmt']:>12s} | side={p['positionSide']:5s}")
    except Exception as e:
        print(f"   ERROR: {e}")
    
    # 3. Test TP order WITHOUT positionSide (One-Way Mode)
    print(f"\n3. Testing TP order submission (One-Way Mode):")
    if active:
        test_pos = active[0]
        symbol = test_pos['symbol']
        amt = abs(float(test_pos['positionAmt']))
        entry = float(test_pos['entryPrice'])
        
        # Calculate TP price (3% profit)
        if float(test_pos['positionAmt']) > 0:  # LONG
            tp_price = entry * 1.03
            side = 'SELL'
        else:  # SHORT
            tp_price = entry * 0.97
            side = 'BUY'
        
        print(f"   Symbol: {symbol}")
        print(f"   Side: {side}")
        print(f"   TP Price: {tp_price}")
        print(f"   Quantity: {amt}")
        print(f"\n   Attempting order WITHOUT positionSide parameter...")
        
        try:
            # This is how it SHOULD be done in One-Way Mode
            order = client.futures_create_order(
                symbol=symbol,
                side=side,
                type='TAKE_PROFIT_MARKET',
                stopPrice=tp_price,
                closePosition=True,  # Use this instead of quantity in One-Way
                workingType='MARK_PRICE'
            )
            print(f"   ✅ SUCCESS! Order ID: {order['orderId']}")
            print(f"   → One-Way Mode confirmed - DO NOT send positionSide")
        except Exception as e:
            print(f"   ❌ FAILED: {e}")
    
    print("\n" + "=" * 60)
    print("CONCLUSION:")
    print("=" * 60)
    if not mode.get('dualSidePosition'):
        print("✅ Account is in ONE-WAY MODE")
        print("❌ Position Monitor is sending positionSide=LONG/SHORT")
        print("🔧 FIX: Remove positionSide parameter from orders")
        print("💡 OR: Use closePosition=True instead of quantity")
    else:
        print("✅ Account is in HEDGE MODE")
        print("✅ positionSide parameter is correct")
        print("❓ Need to investigate other issues")

if __name__ == '__main__':
    main()
