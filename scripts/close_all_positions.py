#!/usr/bin/env python3
"""Close all open positions on Binance Testnet"""

import os
from binance.client import Client

# Initialize Binance client
api_key = os.getenv('BINANCE_API_KEY')
api_secret = os.getenv('BINANCE_API_SECRET')

if not api_key or not api_secret:
    print("❌ BINANCE_API_KEY or BINANCE_API_SECRET not set")
    exit(1)

client = Client(api_key, api_secret, testnet=True)

print("📊 Fetching all positions...")
positions = client.futures_position_information()

active_positions = [p for p in positions if float(p['positionAmt']) != 0]

print(f"\n✅ Found {len(active_positions)} active positions\n")

if not active_positions:
    print("No positions to close.")
    exit(0)

print("Active Positions:")
print("-" * 80)
for p in active_positions:
    symbol = p['symbol']
    amt = float(p['positionAmt'])
    entry = float(p['entryPrice'])
    leverage = p['leverage']
    unrealized_pnl = float(p['unRealizedProfit'])
    
    side = "LONG" if amt > 0 else "SHORT"
    print(f"{symbol:12} {side:5} {abs(amt):10.4f} @ ${entry:10.4f} | {leverage:2}x | PnL: ${unrealized_pnl:8.2f}")

print("-" * 80)
print(f"\n🚨 Closing all {len(active_positions)} positions...\n")

for p in active_positions:
    symbol = p['symbol']
    amt = float(p['positionAmt'])
    
    if amt == 0:
        continue
    
    # Determine side to close and position side for hedge mode
    if amt > 0:
        side = 'SELL'
        position_side = 'LONG'
    else:
        side = 'BUY'
        position_side = 'SHORT'
    
    try:
        # Close position with market order (hedge mode compatible)
        order = client.futures_create_order(
            symbol=symbol,
            side=side,
            positionSide=position_side,
            type='MARKET',
            quantity=abs(amt)
        )
        print(f"✅ Closed {symbol} {position_side} ({side} {abs(amt)})")
    except Exception as e:
        print(f"❌ Failed to close {symbol} {position_side}: {e}")

print("\n✅ All positions closed!")
