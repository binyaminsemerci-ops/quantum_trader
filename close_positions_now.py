#!/usr/bin/env python3
"""
Emergency script to close all positions and cancel all orders
"""
import os
from binance.client import Client
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

api_key = os.getenv('BINANCE_API_KEY') or os.getenv('BINANCE_TESTNET_API_KEY')
api_secret = os.getenv('BINANCE_API_SECRET') or os.getenv('BINANCE_TESTNET_SECRET_KEY')

if not api_key or not api_secret:
    print("❌ ERROR: API keys not found in .env file")
    exit(1)

client = Client(api_key, api_secret, testnet=True)

print("\n" + "="*60)
print("🗑️  EMERGENCY CLOSE - Cancelling orders and closing positions")
print("="*60 + "\n")

# Step 1: Cancel all open orders
print("📋 Step 1: Cancelling all open orders...")
try:
    open_orders = client.futures_get_open_orders()
    print(f"   Found {len(open_orders)} open orders")
    
    for order in open_orders:
        try:
            result = client.futures_cancel_order(
                symbol=order['symbol'], 
                orderId=order['orderId']
            )
            print(f"   ✅ Cancelled {order['symbol']} order #{order['orderId']}")
        except Exception as e:
            print(f"   ❌ Error cancelling {order['symbol']}: {e}")
    
    print(f"\n✅ Cancelled {len(open_orders)} orders\n")
except Exception as e:
    print(f"❌ Error getting open orders: {e}\n")

# Step 2: Close all positions
print("📦 Step 2: Closing all positions...")
try:
    positions = client.futures_position_information()
    open_positions = [p for p in positions if float(p['positionAmt']) != 0]
    print(f"   Found {len(open_positions)} open positions")
    
    for pos in open_positions:
        amt = float(pos['positionAmt'])
        symbol = pos['symbol']
        side = 'SELL' if amt > 0 else 'BUY'
        
        try:
            result = client.futures_create_order(
                symbol=symbol,
                side=side,
                type='MARKET',
                quantity=abs(amt),
                reduceOnly=True
            )
            print(f"   ✅ Closed {symbol}: {amt} units (Order #{result['orderId']})")
        except Exception as e:
            print(f"   ❌ Error closing {symbol}: {e}")
    
    print(f"\n✅ Closed {len(open_positions)} positions\n")
except Exception as e:
    print(f"❌ Error getting positions: {e}\n")

print("="*60)
print("✅ EMERGENCY CLOSE COMPLETE")
print("="*60)
