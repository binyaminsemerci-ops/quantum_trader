#!/usr/bin/env python3
"""Verify executed testnet order"""

from binance.client import Client
import os
import json
from datetime import datetime

api_key = os.getenv('BINANCE_API_KEY')
api_secret = os.getenv('BINANCE_API_SECRET')
client = Client(api_key, api_secret, testnet=True)

print("\n" + "="*70)
print("🔍 BINANCE TESTNET ORDER VERIFICATION")
print("="*70 + "\n")

# Get order details
order_id = 10937490288

try:
    order = client.futures_get_order(symbol='BTCUSDT', orderId=order_id)
    
    print(f"📋 Order Details:")
    print(f"   Order ID: {order['orderId']}")
    print(f"   Symbol: {order['symbol']}")
    print(f"   Status: {order['status']}")
    print(f"   Side: {order['side']}")
    print(f"   Type: {order['type']}")
    print(f"   Quantity: {order['origQty']} BTC")
    print(f"   Executed: {order['executedQty']} BTC")
    print(f"   Average Price: ${float(order['avgPrice']):,.2f}")
    
    # Calculate notional value
    notional = float(order['executedQty']) * float(order['avgPrice'])
    print(f"   Notional Value: ${notional:.2f}")
    
    # Time
    timestamp = int(order['updateTime']) / 1000
    dt = datetime.fromtimestamp(timestamp)
    print(f"   Execution Time: {dt.strftime('%Y-%m-%d %H:%M:%S')} UTC")
    
    print(f"\n✅ Order Status: {order['status']}")
    
    if order['status'] == 'FILLED':
        print("🎉 Order FILLED and CONFIRMED!\n")
    
    # Get current position
    print("="*70)
    print("📊 CURRENT POSITION")
    print("="*70 + "\n")
    
    positions = client.futures_position_information(symbol='BTCUSDT')
    for pos in positions:
        if float(pos['positionAmt']) != 0:
            print(f"   Symbol: {pos['symbol']}")
            print(f"   Position: {pos['positionAmt']} BTC")
            print(f"   Entry Price: ${float(pos['entryPrice']):,.2f}")
            print(f"   Unrealized P&L: ${float(pos['unRealizedProfit']):.2f}")
            print(f"   Leverage: {pos['leverage']}x")
            print(f"   Margin Type: {pos['marginType']}")
            
    # Get account balance
    print("\n" + "="*70)
    print("💰 ACCOUNT BALANCE")
    print("="*70 + "\n")
    
    account = client.futures_account()
    print(f"   Total Balance: ${float(account['totalWalletBalance']):,.2f} USDT")
    print(f"   Available: ${float(account['availableBalance']):,.2f} USDT")
    print(f"   Used Margin: ${float(account['totalInitialMargin']):,.2f} USDT")
    
    print("\n" + "="*70)
    print("✅ VERIFICATION COMPLETE")
    print("="*70 + "\n")
    
except Exception as e:
    print(f"❌ Error verifying order: {e}\n")
