#!/usr/bin/env python3
import os
import sys
sys.path.insert(0, "/home/qt/quantum_trader")
from binance.client import Client

# Load manually since dotenv not installed
api_key = "e9ZqWhGhAEhDPfNBfQMiJv8zULKJZBIwaaJdfbbUQ8ZNj1WUMumrjenHoRzpzUPD"
api_secret = "ZowBZEfL1R1ValcYLkbxjMfZ1tOxfEDRW4eloWRGGjk5etn0vSFFSU3gCTdCFoja"

# Load manually since dotenv not installed
api_key = "e9ZqWhGhAEhDPfNBfQMiJv8zULKJZBIwaaJdfbbUQ8ZNj1WUMumrjenHoRzpzUPD"
api_secret = "ZowBZEfL1R1ValcYLkbxjMfZ1tOxfEDRW4eloWRGGjk5etn0vSFFSU3gCTdCFoja"

client = Client(
    api_key=api_key,
    api_secret=api_secret,
    testnet=True
)

# Close all open positions
print("🔍 Fetching open positions...")
account = client.futures_account()
positions = [p for p in account["positions"] if float(p["positionAmt"]) != 0]

print(f"📊 Found {len(positions)} open positions\n")

for pos in positions:
    symbol = pos["symbol"]
    amt = float(pos["positionAmt"])
    side = "SELL" if amt > 0 else "BUY"
    qty = abs(amt)
    
    print(f"📍 Closing {symbol}: {side} {qty}")
    try:
        result = client.futures_create_order(
            symbol=symbol,
            side=side,
            type="MARKET",
            quantity=qty,
            reduceOnly=True
        )
        print(f"   ✅ Closed (Order ID: {result['orderId']})")
    except Exception as e:
        print(f"   ❌ Error: {e}")

# Cancel all open orders
print("\n🔍 Fetching open orders...")
orders = client.futures_get_open_orders()

print(f"📊 Found {len(orders)} open orders\n")

for order in orders:
    symbol = order["symbol"]
    order_id = order["orderId"]
    
    print(f"🗑️  Canceling {symbol} order {order_id}")
    try:
        client.futures_cancel_order(symbol=symbol, orderId=order_id)
        print("   ✅ Canceled")
    except Exception as e:
        print(f"   ❌ Error: {e}")

print("\n✅ Done! Testnet cleaned.")
