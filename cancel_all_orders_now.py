#!/usr/bin/env python3
"""Cancel ALL open orders on Binance Futures"""
import os
from dotenv import load_dotenv
from binance.client import Client

load_dotenv()

client = Client(
    os.getenv("BINANCE_API_KEY"),
    os.getenv("BINANCE_API_SECRET")
)

print("[SEARCH] Fetching all open orders...")
open_orders = client.futures_get_open_orders()

if not open_orders:
    print("[OK] No open orders to cancel")
else:
    print(f"\n[WARNING]  Found {len(open_orders)} open orders:")
    for order in open_orders:
        print(f"  {order['symbol']}: {order['side']} {order['origQty']} @ {order['price']}")
    
    print(f"\n🗑️  Cancelling all {len(open_orders)} orders...")
    for order in open_orders:
        try:
            client.futures_cancel_order(
                symbol=order['symbol'],
                orderId=order['orderId']
            )
            print(f"  [OK] Cancelled {order['symbol']} order #{order['orderId']}")
        except Exception as e:
            print(f"  ❌ Failed to cancel {order['symbol']}: {e}")
    
    print(f"\n[OK] Done! Cancelled {len(open_orders)} orders")
