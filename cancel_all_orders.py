"""Cancel all open orders on Binance Futures"""
import os
from dotenv import load_dotenv
from binance.client import Client

load_dotenv()

api_key = os.getenv("BINANCE_API_KEY")
api_secret = os.getenv("BINANCE_API_SECRET")

client = Client(api_key, api_secret)

print("\n🗑️ CANCELLING ALL OPEN ORDERS...")
print("=" * 80)

# Get all open orders
open_orders = client.futures_get_open_orders()

if not open_orders:
    print("[OK] No open orders to cancel")
else:
    print(f"Found {len(open_orders)} open orders\n")
    
    for order in open_orders:
        symbol = order['symbol']
        order_id = order['orderId']
        order_type = order['type']
        side = order['side']
        
        try:
            client.futures_cancel_order(symbol=symbol, orderId=order_id)
            print(f"[OK] Cancelled {symbol} {order_type} {side} (ID: {order_id})")
        except Exception as e:
            print(f"❌ Failed to cancel {symbol} order: {e}")

print("\n" + "=" * 80)
print("[OK] All orders cancelled!")
