"""
Cancel ALL open orders on Binance Futures Testnet
"""
import os
from binance.client import Client
from dotenv import load_dotenv

load_dotenv()

api_key = os.getenv('BINANCE_TESTNET_API_KEY')
api_secret = os.getenv('BINANCE_TESTNET_SECRET_KEY')

client = Client(api_key, api_secret, testnet=True)

print("🔍 Fetching all open orders...")
open_orders = client.futures_get_open_orders()

if not open_orders:
    print("✅ No open orders found!")
else:
    print(f"📋 Found {len(open_orders)} open orders:")
    for order in open_orders:
        print(f"   {order['symbol']}: {order['type']} {order['side']} - OrderID: {order['orderId']}")
    
    print("\n🗑️  Canceling all orders...")
    for order in open_orders:
        try:
            result = client.futures_cancel_order(
                symbol=order['symbol'],
                orderId=order['orderId']
            )
            print(f"   ✅ Cancelled {order['symbol']} order {order['orderId']}")
        except Exception as e:
            print(f"   ❌ Failed to cancel {order['symbol']}: {e}")
    
    print("\n✅ All orders cancelled!")
