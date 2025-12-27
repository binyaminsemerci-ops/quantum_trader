"""
Quick check of live orders for current positions
"""
import os
from binance.client import Client

api_key = os.getenv("BINANCE_API_KEY")
api_secret = os.getenv("BINANCE_API_SECRET")

client = Client(api_key, api_secret)

symbols = ["AVAXUSDT", "ADAUSDT", "BTCUSDT", "SOLUSDT", "DOTUSDT"]

print("\n" + "="*80)
print("LIVE ORDERS FOR CURRENT POSITIONS")
print("="*80)

for symbol in symbols:
    try:
        orders = client.futures_get_open_orders(symbol=symbol)
        print(f"\n{symbol} ({len(orders)} orders):")
        
        if not orders:
            print("  ⚠️  NO ORDERS - UNPROTECTED!")
            continue
        
        for order in orders:
            order_type = order['type']
            side = order['side']
            qty = float(order.get('origQty', 0))
            
            # Extract price info
            if order_type == 'TRAILING_STOP_MARKET':
                callback = order.get('callbackRate', 'N/A')
                activation = order.get('activatePrice', 'N/A')
                print(f"  ✅ {order_type} ({side})")
                print(f"     Callback: {callback}% | Activation: ${activation}")
            elif 'STOP' in order_type:
                stop_price = order.get('stopPrice', 'N/A')
                close_position = order.get('closePosition', False)
                print(f"  ✅ {order_type} ({side}) @ ${stop_price}")
                print(f"     Close Position: {close_position}")
            elif 'TAKE_PROFIT' in order_type or order_type == 'LIMIT':
                price = order.get('stopPrice') or order.get('price', 'N/A')
                print(f"  ✅ {order_type} ({side}) @ ${price}")
                print(f"     Quantity: {qty}")
        
    except Exception as e:
        print(f"  ❌ Error: {e}")

print("\n" + "="*80)
print("\nLEGEND:")
print("  TRAILING_STOP_MARKET = Dynamic profit protection (follows price)")
print("  STOP_MARKET = Fixed stop loss (protects against losses)")
print("  TAKE_PROFIT_MARKET/LIMIT = Fixed take profit targets")
print("="*80 + "\n")
