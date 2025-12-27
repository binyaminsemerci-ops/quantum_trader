#!/usr/bin/env python3
"""
Show ALL open orders on Binance Futures
"""
import os
from binance.client import Client

api_key = os.getenv("BINANCE_API_KEY")
api_secret = os.getenv("BINANCE_API_SECRET")

client = Client(api_key, api_secret)

print("\n" + "="*90)
print("[CLIPBOARD] ALL OPEN ORDERS ON BINANCE FUTURES")
print("="*90 + "\n")

try:
    # Get ALL open orders (all symbols)
    orders = client.futures_get_open_orders()
    
    if not orders:
        print("[ALERT] NO OPEN ORDERS FOUND!")
        print("\n[WARNING]  This means TP/SL orders are NOT on Binance!")
        print("[WARNING]  Position monitor may think they exist but they don't!\n")
    else:
        print(f"[OK] Found {len(orders)} open orders:\n")
        
        # Group by symbol
        by_symbol = {}
        for order in orders:
            symbol = order['symbol']
            if symbol not in by_symbol:
                by_symbol[symbol] = []
            by_symbol[symbol].append(order)
        
        for symbol, symbol_orders in by_symbol.items():
            print(f"[CHART] {symbol}: {len(symbol_orders)} orders")
            for order in symbol_orders:
                order_type = order['type']
                side = order['side']
                qty = order['origQty']
                stop_price = order.get('stopPrice', 'N/A')
                price = order.get('price', 'MARKET')
                close_position = order.get('closePosition', False)
                
                print(f"   • {order_type:25s} {side:4s} {qty:>15s}")
                if stop_price != 'N/A':
                    print(f"     Stop: ${stop_price}")
                if price != 'MARKET':
                    print(f"     Price: ${price}")
                if close_position:
                    print(f"     [WARNING]  STOP_LOSS - Closes entire position!")
                    
            print()
    
    print("="*90 + "\n")
    
except Exception as e:
    print(f"❌ Error: {e}")
    import traceback
    traceback.print_exc()
