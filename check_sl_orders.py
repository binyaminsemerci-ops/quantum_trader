import os
from binance.um_futures import UMFutures

c = UMFutures(key=os.getenv('BINANCE_API_KEY'), secret=os.getenv('BINANCE_API_SECRET'))

print("\n" + "="*80)
print("[SEARCH] CHECKING STOP LOSS ORDERS ON BINANCE")
print("="*80 + "\n")

for symbol in ['AIUSDT', 'APTUSDT']:
    print(f"[CHART] {symbol}:")
    try:
        orders = c.get_orders(symbol=symbol)
        sl_orders = [o for o in orders if 'STOP' in o['type']]
        
        if sl_orders:
            print(f"   [OK] {len(sl_orders)} Stop Loss Order(s) Found:")
            for o in sl_orders:
                order_type = o['type']
                stop_price = o.get('stopPrice', 'N/A')
                price = o.get('price', 'N/A')
                tif = o.get('timeInForce', 'N/A')
                
                print(f"      • Type: {order_type}")
                print(f"      • Stop Price: ${stop_price}")
                print(f"      • Limit Price: ${price}")
                print(f"      • Time In Force: {tif}")
                
                if price != 'N/A' and stop_price != 'N/A':
                    print(f"      [OK] STOP_LOSS with limit = GUARANTEED EXECUTION!")
                else:
                    print(f"      [WARNING]  STOP_MARKET = Can be skipped!")
        else:
            print(f"   [ALERT] NO STOP LOSS ORDERS FOUND!")
            print(f"      Position is UNPROTECTED!")
    except Exception as e:
        print(f"   ❌ Error: {e}")
    print()

print("="*80)
