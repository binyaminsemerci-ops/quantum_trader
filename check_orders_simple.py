from binance.client import Client
import os

c = Client(os.getenv('BINANCE_API_KEY'), os.getenv('BINANCE_API_SECRET'), testnet=True)
orders = c.futures_get_open_orders()

print(f"\n=== TOTAL OPEN ORDERS: {len(orders)} ===\n")

if len(orders) == 0:
    print("❌ INGEN ORDRER PÅ BINANCE!")
    print("Dette betyr at Exit Brain ikke har plassert ordrer, eller at de ble cancelled.")
else:
    for o in orders:
        print(f"{o['symbol']:12} {o['type']:15} {o['positionSide']:5} @ {o.get('stopPrice', o.get('price', 'N/A'))}")
