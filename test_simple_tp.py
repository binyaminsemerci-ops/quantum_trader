"""Test simple TP order placement without positionSide"""
import os
from binance.client import Client

# Get credentials
api_key = os.getenv('BINANCE_TEST_API_KEY')
api_secret = os.getenv('BINANCE_TEST_API_SECRET')

client = Client(api_key, api_secret, testnet=True)

# Get position
positions = client.futures_position_information(symbol='SOLUSDT')
position = [p for p in positions if float(p['positionAmt']) != 0][0]

amt = float(position['positionAmt'])
entry = float(position['entryPrice'])

print(f"Position: {amt} @ ${entry}")

# Calculate TP (3% above entry for LONG)
tp_price = entry * 1.03
partial_qty = abs(amt) * 0.5

print(f"TP Price: ${tp_price:.2f}")
print(f"Partial Qty: {partial_qty}")

# Try placing TP WITHOUT positionSide
try:
    print("\n1️⃣ Placing TP WITHOUT positionSide...")
    order = client.futures_create_order(
        symbol='SOLUSDT',
        side='SELL',
        type='TAKE_PROFIT_MARKET',
        stopPrice=tp_price,
        quantity=partial_qty,
        workingType='MARK_PRICE'
    )
    print(f"✅ SUCCESS! Order ID: {order['orderId']}")
except Exception as e:
    print(f"❌ FAILED: {e}")

print("\nChecking orders...")
orders = client.futures_get_open_orders(symbol='SOLUSDT')
print(f"Open orders: {len(orders)}")
for o in orders:
    print(f"  - {o['type']} {o['side']} @ ${o.get('stopPrice', o.get('price', 'N/A'))}")
