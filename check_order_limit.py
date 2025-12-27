"""Check Binance stop order limit issue"""
from dotenv import load_dotenv
load_dotenv()

from binance.client import Client
import os

c = Client(os.getenv('BINANCE_API_KEY'), os.getenv('BINANCE_API_SECRET'), testnet=True)

print('=' * 80)
print('CHECKING BINANCE ORDER LIMITS')
print('=' * 80)

# Get ALL orders
try:
    all_orders = c.futures_get_all_orders()
    print(f'\n✅ Total orders in history: {len(all_orders)}')
except Exception as e:
    print(f'\n❌ Failed to get all orders: {e}')
    all_orders = []

# Check open orders
open_orders = c.futures_get_open_orders()
print(f'✅ Current open orders: {len(open_orders)}')

# Filter stop orders
stop_orders = [o for o in all_orders if 'STOP' in o.get('type', '') or 'TAKE_PROFIT' in o.get('type', '')]
open_stop_orders = [o for o in open_orders if 'STOP' in o.get('type', '') or 'TAKE_PROFIT' in o.get('type', '')]

print(f'\n📊 STOP/TP ORDERS:')
print(f'  Total in history: {len(stop_orders)}')
print(f'  Currently open: {len(open_stop_orders)}')

# Show recent stop orders
print(f'\n📋 RECENT STOP ORDERS (last 20):')
for order in stop_orders[-20:]:
    symbol = order['symbol']
    order_type = order['type']
    status = order['status']
    print(f'  {symbol:12} {order_type:25} Status: {status}')

print(f'\n🚨 BINANCE LIMITS:')
print(f'  Testnet typically allows 10-20 stop orders per symbol')
print(f'  You have {len(open_stop_orders)} open stop orders')
print(f'\n💡 SOLUTION: Cancel old/stale stop orders to free up limit')
