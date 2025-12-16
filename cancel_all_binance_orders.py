"""Cancel all open Binance orders directly."""
import requests
import time
import hmac
import hashlib
import os

API_KEY = os.getenv('BINANCE_API_KEY')
API_SECRET = os.getenv('BINANCE_API_SECRET')

if not API_KEY or not API_SECRET:
    print('❌ Missing API credentials')
    exit(1)

timestamp = int(time.time() * 1000)
params = f'recvWindow=10000&timestamp={timestamp}'
signature = hmac.new(API_SECRET.encode(), params.encode(), hashlib.sha256).hexdigest()

url = f'https://fapi.binance.com/fapi/v1/openOrders?{params}&signature={signature}'
headers = {'X-MBX-APIKEY': API_KEY}

try:
    r = requests.get(url, headers=headers, timeout=10)
    orders = r.json()
    print('\n[CLIPBOARD] BINANCE OPEN ORDERS (DIREKTE FRA BINANCE):\n')
    print('═' * 80)
    if isinstance(orders, list) and len(orders) > 0:
        for order in orders:
            symbol = order.get('symbol')
            order_id = order.get('orderId')
            order_type = order.get('type')
            side = order.get('side')
            stop_price = order.get('stopPrice', 'N/A')
            price = order.get('price', 'N/A')
            print(f'Kansellerer: {symbol:12s} | {order_type:20s} | {side:4s} | ID: {order_id}')
            cancel_params = f'symbol={symbol}&orderId={order_id}&recvWindow=10000&timestamp={int(time.time() * 1000)}'
            cancel_signature = hmac.new(API_SECRET.encode(), cancel_params.encode(), hashlib.sha256).hexdigest()
            cancel_url = f'https://fapi.binance.com/fapi/v1/order?{cancel_params}&signature={cancel_signature}'
            cancel_resp = requests.delete(cancel_url, headers=headers, timeout=10)
            if cancel_resp.status_code == 200:
                print(f'[OK] Kansellert: {symbol} ID {order_id}')
            else:
                print(f'❌ Feil ved kansellering: {symbol} ID {order_id} - {cancel_resp.text}')
    else:
        print('❌ INGEN ÅPNE ORDRER!')
    print('\n' + '═' * 80)
except Exception as e:
    print(f'❌ Feil: {e}')
