"""Check Binance open orders directly."""
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
    
    if isinstance(orders, list):
        if len(orders) == 0:
            print('❌ INGEN ÅPNE ORDRER!')
            print('[WARNING]  DETTE ER PROBLEMET: Position monitor sier "protected",')
            print('[WARNING]  men ordrene eksisterer IKKE på Binance!')
        else:
            for order in orders:
                symbol = order.get('symbol')
                order_type = order.get('type')
                side = order.get('side')
                stop_price = order.get('stopPrice', 'N/A')
                price = order.get('price', 'N/A')
                
                print(f'{symbol:12s} | {order_type:20s} | {side:4s}')
                print(f'  Stop: ${stop_price}, Limit: ${price}')
    else:
        print(f'❌ API Error: {orders}')
    
    print('\n' + '═' * 80)
    
except Exception as e:
    print(f'❌ Feil: {e}')
