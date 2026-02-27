#!/usr/bin/env python3
import requests
import hmac
import hashlib
import time
from urllib.parse import urlencode

print('=== TESTING APPLY-LAYER CREDENTIALS ===')

# Apply layer credentials from apply-layer.env
api_key = 'w2W60kzuCfPJKGIqSvmp0pqISUO8XKICjc5sD8QyJuJpp9LKQgvXKhtd09Ii3rwg'
api_secret = 'QI18cg4zcbApc9uaDL8ZUmoAJQthQczZ9cKzORlSJfnK2zBEdLvSLb5ZEgZ6R1Kg'

base_url = 'https://testnet.binancefuture.com'

def sign_request(params, secret):
    query_string = urlencode(sorted(params.items()))
    return hmac.new(secret.encode(), query_string.encode(), hashlib.sha256).hexdigest()

try:
    # Test account info endpoint
    timestamp = int(time.time() * 1000)
    params = {'timestamp': timestamp}
    signature = sign_request(params, api_secret)
    params['signature'] = signature
    
    headers = {'X-MBX-APIKEY': api_key}
    url = f'{base_url}/fapi/v2/account'
    
    print(f'Testing URL: {url}')
    print(f'API Key: {api_key[:20]}...')
    
    response = requests.get(url, headers=headers, params=params, timeout=10)
    
    print(f'Response Status: {response.status_code}')
    
    if response.status_code == 200:
        data = response.json()
        print('✅ SUCCESS: Apply-layer credentials work!')
        print(f'Total Balance: {data.get("totalWalletBalance", "N/A")} USDT')
        print(f'Available Balance: {data.get("availableBalance", "N/A")} USDT')
        print(f'Total Position Margin: {data.get("totalPositionInitialMargin", "N/A")} USDT')
        
        # Also test position info
        print('\n--- Testing Position Info ---')
        pos_url = f'{base_url}/fapi/v2/positionRisk'
        pos_response = requests.get(pos_url, headers=headers, params=params, timeout=10)
        if pos_response.status_code == 200:
            positions = pos_response.json()
            active_positions = [p for p in positions if float(p.get('positionAmt', 0)) != 0]
            print(f'Active Positions: {len(active_positions)}')
            for pos in active_positions[:3]:  # Show first 3
                symbol = pos.get('symbol', 'Unknown')
                size = pos.get('positionAmt', 'N/A')
                pnl = pos.get('unRealizedProfit', 'N/A')
                print(f'  {symbol}: {size}, PnL: {pnl}')
        else:
            print(f'Position request failed: {pos_response.status_code}')
        
    else:
        print(f'❌ API Error {response.status_code}: {response.text}')
        
except Exception as e:
    print(f'❌ Connection Error: {e}')
    import traceback
    traceback.print_exc()