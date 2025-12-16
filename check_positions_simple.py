#!/usr/bin/env python3
import os
import requests

api_key = os.getenv('BINANCE_API_KEY')
base_url = 'https://testnet.binancefuture.com'
headers = {'X-MBX-APIKEY': api_key}

response = requests.get(f'{base_url}/fapi/v2/positionRisk', headers=headers)
positions = [p for p in response.json() if float(p.get('positionAmt', 0)) != 0]

print(f'\n📊 OPEN POSITIONS: {len(positions)}\n')

for pos in positions:
    symbol = pos['symbol']
    entry = float(pos['entryPrice'])
    mark = float(pos['markPrice'])
    amt = float(pos['positionAmt'])
    pnl = float(pos['unRealizedProfit'])
    pnl_pct = (pnl / (abs(amt) * entry)) * 100 if entry > 0 else 0
    price_move = ((mark - entry) / entry) * 100
    
    side = 'LONG' if amt > 0 else 'SHORT'
    
    print(f'{symbol} {side}:')
    print(f'  Entry: ${entry:.4f}, Mark: ${mark:.4f}')
    print(f'  Price move: {price_move:+.2f}%')
    print(f'  PnL: ${pnl:.2f} ({pnl_pct:+.2f}%)')
    print()
