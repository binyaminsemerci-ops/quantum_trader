#!/usr/bin/env python3
"""Check trade history and realized PnL on Binance Futures Testnet"""

import os
import time
import hmac
import hashlib
import requests
from datetime import datetime, timedelta

# Get credentials
api_key = os.getenv('BINANCE_API_KEY')
api_secret = os.getenv('BINANCE_API_SECRET')

if not api_key or not api_secret:
    print("❌ Error: Credentials not found")
    exit(1)

base_url = "https://testnet.binancefuture.com"

def sign_request(params):
    query_string = '&'.join([f"{k}={v}" for k, v in params.items()])
    signature = hmac.new(api_secret.encode(), query_string.encode(), hashlib.sha256).hexdigest()
    return query_string + f'&signature={signature}'

headers = {'X-MBX-APIKEY': api_key}

# Get today's trades (last 24 hours)
print("\n" + "="*70)
print("  📊 BINANCE FUTURES TESTNET - DAGENS TRADING RESULTAT")
print("="*70 + "\n")

# Get income history (realized PnL)
params = {
    'incomeType': 'REALIZED_PNL',
    'startTime': int((datetime.now() - timedelta(hours=24)).timestamp() * 1000),
    'timestamp': int(time.time() * 1000)
}

url = f'{base_url}/fapi/v1/income?' + sign_request(params)
response = requests.get(url, headers=headers)

if response.status_code == 200:
    income_data = response.json()
    
    total_realized_pnl = sum(float(item['income']) for item in income_data)
    
    print(f"💰 Realized PnL (siste 24t): ${total_realized_pnl:,.2f} USDT")
    print(f"📊 Antall trades: {len(income_data)}")
    
    if income_data:
        print("\n📜 Trade Historie:")
        print("-"*70)
        for item in income_data:
            symbol = item['symbol']
            income = float(item['income'])
            time_str = datetime.fromtimestamp(int(item['time'])/1000).strftime('%H:%M:%S')
            emoji = "🟢" if income >= 0 else "🔴"
            print(f"{emoji} {time_str} | {symbol:12} | PnL: ${income:+.2f}")
else:
    print(f"❌ Error fetching income: {response.status_code}")

# Get all trades today
params = {
    'startTime': int((datetime.now() - timedelta(hours=24)).timestamp() * 1000),
    'timestamp': int(time.time() * 1000)
}

print("\n" + "="*70)
print("  🔍 ALLE TRADES (siste 24t)")
print("="*70 + "\n")

# We need to check each symbol individually
symbols = ['SOLUSDT', 'BTCUSDT', 'COMPUSDT', 'ETHUSDT', 'BNBUSDT']
all_trades = []

for symbol in symbols:
    params_sym = params.copy()
    params_sym['symbol'] = symbol
    url = f'{base_url}/fapi/v1/userTrades?' + sign_request(params_sym)
    response = requests.get(url, headers=headers)
    
    if response.status_code == 200:
        trades = response.json()
        all_trades.extend(trades)

if all_trades:
    print(f"📊 Totalt {len(all_trades)} trades utført")
    for trade in sorted(all_trades, key=lambda x: x['time'])[-10:]:  # Last 10 trades
        symbol = trade['symbol']
        side = trade['side']
        price = float(trade['price'])
        qty = float(trade['qty'])
        realized_pnl = float(trade.get('realizedPnl', 0))
        time_str = datetime.fromtimestamp(int(trade['time'])/1000).strftime('%H:%M:%S')
        emoji = "📈" if side == "BUY" else "📉"
        
        print(f"{emoji} {time_str} | {symbol:12} {side:5} {qty:.4f} @ ${price:.2f} | PnL: ${realized_pnl:+.2f}")
else:
    print("Ingen trades funnet i dag")

print("\n" + "="*70 + "\n")
