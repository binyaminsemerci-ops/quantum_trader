#!/usr/bin/env python3
"""Check SPOT trades on Binance Testnet"""

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

# SPOT TESTNET URL
base_url = "https://testnet.binance.vision"

def sign_request(params):
    query_string = '&'.join([f"{k}={v}" for k, v in params.items()])
    signature = hmac.new(api_secret.encode(), query_string.encode(), hashlib.sha256).hexdigest()
    return query_string + f'&signature={signature}'

headers = {'X-MBX-APIKEY': api_key}

print("\n" + "="*70)
print("  📊 BINANCE SPOT TESTNET - TRADE HISTORY")
print("="*70 + "\n")

# Get account info first
params = {
    'timestamp': int(time.time() * 1000)
}

url = f'{base_url}/api/v3/account?' + sign_request(params)
response = requests.get(url, headers=headers)

if response.status_code == 200:
    account_data = response.json()
    
    print("💰 SPOT BALANCES:")
    print("-"*70)
    
    balances = [b for b in account_data.get('balances', []) if float(b['free']) > 0 or float(b['locked']) > 0]
    
    if balances:
        for b in balances:
            asset = b['asset']
            free = float(b['free'])
            locked = float(b['locked'])
            total = free + locked
            print(f"  {asset:8} | Free: {free:15.8f} | Locked: {locked:15.8f} | Total: {total:15.8f}")
    else:
        print("  Ingen balances funnet")
    
    print("\n" + "="*70)
    print("  📜 SPOT TRADE HISTORY (siste 24 timer)")
    print("="*70 + "\n")
    
    # Try to get trades for common symbols
    symbols = ['BTCUSDT', 'ETHUSDT', 'BNBUSDT', 'SOLUSDT', 'ADAUSDT', 'DOGEUSDT', 'XRPUSDT']
    all_trades = []
    
    for symbol in symbols:
        params_sym = {
            'symbol': symbol,
            'startTime': int((datetime.now() - timedelta(hours=24)).timestamp() * 1000),
            'timestamp': int(time.time() * 1000)
        }
        
        url = f'{base_url}/api/v3/myTrades?' + sign_request(params_sym)
        response = requests.get(url, headers=headers)
        
        if response.status_code == 200:
            trades = response.json()
            all_trades.extend(trades)
    
    if all_trades:
        # Sort by time
        all_trades.sort(key=lambda x: x['time'])
        
        print(f"📊 Totalt {len(all_trades)} SPOT trades funnet!\n")
        
        for trade in all_trades:
            symbol = trade['symbol']
            side = "KJØP" if trade['isBuyer'] else "SELG"
            price = float(trade['price'])
            qty = float(trade['qty'])
            commission = float(trade['commission'])
            commission_asset = trade['commissionAsset']
            trade_time = datetime.fromtimestamp(int(trade['time'])/1000)
            time_str = trade_time.strftime('%Y-%m-%d %H:%M:%S')
            
            emoji = "🟢" if trade['isBuyer'] else "🔴"
            
            print(f"{emoji} {time_str} | {symbol:10} {side:5} | Qty: {qty:12.8f} @ ${price:.2f}")
            print(f"   → Commission: {commission:.8f} {commission_asset}")
            print()
        
        # Last trade info
        last_trade = all_trades[-1]
        last_time = datetime.fromtimestamp(int(last_trade['time'])/1000)
        
        print("="*70)
        print(f"⏰ SISTE TRADE: {last_time.strftime('%Y-%m-%d kl. %H:%M:%S')}")
        print(f"   Symbol: {last_trade['symbol']}")
        print(f"   Side: {'KJØP' if last_trade['isBuyer'] else 'SELG'}")
        print(f"   Pris: ${float(last_trade['price']):.2f}")
        print(f"   Mengde: {float(last_trade['qty']):.8f}")
        print("="*70)
        
    else:
        print("❌ Ingen SPOT trades funnet i siste 24 timer")
        print("    (Dette er normalt hvis du kun handler Futures)")
    
else:
    print(f"❌ API Error: {response.status_code}")
    print(response.text)

print("\n" + "="*70)
print("  ℹ️  VIKTIG INFORMASJON")
print("="*70)
print()
print("  Quantum Trader er konfigurert for FUTURES trading, ikke SPOT.")
print("  Hvis du ser SPOT trades på Binance Testnet, er disse sannsynligvis:")
print()
print("  1. ❌ IKKE gjort av denne AI-boten")
print("  2. 🤔 Gjort manuelt av deg")
print("  3. 🤔 Gjort av en annen bot/script")
print("  4. 🤔 Test-trades fra tidligere")
print()
print("  Denne boten bruker USDⓈ-M Perpetual Futures (30x leverage)")
print("  URL: https://testnet.binancefuture.com (ikke binance.vision)")
print()
print("="*70 + "\n")
