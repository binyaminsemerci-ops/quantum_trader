#!/usr/bin/env python3
"""Quick script to check Binance Futures Testnet balance and PnL"""

import os
import time
import hmac
import hashlib
import requests
from datetime import datetime

# Get credentials from environment
api_key = os.getenv('BINANCE_API_KEY')
api_secret = os.getenv('BINANCE_API_SECRET')

if not api_key or not api_secret:
    print("❌ Error: BINANCE_API_KEY or BINANCE_API_SECRET not found in environment")
    exit(1)

# Testnet URL
base_url = "https://testnet.binancefuture.com"

# Create signature
timestamp = int(time.time() * 1000)
params = f'timestamp={timestamp}'
signature = hmac.new(api_secret.encode(), params.encode(), hashlib.sha256).hexdigest()

# Request headers
headers = {'X-MBX-APIKEY': api_key}

# Get account info
url = f'{base_url}/fapi/v2/account?{params}&signature={signature}'
response = requests.get(url, headers=headers)

if response.status_code != 200:
    print(f"❌ API Error: {response.status_code}")
    print(response.text)
    exit(1)

data = response.json()

# Display account info
print("\n" + "="*70)
print("  📊 BINANCE FUTURES TESTNET - ACCOUNT STATUS")
print("="*70 + "\n")

total_balance = float(data.get('totalWalletBalance', 0))
unrealized_pnl = float(data.get('totalUnrealizedProfit', 0))
available_balance = float(data.get('availableBalance', 0))

print(f"💰 Total Balance:      ${total_balance:,.2f} USDT")
print(f"📈 Unrealized PnL:     ${unrealized_pnl:,.2f} USDT")
print(f"✅ Available Balance:  ${available_balance:,.2f} USDT")

# Get open positions
positions = [p for p in data.get('positions', []) if float(p.get('positionAmt', 0)) != 0]

print(f"\n📌 Open Positions: {len(positions)}")
print("-"*70)

if positions:
    for p in positions:
        symbol = p['symbol']
        amt = float(p['positionAmt'])
        entry_price = float(p['entryPrice'])
        mark_price = float(p['markPrice'])
        unrealized = float(p['unRealizedProfit'])
        leverage = p['leverage']
        side = "LONG" if amt > 0 else "SHORT"
        
        # Calculate PnL %
        if side == "LONG":
            pnl_pct = ((mark_price - entry_price) / entry_price) * 100
        else:
            pnl_pct = ((entry_price - mark_price) / entry_price) * 100
        
        pnl_emoji = "🟢" if unrealized >= 0 else "🔴"
        
        print(f"\n{pnl_emoji} {symbol} {side}")
        print(f"   Quantity: {abs(amt):.4f}")
        print(f"   Entry: ${entry_price:.4f}")
        print(f"   Mark:  ${mark_price:.4f}")
        print(f"   PnL:   ${unrealized:,.2f} ({pnl_pct:+.2f}%)")
        print(f"   Leverage: {leverage}x")
else:
    print("   No open positions")

print("\n" + "="*70 + "\n")
