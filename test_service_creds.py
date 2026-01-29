#!/usr/bin/env python3
import os
import sys

# Load service environment
with open('/etc/quantum/position-monitor.env') as f:
    for line in f:
        line = line.strip()
        if line and not line.startswith('#') and '=' in line:
            key, val = line.split('=', 1)
            os.environ[key] = val

from binance.client import Client

api_key = os.getenv('BINANCE_API_KEY')
api_secret = os.getenv('BINANCE_API_SECRET')

if not api_key or not api_secret:
    print('❌ Credentials not loaded')
    sys.exit(1)

print(f'Using API Key: {api_key[:10]}...')
print('')

# Test mainnet
print('Testing MAINNET...')
try:
    client = Client(api_key, api_secret, testnet=False)
    account = client.futures_account()
    balance = account.get('totalWalletBalance', 'N/A')
    print(f'✅ MAINNET WORKING')
    print(f'   Balance: {balance} USDT')
    
    positions = [p for p in client.futures_position_information() if float(p['positionAmt']) != 0]
    print(f'   Open positions: {len(positions)}')
    
    if positions:
        for p in positions[:5]:
            symbol = p['symbol']
            qty = float(p['positionAmt'])
            entry = float(p['entryPrice'])
            mark = float(p['markPrice'])
            pnl = float(p['unRealizedProfit'])
            print(f'     {symbol}: qty={qty:+.4f}, entry=${entry:.2f}, mark=${mark:.2f}, PnL=${pnl:.2f}')
    
    print('')
    print('✅ VERDICT: PASS - Exit Brain v3.5 can access Binance API')
    sys.exit(0)
    
except Exception as e:
    print(f'❌ MAINNET FAILED: {e}')
    print('')

# Try testnet
print('Testing TESTNET...')
try:
    client = Client(api_key, api_secret, testnet=True)
    account = client.futures_account()
    balance = account.get('totalWalletBalance', 'N/A')
    print(f'✅ TESTNET WORKING')
    print(f'   Balance: {balance} USDT')
    
    positions = [p for p in client.futures_position_information() if float(p['positionAmt']) != 0]
    print(f'   Open positions: {len(positions)}')
    
    if positions:
        for p in positions[:5]:
            symbol = p['symbol']
            qty = float(p['positionAmt'])
            entry = float(p['entryPrice'])
            mark = float(p['markPrice'])
            pnl = float(p['unRealizedProfit'])
            print(f'     {symbol}: qty={qty:+.4f}, entry=${entry:.2f}, mark=${mark:.2f}, PnL=${pnl:.2f}')
    
    print('')
    print('⚠️  VERDICT: PARTIAL - Service using TESTNET credentials')
    sys.exit(0)
    
except Exception as e:
    print(f'❌ TESTNET ALSO FAILED: {e}')
    print('')

print('❌ VERDICT: FAIL - Credentials invalid/IP-restricted')
print('   Action required: Check Binance API key whitelist or generate new keys')
sys.exit(1)
