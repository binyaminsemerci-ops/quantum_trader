#!/usr/bin/env python3
import os

env = {}
with open('/home/qt/quantum_trader/.env', encoding='utf-8') as f:
    for line in f:
        line = line.strip()
        if not line or line.startswith('#') or '=' not in line:
            continue
        k, v = line.split('=', 1)
        env[k.strip()] = v.strip()

key = env.get('BINANCE_API_KEY')
secret = env.get('BINANCE_API_SECRET')
if not key or not secret:
    print('Missing API keys in .env')
    raise SystemExit(1)

from binance.client import Client
client = Client(key, secret, testnet=True)
client.FUTURES_URL = 'https://testnet.binancefuture.com'

bal = client.futures_account_balance()
print('Futures account balance (asset, balance, available):')
for b in bal:
    print(f"  {b['asset']}: balance={b['balance']} available={b.get('availableBalance', 'n/a')}")

positions = client.futures_position_information()
print('\nOpen positions (non-zero):')
any_pos = False
for p in positions:
    amt = float(p['positionAmt'])
    if amt != 0.0:
        any_pos = True
        print(f"  {p['symbol']}: amt={p['positionAmt']} entry={p['entryPrice']} unrealized={p['unRealizedProfit']}")
if not any_pos:
    print('  (none)')
