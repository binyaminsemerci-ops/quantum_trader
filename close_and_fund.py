#!/usr/bin/env python3
import os
from binance.client import Client

TRANSFER_AMOUNT = 2000  # USDT to transfer SPOT -> FUTURES
TESTNET_URL = 'https://testnet.binancefuture.com'

# Load env
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
    raise SystemExit('Missing BINANCE_API_KEY/SECRET in .env')

client = Client(key, secret, testnet=True)
client.FUTURES_URL = TESTNET_URL

print('--- Closing open futures positions (reduceOnly market) ---')
pos = client.futures_position_information()
closed_any = False
for p in pos:
    amt = float(p['positionAmt'])
    if amt == 0.0:
        continue
    sym = p['symbol']
    side = 'SELL' if amt > 0 else 'BUY'
    qty = abs(amt)
    try:
        r = client.futures_create_order(symbol=sym, side=side, type='MARKET', quantity=qty, reduceOnly=True)
        print(f"  Closed {sym}: side={side} qty={qty} orderId={r.get('orderId')}")
        closed_any = True
    except Exception as e:
        print(f"  ERROR closing {sym}: {e}")
if not closed_any:
    print('  No open positions to close')

print('\n--- Transferring USDT SPOT -> FUTURES ---')
try:
    res = client.futures_transfer(asset='USDT', amount=TRANSFER_AMOUNT, type=1)
    print(f"  Transfer requested: {TRANSFER_AMOUNT} USDT | tranId={res}")
except Exception as e:
    print(f"  ERROR transfer: {e}")

print('\n--- Balances after operations ---')
bal = client.futures_account_balance()
for b in bal:
    if b['asset'] in ('USDT', 'USDC', 'FDUSD', 'BFUSD'):
        print(f"  {b['asset']}: balance={b['balance']} available={b.get('availableBalance', 'n/a')}")
