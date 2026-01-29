#!/usr/bin/env python3
import os
from binance.client import Client

key = os.getenv('BINANCE_API_KEY')
secret = os.getenv('BINANCE_API_SECRET')

if not key or not secret:
    print("❌ Credentials not in environment")
    exit(1)

print(f'Key (first 10 chars): {key[:10]}...')
print('')

# Try mainnet
print("Testing MAINNET API...")
try:
    client = Client(key, secret, testnet=False)
    account = client.futures_account()
    print(f'✅ MAINNET API Working!')
    print(f'   Balance: {account.get("totalWalletBalance")} USDT')
    positions = [p for p in client.futures_position_information() if float(p["positionAmt"]) != 0]
    print(f'   Open positions: {len(positions)}')
    for p in positions[:3]:
        print(f'     {p["symbol"]}: {p["positionAmt"]} @ ${p["entryPrice"]}')
except Exception as e:
    print(f'❌ Mainnet failed: {e}')
    print('')
    
    # Try testnet
    print("Testing TESTNET API...")
    try:
        client = Client(key, secret, testnet=True)
        account = client.futures_account()
        print(f'✅ TESTNET API Working!')
        print(f'   Balance: {account.get("totalWalletBalance")} USDT')
        positions = [p for p in client.futures_position_information() if float(p["positionAmt"]) != 0]
        print(f'   Open positions: {len(positions)}')
        for p in positions[:3]:
            print(f'     {p["symbol"]}: {p["positionAmt"]} @ ${p["entryPrice"]}')
    except Exception as e2:
        print(f'❌ Testnet also failed: {e2}')
        exit(1)
