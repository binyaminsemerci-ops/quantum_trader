#!/usr/bin/env python3
import redis
import requests
import hmac
import hashlib
import time

r = redis.Redis(decode_responses=True)
api_key = r.get('quantum:config:binance_api_key')
api_secret = r.get('quantum:config:binance_api_secret')

timestamp = int(time.time() * 1000)
query = f'timestamp={timestamp}'
sig = hmac.new(api_secret.encode(), query.encode(), hashlib.sha256).hexdigest()

url = f'https://testnet.binancefuture.com/fapi/v2/account?{query}&signature={sig}'
headers = {'X-MBX-APIKEY': api_key}

try:
    resp = requests.get(url, headers=headers, timeout=10)
    resp.raise_for_status()
    data = resp.json()
    
    print("BINANCE TESTNET ACCOUNT:")
    print(f"  Wallet Balance: {float(data['totalWalletBalance']):.2f} USDT")
    print(f"  Unrealized PNL: {float(data['totalUnrealizedProfit']):.2f} USDT")
    print(f"  Available: {float(data['availableBalance']):.2f} USDT")
    
except Exception as e:
    print(f"ERROR: {e}")
