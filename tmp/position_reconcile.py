#!/usr/bin/env python3
"""Position reconciliation script - compare Redis vs Binance reality"""
import os
import sys
import redis
import requests
import hashlib
import hmac
import time
from urllib.parse import urlencode

# Redis connection
r = redis.Redis(host='localhost', port=6379, db=0, decode_responses=True)

# Binance testnet credentials
API_KEY = os.getenv("BINANCE_TESTNET_API_KEY", "")
API_SECRET = os.getenv("BINANCE_TESTNET_API_SECRET", "")
BASE_URL = "https://testnet.binancefuture.com"

def binance_request(method, endpoint, params=None, signed=False):
    """Make Binance API request"""
    if params is None:
        params = {}
    
    headers = {"X-MBX-APIKEY": API_KEY}
    
    if signed:
        params['timestamp'] = int(time.time() * 1000)
        query_string = urlencode(params)
        signature = hmac.new(
            API_SECRET.encode('utf-8'),
            query_string.encode('utf-8'),
            hashlib.sha256
        ).hexdigest()
        params['signature'] = signature
    
    url = f"{BASE_URL}{endpoint}"
    
    if method == "GET":
        resp = requests.get(url, params=params, headers=headers, timeout=10)
    else:
        resp = requests.post(url, params=params, headers=headers, timeout=10)
    
    resp.raise_for_status()
    return resp.json()

print("=== POSITION RECONCILIATION ===\n")

# 1. Get all Redis positions
print("--- Redis Positions ---")
redis_positions = {}
for key in r.keys("quantum:position:*"):
    if key.startswith("quantum:position:snapshot:") or key.startswith("quantum:position:ledger:"):
        continue  # Skip snapshot/ledger keys
    
    symbol = key.replace("quantum:position:", "")
    data = r.hgetall(key)
    qty = float(data.get('quantity', 0.0))
    side = data.get('side', 'NONE')
    
    if qty != 0 or side not in ['NONE', 'FLAT', '']:
        redis_positions[symbol] = {
            'qty': qty,
            'side': side,
            'entry_price': float(data.get('entry_price', 0.0)),
            'leverage': float(data.get('leverage', 0.0))
        }
        print(f"  {symbol}: {side} qty={qty}")

print(f"\nTotal Redis active: {len(redis_positions)}")

# 2. Get Binance positions
print("\n--- Binance Testnet Positions ---")
try:
    binance_positions = {}
    account_info = binance_request("GET", "/fapi/v2/account", signed=True)
    
    for pos in account_info.get('positions', []):
        symbol = pos['symbol']
        amt = float(pos['positionAmt'])
        if amt != 0:
            side = 'LONG' if amt > 0 else 'SHORT'
            binance_positions[symbol] = {
                'qty': abs(amt),
                'side': side,
                'entry_price': float(pos['entryPrice']),
                'unrealized_pnl': float(pos['unRealizedProfit']),
                'leverage': int(pos['leverage'])
            }
            print(f"  {symbol}: {side} qty={abs(amt)} entry={pos['entryPrice']} pnl={pos['unRealizedProfit']}")
    
    print(f"\nTotal Binance active: {len(binance_positions)}")
    
except Exception as e:
    print(f"ERROR fetching Binance positions: {e}")
    binance_positions = {}

# 3. Compare and find discrepancies
print("\n--- Discrepancies ---")
discrepancies = []

# Redis positions not in Binance
for symbol, data in redis_positions.items():
    if symbol not in binance_positions:
        print(f"  GHOST: {symbol} exists in Redis but NOT on Binance (qty={data['qty']})")
        discrepancies.append(('ghost', symbol, data))

# Binance positions not in Redis
for symbol, data in binance_positions.items():
    if symbol not in redis_positions:
        print(f"  MISSING: {symbol} exists on Binance but NOT in Redis (qty={data['qty']})")
        discrepancies.append(('missing', symbol, data))

# Quantity/side mismatches
for symbol in set(redis_positions.keys()) & set(binance_positions.keys()):
    redis_data = redis_positions[symbol]
    binance_data = binance_positions[symbol]
    
    if abs(redis_data['qty'] - binance_data['qty']) > 0.01:
        print(f"  QTY_MISMATCH: {symbol} Redis={redis_data['qty']} Binance={binance_data['qty']}")
        discrepancies.append(('qty_mismatch', symbol, {'redis': redis_data, 'binance': binance_data}))
    
    if redis_data['side'] != binance_data['side']:
        print(f"  SIDE_MISMATCH: {symbol} Redis={redis_data['side']} Binance={binance_data['side']}")
        discrepancies.append(('side_mismatch', symbol, {'redis': redis_data, 'binance': binance_data}))

if not discrepancies:
    print("  None - Redis and Binance are in sync!")

# 4. Ghost key cleanup recommendations
print("\n--- Cleanup Recommendations ---")
ghost_keys = []
for key in r.keys("quantum:position:*"):
    symbol = key.replace("quantum:position:", "")
    
    # Skip if already in active positions
    if symbol in redis_positions or symbol in binance_positions:
        continue
    
    data = r.hgetall(key)
    qty = float(data.get('quantity', data.get('position_amt', data.get('qty', 0.0))))
    
    if qty == 0 or not data:
        ghost_keys.append(key)

print(f"Found {len(ghost_keys)} ghost keys to delete:")
for key in ghost_keys[:20]:  # Show first 20
    print(f"  {key}")

if len(ghost_keys) > 20:
    print(f"  ... and {len(ghost_keys) - 20} more")

print(f"\nTo clean up: redis-cli DEL {' '.join([f'\"{k}\"' for k in ghost_keys[:10]])}")

print("\n=== END RECONCILIATION ===")
