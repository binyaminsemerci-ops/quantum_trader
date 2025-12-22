import os
import requests
import hmac
import hashlib
import time

# Get credentials from environment
api_key = os.environ.get('BINANCE_API_KEY')
api_secret = os.environ.get('BINANCE_API_SECRET')

if not api_key or not api_secret:
    print("❌ API credentials not found in environment")
    exit(1)

base_url = 'https://testnet.binancefuture.com'

def sign_request(params):
    """Sign request with API secret"""
    query_string = '&'.join([f"{k}={v}" for k, v in params.items()])
    signature = hmac.new(
        api_secret.encode('utf-8'),
        query_string.encode('utf-8'),
        hashlib.sha256
    ).hexdigest()
    return signature

print("=" * 60)
print("BINANCE POSITION MODE TEST")
print("=" * 60)

# 1. Check position mode (dual side or one-way)
print("\n1️⃣ Checking Position Mode...")
try:
    timestamp = int(time.time() * 1000)
    params = {'timestamp': timestamp}
    params['signature'] = sign_request(params)
    
    headers = {'X-MBX-APIKEY': api_key}
    response = requests.get(
        f"{base_url}/fapi/v1/positionSide/dual",
        headers=headers,
        params=params
    )
    
    if response.status_code == 200:
        data = response.json()
        dual_side = data.get('dualSidePosition', False)
        print(f"   dualSidePosition: {dual_side}")
        if dual_side:
            print("   ✅ HEDGE MODE (can have LONG and SHORT simultaneously)")
        else:
            print("   ✅ ONE-WAY MODE (single position per symbol)")
    else:
        print(f"   ❌ Error: {response.status_code} - {response.text}")
except Exception as e:
    print(f"   ❌ Exception: {e}")

# 2. Get account info and active positions
print("\n2️⃣ Checking Active Positions...")
try:
    timestamp = int(time.time() * 1000)
    params = {'timestamp': timestamp}
    params['signature'] = sign_request(params)
    
    headers = {'X-MBX-APIKEY': api_key}
    response = requests.get(
        f"{base_url}/fapi/v2/account",
        headers=headers,
        params=params
    )
    
    if response.status_code == 200:
        account = response.json()
        positions = account.get('positions', [])
        active_positions = [p for p in positions if float(p.get('positionAmt', 0)) != 0]
        
        print(f"   Total positions: {len(positions)}")
        print(f"   Active positions (qty != 0): {len(active_positions)}")
        
        if active_positions:
            print("\n   Active Positions:")
            for pos in active_positions:
                symbol = pos.get('symbol')
                amt = float(pos.get('positionAmt', 0))
                side = "LONG" if amt > 0 else "SHORT"
                position_side = pos.get('positionSide', 'BOTH')
                entry_price = pos.get('entryPrice', 'N/A')
                
                print(f"   • {symbol}:")
                print(f"     - Amount: {amt} ({side})")
                print(f"     - positionSide field: {position_side}")
                print(f"     - Entry Price: {entry_price}")
    else:
        print(f"   ❌ Error: {response.status_code} - {response.text}")
except Exception as e:
    print(f"   ❌ Exception: {e}")

print("\n" + "=" * 60)
print("TEST COMPLETE")
print("=" * 60)

# 3. Check open orders
print("\n3️⃣ Checking Open Orders...")
try:
    timestamp = int(time.time() * 1000)
    params = {'timestamp': timestamp}
    params['signature'] = sign_request(params)
    
    headers = {'X-MBX-APIKEY': api_key}
    response = requests.get(
        f"{base_url}/fapi/v1/openOrders",
        headers=headers,
        params=params
    )
    
    if response.status_code == 200:
        orders = response.json()
        print(f"   Total open orders: {len(orders)}")
        
        # Group by symbol
        by_symbol = {}
        for order in orders:
            symbol = order.get('symbol')
            if symbol not in by_symbol:
                by_symbol[symbol] = []
            by_symbol[symbol].append(order)
        
        if by_symbol:
            print(f"\n   Orders by Symbol:")
            for symbol in sorted(by_symbol.keys()):
                symbol_orders = by_symbol[symbol]
                print(f"\n   {symbol}: {len(symbol_orders)} orders")
                for order in symbol_orders:
                    order_type = order.get('type')
                    side = order.get('side')
                    position_side = order.get('positionSide', 'N/A')
                    status = order.get('status')
                    print(f"      • {order_type} {side} (positionSide={position_side}, status={status})")
        else:
            print("   ⚠️  No open orders found")
    else:
        print(f"   ❌ Error: {response.status_code} - {response.text}")
except Exception as e:
    print(f"   ❌ Exception: {e}")

print("\n" + "=" * 60)
