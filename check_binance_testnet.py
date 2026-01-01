import requests
import hmac
import hashlib
import time
from urllib.parse import urlencode

api_key = "e9ZqWhGhAEhDPfNBfQMiJv8zULKJZBIwaaJdfbbUQ8ZNj1WUMumrjenHoRzpzUPD"
api_secret = "ZowBZEfL1R1ValcYLkbxjMfZ1tOxfEDRW4eloWRGGjk5etn0vSFFSU3gCTdCFoja"
base_url = "https://testnet.binancefuture.com"

timestamp = int(time.time() * 1000)
params = {"timestamp": timestamp}
query_string = urlencode(params)
signature = hmac.new(api_secret.encode(), query_string.encode(), hashlib.sha256).hexdigest()
params["signature"] = signature
headers = {"X-MBX-APIKEY": api_key}

response = requests.get(f"{base_url}/fapi/v2/account", params=params, headers=headers)
data = response.json()

print("=" * 50)
print("BINANCE TESTNET ACCOUNT STATUS")
print("=" * 50)
print(f"Total Wallet Balance: {data.get('totalWalletBalance', 0)} USDT")
print(f"Available Balance: {data.get('availableBalance', 0)} USDT")
print(f"Total Unrealized PnL: {data.get('totalUnrealizedProfit', 0)} USDT")
print(f"Total Position Margin: {data.get('totalPositionInitialMargin', 0)} USDT")
print()

print("=" * 50)
print("ACTIVE POSITIONS")
print("=" * 50)
active = [p for p in data.get("positions", []) if float(p.get("positionAmt", 0)) != 0]
print(f"Total Active Positions: {len(active)}")
print()

for i, pos in enumerate(active, 1):
    symbol = pos["symbol"]
    side = pos["positionSide"]
    amt = pos["positionAmt"]
    entry = pos["entryPrice"]
    pnl = pos["unrealizedProfit"]
    leverage = pos["leverage"]
    notional = pos["notional"]
    
    print(f"{i}. {symbol} ({side})")
    print(f"   Amount: {amt}")
    print(f"   Entry Price: {entry}")
    print(f"   Notional Value: {notional} USDT")
    print(f"   Leverage: {leverage}x")
    print(f"   Unrealized PnL: {pnl} USDT")
    print()

# Get recent trades
print("=" * 50)
print("RECENT TRADES (Last 5)")
print("=" * 50)
if active:
    # Get trades for first active symbol
    symbol = active[0]["symbol"]
    timestamp = int(time.time() * 1000)
    params = {"symbol": symbol, "limit": 5, "timestamp": timestamp}
    query_string = urlencode(params)
    signature = hmac.new(api_secret.encode(), query_string.encode(), hashlib.sha256).hexdigest()
    params["signature"] = signature
    
    response = requests.get(f"{base_url}/fapi/v1/userTrades", params=params, headers=headers)
    trades = response.json()
    
    for trade in trades:
        print(f"Symbol: {trade['symbol']}, Side: {trade['side']}, Price: {trade['price']}, Qty: {trade['qty']}, Time: {trade['time']}")
