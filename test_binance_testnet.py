#!/usr/bin/env python3
from binance.client import Client
import time
import requests

key = "6WxDRmtUQyWLEFf0hCwo33nSdS5nD5BUcGsFxwAQqsx9DkMNaMDCDYQZbSNQ3fIR"
secret = "XDQqqxprJTK9KJCu0Mpp9R5yYDDZV03VD6Ldxu71VQOVREmj6mz2paSjv7TJkBaR"

print("🔍 Testing Binance Testnet Connection...\n")

# Test 1: Timestamp sync
print("1. Checking timestamp sync...")
server_time = requests.get("https://testnet.binancefuture.com/fapi/v1/time").json()
local_time = int(time.time() * 1000)
diff = abs(server_time['serverTime'] - local_time)
print(f"   Server time: {server_time['serverTime']}")
print(f"   Local time:  {local_time}")
print(f"   Difference:  {diff}ms")
if diff > 5000:
    print(f"   ⚠️  WARNING: Time difference > 5s!")
else:
    print(f"   ✅ Time sync OK\n")

# Test 2: Connection with large recvWindow
print("2. Testing API connection (recvWindow=60000)...")
try:
    client = Client(key, secret, testnet=True, tld="com")
    client.FUTURES_URL = "https://testnet.binancefuture.com/fapi"
    account = client.futures_account(recvWindow=60000)
    print(f"   ✅ SUCCESS!")
    print(f"   Balance: {account['totalWalletBalance']} USDT")
    print(f"   Available: {account['availableBalance']} USDT\n")
except Exception as e:
    print(f"   ❌ FAILED: {e}\n")

# Test 3: Connection with default recvWindow
print("3. Testing API connection (default recvWindow)...")
try:
    client = Client(key, secret, testnet=True, tld="com")
    client.FUTURES_URL = "https://testnet.binancefuture.com/fapi"
    account = client.futures_account()
    print(f"   ✅ SUCCESS!")
    print(f"   Balance: {account['totalWalletBalance']} USDT\n")
except Exception as e:
    print(f"   ❌ FAILED: {e}\n")

print("✅ Test complete!")
