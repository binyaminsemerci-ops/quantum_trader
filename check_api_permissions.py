#!/usr/bin/env python3
import requests
import hmac
import hashlib
import time
from urllib.parse import urlencode

API_KEY = "6WxDRmtUQyWLEFf0hCwo33nSdS5nD5BUcGsFxwAQqsx9DkMNaMDCDYQZbSNQ3fIR"
API_SECRET = "XDQqqxprJTK9KJCu0Mpp9R5yYDDZV03VD6Ldxu71VQOVREmj6mz2paSjv7TJkBaR"

print("🔍 Checking API Key Permissions\n")

# Try SPOT testnet first
print("1. Testing SPOT testnet...")
spot_url = "https://testnet.binance.vision"
try:
    params = {"timestamp": int(time.time() * 1000), "recvWindow": 60000}
    query = urlencode(params)
    sig = hmac.new(API_SECRET.encode(), query.encode(), hashlib.sha256).hexdigest()
    resp = requests.get(f"{spot_url}/api/v3/account?{query}&signature={sig}", 
                       headers={"X-MBX-APIKEY": API_KEY})
    print(f"   Status: {resp.status_code}")
    print(f"   Response: {resp.json()}")
except Exception as e:
    print(f"   Error: {e}")

print("\n2. Testing FUTURES testnet...")
futures_url = "https://testnet.binancefuture.com"
try:
    params = {"timestamp": int(time.time() * 1000), "recvWindow": 60000}
    query = urlencode(params)
    sig = hmac.new(API_SECRET.encode(), query.encode(), hashlib.sha256).hexdigest()
    resp = requests.get(f"{futures_url}/fapi/v2/account?{query}&signature={sig}", 
                       headers={"X-MBX-APIKEY": API_KEY})
    print(f"   Status: {resp.status_code}")
    print(f"   Response: {resp.json()}")
except Exception as e:
    print(f"   Error: {e}")

print("\n3. Checking API info endpoint (if exists)...")
try:
    resp = requests.get(f"{futures_url}/fapi/v1/apiTradingStatus",
                       headers={"X-MBX-APIKEY": API_KEY})
    print(f"   Status: {resp.status_code}")
    print(f"   Response: {resp.json()}")
except Exception as e:
    print(f"   Error: {e}")
