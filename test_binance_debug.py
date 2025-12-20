#!/usr/bin/env python3
import hashlib
import hmac
import time
import requests
from urllib.parse import urlencode

API_KEY = "6WxDRmtUQyWLEFf0hCwo33nSdS5nD5BUcGsFxwAQqsx9DkMNaMDCDYQZbSNQ3fIR"
API_SECRET = "XDQqqxprJTK9KJCu0Mpp9R5yYDDZV03VD6Ldxu71VQOVREmj6mz2paSjv7TJkBaR"
BASE_URL = "https://testnet.binancefuture.com"

print("🔍 Manual Binance Testnet Signature Test\n")

# Step 1: Get server time
server_time_resp = requests.get(f"{BASE_URL}/fapi/v1/time")
server_time = server_time_resp.json()["serverTime"]
print(f"1. Server time: {server_time}")

# Step 2: Build query string
params = {
    "timestamp": server_time,
    "recvWindow": 60000
}
query_string = urlencode(params)
print(f"2. Query string: {query_string}")

# Step 3: Generate signature
signature = hmac.new(
    API_SECRET.encode('utf-8'),
    query_string.encode('utf-8'),
    hashlib.sha256
).hexdigest()
print(f"3. Signature: {signature}")

# Step 4: Make request
url = f"{BASE_URL}/fapi/v2/account?{query_string}&signature={signature}"
headers = {"X-MBX-APIKEY": API_KEY}

print(f"\n4. Making request to: {BASE_URL}/fapi/v2/account")
print(f"   Headers: {headers}")
print(f"   Params: {query_string}&signature={signature[:20]}...")

try:
    response = requests.get(url, headers=headers)
    print(f"\n✅ Status: {response.status_code}")
    if response.status_code == 200:
        print(f"✅ SUCCESS! Response: {response.json()}")
    else:
        print(f"❌ FAILED: {response.json()}")
except Exception as e:
    print(f"\n❌ Exception: {e}")
