#!/usr/bin/env python3
"""Test Binance Testnet API Keys"""
from binance.client import Client

# Les credentials fra .env
env_vars = {}
with open(".env") as f:
    for line in f:
        if "=" in line and not line.strip().startswith("#"):
            key, val = line.strip().split("=", 1)
            env_vars[key] = val.strip()

api_key = env_vars.get("BINANCE_API_KEY", "")
api_secret = env_vars.get("BINANCE_API_SECRET", "")

print(f"Key length: {len(api_key)}")
print(f"Key: {api_key[:15]}...{api_key[-15:]}")
print(f"Secret: {api_secret[:15]}...{api_secret[-15:]}\n")

print("Testing Binance Futures Testnet...")
try:
    client = Client(api_key, api_secret)
    client.FUTURES_URL = "https://testnet.binancefuture.com/fapi"
    print(f"FUTURES_URL set to: {client.FUTURES_URL}")
    
    account = client.futures_account()
    print("✅ SUCCESS!")
    print(f"Balance: ${account.get('totalWalletBalance')}")
    print(f"Available: ${account.get('availableBalance')}")
except Exception as e:
    print(f"❌ Error: {e}")
    import traceback
    traceback.print_exc()
