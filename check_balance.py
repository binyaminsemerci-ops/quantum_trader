"""Check Binance Futures balance"""
import os
from binance.client import Client

api_key = os.getenv("BINANCE_API_KEY")
api_secret = os.getenv("BINANCE_API_SECRET")

if not api_key:
    print("Loading from .env...")
    with open(".env") as f:
        for line in f:
            if line.startswith("BINANCE_API_KEY="):
                api_key = line.split("=", 1)[1].strip()
            elif line.startswith("BINANCE_API_SECRET="):
                api_secret = line.split("=", 1)[1].strip()

client = Client(api_key, api_secret)

account = client.futures_account()

print("\n[MONEY] BINANCE FUTURES BALANCE")
print("=" * 80)

for asset in account['assets']:
    balance = float(asset['availableBalance'])
    if balance > 0:
        print(f"\n{asset['asset']}:")
        print(f"  Available: ${balance:.2f}")
        print(f"  Wallet: ${float(asset['walletBalance']):.2f}")
        print(f"  Unrealized PnL: ${float(asset['unrealizedProfit']):.2f}")

print("\n" + "=" * 80)
print(f"Total Available (USDT): ${float([a for a in account['assets'] if a['asset'] == 'USDT'][0]['availableBalance']):.2f}")
