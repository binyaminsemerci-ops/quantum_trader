#!/usr/bin/env python3
"""
Convert all Binance Futures assets to USDT
"""
import os
from dotenv import load_dotenv
from binance.client import Client

load_dotenv()

client = Client(
    os.getenv("BINANCE_API_KEY"),
    os.getenv("BINANCE_API_SECRET")
)

print("[MONEY] Current Binance Futures Balance:")
print("=" * 80)

account = client.futures_account()
assets_to_convert = []

for asset_info in account['assets']:
    asset = asset_info['asset']
    wallet = float(asset_info['walletBalance'])
    available = float(asset_info['availableBalance'])
    
    if available > 0 and asset != 'USDT':
        print(f"{asset}: Available ${available:.2f}, Wallet ${wallet:.2f}")
        assets_to_convert.append((asset, available))
    elif asset == 'USDT':
        print(f"[OK] USDT: Available ${available:.2f}, Wallet ${wallet:.2f}")

print("=" * 80)

if not assets_to_convert:
    print("\n[OK] All assets are already in USDT or zero balance")
    exit(0)

print(f"\n[WARNING]  Found {len(assets_to_convert)} assets to convert:")
for asset, amount in assets_to_convert:
    print(f"  - {asset}: ${amount:.2f}")

print("\n🔧 MANUAL CONVERSION REQUIRED:")
print("These assets cannot be auto-converted via API:")
print("  1. Log in to Binance Futures")
print("  2. Go to Wallet → Futures Wallet")
print("  3. Convert each asset to USDT manually")
print("\nAlternatively:")
print("  - BNFCR: Use as collateral directly (Multi-Assets Mode)")
print("  - BFUSD: Convert to USDT via Binance Convert")
print("  - LDUSDT/RWUSD: Redeem to USDT if possible")
