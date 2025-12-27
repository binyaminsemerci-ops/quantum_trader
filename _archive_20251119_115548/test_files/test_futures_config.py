#!/usr/bin/env python3
"""
Test script for futures configuration
"""
import os
import sys

# Set environment variables
os.environ["QT_MARKET_TYPE"] = "usdm_perp"
os.environ["QT_MARGIN_MODE"] = "cross"
os.environ["QT_DEFAULT_LEVERAGE"] = "5"
os.environ["QT_LIQUIDITY_STABLE_QUOTES"] = "USDT,USDC"
os.environ["STAGING_MODE"] = "true"

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "backend"))

from backend.config.liquidity import load_liquidity_config
from backend.services.liquidity import _select_binance_endpoint

print("\n" + "="*60)
print("[ROCKET] QUANTUM TRADER - FUTURES CONFIG TEST")
print("="*60)

# Load config
config = load_liquidity_config()

print(f"\n[CHART] Market Configuration:")
print(f"  Market Type:      {config.market_type}")
print(f"  Margin Mode:      {config.margin_mode}")
print(f"  Default Leverage: {config.default_leverage}x")
print(f"  Stable Quotes:    {', '.join(config.stable_quote_assets)}")

# Test endpoint selection
endpoint = _select_binance_endpoint(config)
print(f"\n🌐 API Endpoint:")
print(f"  URL: {endpoint}")

# Verify it's the futures endpoint
if "fapi.binance.com" in endpoint:
    print("  [OK] Correct: Using USDT-M Futures endpoint")
elif "dapi.binance.com" in endpoint:
    print("  [OK] Correct: Using COIN-M Futures endpoint")
elif "api.binance.com" in endpoint:
    print("  ❌ Error: Still using Spot endpoint")
else:
    print(f"  [WARNING]  Unknown endpoint: {endpoint}")

print(f"\n[MONEY] Quote Assets:")
for quote in config.stable_quote_assets:
    print(f"  • {quote}")

if len(config.stable_quote_assets) == 2 and "USDT" in config.stable_quote_assets and "USDC" in config.stable_quote_assets:
    print("  [OK] Correct: Limited to USDT and USDC only")
else:
    print(f"  [WARNING]  Expected only USDT and USDC, got: {config.stable_quote_assets}")

print(f"\n🔒 Risk Settings:")
print(f"  Staging Mode:     {os.getenv('STAGING_MODE', 'false')}")
print(f"  Live Trading:     {'DISABLED' if os.getenv('STAGING_MODE') == 'true' else 'ENABLED [WARNING]'}")

print("\n" + "="*60)
print("[OK] Configuration test completed successfully!")
print("="*60 + "\n")
