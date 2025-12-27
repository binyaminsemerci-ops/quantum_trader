#!/usr/bin/env python3
"""
Debug why BinanceFuturesExecutionAdapter is not ready
"""
import sys
sys.path.insert(0, "C:\\quantum_trader")

from dotenv import load_dotenv
load_dotenv()

from backend.config import load_config, load_liquidity_config
from backend.services.execution.execution import BinanceFuturesExecutionAdapter

print("=" * 70)
print("DEBUGGING BINANCE FUTURES ADAPTER")
print("=" * 70)

try:
    # Load configs
    cfg = load_config()
    liq = load_liquidity_config()
    
    print(f"\n📊 Config:")
    print(f"   API Key: {getattr(cfg, 'binance_api_key', None)[:20]}..." if getattr(cfg, 'binance_api_key', None) else "   API Key: ❌ MISSING")
    print(f"   API Secret: {'✅ Present' if getattr(cfg, 'binance_api_secret', None) else '❌ MISSING'}")
    print(f"   Market Type: {getattr(liq, 'market_type', 'usdm_perp')}")
    print(f"   Leverage: {getattr(liq, 'default_leverage', 30)}")
    
    # Try to build adapter
    print(f"\n🔨 Building BinanceFuturesExecutionAdapter...")
    adapter = BinanceFuturesExecutionAdapter(
        api_key=getattr(cfg, 'binance_api_key', None),
        api_secret=getattr(cfg, 'binance_api_secret', None),
        quote_asset="USDT",
        market_type=getattr(liq, 'market_type', 'usdm_perp'),
        default_leverage=getattr(liq, 'default_leverage', 30),
        margin_mode=getattr(liq, 'margin_mode', 'cross'),
    )
    
    print(f"✅ Adapter created")
    print(f"   Ready: {adapter.ready}")
    print(f"   Base URL: {adapter._base_url}")
    
    if not adapter.ready:
        print(f"\n❌ Adapter NOT READY")
        print(f"   API Key empty: {not adapter._api_key}")
        print(f"   API Secret empty: {not adapter._api_secret}")
    else:
        print(f"\n✅ Adapter READY - Will use Binance Futures!")
    
    print("\n" + "=" * 70)
    
except Exception as e:
    print(f"\n❌ ERROR: {e}")
    import traceback
    traceback.print_exc()
    print("=" * 70)
