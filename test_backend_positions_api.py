#!/usr/bin/env python3
"""
Test backend /positions API endpoint directly
"""
import asyncio
import sys
sys.path.insert(0, "C:\\quantum_trader")

from backend.services.execution.execution import build_execution_adapter
from backend.config.execution import load_execution_config

async def test_positions():
    print("=" * 70)
    print("TESTING BACKEND POSITIONS API")
    print("=" * 70)
    
    try:
        # Load config
        config = load_execution_config()
        print(f"\n📊 Config:")
        print(f"   Exchange: {config.exchange}")
        print(f"   Quote: {config.quote_asset}")
        
        # Build adapter
        adapter = build_execution_adapter(config)
        print(f"\n✅ Adapter: {type(adapter).__name__}")
        
        # Get positions
        print(f"\n🔍 Calling get_positions()...")
        positions = await adapter.get_positions()
        
        print(f"\n📊 Result: {len(positions)} positions")
        
        if positions:
            for symbol, qty in positions.items():
                side = "LONG" if qty > 0 else "SHORT"
                print(f"   {symbol:12} {side:6} {qty:>15,.2f}")
        else:
            print("   ⚠️ No positions found")
            print("\n   Possible causes:")
            print("   - Backend using wrong API keys")
            print("   - Backend not in testnet mode")
            print("   - Positions were closed")
        
        print("\n" + "=" * 70)
        
    except Exception as e:
        print(f"\n❌ ERROR: {e}")
        import traceback
        traceback.print_exc()
        print("=" * 70)

if __name__ == "__main__":
    asyncio.run(test_positions())
