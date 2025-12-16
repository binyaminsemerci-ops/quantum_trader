#!/usr/bin/env python3
"""
Manual test: Try to start Exit Brain executor
"""
import sys
sys.path.insert(0, "C:\\quantum_trader")

from dotenv import load_dotenv
load_dotenv()

import asyncio
import os
from pathlib import Path

print("=" * 70)
print("MANUAL EXIT BRAIN EXECUTOR TEST")
print("=" * 70)

async def test_executor():
    try:
        print("\n1️⃣ Importing modules...")
        from backend.config.exit_mode import is_exit_brain_mode
        from backend.domains.exits.exit_brain_v3.adapter import ExitBrainAdapter
        from backend.domains.exits.exit_brain_v3.dynamic_executor import ExitBrainDynamicExecutor
        from binance.client import Client
        print("✅ Imports successful")
        
        print("\n2️⃣ Checking configuration...")
        should_start = is_exit_brain_mode()
        print(f"   is_exit_brain_mode(): {should_start}")
        
        api_key = os.getenv("BINANCE_API_KEY")
        api_secret = os.getenv("BINANCE_API_SECRET")
        print(f"   Has API keys: {bool(api_key and api_secret)}")
        
        if not should_start:
            print("❌ EXIT_MODE not set to EXIT_BRAIN_V3")
            return
            
        if not (api_key and api_secret):
            print("❌ Missing API keys")
            return
        
        print("\n3️⃣ Creating Binance client...")
        go_live_active = Path("go_live.active").exists()
        use_testnet = (os.getenv("QT_PAPER_TRADING", "false").lower() == "true") and not go_live_active
        
        print(f"   go_live.active: {go_live_active}")
        print(f"   QT_PAPER_TRADING: {os.getenv('QT_PAPER_TRADING', 'NOT SET')}")
        print(f"   Using testnet: {use_testnet}")
        
        binance_client = Client(api_key, api_secret, testnet=use_testnet)
        print(f"✅ Binance client created ({'TESTNET' if use_testnet else 'PRODUCTION'})")
        
        print("\n4️⃣ Creating Exit Brain adapter...")
        exit_brain_adapter = ExitBrainAdapter()
        print("✅ Adapter created")
        
        print("\n5️⃣ Creating dynamic executor...")
        exit_brain_executor = ExitBrainDynamicExecutor(
            adapter=exit_brain_adapter,
            position_source=binance_client,
            loop_interval_sec=10.0,
            shadow_mode=False
        )
        print(f"✅ Executor created (mode={exit_brain_executor.effective_mode})")
        
        print("\n6️⃣ Starting executor (will run ONE cycle then stop)...")
        # Run one monitoring cycle
        await exit_brain_executor._monitoring_cycle(1)
        print("✅ Monitoring cycle completed!")
        
        print("\n" + "=" * 70)
        print("✅ EXIT BRAIN EXECUTOR TEST PASSED")
        print("   Executor CAN start and run successfully!")
        print("=" * 70)
        
    except Exception as e:
        print(f"\n❌ ERROR: {e}")
        import traceback
        traceback.print_exc()
        print("=" * 70)

if __name__ == "__main__":
    asyncio.run(test_executor())
