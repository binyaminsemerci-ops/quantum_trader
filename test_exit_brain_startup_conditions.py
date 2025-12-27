#!/usr/bin/env python3
"""
Test: Load config and check if Exit Brain should start
"""
from dotenv import load_dotenv
load_dotenv()

import os
import sys
sys.path.insert(0, "C:\\quantum_trader")

print("=" * 70)
print("EXIT BRAIN STARTUP CONDITIONS CHECK")
print("=" * 70)

# Check all relevant environment variables
print("\n📊 Environment Variables:")
print(f"   EXIT_MODE: {os.getenv('EXIT_MODE', 'NOT SET')}")
print(f"   EXIT_EXECUTOR_MODE: {os.getenv('EXIT_EXECUTOR_MODE', 'NOT SET')}")
print(f"   EXIT_BRAIN_V3_LIVE_ROLLOUT: {os.getenv('EXIT_BRAIN_V3_LIVE_ROLLOUT', 'NOT SET')}")
print(f"   BINANCE_API_KEY: {'SET (' + os.getenv('BINANCE_API_KEY', '')[:20] + '...)' if os.getenv('BINANCE_API_KEY') else 'NOT SET'}")
print(f"   BINANCE_API_SECRET: {'SET' if os.getenv('BINANCE_API_SECRET') else 'NOT SET'}")
print(f"   QT_PAPER_TRADING: {os.getenv('QT_PAPER_TRADING', 'NOT SET')}")
print(f"   BINANCE_TESTNET: {os.getenv('BINANCE_TESTNET', 'NOT SET')}")

# Check startup conditions
print("\n🔍 Startup Condition Check:")

from backend.config.exit_mode import is_exit_brain_mode

should_start = is_exit_brain_mode()
print(f"   1. is_exit_brain_mode(): {should_start}")

has_api_keys = bool(os.getenv("BINANCE_API_KEY") and os.getenv("BINANCE_API_SECRET"))
print(f"   2. Has Binance API keys: {has_api_keys}")

from pathlib import Path
go_live_active = Path("go_live.active").exists()
print(f"   3. go_live.active file exists: {go_live_active}")

qt_paper = os.getenv("QT_PAPER_TRADING", "false").lower() == "true"
print(f"   4. QT_PAPER_TRADING == 'true': {qt_paper}")

use_testnet = qt_paper and not go_live_active
print(f"   5. Should use testnet: {use_testnet}")

print("\n📋 RESULT:")
if should_start and has_api_keys:
    print(f"   ✅ Exit Brain SHOULD START")
    print(f"   ✅ Will use {'TESTNET' if use_testnet else 'PRODUCTION'}")
else:
    print(f"   ❌ Exit Brain WILL NOT START")
    if not should_start:
        print(f"      Reason: EXIT_MODE is not EXIT_BRAIN_V3")
    if not has_api_keys:
        print(f"      Reason: Missing BINANCE_API_KEY or BINANCE_API_SECRET")

print("=" * 70)
