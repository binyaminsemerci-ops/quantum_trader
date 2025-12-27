#!/usr/bin/env python3
"""
Debug: Check app.state for Exit Brain executor
"""
import requests

print("=" * 70)
print("CHECKING APP STATE FOR EXIT BRAIN EXECUTOR")
print("=" * 70)

try:
    # Try debug endpoint if it exists
    resp = requests.get("http://localhost:8000/debug/app_state", timeout=5)
    
    if resp.status_code == 200:
        import json
        print("\n📊 App State:")
        print(json.dumps(resp.json(), indent=2))
    else:
        print(f"\n⚠️ Debug endpoint returned {resp.status_code}")
        
        # Try to infer from logs
        print("\n🔍 Checking if Exit Brain logged startup...")
        print("   Expected log: '[EXIT_BRAIN] EXIT_MODE=EXIT_BRAIN_V3 detected'")
        print("   Expected log: '[EXIT_BRAIN] Dynamic Executor started'")
        print("\n   If these logs are missing, executor did not start!")
        print("   Possible reasons:")
        print("   1. EXIT_MODE not set to EXIT_BRAIN_V3")
        print("   2. BINANCE_API_KEY/SECRET missing")
        print("   3. QT_PAPER_TRADING logic failed")
        print("   4. Import error in Exit Brain modules")
        
except Exception as e:
    print(f"\n❌ ERROR: {e}")

print("\n" + "=" * 70)
