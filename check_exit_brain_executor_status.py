#!/usr/bin/env python3
"""
Check if Exit Brain is actually running and making decisions
"""
import requests
import sys

print("=" * 70)
print("EXIT BRAIN V3 EXECUTOR STATUS CHECK")
print("=" * 70)

try:
    # Check backend health
    health = requests.get("http://localhost:8000/health", timeout=5).json()
    print(f"\n✅ Backend: {health['status']}")
    
    # Try to get Exit Brain specific status endpoint if it exists
    try:
        status = requests.get("http://localhost:8000/health/exit_brain_status", timeout=5)
        if status.status_code == 200:
            print(f"\n📊 Exit Brain Status:")
            import json
            print(json.dumps(status.json(), indent=2))
        else:
            print(f"\n⚠️ Exit Brain status endpoint not available (404)")
    except:
        print(f"\n⚠️ No Exit Brain status endpoint")
    
    # Check if there are any open orders (TP/SL)
    try:
        orders = requests.get("http://localhost:8000/orders", timeout=5)
        if orders.status_code == 200:
            order_list = orders.json()
            print(f"\n📋 Open Orders: {len(order_list)}")
            
            if order_list:
                for order in order_list:
                    print(f"   {order.get('symbol')} {order.get('type')} {order.get('side')} @ ${order.get('price')}")
            else:
                print(f"   ⚠️ NO OPEN ORDERS FOUND!")
                print(f"   This means Exit Brain has NOT placed TP/SL orders yet")
        else:
            print(f"\n⚠️ Orders endpoint returned {orders.status_code}")
    except Exception as e:
        print(f"\n⚠️ Cannot check orders: {e}")
    
    print("\n" + "=" * 70)
    print("🔍 ANALYSIS:")
    print("   If Exit Brain is LIVE and running, it should have:")
    print("   1. Placed STOP_MARKET orders for stop loss")
    print("   2. Placed TAKE_PROFIT_MARKET orders for take profit")
    print("   3. Created at least 8 orders (2 per position)")
    print("\n   If NO orders exist, Exit Brain might:")
    print("   - Not be running yet (still initializing)")
    print("   - Be in SHADOW mode (observing only)")
    print("   - Have errors preventing order placement")
    print("=" * 70)
    
except Exception as e:
    print(f"\n❌ ERROR: {e}")
    print("=" * 70)
    sys.exit(1)
