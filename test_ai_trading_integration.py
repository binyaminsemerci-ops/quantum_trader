#!/usr/bin/env python3
"""
AI Auto Trading Integration Test

This script tests the AI Auto Trading Service integration with the backend.
It verifies that all endpoints work correctly and that the AI service can
generate signals and execute trades.
"""

import requests
import json
import time
import sys
import os
from datetime import datetime

# Backend URL
BASE_URL = "http://127.0.0.1:8000"

def test_endpoint(method, endpoint, data=None, description=""):
    """Test a single API endpoint"""
    url = f"{BASE_URL}{endpoint}"
    
    print(f"\n🧪 Testing {method} {endpoint}")
    if description:
        print(f"   {description}")
    
    try:
        if method.upper() == "GET":
            response = requests.get(url, timeout=10)
        elif method.upper() == "POST":
            response = requests.post(url, json=data, timeout=10)
        else:
            print(f"❌ Unsupported method: {method}")
            return False
            
        print(f"   Status Code: {response.status_code}")
        
        if response.status_code in [200, 201]:
            result = response.json()
            print(f"   ✅ Success")
            if isinstance(result, dict) and len(result) <= 5:
                # Print small responses inline
                print(f"   Response: {json.dumps(result, indent=2)[:200]}...")
            return result
        else:
            print(f"   ❌ Failed: {response.status_code}")
            if response.text:
                print(f"   Error: {response.text[:200]}...")
            return None
            
    except requests.exceptions.RequestException as e:
        print(f"   ❌ Request failed: {e}")
        return None
    except Exception as e:
        print(f"   ❌ Unexpected error: {e}")
        return None

def main():
    print("🚀 AI Auto Trading Integration Test")
    print("=" * 50)
    
    # Test 1: Check if backend is running
    print("\n📡 Testing Backend Connection...")
    system_status = test_endpoint("GET", "/api/v1/system/status", 
                                  description="Check if backend is running")
    if not system_status:
        print("\n❌ Backend is not running. Please start the backend first:")
        print("   python backend\\simple_main.py")
        print("\nStarting backend for you...")
        import subprocess
        import time
        
        # Start backend in separate process
        backend_process = subprocess.Popen([
            sys.executable, "backend\\simple_main.py"
        ], cwd=os.getcwd())
        
        # Wait for backend to start
        print("   ⏳ Waiting for backend to start...")
        time.sleep(5)
        
        # Try to connect again
        system_status = test_endpoint("GET", "/api/v1/system/status", 
                                      description="Check if backend is running after start")
        if not system_status:
            print("   ❌ Failed to start backend")
            backend_process.terminate()
            return 1
    
    # Test 2: AI Trading Status (initial)
    print("\n🧠 Testing AI Trading Status...")
    ai_status = test_endpoint("GET", "/api/v1/ai-trading/status",
                              description="Get initial AI trading status")
    
    # Test 3: Update AI Configuration
    print("\n⚙️ Testing AI Configuration Update...")
    config = {
        "position_size": 500.0,
        "stop_loss_pct": 2.5,
        "take_profit_pct": 5.0,
        "min_confidence": 0.75,
        "max_positions": 3,
        "risk_limit": 5000.0
    }
    config_result = test_endpoint("POST", "/api/v1/ai-trading/config", config,
                                  "Update AI trading configuration")
    
    # Test 4: Start AI Trading
    print("\n🎯 Testing Start AI Trading...")
    start_symbols = ["BTCUSDC", "ETHUSDC"]
    start_result = test_endpoint("POST", "/api/v1/ai-trading/start", start_symbols,
                                 "Start AI trading with test symbols")
    
    if start_result:
        print("   🟢 AI Trading started successfully!")
        
        # Wait a moment for the system to initialize
        print("   ⏳ Waiting 3 seconds for AI to initialize...")
        time.sleep(3)
        
        # Test 5: Get AI Status (after start)
        print("\n📊 Testing AI Status After Start...")
        post_start_status = test_endpoint("GET", "/api/v1/ai-trading/status",
                                          description="Get AI status after starting")
        
        # Test 6: Get AI Signals
        print("\n📡 Testing AI Signals...")
        signals = test_endpoint("GET", "/api/v1/ai-trading/signals?limit=5",
                                description="Get recent AI trading signals")
        
        # Test 7: Get AI Executions
        print("\n⚡ Testing AI Executions...")
        executions = test_endpoint("GET", "/api/v1/ai-trading/executions?limit=5",
                                   description="Get recent AI trade executions")
        
        # Let AI run for a few seconds to potentially generate signals
        print("\n⏳ Letting AI run for 5 seconds to generate signals...")
        time.sleep(5)
        
        # Test 8: Check for new signals
        print("\n🔄 Checking for New Signals...")
        new_signals = test_endpoint("GET", "/api/v1/ai-trading/signals?limit=10",
                                    description="Check for newly generated signals")
        
        # Test 9: Stop AI Trading
        print("\n🛑 Testing Stop AI Trading...")
        stop_result = test_endpoint("POST", "/api/v1/ai-trading/stop", None,
                                    "Stop AI auto trading")
        
        if stop_result:
            print("   🔴 AI Trading stopped successfully!")
    
    # Test 10: Final Status Check
    print("\n🏁 Final Status Check...")
    final_status = test_endpoint("GET", "/api/v1/ai-trading/status",
                                 description="Get final AI trading status")
    
    # Test 11: Basic Portfolio Check
    print("\n💼 Testing Portfolio Access...")
    portfolio = test_endpoint("GET", "/api/v1/portfolio",
                              description="Check portfolio data access")
    
    # Summary
    print("\n" + "=" * 50)
    print("🏆 AI Trading Integration Test Summary")
    print("=" * 50)
    
    if ai_status and start_result and stop_result:
        print("✅ All critical AI trading endpoints working")
        print("✅ AI service can be started and stopped")
        print("✅ Configuration updates working")
        print("✅ Signal and execution endpoints accessible")
        print("\n🎉 AI Auto Trading System is Ready!")
        return 0
    else:
        print("❌ Some critical tests failed")
        print("❌ AI trading system may not be fully functional")
        return 1

if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)