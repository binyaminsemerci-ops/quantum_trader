#!/usr/bin/env python3
"""
Proof Script: Conditional Order Block Enforcement

Tests:
1. Gateway blocks STOP_MARKET submission (must raise ValueError)
2. No conditional orders in recent logs
3. TPSL shield disabled by default (env flag check)

Exit codes:
  0 = PASS (policy enforced)
  1 = FAIL (policy violation detected)
"""

import os
import sys
import subprocess
import asyncio
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

async def test_gateway_blocks_conditional():
    """Test 1: Gateway must reject STOP_MARKET orders"""
    print("=" * 70)
    print("TEST 1: Gateway Block for Conditional Orders")
    print("=" * 70)
    
    try:
        from backend.services.execution.exit_order_gateway import submit_exit_order
        from unittest.mock import MagicMock
        
        # Create mock Binance client
        mock_client = MagicMock()
        
        # Attempt to submit STOP_MARKET order (should fail)
        conditional_params = {
            'symbol': 'BTCUSDT',
            'side': 'SELL',
            'type': 'STOP_MARKET',  # ❌ CONDITIONAL (should be blocked)
            'stopPrice': 50000.0,
            'closePosition': True
        }
        
        try:
            await submit_exit_order(
                module_name="proof_test",
                symbol="BTCUSDT",
                order_params=conditional_params,
                order_kind="sl",
                client=mock_client,
                explanation="Proof test - conditional order (should fail)"
            )
            # If we reach here, gateway DID NOT block the order
            print("❌ FAIL: Gateway allowed STOP_MARKET order (policy violated)")
            return False
            
        except ValueError as e:
            if "Conditional orders not allowed" in str(e):
                print(f"✅ PASS: Gateway blocked STOP_MARKET order")
                print(f"   Error message: {e}")
                return True
            else:
                print(f"❌ FAIL: Wrong error type: {e}")
                return False
                
    except Exception as e:
        print(f"❌ FAIL: Test exception: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_no_conditional_in_logs():
    """Test 2: Check recent logs for conditional orders"""
    print("\n" + "=" * 70)
    print("TEST 2: No Conditional Orders in Recent Logs")
    print("=" * 70)
    
    # Check if running locally or on VPS
    log_files = [
        "/var/log/quantum/trading_bot.log",
        "/var/log/quantum/exitbrain_v35.log",
        "/var/log/quantum/apply_layer.log"
    ]
    
    conditional_types = [
        "STOP_MARKET", "TAKE_PROFIT_MARKET",
        "TRAILING_STOP_MARKET", "STOP_LOSS", "TAKE_PROFIT"
    ]
    
    found_violations = []
    
    for log_file in log_files:
        if not os.path.exists(log_file):
            continue
            
        print(f"\nChecking {log_file}...")
        
        # Grep last 1000 lines for conditional order types
        for order_type in conditional_types:
            try:
                result = subprocess.run(
                    ["tail", "-n", "1000", log_file, "|", "grep", "-i", order_type],
                    shell=True,
                    capture_output=True,
                    text=True
                )
                if result.returncode == 0 and result.stdout.strip():
                    # Check if it's a policy violation log (allowed)
                    # or actual order placement (violation)
                    lines = result.stdout.strip().split('\n')
                    for line in lines:
                        # Allowed: Gateway block logs
                        if "POLICY VIOLATION" in line or "blocked" in line.lower():
                            continue
                        # Allowed: Exit Brain MARKET orders
                        if "type='MARKET'" in line:
                            continue
                        # Violation: Actual conditional order
                        found_violations.append((log_file, order_type, line[:100]))
                        
            except Exception as e:
                print(f"   Warning: Could not check {order_type}: {e}")
    
    if found_violations:
        print("\n❌ FAIL: Found conditional orders in logs:")
        for log, otype, line in found_violations:
            print(f"   {log}: {otype}")
            print(f"      {line}")
        return False
    else:
        print("\n✅ PASS: No conditional orders found in recent logs")
        return True

def test_tpsl_shield_disabled():
    """Test 3: TPSL shield must be disabled by default"""
    print("\n" + "=" * 70)
    print("TEST 3: TPSL Shield Disabled by Default")
    print("=" * 70)
    
    # Check environment variable
    shield_enabled = os.getenv("EXECUTION_TPSL_SHIELD_ENABLED", "false").lower()
    
    print(f"EXECUTION_TPSL_SHIELD_ENABLED={shield_enabled}")
    
    if shield_enabled in ("true", "1", "yes", "enabled"):
        print("❌ FAIL: TPSL shield is ENABLED (should be disabled)")
        return False
    else:
        print("✅ PASS: TPSL shield is DISABLED (policy compliant)")
        return True

async def test_market_orders_work():
    """Test 4: Gateway allows MARKET orders (baseline)"""
    print("\n" + "=" * 70)
    print("TEST 4: Gateway Allows MARKET Orders (Baseline)")
    print("=" * 70)
    
    try:
        from backend.services.execution.exit_order_gateway import submit_exit_order
        from unittest.mock import MagicMock
        
        # Create mock Binance client
        mock_client = MagicMock()
        mock_client.futures_create_order.return_value = {'orderId': 999999}
        
        # Attempt to submit MARKET order (should succeed)
        market_params = {
            'symbol': 'BTCUSDT',
            'side': 'SELL',
            'type': 'MARKET',  # ✅ ALLOWED
            'quantity': 0.001,
            'reduceOnly': True
        }
        
        result = await submit_exit_order(
            module_name="proof_test",
            symbol="BTCUSDT",
            order_params=market_params,
            order_kind="exit",
            client=mock_client,
            explanation="Proof test - MARKET order (should succeed)"
        )
        
        if result and result.get('orderId'):
            print("✅ PASS: Gateway allowed MARKET order")
            return True
        else:
            print("❌ FAIL: Gateway blocked MARKET order (should allow)")
            return False
            
    except Exception as e:
        print(f"❌ FAIL: MARKET order test exception: {e}")
        import traceback
        traceback.print_exc()
        return False

async def main_async():
    print("\n" + "=" * 70)
    print("🔒 CONDITIONAL ORDER POLICY ENFORCEMENT - PROOF")
    print("=" * 70)
    print("Policy: NO conditional orders. Internal intents + MARKET only.")
    print("Owner: Exit Brain v3.5 (sole exit decision authority)")
    print("=" * 70 + "\n")
    
    # Run tests (async ones need await)
    results = {
        "Gateway blocks STOP_MARKET": await test_gateway_blocks_conditional(),
        "No conditionals in logs": test_no_conditional_in_logs(),
        "TPSL shield disabled": test_tpsl_shield_disabled(),
        "Gateway allows MARKET": await test_market_orders_work()
    }
    
    # Summary
    print("\n" + "=" * 70)
    print("TEST SUMMARY")
    print("=" * 70)
    
    all_passed = True
    for test_name, passed in results.items():
        status = "✅ PASS" if passed else "❌ FAIL"
        print(f"{status}: {test_name}")
        if not passed:
            all_passed = False
    
    print("=" * 70)
    
    if all_passed:
        print("\n✅ ALL TESTS PASSED - Policy enforcement working")
        print("   Exit code: 0")
        return 0
    else:
        print("\n❌ SOME TESTS FAILED - Policy enforcement incomplete")
        print("   Exit code: 1")
        return 1

def main():
    return asyncio.run(main_async())

if __name__ == "__main__":
    sys.exit(main())
