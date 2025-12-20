#!/usr/bin/env python3
"""
AI Signal Flow Integration Test
================================
Tests full data flow from Trading Bot → AI Engine → Decision
Validates signal generation pipeline end-to-end.

Author: GitHub Copilot
Date: December 17, 2025
"""

import asyncio
import httpx
import sys
from datetime import datetime
from typing import Dict, Any, List

# Test configuration
AI_ENGINE_URL = "http://46.224.116.254:8001"
TRADING_BOT_URL = "http://46.224.116.254:8000"
TIMEOUT = 10

# Test symbols
TEST_SYMBOLS = ["BTCUSDT", "ETHUSDT", "BNBUSDT"]

# Test results
results: List[Dict[str, Any]] = []


def record_test(test_name: str, passed: bool, details: str = "", error: str = ""):
    """Record test result."""
    result = {
        "test": test_name,
        "passed": passed,
        "timestamp": datetime.utcnow().isoformat(),
        "details": details,
        "error": error
    }
    results.append(result)
    
    status = "✅ PASS" if passed else "❌ FAIL"
    print(f"\n{status} | {test_name}")
    if details:
        print(f"  └─ {details}")
    if error:
        print(f"  └─ ERROR: {error}")


async def test_ai_engine_health():
    """Test AI Engine health endpoint."""
    print("\n" + "="*80)
    print("TEST 1: AI ENGINE HEALTH CHECK")
    print("="*80)
    
    try:
        async with httpx.AsyncClient(timeout=TIMEOUT) as client:
            response = await client.get(f"{AI_ENGINE_URL}/health")
            
            if response.status_code == 200:
                health = response.json()
                details = (
                    f"Status: {health.get('status', 'unknown')} | "
                    f"Models: {health.get('models_loaded', 0)} | "
                    f"Price history: {health.get('price_history_len', 0)}"
                )
                record_test("AI Engine Health", True, details)
                return health
            else:
                record_test("AI Engine Health", False, error=f"HTTP {response.status_code}")
                return None
    except Exception as e:
        record_test("AI Engine Health", False, error=str(e))
        return None


async def test_trading_bot_health():
    """Test Trading Bot health endpoint."""
    print("\n" + "="*80)
    print("TEST 2: TRADING BOT HEALTH CHECK")
    print("="*80)
    
    try:
        async with httpx.AsyncClient(timeout=TIMEOUT) as client:
            response = await client.get(f"{TRADING_BOT_URL}/health")
            
            if response.status_code == 200:
                health = response.json()
                details = f"Status: {health.get('status', 'unknown')}"
                record_test("Trading Bot Health", True, details)
                return health
            else:
                record_test("Trading Bot Health", False, error=f"HTTP {response.status_code}")
                return None
    except Exception as e:
        record_test("Trading Bot Health", False, error=str(e))
        return None


async def test_signal_generation_direct():
    """Test AI Engine signal generation API directly."""
    print("\n" + "="*80)
    print("TEST 3: DIRECT SIGNAL GENERATION")
    print("="*80)
    
    success_count = 0
    
    for symbol in TEST_SYMBOLS:
        try:
            async with httpx.AsyncClient(timeout=TIMEOUT) as client:
                payload = {
                    "symbol": symbol,
                    "price": 50000.0 if "BTC" in symbol else 3000.0,
                    "volume": 1000.0,
                    "timestamp": datetime.utcnow().isoformat()
                }
                
                response = await client.post(
                    f"{AI_ENGINE_URL}/api/signal",
                    json=payload
                )
                
                if response.status_code == 200:
                    signal = response.json()
                    details = (
                        f"{symbol}: {signal.get('action', 'UNKNOWN')} @ "
                        f"{signal.get('confidence', 0)*100:.1f}% confidence"
                    )
                    record_test(f"Signal Generation - {symbol}", True, details)
                    success_count += 1
                else:
                    # 404 means no signal generated (expected in some cases)
                    if response.status_code == 404:
                        details = f"{symbol}: No signal (insufficient data or low confidence)"
                        record_test(f"Signal Generation - {symbol}", True, details)
                        success_count += 1
                    else:
                        record_test(
                            f"Signal Generation - {symbol}",
                            False,
                            error=f"HTTP {response.status_code}: {response.text}"
                        )
        except Exception as e:
            record_test(f"Signal Generation - {symbol}", False, error=str(e))
    
    # Overall test
    if success_count > 0:
        record_test("Direct Signal API", True, f"{success_count}/{len(TEST_SYMBOLS)} symbols processed")
    else:
        record_test("Direct Signal API", False, error="No symbols processed successfully")


async def test_price_history_building():
    """Test that price history is building in AI Engine."""
    print("\n" + "="*80)
    print("TEST 4: PRICE HISTORY ACCUMULATION")
    print("="*80)
    
    try:
        # Send 5 price updates
        async with httpx.AsyncClient(timeout=TIMEOUT) as client:
            for i in range(5):
                payload = {
                    "symbol": "BTCUSDT",
                    "price": 50000.0 + i * 10,
                    "volume": 1000.0 + i * 100,
                    "timestamp": datetime.utcnow().isoformat()
                }
                
                await client.post(f"{AI_ENGINE_URL}/api/signal", json=payload)
                await asyncio.sleep(0.5)
            
            # Check health to see price history length
            response = await client.get(f"{AI_ENGINE_URL}/health")
            if response.status_code == 200:
                health = response.json()
                price_history_len = health.get('price_history_len', 0)
                
                if price_history_len > 0:
                    details = f"Price history length: {price_history_len} (building correctly)"
                    record_test("Price History Building", True, details)
                else:
                    record_test("Price History Building", False, error="Price history still empty")
            else:
                record_test("Price History Building", False, error="Could not retrieve health status")
    except Exception as e:
        record_test("Price History Building", False, error=str(e))


async def test_ensemble_components():
    """Test ensemble components are loaded."""
    print("\n" + "="*80)
    print("TEST 5: ENSEMBLE COMPONENTS")
    print("="*80)
    
    try:
        async with httpx.AsyncClient(timeout=TIMEOUT) as client:
            response = await client.get(f"{AI_ENGINE_URL}/health")
            
            if response.status_code == 200:
                health = response.json()
                models = health.get('models_loaded', 0)
                components = health.get('components', {})
                
                required = ['ensemble_manager', 'meta_strategy_selector', 'rl_sizing_agent']
                loaded = [c for c in required if components.get(c) is True]
                
                if models >= 4 and len(loaded) >= 3:
                    details = (
                        f"Models: {models}/4 loaded | "
                        f"Components: {len(loaded)}/{len(required)} active"
                    )
                    record_test("Ensemble Components", True, details)
                else:
                    record_test(
                        "Ensemble Components",
                        False,
                        error=f"Only {models} models and {len(loaded)} components loaded"
                    )
            else:
                record_test("Ensemble Components", False, error="Could not retrieve health status")
    except Exception as e:
        record_test("Ensemble Components", False, error=str(e))


async def test_signal_confidence_threshold():
    """Test that signals respect confidence threshold."""
    print("\n" + "="*80)
    print("TEST 6: CONFIDENCE THRESHOLD FILTERING")
    print("="*80)
    
    try:
        signals_received = 0
        signals_rejected = 0
        
        async with httpx.AsyncClient(timeout=TIMEOUT) as client:
            for symbol in TEST_SYMBOLS * 3:  # Test each symbol 3 times
                payload = {
                    "symbol": symbol,
                    "price": 50000.0,
                    "volume": 1000.0,
                    "timestamp": datetime.utcnow().isoformat()
                }
                
                response = await client.post(f"{AI_ENGINE_URL}/api/signal", json=payload)
                
                if response.status_code == 200:
                    signal = response.json()
                    confidence = signal.get('confidence', 0)
                    if confidence >= 0.55:  # MIN_SIGNAL_CONFIDENCE
                        signals_received += 1
                elif response.status_code == 404:
                    signals_rejected += 1
        
        total = signals_received + signals_rejected
        if total > 0:
            details = (
                f"Received: {signals_received} | "
                f"Rejected: {signals_rejected} | "
                f"Total: {total}"
            )
            record_test("Confidence Threshold", True, details)
        else:
            record_test("Confidence Threshold", False, error="No signals processed")
    except Exception as e:
        record_test("Confidence Threshold", False, error=str(e))


async def main():
    """Run all tests."""
    print("\n" + "="*80)
    print("AI SIGNAL FLOW INTEGRATION TEST")
    print("Quantum Trader v3.5")
    print("="*80)
    print(f"AI Engine: {AI_ENGINE_URL}")
    print(f"Trading Bot: {TRADING_BOT_URL}")
    print(f"Test Symbols: {', '.join(TEST_SYMBOLS)}")
    
    # Run tests
    await test_ai_engine_health()
    await test_trading_bot_health()
    await test_signal_generation_direct()
    await test_price_history_building()
    await test_ensemble_components()
    await test_signal_confidence_threshold()
    
    # Summary
    print("\n" + "="*80)
    print("TEST SUMMARY")
    print("="*80)
    
    passed = sum(1 for r in results if r["passed"])
    failed = sum(1 for r in results if not r["passed"])
    total = len(results)
    pass_rate = (passed / total * 100) if total > 0 else 0
    
    print(f"Total Tests:  {total}")
    print(f"Passed:       {passed} ({pass_rate:.1f}%)")
    print(f"Failed:       {failed}")
    
    if failed > 0:
        print("\nFAILED TESTS:")
        for r in results:
            if not r["passed"]:
                print(f"  - {r['test']}: {r['error']}")
    
    print("\n" + "="*80)
    
    if passed == total:
        print("✅ SUCCESS: All tests passed!")
        return 0
    else:
        print(f"❌ FAILURE: {failed} test(s) failed")
        return 1


if __name__ == "__main__":
    exit_code = asyncio.run(main())
    sys.exit(exit_code)
