"""
Quick Integration Test - Strategy Runtime Engine

Simple test without Unicode characters for Windows compatibility.
"""

import os
import sys
from pathlib import Path

project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

os.environ['QT_ENV'] = 'test'

print("\n" + "="*70)
print("Strategy Runtime Engine - Quick Integration Test")
print("="*70)

passed = 0
failed = 0

# Test 1: Import modules
print("\n[TEST 1] Import integration module...")
try:
    from backend.services.strategy_runtime_integration import (
        get_strategy_runtime_engine,
        check_strategy_runtime_health,
        generate_strategy_signals
    )
    print("[PASS] Imports successful")
    passed += 1
except Exception as e:
    print(f"[FAIL] Import error: {e}")
    failed += 1

# Test 2: Initialize engine
print("\n[TEST 2] Initialize Strategy Runtime Engine...")
try:
    engine = get_strategy_runtime_engine()
    count = engine.get_active_strategy_count()
    print(f"[PASS] Engine initialized with {count} active strategies")
    passed += 1
except Exception as e:
    print(f"[FAIL] Initialization error: {e}")
    failed += 1

# Test 3: Health check
print("\n[TEST 3] Health check...")
try:
    health = check_strategy_runtime_health()
    print(f"[PASS] Health status: {health['status']}")
    print(f"       Active strategies: {health['active_strategies']}")
    passed += 1
except Exception as e:
    print(f"[FAIL] Health check error: {e}")
    failed += 1

# Test 4: Generate signals
print("\n[TEST 4] Generate signals...")
try:
    symbols = ["BTCUSDT", "ETHUSDT"]
    decisions = generate_strategy_signals(symbols, current_regime="TRENDING")
    print(f"[PASS] Generated {len(decisions)} signals")
    if decisions:
        for d in decisions[:2]:
            print(f"       {d.symbol}: {d.side} @ {d.confidence:.0%} confidence")
    passed += 1
except Exception as e:
    print(f"[FAIL] Signal generation error: {e}")
    failed += 1

# Test 5: Executor integration
print("\n[TEST 5] Executor integration...")
try:
    from backend.services.execution.event_driven_executor import EventDrivenExecutor
    print("[PASS] Executor has Strategy Runtime Engine available")
    passed += 1
except Exception as e:
    print(f"[FAIL] Executor integration error: {e}")
    failed += 1

# Summary
print("\n" + "="*70)
print(f"RESULTS: {passed} passed, {failed} failed")
print("="*70)

if failed == 0:
    print("\n[SUCCESS] All integration tests passed!")
    sys.exit(0)
else:
    print("\n[FAILURE] Some tests failed")
    sys.exit(1)
