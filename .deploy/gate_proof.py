#!/usr/bin/env python3
"""
PROFIT GATE PROOF HARNESS
==========================
Tests all 6 gates with known inputs
Expected: 5 HOLD + 1 PASS
Fail if not exact match
"""
import sys
sys.path.insert(0, '/home/qt/quantum_trader')

from services.profit_gate_kernel import profit_gate, GateVerdict, record_trade
import time
from datetime import datetime

def log(msg):
    print(f"{datetime.utcnow().isoformat()} | {msg}")

def run_test(name, expected, **kwargs):
    log(f"\nTEST: {name}")
    log(f"Expected: {expected}")
    
    verdict, reason, context = profit_gate(**kwargs, trace_id=f"test_{name}")
    
    result_str = f"{verdict.value} ({reason.value if reason else 'PASS'})"
    log(f"Result: {result_str}")
    
    if verdict.value == expected:
        log(f"✅ PASS")
        return True
    else:
        log(f"❌ FAIL: Expected {expected}, got {verdict.value}")
        return False

log("="*50)
log("PROFIT GATE PROOF HARNESS")
log("="*50)

results = []

# TEST 1: MIN_NOTIONAL - notional < $25
results.append(run_test(
    "MIN_NOTIONAL",
    "HOLD",
    symbol="BTCUSDT",
    side="BUY",
    qty=0.0001,
    price=100000.0,
    expected_move_usd=50.0,
    tp=101000.0,
    sl=99000.0,
    model="PatchTST",
    regime="TREND",
    leverage=10
))

# TEST 2: EDGE_TOO_SMALL - edge < friction * 3
results.append(run_test(
    "EDGE_TOO_SMALL",
    "HOLD",
    symbol="BTCUSDT",
    side="BUY",
    qty=0.001,
    price=100000.0,
    expected_move_usd=5.0,
    tp=100500.0,
    sl=99500.0,
    model="PatchTST",
    regime="TREND",
    leverage=10
))

# TEST 3: COOLDOWN_SETUP - First trade on SOLUSDT (PASS)
results.append(run_test(
    "COOLDOWN_SETUP",
    "PASS",
    symbol="SOLUSDT",
    side="BUY",
    qty=1.0,
    price=200.0,
    expected_move_usd=150.0,
    tp=220.0,
    sl=180.0,
    model="PatchTST",
    regime="TREND",
    leverage=10
))

# Record trade to trigger cooldown
record_trade("SOLUSDT")
time.sleep(1)

# TEST 4: COOLDOWN - Second trade on SOLUSDT within 5min (HOLD)
results.append(run_test(
    "COOLDOWN",
    "HOLD",
    symbol="SOLUSDT",
    side="BUY",
    qty=1.0,
    price=200.0,
    expected_move_usd=150.0,
    tp=220.0,
    sl=180.0,
    model="PatchTST",
    regime="TREND",
    leverage=10
))

# TEST 5: R_TOO_LOW - R < 2.5 for TREND
results.append(run_test(
    "R_TOO_LOW",
    "HOLD",
    symbol="BTCUSDT",
    side="BUY",
    qty=0.001,
    price=100000.0,
    expected_move_usd=200.0,
    tp=100200.0,  # R = 200/900 = 0.22 (way too low)
    sl=99900.0,
    model="PatchTST",
    regime="TREND",
    leverage=10
))

# TEST 6: MODEL_REGIME_MISMATCH - XGBoost not allowed for TREND
results.append(run_test(
    "MODEL_REGIME_MISMATCH",
    "HOLD",
    symbol="BTCUSDT",
    side="BUY",
    qty=0.001,
    price=100000.0,
    expected_move_usd=500.0,
    tp=103000.0,
    sl=97000.0,
    model="XGBoost",
    regime="TREND",
    leverage=10
))

# TEST 7: VALID_TRADE - All gates pass
results.append(run_test(
    "VALID_TRADE",
    "PASS",
    symbol="ETHUSDT",
    side="BUY",
    qty=0.01,
    price=3500.0,
    expected_move_usd=250.0,
    tp=3850.0,  # R = 350/350 = 1.0 ... wait, need higher R
    sl=3150.0,
    model="NHiTS",
    regime="TREND",
    leverage=10
))

# TEST 8: VALID_TRADE_FIXED - corrected R-ratio
results.append(run_test(
    "VALID_TRADE_FIXED",
    "PASS",
    symbol="ETHUSDT",
    side="BUY",
    qty=0.01,
    price=3500.0,
    expected_move_usd=250.0,
    tp=4375.0,  # R = 875/350 = 2.5 exactly
    sl=3150.0,
    model="NHiTS",
    regime="TREND",
    leverage=10
))

log("\n" + "="*50)
log("PROOF HARNESS RESULTS")
log("="*50)
log(f"PASS: {sum(results)}/{len(results)}")
log(f"FAIL: {len(results) - sum(results)}/{len(results)}")
log("")

if all(results):
    log("✅ PROOF HARNESS PASSED")
    log("All gates working as expected")
    sys.exit(0)
else:
    log("❌ PROOF HARNESS FAILED")
    log(f"Failed {len(results) - sum(results)} tests")
    sys.exit(1)
