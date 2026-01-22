#!/bin/bash
#
# PROFIT GATE PROOF HARNESS
# ==========================
# Tests all 6 gates with known inputs
# Expected: 5 HOLD + 1 PASS
# Fail if not exact match
#
set -euo pipefail

LOGFILE=/tmp/gate_proof_$(date +%Y%m%d_%H%M%S).log

log() { echo "$(date -Iseconds) | $*" | tee -a $LOGFILE; }

log "================================"
log "PROFIT GATE PROOF HARNESS"
log "================================"

# Verify Python environment
if ! python3 -c "import sys; sys.path.insert(0, '/home/qt/quantum_trader'); from services.profit_gate_kernel import profit_gate" 2>/dev/null; then
    log "❌ FAILED: Cannot import profit_gate_kernel"
    exit 1
fi

log "✅ profit_gate_kernel importable"

# Test runner function
run_test() {
    local test_name="$1"
    shift
    local expected_verdict="$1"
    shift
    
    log ""
    log "TEST: $test_name"
    log "Expected: $expected_verdict"
    
    # Run Python test
    RESULT=$(python3 << PYEOF
import sys
sys.path.insert(0, '/home/qt/quantum_trader')
from services.profit_gate_kernel import profit_gate, GateVerdict

verdict, reason, context = profit_gate(
    symbol="$2",
    side="$3",
    qty=$4,
    price=$5,
    expected_move_usd=$6,
    tp=$7,
    sl=$8,
    model="$9",
    regime="${10}",
    leverage=${11},
    trace_id="test_${test_name}"
)

print(f"{verdict.value}|{reason.value if reason else 'NONE'}|{context.get('gate_failed', 'NONE')}")
PYEOF
)
    
    VERDICT=$(echo "$RESULT" | cut -d'|' -f1)
    REASON=$(echo "$RESULT" | cut -d'|' -f2)
    GATE=$(echo "$RESULT" | cut -d'|' -f3)
    
    if [ "$VERDICT" == "$expected_verdict" ]; then
        log "✅ PASS: $test_name -> $VERDICT ($REASON)"
        return 0
    else
        log "❌ FAIL: $test_name -> Expected $expected_verdict, got $VERDICT ($REASON)"
        return 1
    fi
}

PASS_COUNT=0
FAIL_COUNT=0

# TEST 1: Small notional (< $25) -> HOLD
run_test "MIN_NOTIONAL" "HOLD" \
    "BTCUSDT" "BUY" 0.0001 100000 50.0 101000 99000 "PatchTST" "TREND" 10 \
    && PASS_COUNT=$((PASS_COUNT+1)) || FAIL_COUNT=$((FAIL_COUNT+1))

# TEST 2: Edge too small (< friction * 3) -> HOLD
run_test "EDGE_TOO_SMALL" "HOLD" \
    "BTCUSDT" "BUY" 0.001 100000 5.0 100500 99500 "PatchTST" "TREND" 10 \
    && PASS_COUNT=$((PASS_COUNT+1)) || FAIL_COUNT=$((FAIL_COUNT+1))

# TEST 3: Cooldown violation -> HOLD (simulate by running twice quickly)
run_test "COOLDOWN_SETUP" "PASS" \
    "SOLUSDT" "BUY" 1.0 200 150.0 220 180 "PatchTST" "TREND" 10 \
    && PASS_COUNT=$((PASS_COUNT+1)) || FAIL_COUNT=$((FAIL_COUNT+1))

# Record trade to trigger cooldown
python3 << PYEOF
import sys
sys.path.insert(0, '/home/qt/quantum_trader')
from services.profit_gate_kernel import record_trade
record_trade("SOLUSDT")
PYEOF

sleep 1  # Brief pause

run_test "COOLDOWN" "HOLD" \
    "SOLUSDT" "BUY" 1.0 200 150.0 220 180 "PatchTST" "TREND" 10 \
    && PASS_COUNT=$((PASS_COUNT+1)) || FAIL_COUNT=$((FAIL_COUNT+1))

# TEST 4: R-ratio too low -> HOLD
run_test "R_TOO_LOW" "HOLD" \
    "BTCUSDT" "BUY" 0.001 100000 200.0 100200 99900 "PatchTST" "TREND" 10 \
    && PASS_COUNT=$((PASS_COUNT+1)) || FAIL_COUNT=$((FAIL_COUNT+1))

# TEST 5: Wrong model for regime -> HOLD
run_test "MODEL_REGIME_MISMATCH" "HOLD" \
    "BTCUSDT" "BUY" 0.001 100000 500.0 103000 97000 "XGBoost" "TREND" 10 \
    && PASS_COUNT=$((PASS_COUNT+1)) || FAIL_COUNT=$((FAIL_COUNT+1))

# TEST 6: Valid trade -> PASS
run_test "VALID_TRADE" "PASS" \
    "ETHUSDT" "BUY" 0.01 3500 250.0 3850 3150 "NHiTS" "TREND" 10 \
    && PASS_COUNT=$((PASS_COUNT+1)) || FAIL_COUNT=$((FAIL_COUNT+1))

log ""
log "================================"
log "PROOF HARNESS RESULTS"
log "================================"
log "PASS: $PASS_COUNT"
log "FAIL: $FAIL_COUNT"
log ""

if [ $FAIL_COUNT -gt 0 ]; then
    log "❌ PROOF HARNESS FAILED"
    log "Log: $LOGFILE"
    exit 1
else
    log "✅ PROOF HARNESS PASSED"
    log "All gates working as expected"
    log "Log: $LOGFILE"
    exit 0
fi
