#!/usr/bin/env bash
set -euo pipefail

# ==============================================================================
# P2.6 Portfolio Heat Gate - Proof Script
# ==============================================================================
# Tests:
# 1. COLD (heat_score < T_WARM=0.45) → FULL_CLOSE stays FULL_CLOSE (heat_action=NONE)
# 2. WARM (0.45 <= heat_score < 0.70) → FULL_CLOSE → PARTIAL_50 (DOWNGRADE)
# 3. HOT (heat_score >= 0.70) → FULL_CLOSE → PARTIAL_25 (DOWNGRADE)
# 4. UNKNOWN (missing portfolio state) → NONE + reason=missing_inputs
#
# Exit codes: 0 = PASS, 1 = FAIL
# ==============================================================================

# Auto-detect repo root (location-agnostic)
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

REDIS="redis-cli"
PYTHON3="python3"
INJECT_SCRIPT="$REPO_ROOT/scripts/proof_p26_inject_state.py"
TEST_SYMBOL="BTCUSDT"
FAILURES=0

echo "===================================================================="
echo "P2.6 Heat Gate Proof: Portfolio Heat Moderation"
echo "===================================================================="
echo "Test symbol: $TEST_SYMBOL"
echo ""

# ==============================================================================
# Helper Functions
# ==============================================================================

fail() {
    echo "❌ FAIL: $1"
    FAILURES=$((FAILURES + 1))
}

pass() {
    echo "✅ PASS: $1"
}

wait_for_processing() {
    echo "   Waiting 3s for heat gate to process..."
    sleep 3
}

cleanup() {
    echo ""
    echo "Cleanup: Clearing test data..."
    $PYTHON3 "$INJECT_SCRIPT" clear >/dev/null 2>&1 || true
    $REDIS DEL "quantum:harvest:heat:$TEST_SYMBOL" >/dev/null 2>&1 || true
}

# ==============================================================================
# Test 0: Preflight Checks
# ==============================================================================

echo "[0] Preflight: Redis connectivity"
if ! $REDIS PING >/dev/null 2>&1; then
    fail "Redis not available"
    exit 1
fi
pass "Redis available"

echo "[0] Preflight: Heat Gate service running"
if ! systemctl is-active --quiet quantum-heat-gate 2>/dev/null; then
    echo "⚠️  WARNING: Heat Gate service not running, tests may fail"
fi

echo "[0] Preflight: Heat Gate metrics endpoint"
if ! curl -s http://localhost:8068/metrics >/dev/null 2>&1; then
    echo "⚠️  WARNING: Heat Gate metrics not responding"
fi

# ==============================================================================
# Test 1: COLD (heat_score < T_WARM=0.45) → NONE
# ==============================================================================

echo ""
echo "[1] Test: COLD portfolio (heat_score < 0.45) → FULL_CLOSE stays FULL_CLOSE"

cleanup

# Inject low heat portfolio state
# gross=1000 (target=5000) => 0.2
# dd=50 (target=250) => 0.2
# burst=50 (target=250) => 0.2
# fee=10 (target=50) => 0.2
# churn=0.2 (target=1.0) => 0.2
# heat_score = (1.0*0.2 + 1.0*0.2 + 1.0*0.2 + 0.5*0.2 + 0.5*0.2) / (1+1+1+0.5+0.5) = 0.8/4.0 = 0.2
$PYTHON3 "$INJECT_SCRIPT" state 1000 50 50 10 0.2 >/dev/null

# Inject FULL_CLOSE_PROPOSED
$PYTHON3 "$INJECT_SCRIPT" proposal "$TEST_SYMBOL" "FULL_CLOSE_PROPOSED" 0.8 >/dev/null

wait_for_processing

# Check shadow key
SHADOW_KEY="quantum:harvest:heat:$TEST_SYMBOL"
if ! $REDIS EXISTS "$SHADOW_KEY" >/dev/null; then
    fail "Test 1: Shadow key not created"
else
    HEAT_LEVEL=$($REDIS HGET "$SHADOW_KEY" heat_level)
    HEAT_ACTION=$($REDIS HGET "$SHADOW_KEY" heat_action)
    OUT_ACTION=$($REDIS HGET "$SHADOW_KEY" out_action)
    
    if [ "$HEAT_LEVEL" = "cold" ] && [ "$HEAT_ACTION" = "NONE" ] && [ "$OUT_ACTION" = "FULL_CLOSE_PROPOSED" ]; then
        pass "Test 1: COLD → heat_action=NONE, out_action=FULL_CLOSE_PROPOSED"
    else
        fail "Test 1: Expected cold/NONE/FULL_CLOSE, got $HEAT_LEVEL/$HEAT_ACTION/$OUT_ACTION"
    fi
fi

# ==============================================================================
# Test 2: WARM (0.45 <= heat_score < 0.70) → DOWNGRADE to PARTIAL_50
# ==============================================================================

echo ""
echo "[2] Test: WARM portfolio (0.45 <= heat_score < 0.70) → DOWNGRADE to PARTIAL_50"

cleanup

# Inject medium heat portfolio state
# gross=2500 (target=5000) => 0.5
# dd=125 (target=250) => 0.5
# burst=125 (target=250) => 0.5
# fee=25 (target=50) => 0.5
# churn=0.5 (target=1.0) => 0.5
# heat_score = (1.0*0.5 + 1.0*0.5 + 1.0*0.5 + 0.5*0.5 + 0.5*0.5) / 4.0 = 2.0/4.0 = 0.5
$PYTHON3 "$INJECT_SCRIPT" state 2500 125 125 25 0.5 >/dev/null

# Inject FULL_CLOSE_PROPOSED
$PYTHON3 "$INJECT_SCRIPT" proposal "$TEST_SYMBOL" "FULL_CLOSE_PROPOSED" 0.8 >/dev/null

wait_for_processing

# Check shadow key
if ! $REDIS EXISTS "$SHADOW_KEY" >/dev/null; then
    fail "Test 2: Shadow key not created"
else
    HEAT_LEVEL=$($REDIS HGET "$SHADOW_KEY" heat_level)
    HEAT_ACTION=$($REDIS HGET "$SHADOW_KEY" heat_action)
    OUT_ACTION=$($REDIS HGET "$SHADOW_KEY" out_action)
    RECOMMENDED_PARTIAL=$($REDIS HGET "$SHADOW_KEY" recommended_partial)
    
    if [ "$HEAT_LEVEL" = "warm" ] && [ "$HEAT_ACTION" = "DOWNGRADE_FULL_TO_PARTIAL" ] && [ "$OUT_ACTION" = "PARTIAL_50_PROPOSED" ]; then
        pass "Test 2: WARM → DOWNGRADE_FULL_TO_PARTIAL, out_action=PARTIAL_50_PROPOSED, partial=$RECOMMENDED_PARTIAL"
    else
        fail "Test 2: Expected warm/DOWNGRADE/PARTIAL_50, got $HEAT_LEVEL/$HEAT_ACTION/$OUT_ACTION"
    fi
fi

# ==============================================================================
# Test 3: HOT (heat_score >= 0.70) → DOWNGRADE to PARTIAL_25
# ==============================================================================

echo ""
echo "[3] Test: HOT portfolio (heat_score >= 0.70) → DOWNGRADE to PARTIAL_25"

cleanup

# Inject high heat portfolio state
# gross=4000 (target=5000) => 0.8
# dd=200 (target=250) => 0.8
# burst=200 (target=250) => 0.8
# fee=40 (target=50) => 0.8
# churn=0.8 (target=1.0) => 0.8
# heat_score = (1.0*0.8 + 1.0*0.8 + 1.0*0.8 + 0.5*0.8 + 0.5*0.8) / 4.0 = 3.2/4.0 = 0.8
$PYTHON3 "$INJECT_SCRIPT" state 4000 200 200 40 0.8 >/dev/null

# Inject FULL_CLOSE_PROPOSED
$PYTHON3 "$INJECT_SCRIPT" proposal "$TEST_SYMBOL" "FULL_CLOSE_PROPOSED" 0.8 >/dev/null

wait_for_processing

# Check shadow key
if ! $REDIS EXISTS "$SHADOW_KEY" >/dev/null; then
    fail "Test 3: Shadow key not created"
else
    HEAT_LEVEL=$($REDIS HGET "$SHADOW_KEY" heat_level)
    HEAT_ACTION=$($REDIS HGET "$SHADOW_KEY" heat_action)
    OUT_ACTION=$($REDIS HGET "$SHADOW_KEY" out_action)
    RECOMMENDED_PARTIAL=$($REDIS HGET "$SHADOW_KEY" recommended_partial)
    
    if [ "$HEAT_LEVEL" = "hot" ] && [ "$HEAT_ACTION" = "DOWNGRADE_FULL_TO_PARTIAL" ] && [ "$OUT_ACTION" = "PARTIAL_25_PROPOSED" ]; then
        pass "Test 3: HOT → DOWNGRADE_FULL_TO_PARTIAL, out_action=PARTIAL_25_PROPOSED, partial=$RECOMMENDED_PARTIAL"
    else
        fail "Test 3: Expected hot/DOWNGRADE/PARTIAL_25, got $HEAT_LEVEL/$HEAT_ACTION/$OUT_ACTION"
    fi
fi

# ==============================================================================
# Test 4: UNKNOWN (missing portfolio state) → FAIL-OPEN
# ==============================================================================

echo ""
echo "[4] Test: Missing portfolio state → FAIL-OPEN (heat_action=NONE, reason=missing_inputs)"

cleanup

# NO portfolio state injected

# Inject FULL_CLOSE_PROPOSED
$PYTHON3 "$INJECT_SCRIPT" proposal "$TEST_SYMBOL" "FULL_CLOSE_PROPOSED" 0.8 >/dev/null

wait_for_processing

# Check shadow key
if ! $REDIS EXISTS "$SHADOW_KEY" >/dev/null; then
    fail "Test 4: Shadow key not created"
else
    HEAT_LEVEL=$($REDIS HGET "$SHADOW_KEY" heat_level)
    HEAT_ACTION=$($REDIS HGET "$SHADOW_KEY" heat_action)
    OUT_ACTION=$($REDIS HGET "$SHADOW_KEY" out_action)
    REASON=$($REDIS HGET "$SHADOW_KEY" reason)
    
    if [ "$HEAT_LEVEL" = "unknown" ] && [ "$HEAT_ACTION" = "NONE" ] && [ "$OUT_ACTION" = "FULL_CLOSE_PROPOSED" ] && [ "$REASON" = "missing_inputs" ]; then
        pass "Test 4: UNKNOWN → heat_action=NONE, out_action=FULL_CLOSE (fail-open), reason=missing_inputs"
    else
        fail "Test 4: Expected unknown/NONE/FULL_CLOSE/missing_inputs, got $HEAT_LEVEL/$HEAT_ACTION/$OUT_ACTION/$REASON"
    fi
fi

# ==============================================================================
# Test 5: Stream Output Verification
# ==============================================================================

echo ""
echo "[5] Verify heat decision stream output"

STREAM_KEY="quantum:stream:harvest.heat.decision"
STREAM_LEN=$($REDIS XLEN "$STREAM_KEY" 2>/dev/null || echo "0")

if [ "$STREAM_LEN" -ge 4 ]; then
    pass "Test 5: Heat decision stream has $STREAM_LEN entries (>= 4 tests)"
else
    fail "Test 5: Heat decision stream has only $STREAM_LEN entries (expected >= 4)"
fi

# ==============================================================================
# Test 6: Metrics Verification
# ==============================================================================

echo ""
echo "[6] Verify Heat Gate metrics"

if ! curl -s http://localhost:8068/metrics >/dev/null 2>&1; then
    echo "⚠️  WARNING: Heat Gate metrics endpoint not responding"
else
    # Check if P2.6 metrics are present
    LOOPS_METRIC=$(curl -s http://localhost:8068/metrics | grep "p26_loops_total" || echo "")
    if [ -n "$LOOPS_METRIC" ]; then
        pass "Test 6: P2.6 metrics registered (p26_loops_total)"
    else
        fail "Test 6: P2.6 metrics not found in Prometheus output"
    fi
fi

# ==============================================================================
# Test 7: Service Health
# ==============================================================================

echo ""
echo "[7] Heat Gate service health"

if systemctl is-active --quiet quantum-heat-gate 2>/dev/null; then
    pass "Heat Gate service is active"
else
    fail "Heat Gate service is not active"
fi

# Check for errors in logs
ERROR_COUNT=$(journalctl -u quantum-heat-gate -n 30 --no-pager 2>/dev/null | grep -ic "error" || echo 0)
if [ "$ERROR_COUNT" -eq 0 ]; then
    pass "No errors in recent heat gate logs"
else
    echo "⚠️  WARNING: Found $ERROR_COUNT errors in recent logs"
fi

# ==============================================================================
# Cleanup and Summary
# ==============================================================================

cleanup

echo ""
echo "===================================================================="
if [ $FAILURES -eq 0 ]; then
    echo "✅ SUMMARY: PASS (all tests passed)"
    echo "===================================================================="
    exit 0
else
    echo "❌ SUMMARY: FAIL ($FAILURES test(s) failed)"
    echo "===================================================================="
    echo ""
    echo "Debugging commands:"
    echo "  journalctl -u quantum-heat-gate -n 50 --no-pager | tail -30"
    echo "  redis-cli HGETALL quantum:harvest:heat:$TEST_SYMBOL"
    echo "  redis-cli XREVRANGE quantum:stream:harvest.heat.decision + - COUNT 5"
    echo "  curl -s http://localhost:8068/metrics | grep p26_"
    exit 1
fi
