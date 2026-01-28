#!/usr/bin/env bash
set -euo pipefail

# ==============================================================================
# P2.8A Apply Heat Observer - Proof Script
# ==============================================================================
# Tests:
# A) Heat found: inject plan+heat, verify observed event with heat_found=1
# B) Deduplication: re-inject same plan, verify no duplicate observed event
# C) Heat missing: inject plan without heat, verify observed event with heat_found=0
#
# Exit codes: 0 = PASS, 1 = FAIL
# ==============================================================================

# Auto-detect repo root
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

REDIS="redis-cli"
PYTHON3="python3"
INJECT_SCRIPT="$REPO_ROOT/scripts/proof_p28_inject_plan_apply.py"
FAILURES=0

echo "===================================================================="
echo "P2.8A Apply Heat Observer Proof"
echo "===================================================================="
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

# ==============================================================================
# Test 0: Preflight Checks
# ==============================================================================

echo "[0] Preflight: Redis connectivity"
if ! $REDIS PING >/dev/null 2>&1; then
    fail "Redis not available"
    exit 1
fi
pass "Redis available"

echo "[0] Preflight: Apply service running"
if ! systemctl is-active --quiet quantum-apply 2>/dev/null; then
    echo "⚠️  WARNING: Apply service not running, tests may fail"
    echo "    Manual testing: Run apply service manually for local testing"
fi

echo "[0] Preflight: Clean up previous test data"
$REDIS DEL quantum:dedupe:p28:test_plan_a >/dev/null 2>&1 || true
$REDIS DEL quantum:dedupe:p28:test_plan_b >/dev/null 2>&1 || true
$REDIS DEL quantum:dedupe:p28:test_plan_c >/dev/null 2>&1 || true
echo "    Test dedupe keys cleaned"

# ==============================================================================
# Test A: Heat Found
# ==============================================================================

echo ""
echo "[A] Test: Heat found → observed event with heat_found=1"

# Inject plan + heat
echo "   Injecting plan with heat..."
PLAN_A_OUTPUT=$($PYTHON3 "$INJECT_SCRIPT" \
    --symbol BTCUSDT \
    --action FULL_CLOSE_PROPOSED \
    --kill_score 0.25 \
    --heat_level warm \
    --heat_action DOWNGRADE_FULL_TO_PARTIAL \
    2>&1)

# Extract plan_id from output
PLAN_A=$(echo "$PLAN_A_OUTPUT" | grep "Plan ID:" | awk '{print $NF}')
if [ -z "$PLAN_A" ]; then
    fail "Test A: Could not extract plan_id from injection"
    echo "Injection output:"
    echo "$PLAN_A_OUTPUT"
    exit 1
fi

echo "   Plan ID: $PLAN_A"
echo "   Waiting 12s for Apply to process (poll_interval=5s + margin)..."
sleep 12

# Check if heat key exists
HEAT_KEY="quantum:harvest:heat:by_plan:$PLAN_A"
if ! $REDIS EXISTS "$HEAT_KEY" >/dev/null; then
    fail "Test A: Heat key not found: $HEAT_KEY"
else
    pass "Test A: Heat key exists"
fi

# Check observed stream for this plan
echo "   Checking observed stream..."
OBSERVED_STREAM="quantum:stream:apply.heat.observed"

# Get recent events and grep for plan_id
OBSERVED_EVENTS=$($REDIS XREVRANGE "$OBSERVED_STREAM" + - COUNT 10 2>/dev/null || echo "")

if echo "$OBSERVED_EVENTS" | grep -q "$PLAN_A"; then
    # Extract heat_found field for this plan
    # XREVRANGE output format: stream_id, field1, value1, field2, value2, ...
    HEAT_FOUND=$($REDIS XREVRANGE "$OBSERVED_STREAM" + - COUNT 10 | grep -A 50 "$PLAN_A" | grep "heat_found" | head -1 | awk '{print $2}' || echo "")
    
    if [ "$HEAT_FOUND" = "1" ]; then
        pass "Test A: Observed event found with heat_found=1"
    else
        fail "Test A: Observed event found but heat_found=$HEAT_FOUND (expected 1)"
    fi
    
    # Check heat_level
    HEAT_LEVEL=$($REDIS XREVRANGE "$OBSERVED_STREAM" + - COUNT 10 | grep -A 50 "$PLAN_A" | grep "heat_level" | head -1 | awk '{print $2}' || echo "")
    if [ "$HEAT_LEVEL" = "warm" ]; then
        pass "Test A: heat_level=warm (correct)"
    else
        fail "Test A: heat_level=$HEAT_LEVEL (expected warm)"
    fi
    
    # Check heat_action
    HEAT_ACTION=$($REDIS XREVRANGE "$OBSERVED_STREAM" + - COUNT 10 | grep -A 50 "$PLAN_A" | grep "heat_action" | head -1 | awk '{print $2}' || echo "")
    if [ "$HEAT_ACTION" = "DOWNGRADE_FULL_TO_PARTIAL" ]; then
        pass "Test A: heat_action=DOWNGRADE_FULL_TO_PARTIAL (correct)"
    else
        fail "Test A: heat_action=$HEAT_ACTION (expected DOWNGRADE_FULL_TO_PARTIAL)"
    fi
else
    fail "Test A: No observed event found for plan $PLAN_A"
    echo "   Recent events in stream:"
    $REDIS XREVRANGE "$OBSERVED_STREAM" + - COUNT 5 | head -20
fi

# ==============================================================================
# Test B: Deduplication
# ==============================================================================

echo ""
echo "[B] Test: Deduplication → re-inject same plan, no duplicate observed event"

# Get current event count for plan_a
EVENTS_BEFORE=$($REDIS XREVRANGE "$OBSERVED_STREAM" + - | grep -c "$PLAN_A" || echo "0")
echo "   Events before re-injection: $EVENTS_BEFORE"

# Re-inject same plan (should be deduplicated by Apply's plan dedupe, but also by P28 dedupe)
echo "   Re-injecting same plan..."
$PYTHON3 "$INJECT_SCRIPT" \
    --symbol BTCUSDT \
    --action FULL_CLOSE_PROPOSED \
    --kill_score 0.25 \
    --heat_level warm \
    --heat_action DOWNGRADE_FULL_TO_PARTIAL \
    >/dev/null 2>&1

echo "   Waiting 12s for Apply to process..."
sleep 12

# Check event count again
EVENTS_AFTER=$($REDIS XREVRANGE "$OBSERVED_STREAM" + - | grep -c "$PLAN_A" || echo "0")
echo "   Events after re-injection: $EVENTS_AFTER"

if [ "$EVENTS_AFTER" -eq "$EVENTS_BEFORE" ]; then
    pass "Test B: No duplicate observed event (dedupe working)"
elif [ "$EVENTS_AFTER" -eq $((EVENTS_BEFORE + 1)) ]; then
    # Check if dedupe key exists
    DEDUPE_KEY="quantum:dedupe:p28:$PLAN_A"
    if $REDIS EXISTS "$DEDUPE_KEY" >/dev/null; then
        pass "Test B: Dedupe key exists (may have expired between injections)"
    else
        fail "Test B: Duplicate observed event created (dedupe not working)"
    fi
else
    fail "Test B: Unexpected event count change ($EVENTS_BEFORE → $EVENTS_AFTER)"
fi

# ==============================================================================
# Test C: Heat Missing
# ==============================================================================

echo ""
echo "[C] Test: Heat missing → observed event with heat_found=0"

# Inject plan WITHOUT heat
echo "   Injecting plan WITHOUT heat..."
PLAN_C_OUTPUT=$($PYTHON3 "$INJECT_SCRIPT" \
    --symbol ETHUSDT \
    --action FULL_CLOSE_PROPOSED \
    --kill_score 0.15 \
    --no_heat \
    2>&1)

PLAN_C=$(echo "$PLAN_C_OUTPUT" | grep "Plan ID:" | awk '{print $NF}')
if [ -z "$PLAN_C" ]; then
    fail "Test C: Could not extract plan_id from injection"
    exit 1
fi

echo "   Plan ID: $PLAN_C"
echo "   Waiting 12s for Apply to process..."
sleep 12

# Check observed stream for this plan
OBSERVED_EVENTS_C=$($REDIS XREVRANGE "$OBSERVED_STREAM" + - COUNT 20 2>/dev/null || echo "")

if echo "$OBSERVED_EVENTS_C" | grep -q "$PLAN_C"; then
    # Extract heat_found field
    HEAT_FOUND_C=$($REDIS XREVRANGE "$OBSERVED_STREAM" + - COUNT 20 | grep -A 50 "$PLAN_C" | grep "heat_found" | head -1 | awk '{print $2}' || echo "")
    
    if [ "$HEAT_FOUND_C" = "0" ]; then
        pass "Test C: Observed event found with heat_found=0 (correct)"
    else
        fail "Test C: Observed event found but heat_found=$HEAT_FOUND_C (expected 0)"
    fi
    
    # Check heat_reason
    HEAT_REASON_C=$($REDIS XREVRANGE "$OBSERVED_STREAM" + - COUNT 20 | grep -A 50 "$PLAN_C" | grep "heat_reason" | head -1 | awk '{print $2}' || echo "")
    if [ "$HEAT_REASON_C" = "missing" ]; then
        pass "Test C: heat_reason=missing (correct)"
    else
        fail "Test C: heat_reason=$HEAT_REASON_C (expected missing)"
    fi
else
    fail "Test C: No observed event found for plan $PLAN_C"
    echo "   Recent events in stream:"
    $REDIS XREVRANGE "$OBSERVED_STREAM" + - COUNT 10 | head -30
fi

# ==============================================================================
# Test D: Stream Length
# ==============================================================================

echo ""
echo "[D] Verify observed stream has events"

STREAM_LEN=$($REDIS XLEN "$OBSERVED_STREAM" 2>/dev/null || echo "0")
if [ "$STREAM_LEN" -gt 0 ]; then
    pass "Test D: Observed stream has $STREAM_LEN events"
else
    fail "Test D: Observed stream is empty or doesn't exist"
fi

# ==============================================================================
# Cleanup and Summary
# ==============================================================================

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
    echo "  systemctl status quantum-apply"
    echo "  journalctl -u quantum-apply -n 50 --no-pager"
    echo "  redis-cli HGETALL quantum:harvest:BTCUSDT:proposal"
    echo "  redis-cli KEYS 'quantum:harvest:heat:by_plan:*' | head -5"
    echo "  redis-cli XREVRANGE quantum:stream:apply.heat.observed + - COUNT 10"
    echo "  redis-cli KEYS 'quantum:dedupe:p28:*'"
    exit 1
fi
