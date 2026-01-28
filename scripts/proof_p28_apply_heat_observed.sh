#!/usr/bin/env bash
set -euo pipefail

# ==============================================================================
# P2.8A.1 Apply Heat Observer - Deterministic Proof Script
# ==============================================================================
# Uses direct reconcile.close injection for deterministic plan_id matching.
# Tests:
# A) Heat found: inject plan+heat to reconcile.close, verify observed event with heat_found=1
# B) Deduplication: re-inject same plan_id, verify no duplicate observed event
# C) Heat missing: inject plan to reconcile.close without heat, verify heat_found=0
# D) Stream integrity: verify observed stream has events
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
echo "P2.8A.1 Apply Heat Observer Proof (Deterministic)"
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

check_observed_event() {
    local plan_id="$1"
    local expected_heat_found="$2"
    local stream="quantum:stream:apply.heat.observed"
    
    # Search for plan_id in stream
    local events=$($REDIS XRANGE "$stream" - + | grep -A 30 "$plan_id" || echo "")
    
    if [ -z "$events" ]; then
        echo ""
        return 1
    fi
    
    # Extract heat_found value (appears after "heat_found" line)
    local heat_found=$(echo "$events" | grep -A 1 "^heat_found$" | tail -1 || echo "")
    
    if [ "$heat_found" = "$expected_heat_found" ]; then
        echo "$heat_found"
        return 0
    else
        echo "$heat_found"
        return 1
    fi
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
if ! systemctl is-active --quiet quantum-apply-layer 2>/dev/null; then
    if ! systemctl is-active --quiet quantum-apply 2>/dev/null; then
        echo "⚠️  WARNING: Apply service not running, tests may fail"
        echo "    Try: systemctl start quantum-apply-layer"
    fi
fi

echo "[0] Preflight: Clean up previous test data"
# Hard cleanup: Delete all test plan keys (heat + dedupe for both obs_points)
$REDIS DEL quantum:harvest:heat:by_plan:test_plan_a >/dev/null 2>&1 || true
$REDIS DEL quantum:harvest:heat:by_plan:test_plan_b >/dev/null 2>&1 || true
$REDIS DEL quantum:harvest:heat:by_plan:test_plan_c >/dev/null 2>&1 || true
$REDIS DEL quantum:dedupe:p28:create_apply_plan:test_plan_a >/dev/null 2>&1 || true
$REDIS DEL quantum:dedupe:p28:create_apply_plan:test_plan_b >/dev/null 2>&1 || true
$REDIS DEL quantum:dedupe:p28:create_apply_plan:test_plan_c >/dev/null 2>&1 || true
$REDIS DEL quantum:dedupe:p28:reconcile_close_consume:test_plan_a >/dev/null 2>&1 || true
$REDIS DEL quantum:dedupe:p28:reconcile_close_consume:test_plan_b >/dev/null 2>&1 || true
$REDIS DEL quantum:dedupe:p28:reconcile_close_consume:test_plan_c >/dev/null 2>&1 || true
echo "    Test keys cleaned (heat + dedupe for both obs_points)"

# ==============================================================================
# Test A: Heat Found (apply.plan + heat key)
# ==============================================================================

echo ""
echo "[A] Test: Heat found → observed event with heat_found=1"

PLAN_A="test_plan_a"

echo "   Injecting plan+heat to reconcile.close stream (deterministic)..."
$PYTHON3 "$INJECT_SCRIPT" \
    --inject_apply_plan \
    --plan_id "$PLAN_A" \
    --symbol BTCUSDT \
    --action FULL_CLOSE_PROPOSED \
    --decision EXECUTE \
    --kill_score 0.25 \
    --heat_level hot \
    --heat_action PASS_THROUGH \
    >/dev/null 2>&1

echo "   Waiting 8s for Apply to consume reconcile.close..."
sleep 8

# Verify heat key exists
HEAT_KEY="quantum:harvest:heat:by_plan:$PLAN_A"
if ! $REDIS EXISTS "$HEAT_KEY" >/dev/null; then
    fail "Test A: Heat key not found: $HEAT_KEY"
else
    pass "Test A: Heat key exists"
fi

# Check observed stream
echo "   Checking observed stream for plan_id=$PLAN_A..."
HEAT_FOUND=$(check_observed_event "$PLAN_A" "1")
if [ $? -eq 0 ] && [ "$HEAT_FOUND" = "1" ]; then
    pass "Test A: Observed event found with heat_found=1"
else
    fail "Test A: No observed event or heat_found=$HEAT_FOUND (expected 1)"
    echo "   Recent stream events:"
    $REDIS XREVRANGE quantum:stream:apply.heat.observed + - COUNT 5
fi

# ==============================================================================
# Test B: Deduplication (re-inject same plan_id)
# ==============================================================================

echo ""
echo "[B] Test: Deduplication → re-inject same plan_id, no duplicate event"

# Check dedupe key exists (for reconcile_close_consume obs_point)
DEDUPE_KEY="quantum:dedupe:p28:reconcile_close_consume:$PLAN_A"
if ! $REDIS EXISTS "$DEDUPE_KEY" >/dev/null; then
    fail "Test B: Dedupe key not found (observer may not have run)"
else
    pass "Test B: Dedupe key exists"
fi

# Count events for plan_a BEFORE re-injection
STREAM="quantum:stream:apply.heat.observed"
EVENTS_BEFORE=$($REDIS XRANGE "$STREAM" - + | grep -c "$PLAN_A" || echo "0")
echo "   Events before re-injection: $EVENTS_BEFORE"

# Re-inject same plan
echo "   Re-injecting same plan_id..."
$PYTHON3 "$INJECT_SCRIPT" \
    --inject_apply_plan \
    --plan_id "$PLAN_A" \
    --symbol BTCUSDT \
    --action FULL_CLOSE_PROPOSED \
    --decision EXECUTE \
    --kill_score 0.25 \
    --heat_level hot \
    --heat_action PASS_THROUGH \
    >/dev/null 2>&1

echo "   Waiting 8s for Apply to process..."
sleep 8

# Count events AFTER re-injection
EVENTS_AFTER=$($REDIS XRANGE "$STREAM" - + | grep -c "$PLAN_A" || echo "0")
echo "   Events after re-injection: $EVENTS_AFTER"

if [ "$EVENTS_BEFORE" -eq "$EVENTS_AFTER" ]; then
    pass "Test B: No duplicate observed event (dedupe working)"
else
    fail "Test B: Event count changed (before=$EVENTS_BEFORE, after=$EVENTS_AFTER)"
fi

# ==============================================================================
# Test C: Heat Missing (apply.plan without heat key)
# ==============================================================================

echo ""
echo "[C] Test: Heat missing → observed event with heat_found=0"

PLAN_C="test_plan_c"

echo "   Injecting plan WITHOUT heat to reconcile.close stream..."
$PYTHON3 "$INJECT_SCRIPT" \
    --inject_apply_plan \
    --plan_id "$PLAN_C" \
    --symbol ETHUSDT \
    --action FULL_CLOSE_PROPOSED \
    --decision EXECUTE \
    --kill_score 0.15 \
    --heat_level none \
    >/dev/null 2>&1

echo "   Waiting 8s for Apply to consume..."
sleep 8

# Verify heat key does NOT exist
HEAT_KEY_C="quantum:harvest:heat:by_plan:$PLAN_C"
if $REDIS EXISTS "$HEAT_KEY_C" >/dev/null; then
    fail "Test C: Heat key should not exist: $HEAT_KEY_C"
else
    pass "Test C: Heat key correctly missing"
fi

# Check observed stream
echo "   Checking observed stream for plan_id=$PLAN_C..."
HEAT_FOUND_C=$(check_observed_event "$PLAN_C" "0")
if [ $? -eq 0 ] && [ "$HEAT_FOUND_C" = "0" ]; then
    pass "Test C: Observed event found with heat_found=0"
    
    # Check heat_reason
    EVENTS_C=$($REDIS XRANGE "$STREAM" - + | grep -A 30 "$PLAN_C" || echo "")
    HEAT_REASON=$(echo "$EVENTS_C" | grep -A 1 "^heat_reason$" | tail -1 || echo "")
    if [ "$HEAT_REASON" = "missing" ]; then
        pass "Test C: heat_reason=missing (correct)"
    else
        fail "Test C: heat_reason=$HEAT_REASON (expected missing)"
    fi
else
    fail "Test C: No observed event or heat_found=$HEAT_FOUND_C (expected 0)"
    echo "   Recent stream events:"
    $REDIS XREVRANGE quantum:stream:apply.heat.observed + - COUNT 5
fi

# ==============================================================================
# Test D: Stream Integrity
# ==============================================================================

echo ""
echo "[D] Verify observed stream has events"

STREAM_LEN=$($REDIS XLEN "$STREAM" || echo "0")
if [ "$STREAM_LEN" -gt 0 ]; then
    pass "Test D: Observed stream has $STREAM_LEN events"
else
    fail "Test D: Observed stream is empty"
fi

# ==============================================================================
# Summary
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
    echo "  systemctl status quantum-apply-layer"
    echo "  journalctl -u quantum-apply-layer -n 50 --no-pager"
    echo "  redis-cli KEYS 'quantum:harvest:heat:by_plan:*' | head -5"
    echo "  redis-cli XREVRANGE quantum:stream:apply.heat.observed + - COUNT 10"
    echo "  redis-cli KEYS 'quantum:dedupe:p28:*'"
    echo ""
    exit 1
fi
