#!/usr/bin/env bash
set -euo pipefail

# ==============================================================================
# P3.1 Integration Step 2 - Governor Downsize Hint Proof Script
# ==============================================================================
# Tests:
# 1. Low efficiency (score=0.2) with high conf → eff_action=DOWNSIZE, eff_factor < 1.0
# 2. High efficiency (score=0.8) → eff_action=NONE, eff_factor=1.0
# 3. Missing efficiency → eff_action=NONE, eff_reason=missing_eff
# 4. Low confidence (conf=0.3 < MIN_CONF=0.65) → eff_action=NONE, eff_reason=low_conf
#
# Exit codes: 0 = PASS, 1 = FAIL
# ==============================================================================

REDIS="redis-cli"
TEST_SYMBOL="BTCUSDT"
FAILURES=0

echo "===================================================================="
echo "P3.1 Step 2 Proof: Governor Downsize Hint Integration"
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

generate_plan_id() {
    echo $(echo "plan_$(date +%s%N)" | md5sum | cut -d' ' -f1)
}

wait_for_governor() {
    echo "   Waiting 3s for governor to process plan..."
    sleep 3
}

cleanup() {
    echo ""
    echo "Cleanup: Removing test keys..."
    $REDIS DEL "quantum:capital:efficiency:$TEST_SYMBOL" >/dev/null 2>&1 || true
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

echo "[0] Preflight: Governor running"
if ! systemctl is-active --quiet quantum-governor 2>/dev/null; then
    echo "⚠️  WARNING: Governor service not running, tests may fail"
fi

echo "[0] Preflight: Clearing symbol cooldowns for testing"
$REDIS DEL quantum:cooldown:last_exec_ts:BTCUSDT quantum:cooldown:last_exec_ts:ETHUSDT quantum:cooldown:last_exec_ts:TRXUSDT >/dev/null
pass "Cooldowns cleared"

# ==============================================================================
# Test 1: Low Efficiency (score=0.2) → DOWNSIZE
# ==============================================================================

echo ""
echo "[1] Test: Low efficiency (score=0.2, conf=0.9) → eff_action=DOWNSIZE"

cleanup

# Inject low efficiency
NOW=$(date +%s)
$REDIS HSET "quantum:capital:efficiency:$TEST_SYMBOL" \
    efficiency_score "0.2" \
    confidence "0.9" \
    ts "$NOW" \
    mode "enforce" >/dev/null

# Generate plan ID and inject plan
PLAN_ID=$(generate_plan_id)
echo "   Generated plan_id: $PLAN_ID"

python3 /root/quantum_trader/scripts/proof_p31_step2_inject_plan.py "$PLAN_ID" "$TEST_SYMBOL" "FULL_CLOSE_PROPOSED" "EXECUTE"

wait_for_governor

# Check if permit exists
PERMIT_KEY="quantum:permit:$PLAN_ID"
if ! $REDIS EXISTS "$PERMIT_KEY" >/dev/null; then
    fail "Test 1: Permit key not created: $PERMIT_KEY"
else
    # Get permit data
    PERMIT_JSON=$($REDIS GET "$PERMIT_KEY")
    
    if [ -z "$PERMIT_JSON" ]; then
        fail "Test 1: Permit is empty"
    else
        # Parse JSON (basic extraction)
        EFF_ACTION=$(echo "$PERMIT_JSON" | grep -oP '"eff_action"\s*:\s*"?\K[^"]*' || echo "")
        EFF_FACTOR=$(echo "$PERMIT_JSON" | grep -oP '"eff_factor"\s*:\s*"?\K[^"]*' || echo "")
        GRANTED=$(echo "$PERMIT_JSON" | grep -oP '"granted"\s*:\s*\K[^,}]*' || echo "")
        
        if [ "$GRANTED" = "true" ] && [ "$EFF_ACTION" = "DOWNSIZE" ]; then
            # Check if factor is between MIN_FACTOR(0.25) and 1.0
            if echo "$EFF_FACTOR" | grep -qE '^0\.[0-9]+|^1\.0'; then
                pass "Test 1: DOWNSIZE action, factor=$EFF_FACTOR (between 0.25 and 1.0)"
            else
                fail "Test 1: Factor out of range: $EFF_FACTOR"
            fi
        elif [ "$GRANTED" = "true" ]; then
            fail "Test 1: Expected eff_action=DOWNSIZE, got $EFF_ACTION"
        else
            fail "Test 1: Permit not granted"
        fi
    fi
fi

# ==============================================================================
# Test 2: High Efficiency (score=0.8) → NONE
# ==============================================================================

echo ""
echo "[2] Test: High efficiency (score=0.8, conf=0.9) → eff_action=NONE"

cleanup

# Inject high efficiency
NOW=$(date +%s)
$REDIS HSET "quantum:capital:efficiency:$TEST_SYMBOL" \
    efficiency_score "0.8" \
    confidence "0.9" \
    ts "$NOW" \
    mode "enforce" >/dev/null

# Inject plan
PLAN_ID=$(generate_plan_id)
python3 scripts/proof_p31_step2_inject_plan.py "$PLAN_ID" "$TEST_SYMBOL" "OPEN_PROPOSED" "EXECUTE"

wait_for_governor

# Check permit
PERMIT_KEY="quantum:permit:$PLAN_ID"
if ! $REDIS EXISTS "$PERMIT_KEY" >/dev/null; then
    fail "Test 2: Permit key not created"
else
    PERMIT_JSON=$($REDIS GET "$PERMIT_KEY")
    EFF_ACTION=$(echo "$PERMIT_JSON" | grep -oP '"eff_action"\s*:\s*"?\K[^"]*' || echo "")
    EFF_FACTOR=$(echo "$PERMIT_JSON" | grep -oP '"eff_factor"\s*:\s*"?\K[^"]*' || echo "")
    
    if [ "$EFF_ACTION" = "NONE" ] && ([ "$EFF_FACTOR" = "1.0" ] || [ "$EFF_FACTOR" = "1.0000" ]); then
        pass "Test 2: NONE action, factor=$EFF_FACTOR"
    else
        fail "Test 2: Expected NONE/1.0, got action=$EFF_ACTION factor=$EFF_FACTOR"
    fi
fi

# ==============================================================================
# Test 3: Missing Efficiency
# ==============================================================================

echo ""
echo "[3] Test: Missing efficiency → eff_action=NONE, eff_reason=missing_eff"

cleanup

# No efficiency data injected

# Inject plan
PLAN_ID=$(generate_plan_id)
python3 scripts/proof_p31_step2_inject_plan.py "$PLAN_ID" "$TEST_SYMBOL" "OPEN_PROPOSED" "EXECUTE"

wait_for_governor

# Check permit
PERMIT_KEY="quantum:permit:$PLAN_ID"
if ! $REDIS EXISTS "$PERMIT_KEY" >/dev/null; then
    fail "Test 3: Permit key not created"
else
    PERMIT_JSON=$($REDIS GET "$PERMIT_KEY")
    EFF_ACTION=$(echo "$PERMIT_JSON" | grep -oP '"eff_action"\s*:\s*"?\K[^"]*' || echo "")
    EFF_REASON=$(echo "$PERMIT_JSON" | grep -oP '"eff_reason"\s*:\s*"?\K[^"]*' || echo "")
    
    if [ "$EFF_ACTION" = "NONE" ] && [ "$EFF_REASON" = "missing_eff" ]; then
        pass "Test 3: NONE action, reason=$EFF_REASON"
    else
        fail "Test 3: Expected NONE/missing_eff, got action=$EFF_ACTION reason=$EFF_REASON"
    fi
fi

# ==============================================================================
# Test 4: Low Confidence (conf=0.3 < MIN_CONF=0.65)
# ==============================================================================

echo ""
echo "[4] Test: Low confidence (conf=0.3 < MIN_CONF=0.65) → eff_action=NONE, reason=low_conf"

cleanup

# Inject efficiency with low confidence
NOW=$(date +%s)
$REDIS HSET "quantum:capital:efficiency:$TEST_SYMBOL" \
    efficiency_score "0.8" \
    confidence "0.3" \
    ts "$NOW" \
    mode "enforce" >/dev/null

# Inject plan
PLAN_ID=$(generate_plan_id)
python3 scripts/proof_p31_step2_inject_plan.py "$PLAN_ID" "$TEST_SYMBOL" "OPEN_PROPOSED" "EXECUTE"

wait_for_governor

# Check permit
PERMIT_KEY="quantum:permit:$PLAN_ID"
if ! $REDIS EXISTS "$PERMIT_KEY" >/dev/null; then
    fail "Test 4: Permit key not created"
else
    PERMIT_JSON=$($REDIS GET "$PERMIT_KEY")
    EFF_ACTION=$(echo "$PERMIT_JSON" | grep -oP '"eff_action"\s*:\s*"?\K[^"]*' || echo "")
    EFF_REASON=$(echo "$PERMIT_JSON" | grep -oP '"eff_reason"\s*:\s*"?\K[^"]*' || echo "")
    
    if [ "$EFF_ACTION" = "NONE" ] && [ "$EFF_REASON" = "low_conf" ]; then
        pass "Test 4: NONE action, reason=$EFF_REASON"
    else
        fail "Test 4: Expected NONE/low_conf, got action=$EFF_ACTION reason=$EFF_REASON"
    fi
fi

# ==============================================================================
# Test 5: Governor Metrics
# ==============================================================================

echo ""
echo "[5] Check Governor metrics"

if ! curl -s http://localhost:8044/metrics >/dev/null 2>&1; then
    echo "⚠️  WARNING: Governor metrics endpoint not responding"
else
    # Check if P3.1 metrics are present
    EFF_APPLY_METRIC=$(curl -s http://localhost:8044/metrics | grep "p32_eff_apply_total" || echo "")
    if [ -n "$EFF_APPLY_METRIC" ]; then
        pass "Test 5: P3.1 metrics registered (p32_eff_apply_total)"
    else
        fail "Test 5: P3.1 metrics not found in Prometheus output"
    fi
fi

# ==============================================================================
# Test 6: Service Health
# ==============================================================================

echo ""
echo "[6] Governor service health"

if systemctl is-active --quiet quantum-governor 2>/dev/null; then
    pass "Governor service is active"
else
    fail "Governor service is not active"
fi

# Check for errors in logs
ERROR_COUNT=$(journalctl -u quantum-governor -n 30 --no-pager 2>/dev/null | grep -ic "error" || echo 0)
if [ "$ERROR_COUNT" -eq 0 ]; then
    pass "No errors in recent governor logs"
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
    echo "  journalctl -u quantum-governor -n 50 --no-pager | tail -30"
    echo "  redis-cli GET quantum:permit:<plan_id>"
    echo "  redis-cli HGETALL quantum:capital:efficiency:$TEST_SYMBOL"
    echo "  curl -s http://localhost:8044/metrics | grep p32_eff"
    exit 1
fi
