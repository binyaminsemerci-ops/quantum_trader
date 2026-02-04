#!/usr/bin/env bash
set -euo pipefail

# ==============================================================================
# P3.1 Integration Step 1 - Allocation Target Shadow Proposer Proof Script
# ==============================================================================
# Tests:
# 1. High efficiency (0.9) → multiplier > 1.0, proposed > base, reason=ok
# 2. Low efficiency (0.3) → multiplier < 1.0, proposed < base, reason=ok
# 3. Missing efficiency → multiplier = 1.0, proposed = base, reason=missing_eff
# 4. Low confidence (0.5) → multiplier = 1.0, proposed = base, reason=low_conf
# 5. Stale efficiency → multiplier = 1.0, proposed = base, reason=stale_eff
#
# Exit codes: 0 = PASS, 1 = FAIL
# ==============================================================================

# Auto-detect repo root (works whether script run from any directory)
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

REDIS="redis-cli"
PYTHON3_INJECT="python3 $REPO_ROOT/scripts/proof_p31_step1_inject_efficiency.py"
TEST_SYMBOL="BTCUSDT"
BASE_TARGET=1000.0
FAILURES=0

echo "===================================================================="
echo "P3.1 Step 1 Proof: Allocation Target Shadow Proposer"
echo "===================================================================="
echo "Test symbol: $TEST_SYMBOL"
echo "Base target: \$$BASE_TARGET"
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
    echo "   Waiting 12s for service to process..."
    sleep 12
}

cleanup() {
    echo ""
    echo "Cleanup: Removing test keys..."
    $REDIS DEL "quantum:allocation:target:$TEST_SYMBOL" >/dev/null 2>&1 || true
    $REDIS DEL "quantum:capital:efficiency:$TEST_SYMBOL" >/dev/null 2>&1 || true
    $REDIS DEL "quantum:allocation:target:proposed:$TEST_SYMBOL" >/dev/null 2>&1 || true
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

echo "[0] Preflight: Service running"
if ! systemctl is-active --quiet quantum-allocation-target 2>/dev/null; then
    echo "⚠️  WARNING: Service not running, tests may fail"
fi

# ==============================================================================
# Test 1: High Efficiency Score (0.9)
# ==============================================================================

echo ""
echo "[1] Test: High efficiency (score=0.9, conf=0.9) → multiplier > 1.0"

cleanup

# Inject base target
NOW=$(date +%s)
$REDIS HSET "quantum:allocation:target:$TEST_SYMBOL" \
    target_usd "$BASE_TARGET" \
    confidence 0.8 \
    timestamp "$NOW" \
    mode enforce >/dev/null

# Inject high efficiency
# Inject high efficiency
NOW=$(date +%s)
$REDIS HSET "quantum:capital:efficiency:$TEST_SYMBOL" \
    efficiency_score "0.9" \
    confidence "0.9" \
    ts "$NOW" \
    mode "enforce" >/dev/null

wait_for_processing

# Check stream for entry
STREAM_ENTRY=$($REDIS XREVRANGE quantum:stream:allocation.target.proposed + - COUNT 1)
if [ -z "$STREAM_ENTRY" ]; then
    fail "Test 1: No stream entry found"
else
    # Parse stream entry (format: entry_id, field1, value1, field2, value2, ...)
    SYMBOL=$(echo "$STREAM_ENTRY" | grep -oP 'symbol\s+\K\S+' || echo "")
    MULT=$(echo "$STREAM_ENTRY" | grep -oP 'multiplier\s+\K\S+' || echo "")
    REASON=$(echo "$STREAM_ENTRY" | grep -oP 'reason\s+\K\S+' || echo "")
    PROPOSED=$(echo "$STREAM_ENTRY" | grep -oP 'proposed_target\s+\K\S+' || echo "")
    
    if [ "$SYMBOL" != "$TEST_SYMBOL" ]; then
        fail "Test 1: Wrong symbol in stream: $SYMBOL"
    elif [ "$REASON" != "ok" ]; then
        fail "Test 1: Expected reason=ok, got $REASON"
    elif [ -z "$MULT" ]; then
        fail "Test 1: No multiplier in stream"
    else
        # Check if multiplier > 1.0 (bc for float comparison)
        if echo "$MULT > 1.0" | bc -l | grep -q 1; then
            if echo "$PROPOSED > $BASE_TARGET" | bc -l | grep -q 1; then
                pass "Test 1: mult=$MULT > 1.0, proposed=\$$PROPOSED > base, reason=ok"
            else
                fail "Test 1: mult=$MULT but proposed=\$$PROPOSED not > base=\$$BASE_TARGET"
            fi
        else
            fail "Test 1: mult=$MULT not > 1.0"
        fi
    fi
fi

# Check shadow key
SHADOW_KEY="quantum:allocation:target:proposed:$TEST_SYMBOL"
if ! $REDIS EXISTS "$SHADOW_KEY" >/dev/null; then
    fail "Test 1: Shadow key not found: $SHADOW_KEY"
else
    SHADOW_MULT=$($REDIS HGET "$SHADOW_KEY" multiplier)
    SHADOW_REASON=$($REDIS HGET "$SHADOW_KEY" reason)
    if [ "$SHADOW_REASON" = "ok" ] && [ -n "$SHADOW_MULT" ]; then
        pass "Test 1: Shadow key exists with mult=$SHADOW_MULT, reason=ok"
    else
        fail "Test 1: Shadow key has wrong data: mult=$SHADOW_MULT, reason=$SHADOW_REASON"
    fi
fi

# ==============================================================================
# Test 2: Low Efficiency Score (0.3)
# ==============================================================================

echo ""
echo "[2] Test: Low efficiency (score=0.3, conf=0.9) → multiplier < 1.0"

cleanup

# Inject base target
NOW=$(date +%s)
$REDIS HSET "quantum:allocation:target:$TEST_SYMBOL" \
    target_usd "$BASE_TARGET" \
    confidence 0.8 \
    timestamp "$NOW" \
    mode enforce >/dev/null

# Inject low efficiency
# Inject low efficiency
NOW=$(date +%s)
$REDIS HSET "quantum:capital:efficiency:$TEST_SYMBOL" \
    efficiency_score "0.3" \
    confidence "0.9" \
    ts "$NOW" \
    mode "enforce" >/dev/null

wait_for_processing

# Check stream
STREAM_ENTRY=$($REDIS XREVRANGE quantum:stream:allocation.target.proposed + - COUNT 1)
if [ -z "$STREAM_ENTRY" ]; then
    fail "Test 2: No stream entry found"
else
    MULT=$(echo "$STREAM_ENTRY" | grep -oP 'multiplier\s+\K\S+' || echo "")
    REASON=$(echo "$STREAM_ENTRY" | grep -oP 'reason\s+\K\S+' || echo "")
    PROPOSED=$(echo "$STREAM_ENTRY" | grep -oP 'proposed_target\s+\K\S+' || echo "")
    
    if [ "$REASON" != "ok" ]; then
        fail "Test 2: Expected reason=ok, got $REASON"
    elif echo "$MULT < 1.0" | bc -l | grep -q 1; then
        if echo "$PROPOSED < $BASE_TARGET" | bc -l | grep -q 1; then
            pass "Test 2: mult=$MULT < 1.0, proposed=\$$PROPOSED < base, reason=ok"
        else
            fail "Test 2: mult=$MULT but proposed=\$$PROPOSED not < base=\$$BASE_TARGET"
        fi
    else
        fail "Test 2: mult=$MULT not < 1.0"
    fi
fi

# ==============================================================================
# Test 3: Missing Efficiency
# ==============================================================================

echo ""
echo "[3] Test: Missing efficiency → multiplier = 1.0, reason=missing_eff"

cleanup

# Inject only base target (no efficiency)
NOW=$(date +%s)
$REDIS HSET "quantum:allocation:target:$TEST_SYMBOL" \
    target_usd "$BASE_TARGET" \
    confidence 0.8 \
    timestamp "$NOW" \
    mode enforce >/dev/null

wait_for_processing

# Check stream
STREAM_ENTRY=$($REDIS XREVRANGE quantum:stream:allocation.target.proposed + - COUNT 1)
if [ -z "$STREAM_ENTRY" ]; then
    fail "Test 3: No stream entry found"
else
    MULT=$(echo "$STREAM_ENTRY" | grep -oP 'multiplier\s+\K\S+' || echo "")
    REASON=$(echo "$STREAM_ENTRY" | grep -oP 'reason\s+\K\S+' || echo "")
    PROPOSED=$(echo "$STREAM_ENTRY" | grep -oP 'proposed_target\s+\K\S+' || echo "")
    
    if [ "$REASON" != "missing_eff" ]; then
        fail "Test 3: Expected reason=missing_eff, got $REASON"
    elif [ "$MULT" != "1.0000" ] && [ "$MULT" != "1.00" ]; then
        fail "Test 3: Expected mult=1.0, got $MULT"
    elif [ "$PROPOSED" != "$BASE_TARGET" ] && [ "${PROPOSED%.00}" != "${BASE_TARGET%.0}" ]; then
        fail "Test 3: Expected proposed=$BASE_TARGET, got $PROPOSED"
    else
        pass "Test 3: mult=$MULT, proposed=$PROPOSED, reason=missing_eff"
    fi
fi

# ==============================================================================
# Test 4: Low Confidence
# ==============================================================================

echo ""
echo "[4] Test: Low confidence (conf=0.5 < MIN_CONF=0.65) → multiplier = 1.0, reason=low_conf"

cleanup

# Inject base target
NOW=$(date +%s)
$REDIS HSET "quantum:allocation:target:$TEST_SYMBOL" \
    target_usd "$BASE_TARGET" \
    confidence 0.8 \
    timestamp "$NOW" \
    mode enforce >/dev/null

# Inject efficiency with low confidence
# Inject low confidence efficiency
NOW=$(date +%s)
$REDIS HSET "quantum:capital:efficiency:$TEST_SYMBOL" \
    efficiency_score "0.8" \
    confidence "0.5" \
    ts "$NOW" \
    mode "enforce" >/dev/null

wait_for_processing

# Check stream
STREAM_ENTRY=$($REDIS XREVRANGE quantum:stream:allocation.target.proposed + - COUNT 1)
if [ -z "$STREAM_ENTRY" ]; then
    fail "Test 4: No stream entry found"
else
    MULT=$(echo "$STREAM_ENTRY" | grep -oP 'multiplier\s+\K\S+' || echo "")
    REASON=$(echo "$STREAM_ENTRY" | grep -oP 'reason\s+\K\S+' || echo "")
    
    if [ "$REASON" != "low_conf" ]; then
        fail "Test 4: Expected reason=low_conf, got $REASON"
    elif [ "$MULT" != "1.0000" ] && [ "$MULT" != "1.00" ]; then
        fail "Test 4: Expected mult=1.0, got $MULT"
    else
        pass "Test 4: mult=$MULT, reason=low_conf"
    fi
fi

# ==============================================================================
# Test 5: Stale Efficiency
# ==============================================================================

echo ""
echo "[5] Test: Stale efficiency (>600s old) → multiplier = 1.0, reason=stale_eff"

cleanup

# Inject base target
NOW=$(date +%s)
$REDIS HSET "quantum:allocation:target:$TEST_SYMBOL" \
    target_usd "$BASE_TARGET" \
    confidence 0.8 \
    timestamp "$NOW" \
    mode enforce >/dev/null

# Inject stale efficiency (700s ago)
# Inject stale efficiency (timestamp old)
NOW=$(date +%s)
STALE_TS=$((NOW - 700))  # 700 seconds old (past 600s threshold)
$REDIS HSET "quantum:capital:efficiency:$TEST_SYMBOL" \
    efficiency_score "0.9" \
    confidence "0.9" \
    ts "$STALE_TS" \
    mode "enforce" >/dev/null

wait_for_processing

# Check stream
STREAM_ENTRY=$($REDIS XREVRANGE quantum:stream:allocation.target.proposed + - COUNT 1)
if [ -z "$STREAM_ENTRY" ]; then
    fail "Test 5: No stream entry found"
else
    MULT=$(echo "$STREAM_ENTRY" | grep -oP 'multiplier\s+\K\S+' || echo "")
    REASON=$(echo "$STREAM_ENTRY" | grep -oP 'reason\s+\K\S+' || echo "")
    
    if [ "$REASON" != "stale_eff" ]; then
        fail "Test 5: Expected reason=stale_eff, got $REASON"
    elif [ "$MULT" != "1.0000" ] && [ "$MULT" != "1.00" ]; then
        fail "Test 5: Expected mult=1.0, got $MULT"
    else
        pass "Test 5: mult=$MULT, reason=stale_eff"
    fi
fi

# ==============================================================================
# Test 6: Service Health
# ==============================================================================

echo ""
echo "[6] Service health check"

if systemctl is-active --quiet quantum-allocation-target 2>/dev/null; then
    pass "Service is active"
else
    fail "Service is not active"
fi

# Check for errors in last 20 lines
ERROR_COUNT=$(journalctl -u quantum-allocation-target -n 20 --no-pager 2>/dev/null | grep -ic "error" || echo 0)
if [ "$ERROR_COUNT" -eq 0 ]; then
    pass "No errors in recent logs"
else
    fail "Found $ERROR_COUNT errors in recent logs"
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
    echo "  journalctl -u quantum-allocation-target -n 50 --no-pager"
    echo "  redis-cli XREVRANGE quantum:stream:allocation.target.proposed + - COUNT 5"
    echo "  redis-cli HGETALL quantum:allocation:target:proposed:$TEST_SYMBOL"
    echo "  curl -s http://localhost:8065/metrics | grep p29_shadow"
    exit 1
fi
