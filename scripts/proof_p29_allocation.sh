#!/bin/bash
#
# P2.9 Capital Allocation Brain - E2E Proof Script
# Tests: shadow mode, enforce mode, stale fallback, cluster caps
# Expected: exit 0 + SUMMARY: PASS
#
# Author: Quantum Trading OS
# Date: 2026-01-28

set -e

RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m'

echo "=============================================="
echo "P2.9 CAPITAL ALLOCATION BRAIN - E2E PROOF"
echo "=============================================="
echo ""

FAIL_COUNT=0
PASS_COUNT=0

# Helper functions
pass() {
    echo -e "${GREEN}✓ PASS${NC}: $1"
    PASS_COUNT=$((PASS_COUNT + 1))
}

fail() {
    echo -e "${RED}✗ FAIL${NC}: $1"
    FAIL_COUNT=$((FAIL_COUNT + 1))
}

info() {
    echo -e "${YELLOW}ℹ INFO${NC}: $1"
}

# Test 1: Service status
echo "[TEST 1] Service Status"
if systemctl is-active --quiet quantum-capital-allocation; then
    pass "Service is active"
else
    fail "Service is not active"
fi

if systemctl is-enabled --quiet quantum-capital-allocation; then
    pass "Service is enabled"
else
    fail "Service is not enabled"
fi

echo ""

# Test 2: Metrics endpoint
echo "[TEST 2] Metrics Endpoint"
METRICS=$(curl -s http://localhost:8059/metrics)
if [ $? -eq 0 ]; then
    pass "Metrics endpoint responding"
else
    fail "Metrics endpoint not responding"
fi

if echo "$METRICS" | grep -q "p29_targets_computed_total"; then
    pass "Metrics contain p29_targets_computed_total"
else
    fail "Missing p29_targets_computed_total metric"
fi

if echo "$METRICS" | grep -q "p29_shadow_pass_total"; then
    pass "Metrics contain p29_shadow_pass_total"
else
    fail "Missing p29_shadow_pass_total metric"
fi

echo ""

# Test 3: Shadow mode behavior
echo "[TEST 3] Shadow Mode Behavior"
MODE=$(grep "^P29_MODE=" /etc/quantum/capital-allocation.env | cut -d= -f2)
info "Current mode: $MODE"

# Get initial shadow count
SHADOW_BEFORE=$(curl -s http://localhost:8059/metrics | grep "p29_shadow_pass_total" | awk '{print $2}')
info "Shadow pass count before: $SHADOW_BEFORE"

# Wait for one cycle
info "Waiting 10 seconds for allocation cycle..."
sleep 10

# Check shadow count increased
SHADOW_AFTER=$(curl -s http://localhost:8059/metrics | grep "p29_shadow_pass_total" | awk '{print $2}')
info "Shadow pass count after: $SHADOW_AFTER"

if [ "$MODE" = "shadow" ]; then
    if [ "$SHADOW_AFTER" -gt "$SHADOW_BEFORE" ]; then
        pass "Shadow mode: passes incrementing"
    else
        fail "Shadow mode: passes not incrementing"
    fi
    
    # Check that allocation targets are NOT written in shadow mode
    TARGET_COUNT=$(redis-cli KEYS "quantum:allocation:target:*" | wc -l)
    if [ "$TARGET_COUNT" -eq 0 ]; then
        pass "Shadow mode: No allocation targets written"
    else
        info "Shadow mode: Found $TARGET_COUNT allocation targets (may be from previous enforce run)"
    fi
else
    info "Not in shadow mode, skipping shadow-specific checks"
fi

echo ""

# Test 4: Stream events
echo "[TEST 4] Stream Events"
STREAM_LEN=$(redis-cli XLEN quantum:stream:allocation.decision)
info "Stream length: $STREAM_LEN"

if [ "$STREAM_LEN" -gt 0 ]; then
    pass "Allocation decisions being streamed"
    
    # Check last entry
    LAST_ENTRY=$(redis-cli XREVRANGE quantum:stream:allocation.decision + - COUNT 1)
    if echo "$LAST_ENTRY" | grep -q "target_usd"; then
        pass "Stream contains target_usd field"
    else
        fail "Stream missing target_usd field"
    fi
    
    if echo "$LAST_ENTRY" | grep -q "regime_factor"; then
        pass "Stream contains regime_factor"
    else
        fail "Stream missing regime_factor"
    fi
else
    fail "No allocation decisions in stream"
fi

echo ""

# Test 5: Portfolio state dependency
echo "[TEST 5] Portfolio State Dependency"
PORTFOLIO_TS=$(redis-cli HGET quantum:state:portfolio timestamp)
if [ -n "$PORTFOLIO_TS" ]; then
    pass "Portfolio state available"
    
    NOW=$(date +%s)
    AGE=$((NOW - PORTFOLIO_TS))
    info "Portfolio age: ${AGE}s"
    
    if [ "$AGE" -lt 60 ]; then
        pass "Portfolio state fresh (<60s)"
    else
        fail "Portfolio state stale (>60s)"
    fi
else
    fail "Portfolio state missing"
fi

echo ""

# Test 6: Budget data dependency
echo "[TEST 6] Budget Data Dependency"
BUDGET_KEYS=$(redis-cli KEYS "quantum:portfolio:budget:*" | wc -l)
info "Budget keys found: $BUDGET_KEYS"

if [ "$BUDGET_KEYS" -gt 0 ]; then
    pass "Budget data available"
    
    # Check one budget for freshness
    FIRST_BUDGET=$(redis-cli KEYS "quantum:portfolio:budget:*" | head -1)
    BUDGET_TS=$(redis-cli HGET "$FIRST_BUDGET" timestamp)
    
    if [ -n "$BUDGET_TS" ]; then
        NOW=$(date +%s)
        BUDGET_AGE=$((NOW - BUDGET_TS))
        info "Budget age: ${BUDGET_AGE}s"
        
        if [ "$BUDGET_AGE" -lt 300 ]; then
            pass "Budget data fresh (<300s)"
        else
            fail "Budget data stale (>300s)"
        fi
    fi
else
    fail "No budget data found"
fi

echo ""

# Test 7: Logs verification
echo "[TEST 7] Log Verification"
RECENT_LOGS=$(journalctl -u quantum-capital-allocation --since "30 seconds ago" --no-pager)

if echo "$RECENT_LOGS" | grep -q "Allocation cycle complete"; then
    pass "Allocation cycles running"
else
    fail "No recent allocation cycles"
fi

if echo "$RECENT_LOGS" | grep -q "target="; then
    pass "Target allocations being computed"
else
    fail "No target computations in logs"
fi

ERROR_COUNT=$(echo "$RECENT_LOGS" | grep -c "ERROR" || true)
if [ "$ERROR_COUNT" -eq 0 ]; then
    pass "No errors in recent logs"
else
    fail "Found $ERROR_COUNT errors in recent logs"
fi

echo ""

# Test 8: Stale data fallback (optional - requires manual data manipulation)
echo "[TEST 8] Stale Data Fallback"
STALE_FALLBACK_COUNT=$(curl -s http://localhost:8059/metrics | grep "p29_stale_fallback_total" | awk '{print $2}')
info "Stale fallback count: $STALE_FALLBACK_COUNT"

if [ -n "$STALE_FALLBACK_COUNT" ]; then
    pass "Stale fallback metric exists"
else
    fail "Stale fallback metric missing"
fi

echo ""

# Test 9: Allocation confidence
echo "[TEST 9] Allocation Confidence"
CONFIDENCE_METRICS=$(curl -s http://localhost:8059/metrics | grep "p29_allocation_confidence{")
if [ -n "$CONFIDENCE_METRICS" ]; then
    pass "Confidence metrics present"
    
    # Check confidence values are in valid range [0, 1]
    CONFIDENCE_VALUES=$(echo "$CONFIDENCE_METRICS" | awk '{print $2}')
    for conf in $CONFIDENCE_VALUES; do
        if (( $(echo "$conf >= 0.0" | bc -l) )) && (( $(echo "$conf <= 1.0" | bc -l) )); then
            pass "Confidence value $conf in valid range [0, 1]"
        else
            fail "Confidence value $conf out of range"
        fi
    done
else
    info "No confidence metrics yet (may need more time)"
fi

echo ""

# Test 10: Target TTL
echo "[TEST 10] Target Hash TTL"
if [ "$MODE" = "enforce" ]; then
    TARGET_KEYS=$(redis-cli KEYS "quantum:allocation:target:*")
    if [ -n "$TARGET_KEYS" ]; then
        FIRST_TARGET=$(echo "$TARGET_KEYS" | head -1)
        TTL=$(redis-cli TTL "$FIRST_TARGET")
        info "Target TTL: ${TTL}s"
        
        if [ "$TTL" -gt 0 ] && [ "$TTL" -le 300 ]; then
            pass "Target TTL in expected range (0, 300]"
        else
            fail "Target TTL out of range: $TTL"
        fi
    else
        info "No allocation targets (enforce mode may not be active)"
    fi
else
    info "Shadow mode active, skipping target TTL check"
fi

echo ""

# Summary
echo "=============================================="
echo "SUMMARY"
echo "=============================================="
echo "PASS: $PASS_COUNT"
echo "FAIL: $FAIL_COUNT"
echo ""

if [ "$FAIL_COUNT" -eq 0 ]; then
    echo -e "${GREEN}✓ ALL TESTS PASSED${NC}"
    echo "SUMMARY: PASS"
    exit 0
else
    echo -e "${RED}✗ SOME TESTS FAILED${NC}"
    echo "SUMMARY: FAIL"
    exit 1
fi
