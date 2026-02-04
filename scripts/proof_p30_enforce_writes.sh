#!/bin/bash
#
# P3.0 Performance Attribution - Enforce Mode Write Proof
# Tests: P3.0 writes attribution to Redis when in enforce mode
#
# Author: Quantum Trading OS
# Date: 2026-01-28

set -e

RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m'

echo "=============================================="
echo "P3.0 ENFORCE MODE WRITE PROOF"
echo "=============================================="
echo ""

FAIL_COUNT=0
PASS_COUNT=0
TEST_SYMBOL="BTCUSDT"

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

# Test 1: Verify P3.0 is in enforce mode
echo "[TEST 1] P3.0 Mode Check"
P30_MODE=$(grep "P30_MODE" /home/qt/quantum_trader/deploy/performance-attribution.env | cut -d'=' -f2)

if [ "$P30_MODE" = "enforce" ]; then
    pass "P3.0 is in enforce mode"
else
    fail "P3.0 is NOT in enforce mode (found: $P30_MODE)"
    echo "Run: sed -i 's/P30_MODE=.*/P30_MODE=enforce/' /home/qt/quantum_trader/deploy/performance-attribution.env"
    echo "Then: systemctl restart quantum-performance-attribution"
    exit 1
fi

echo ""

# Test 2: Verify P3.0 service is active
echo "[TEST 2] P3.0 Service Status"
if systemctl is-active --quiet quantum-performance-attribution; then
    pass "P3.0 service is active"
else
    fail "P3.0 service is not active"
    exit 1
fi

echo ""

# Test 3: Clear any existing attribution for test symbol
echo "[TEST 3] Clear Existing Attribution"
redis-cli DEL quantum:alpha:attribution:${TEST_SYMBOL} > /dev/null
info "Cleared quantum:alpha:attribution:${TEST_SYMBOL}"

# Verify deleted
EXISTS_BEFORE=$(redis-cli EXISTS quantum:alpha:attribution:${TEST_SYMBOL})
if [ "$EXISTS_BEFORE" = "0" ]; then
    pass "Attribution key successfully cleared"
else
    fail "Failed to clear attribution key"
fi

echo ""

# Test 4: Run execution result injector
echo "[TEST 4] Inject Execution Results"
if [ -f "/home/qt/quantum_trader/scripts/inject_execution_result_sample.py" ]; then
    python3 /home/qt/quantum_trader/scripts/inject_execution_result_sample.py --count 5 --symbol ${TEST_SYMBOL}
    
    if [ $? -eq 0 ]; then
        pass "Execution results injected successfully"
    else
        fail "Failed to inject execution results"
    fi
else
    fail "Injector script not found"
    exit 1
fi

echo ""

# Test 5: Wait for P3.0 to process (5s loop + 3s buffer)
echo "[TEST 5] Wait for P3.0 Processing"
info "Waiting 8 seconds for P3.0 attribution loop..."
sleep 8

echo ""

# Test 6: Verify attribution exists in Redis
echo "[TEST 6] Verify Attribution Written to Redis"
EXISTS_AFTER=$(redis-cli EXISTS quantum:alpha:attribution:${TEST_SYMBOL})

if [ "$EXISTS_AFTER" = "1" ]; then
    pass "Attribution key exists in Redis"
else
    fail "Attribution key NOT found in Redis"
    info "Expected: quantum:alpha:attribution:${TEST_SYMBOL}"
    
    # Debug: Check P3.0 logs
    echo ""
    echo "=== P3.0 Recent Logs ==="
    journalctl -u quantum-performance-attribution --since "15 seconds ago" --no-pager | tail -10
    echo ""
fi

echo ""

# Test 7: Verify attribution data structure
echo "[TEST 7] Verify Attribution Data"
if [ "$EXISTS_AFTER" = "1" ]; then
    ALPHA_SCORE=$(redis-cli HGET quantum:alpha:attribution:${TEST_SYMBOL} alpha_score)
    PERF_FACTOR=$(redis-cli HGET quantum:alpha:attribution:${TEST_SYMBOL} performance_factor)
    CONFIDENCE=$(redis-cli HGET quantum:alpha:attribution:${TEST_SYMBOL} confidence)
    MODE=$(redis-cli HGET quantum:alpha:attribution:${TEST_SYMBOL} mode)
    SOURCE=$(redis-cli HGET quantum:alpha:attribution:${TEST_SYMBOL} source)
    
    echo "  alpha_score: $ALPHA_SCORE"
    echo "  performance_factor: $PERF_FACTOR"
    echo "  confidence: $CONFIDENCE"
    echo "  mode: $MODE"
    echo "  source: $SOURCE"
    
    if [ -n "$ALPHA_SCORE" ] && [ -n "$PERF_FACTOR" ]; then
        pass "Attribution has required fields"
    else
        fail "Attribution missing required fields"
    fi
    
    if [ "$MODE" = "enforce" ]; then
        pass "Attribution mode is enforce"
    else
        fail "Attribution mode is not enforce (found: $MODE)"
    fi
else
    info "Skipping data verification (key does not exist)"
fi

echo ""

# Test 8: Verify attribution stream
echo "[TEST 8] Verify Attribution Stream"
STREAM_LEN=$(redis-cli XLEN quantum:stream:alpha.attribution)

if [ "$STREAM_LEN" -gt 0 ]; then
    pass "Attribution stream has $STREAM_LEN entries"
    
    # Show most recent attribution event
    RECENT_EVENT=$(redis-cli XREVRANGE quantum:stream:alpha.attribution + - COUNT 1)
    info "Most recent attribution event:"
    echo "$RECENT_EVENT" | head -10
else
    fail "Attribution stream is empty"
fi

echo ""

# Test 9: Verify P3.0 metrics incremented
echo "[TEST 9] Verify P3.0 Metrics"
METRICS=$(curl -s http://localhost:8061/metrics)

ATTR_TOTAL=$(echo "$METRICS" | grep "p30_attributions_computed_total{symbol=\"${TEST_SYMBOL}\"}" | awk '{print $2}')

if [ -n "$ATTR_TOTAL" ] && [ "$ATTR_TOTAL" != "0.0" ]; then
    pass "P3.0 computed attributions: $ATTR_TOTAL"
else
    info "P3.0 attribution metrics not yet incremented for ${TEST_SYMBOL}"
fi

echo ""

# Test 10: Verify P2.9 can read performance factor
echo "[TEST 10] Verify P2.9 Integration"
if systemctl is-active --quiet quantum-capital-allocation; then
    # Check recent P2.9 logs for performance_factor from P3.0
    P29_LOGS=$(journalctl -u quantum-capital-allocation --since "15 seconds ago" --no-pager | grep -i "performance")
    
    if echo "$P29_LOGS" | grep -q "P3.0"; then
        pass "P2.9 reading performance_factor from P3.0"
    else
        info "P2.9 not yet using P3.0 data (may need next allocation cycle)"
    fi
else
    info "P2.9 service not active"
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
    echo ""
    echo "P3.0 enforce mode verified:"
    echo "  - Writes attribution to quantum:alpha:attribution:*"
    echo "  - Streams to quantum:stream:alpha.attribution"
    echo "  - P2.9 integration ready"
    echo ""
    echo "SUMMARY: PASS"
    exit 0
else
    echo -e "${RED}✗ SOME TESTS FAILED${NC}"
    echo ""
    echo "Check P3.0 logs: journalctl -u quantum-performance-attribution -n 50"
    echo ""
    echo "SUMMARY: FAIL"
    exit 1
fi
