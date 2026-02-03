#!/usr/bin/env bash
# P3.1 Capital Efficiency Brain - E2E Proof Script
set -euo pipefail

REDIS="${REDIS:-redis-cli}"
P31_PORT="${P31_PORT:-8062}"
P31_KEY_BTC="quantum:capital:efficiency:BTCUSDT"
P31_KEY_ETH="quantum:capital:efficiency:ETHUSDT"
P31_KEY_SOL="quantum:capital:efficiency:SOLUSDT"

RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m'

echo "=============================================="
echo "P3.1 PROOF — Capital Efficiency Brain"
echo "=============================================="
echo ""

FAIL_COUNT=0

fail() {
    echo -e "${RED}✗ FAIL${NC}: $1"
    FAIL_COUNT=$((FAIL_COUNT + 1))
}

pass() {
    echo -e "${GREEN}✓ PASS${NC}: $1"
}

info() {
    echo -e "${YELLOW}ℹ INFO${NC}: $1"
}

# Test 0: Preflight
echo "[0] Preflight: services + redis"
if ! $REDIS PING >/dev/null 2>&1; then
    fail "Redis not responding"
    exit 1
fi
pass "Redis responding"
echo ""

# Test 1: Seed P3.0 attribution + P2.9 targets
echo "[1] Seed P3.0 attribution + P2.9 targets (fresh ts)"
NOW=$(date +%s)

# P3.0 seed (performance_factor 0..1)
$REDIS HSET quantum:alpha:attribution:BTCUSDT performance_factor 0.72 confidence 0.9 ts "$NOW" mode enforce >/dev/null
$REDIS HSET quantum:alpha:attribution:ETHUSDT performance_factor 0.35 confidence 0.8 ts "$NOW" mode enforce >/dev/null
$REDIS HSET quantum:alpha:attribution:SOLUSDT performance_factor 0.85 confidence 0.95 ts "$NOW" mode enforce >/dev/null
pass "P3.0 attribution seeded for 3 symbols"

# P2.9 seed
$REDIS HSET quantum:allocation:target:BTCUSDT target_usd 800 confidence 0.7 ts "$NOW" mode enforce >/dev/null
$REDIS HSET quantum:allocation:target:ETHUSDT target_usd 400 confidence 0.7 ts "$NOW" mode enforce >/dev/null
$REDIS HSET quantum:allocation:target:SOLUSDT target_usd 300 confidence 0.8 ts "$NOW" mode enforce >/dev/null
pass "P2.9 allocation targets seeded for 3 symbols"
echo ""

# Test 2: Inject execution results
echo "[2] Inject execution results"
if python3 /home/qt/quantum_trader/scripts/inject_execution_result_sample_p31.py; then
    pass "Execution results injected"
else
    fail "Failed to inject execution results"
fi
echo ""

# Test 3: Restart P3.1 and wait for processing
echo "[3] Restart P3.1 service and wait for processing"
if systemctl restart quantum-capital-efficiency.service; then
    pass "Service restarted"
else
    fail "Service restart failed"
fi
sleep 4
echo ""

# Test 4: Verify metrics endpoint
echo "[4] Verify metrics endpoint"
if curl -sf "http://127.0.0.1:${P31_PORT}/metrics" | grep -E "^p31_" >/dev/null; then
    pass "Metrics endpoint reachable"
    
    # Check specific metrics
    METRICS=$(curl -sf "http://127.0.0.1:${P31_PORT}/metrics")
    
    if echo "$METRICS" | grep -q "p31_loops_total"; then
        pass "p31_loops_total metric present"
    else
        fail "p31_loops_total metric missing"
    fi
    
    if echo "$METRICS" | grep -q "p31_exec_events_total"; then
        pass "p31_exec_events_total metric present"
    else
        fail "p31_exec_events_total metric missing"
    fi
    
    if echo "$METRICS" | grep -q "p31_efficiency_score"; then
        pass "p31_efficiency_score metric present"
    else
        fail "p31_efficiency_score metric missing"
    fi
else
    fail "Metrics endpoint not responding"
fi
echo ""

# Test 5: Verify Redis decision stream
echo "[5] Verify decision stream populated"
STREAM_LEN=$($REDIS XLEN quantum:stream:capital.efficiency.decision)
if [ "$STREAM_LEN" -gt 0 ]; then
    pass "Decision stream has $STREAM_LEN entries"
    
    # Read latest entry
    LATEST=$($REDIS XREVRANGE quantum:stream:capital.efficiency.decision + - COUNT 1)
    if echo "$LATEST" | grep -q "efficiency_score"; then
        pass "Latest entry contains efficiency_score"
    fi
else
    fail "Decision stream is empty"
fi
echo ""

# Test 6: Check efficiency mode (shadow vs enforce)
echo "[6] Check P3.1 mode (shadow/enforce)"
MODE=$(grep "^P31_MODE=" /etc/quantum/capital-efficiency.env | cut -d'=' -f2)
info "Current mode: $MODE"

if [ "$MODE" = "shadow" ]; then
    info "Shadow mode: hashes may not be written (stream + metrics only)"
elif [ "$MODE" = "enforce" ]; then
    # Test 7: Verify Redis hash outputs exist (enforce mode only)
    echo ""
    echo "[7] Verify Redis hash outputs (enforce mode)"
    for KEY in "$P31_KEY_BTC" "$P31_KEY_ETH" "$P31_KEY_SOL"; do
        if $REDIS EXISTS "$KEY" | grep -q "^1$"; then
            pass "Efficiency key exists: $KEY"
        else
            fail "Efficiency key missing: $KEY"
        fi
    done
    
    # Test 8: Validate required fields
    echo ""
    echo "[8] Validate required fields in hashes"
    REQUIRED_FIELDS="efficiency_score efficiency_raw efficiency_ewma capital_pressure reallocation_weight confidence ts mode version"
    
    for KEY in "$P31_KEY_BTC" "$P31_KEY_ETH"; do
        for FIELD in $REQUIRED_FIELDS; do
            VALUE=$($REDIS HGET "$KEY" "$FIELD" 2>/dev/null || echo "")
            if [ -z "$VALUE" ]; then
                fail "Missing field '$FIELD' in $KEY"
            fi
        done
    done
    pass "All required fields present in hashes"
    
    # Test 9: Sanity check score range
    echo ""
    echo "[9] Sanity check: score range [0, 1]"
    S1=$($REDIS HGET "$P31_KEY_BTC" efficiency_score)
    S2=$($REDIS HGET "$P31_KEY_ETH" efficiency_score)
    
    if python3 - <<PY
s1 = float("$S1")
s2 = float("$S2")
assert 0.0 <= s1 <= 1.0, f"BTC score {s1} out of range"
assert 0.0 <= s2 <= 1.0, f"ETH score {s2} out of range"
print("OK: score range sanity")
PY
    then
        pass "Efficiency scores within [0, 1] range"
        info "  BTCUSDT: $S1"
        info "  ETHUSDT: $S2"
    else
        fail "Efficiency scores out of range"
    fi
    
    # Test 10: Validate capital_pressure values
    echo ""
    echo "[10] Validate capital_pressure values"
    P1=$($REDIS HGET "$P31_KEY_BTC" capital_pressure)
    P2=$($REDIS HGET "$P31_KEY_ETH" capital_pressure)
    
    for P in "$P1" "$P2"; do
        if [[ "$P" =~ ^(INCREASE|HOLD|DECREASE)$ ]]; then
            :
        else
            fail "Invalid capital_pressure value: $P"
        fi
    done
    pass "Capital pressure values valid"
    info "  BTCUSDT: $P1"
    info "  ETHUSDT: $P2"
fi

# Test 11: Check service health
echo ""
echo "[11] Service health check"
if systemctl is-active --quiet quantum-capital-efficiency.service; then
    pass "Service is active"
else
    fail "Service is not active"
fi

# Check recent logs for errors
ERROR_COUNT=$(journalctl -u quantum-capital-efficiency.service --since "2 minutes ago" -n 50 2>/dev/null | grep -ciE "error|exception|fatal" || true)
if [ -z "$ERROR_COUNT" ] || [ "$ERROR_COUNT" -eq 0 ]; then
    pass "No errors in recent logs"
else
    fail "Found $ERROR_COUNT errors in recent logs"
fi
echo ""

# Summary
echo "=============================================="
if [ $FAIL_COUNT -eq 0 ]; then
    echo -e "${GREEN}SUMMARY: PASS${NC}"
    echo "=============================================="
    echo ""
    echo "P3.1 Capital Efficiency Brain Status:"
    echo "  [✓] Metrics endpoint operational"
    echo "  [✓] Decision stream populated"
    echo "  [✓] Mode: $MODE"
    if [ "$MODE" = "enforce" ]; then
        echo "  [✓] Efficiency hashes written"
        echo "  [✓] Required fields validated"
        echo "  [✓] Score range sanity passed"
    fi
    echo "  [✓] Service healthy"
    echo ""
    echo "🎯 P3.1 Capital Efficiency Brain VERIFIED"
    echo ""
    exit 0
else
    echo -e "${RED}SUMMARY: FAIL${NC}"
    echo "=============================================="
    echo ""
    echo "Failed tests: $FAIL_COUNT"
    echo ""
    echo "Debug commands:"
    echo "  systemctl status quantum-capital-efficiency"
    echo "  journalctl -u quantum-capital-efficiency -n 50"
    echo "  curl http://127.0.0.1:${P31_PORT}/metrics | grep p31_"
    echo "  redis-cli HGETALL $P31_KEY_BTC"
    echo ""
    exit 1
fi
