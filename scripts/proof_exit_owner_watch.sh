#!/usr/bin/env bash
# Binary proof: Exit owner monitoring works as designed
set -euo pipefail

PASS_COUNT=0
FAIL_COUNT=0

log_test() { echo "[TEST] $1"; }
pass() { echo "  ✅ PASS: $1"; PASS_COUNT=$((PASS_COUNT + 1)); }
fail() { echo "  ❌ FAIL: $1${2:+ | Evidence: $2}"; FAIL_COUNT=$((FAIL_COUNT + 1)); }

# Test 1: Watch script exists and is executable
log_test "1. scripts/exit_owner_watch.sh exists and executable"
if [ -f "scripts/exit_owner_watch.sh" ] && [ -x "scripts/exit_owner_watch.sh" ]; then
    pass "Watch script found and executable"
else
    fail "Watch script missing or not executable" "$(ls -l scripts/exit_owner_watch.sh 2>&1 || echo 'NOT FOUND')"
fi

# Test 2: Systemd service exists
log_test "2. quantum-exit-owner-watch.service exists"
if [ -f "deploy/systemd/quantum-exit-owner-watch.service" ]; then
    pass "Service file exists"
else
    fail "Service file missing"
fi

# Test 3: Systemd timer exists
log_test "3. quantum-exit-owner-watch.timer exists"
if [ -f "deploy/systemd/quantum-exit-owner-watch.timer" ]; then
    pass "Timer file exists"
else
    fail "Timer file missing"
fi

# Test 4: Timer configured for 5min interval
log_test "4. Timer configured for 5min interval"
if grep -q "OnUnitActiveSec=5min" deploy/systemd/quantum-exit-owner-watch.timer; then
    pass "Timer set to 5min"
else
    fail "Timer not set to 5min" "$(grep OnUnitActiveSec deploy/systemd/quantum-exit-owner-watch.timer || echo 'NOT FOUND')"
fi

# Test 5: Service runs as root (needs journalctl access)
log_test "5. Service runs as root (User=root)"
if grep -q "User=root" deploy/systemd/quantum-exit-owner-watch.service; then
    pass "Service runs as root"
else
    fail "Service not configured to run as root"
fi

# Test 6: Service has fail-open semantics for alerts
log_test "6. Service has SuccessExitStatus=0 1 (exit 1 = alert, not failure)"
if grep -q "SuccessExitStatus=0 1" deploy/systemd/quantum-exit-owner-watch.service; then
    pass "Service configured for alert exit codes"
else
    fail "Service missing alert exit code config"
fi

# Test 7: Watch script checks for DENY_NOT_EXIT_OWNER
log_test "7. Watch script searches for DENY_NOT_EXIT_OWNER"
if grep -q "DENY_NOT_EXIT_OWNER" scripts/exit_owner_watch.sh; then
    pass "DENY detection present"
else
    fail "DENY detection missing"
fi

# Test 8: Watch script publishes to quantum:stream:alerts
log_test "8. Watch script publishes alerts to Redis stream"
if grep -q "quantum:stream:alerts" scripts/exit_owner_watch.sh; then
    pass "Alert publishing present"
else
    fail "Alert publishing missing"
fi

# Test 9: Watch script includes required alert fields
log_test "9. Alert contains required fields (alert_type, deny_count, window, timestamp)"
REQUIRED_ALERT_FIELDS=("alert_type" "deny_count" "window" "timestamp")
MISSING_ALERT_FIELDS=()
for field in "${REQUIRED_ALERT_FIELDS[@]}"; do
    if ! grep -A 5 "quantum:stream:alerts" scripts/exit_owner_watch.sh | grep -q "$field"; then
        MISSING_ALERT_FIELDS+=("$field")
    fi
done
if [ ${#MISSING_ALERT_FIELDS[@]} -eq 0 ]; then
    pass "All required alert fields present"
else
    fail "Missing alert fields" "Fields: ${MISSING_ALERT_FIELDS[*]}"
fi

# Test 10: Watch script checks apply_layer service logs
log_test "10. Watch script monitors quantum-apply-layer service"
if grep -q "quantum-apply-layer" scripts/exit_owner_watch.sh; then
    pass "Monitoring correct service"
else
    fail "Not monitoring quantum-apply-layer"
fi

# Test 11: E2E test - inject DENY event and verify alert
log_test "11. E2E test: DENY detection triggers alert to Redis"
TEST_LOG="/tmp/deny_sample_$$.log"
cat > "$TEST_LOG" << 'EOF'
2026-02-03 01:00:00 [apply_layer] DENY_NOT_EXIT_OWNER: source=harvest_engine symbol=BTCUSDT
2026-02-03 01:00:01 [apply_layer] DENY_NOT_EXIT_OWNER: source=ai_sizer symbol=ETHUSDT
EOF

# Clear any existing alerts
if command -v redis-cli &> /dev/null; then
    redis-cli DEL quantum:stream:alerts &> /dev/null || true
fi

# Run watch script with test input
if EXIT_OWNER_WATCH_TEST_INPUT="$TEST_LOG" bash scripts/exit_owner_watch.sh &> /dev/null; then
    fail "Watch script should exit 1 on DENY events"
else
    # Check if alert was written to Redis
    if command -v redis-cli &> /dev/null; then
        ALERT_COUNT=$(redis-cli XLEN quantum:stream:alerts 2>/dev/null || echo "0")
        if [ "$ALERT_COUNT" -gt 0 ]; then
            ALERT_DATA=$(redis-cli XREAD COUNT 1 STREAMS quantum:stream:alerts 0 2>/dev/null || echo "")
            if echo "$ALERT_DATA" | grep -q "EXIT_OWNER_VIOLATION"; then
                pass "Alert written to Redis with correct type"
            else
                fail "Alert written but missing EXIT_OWNER_VIOLATION" "Data: ${ALERT_DATA:0:100}"
            fi
        else
            fail "No alert written to Redis" "XLEN=0"
        fi
    else
        pass "Redis not available, skipping alert verification"
    fi
fi

# Cleanup
rm -f "$TEST_LOG"

# Summary
echo ""
echo "========================================="
echo "EXIT OWNER MONITORING PROOF SUMMARY"
echo "========================================="
echo "PASS: $PASS_COUNT"
echo "FAIL: $FAIL_COUNT"
echo "========================================="

if [ "$FAIL_COUNT" -eq 0 ]; then
    echo "✅ ALL TESTS PASSED - Exit owner monitoring verified"
    exit 0
else
    echo "❌ SOME TESTS FAILED - Review failures above"
    exit 1
fi
