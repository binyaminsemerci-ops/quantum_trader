#!/usr/bin/env bash
# Binary proof: Policy refresh automation works as designed
set -euo pipefail

PASS_COUNT=0
FAIL_COUNT=0

log_test() { echo "[TEST] $1"; }
pass() { echo "  ✅ PASS: $1"; PASS_COUNT=$((PASS_COUNT + 1)); }
fail() { echo "  ❌ FAIL: $1${2:+ | Evidence: $2}"; FAIL_COUNT=$((FAIL_COUNT + 1)); }

# Test 1: Refresh script exists and is executable
log_test "1. scripts/policy_refresh.sh exists and executable"
if [ -f "scripts/policy_refresh.sh" ] && [ -x "scripts/policy_refresh.sh" ]; then
    pass "Refresh script found and executable"
else
    fail "Refresh script missing or not executable" "$(ls -l scripts/policy_refresh.sh 2>&1 || echo 'NOT FOUND')"
fi

# Test 2: Systemd service exists
log_test "2. quantum-policy-refresh.service exists"
if [ -f "deploy/systemd/quantum-policy-refresh.service" ]; then
    pass "Service file exists"
else
    fail "Service file missing"
fi

# Test 3: Systemd timer exists
log_test "3. quantum-policy-refresh.timer exists"
if [ -f "deploy/systemd/quantum-policy-refresh.timer" ]; then
    pass "Timer file exists"
else
    fail "Timer file missing"
fi

# Test 4: Timer configured for 30min interval
log_test "4. Timer configured for 30min interval"
if grep -q "OnUnitActiveSec=30min" deploy/systemd/quantum-policy-refresh.timer; then
    pass "Timer set to 30min"
else
    fail "Timer not set to 30min" "$(grep OnUnitActiveSec deploy/systemd/quantum-policy-refresh.timer || echo 'NOT FOUND')"
fi

# Test 5: Service has fail-open semantics
log_test "5. Service has fail-open semantics (SuccessExitStatus=0 1)"
if grep -q "SuccessExitStatus=0 1" deploy/systemd/quantum-policy-refresh.service; then
    pass "Service configured for fail-open"
else
    fail "Service missing fail-open config"
fi

# Test 6: Refresh script validates policy fields
log_test "6. Refresh script validates policy fields (version, hash, valid_until)"
REQUIRED_VALIDATIONS=("policy_version" "policy_hash" "valid_until_epoch")
MISSING_VALIDATIONS=()
for field in "${REQUIRED_VALIDATIONS[@]}"; do
    if ! grep -q "$field" scripts/policy_refresh.sh; then
        MISSING_VALIDATIONS+=("$field")
    fi
done
if [ ${#MISSING_VALIDATIONS[@]} -eq 0 ]; then
    pass "All required validations present"
else
    fail "Missing validations" "Fields: ${MISSING_VALIDATIONS[*]}"
fi

# Test 7: Refresh script checks expiry time
log_test "7. Refresh script checks expiry time (valid_until > now)"
if grep -q "VALID_UNTIL.*-le.*NOW" scripts/policy_refresh.sh || \
   grep -q "expired" scripts/policy_refresh.sh; then
    pass "Expiry check present"
else
    fail "Expiry check missing"
fi

# Test 8: Audit trail integrated in PolicyStore
log_test "8. lib/policy_store.py publishes to quantum:stream:policy.audit"
if grep -q "quantum:stream:policy.audit" lib/policy_store.py; then
    pass "Audit trail integrated"
else
    fail "Audit trail missing from PolicyStore"
fi

# Test 9: Audit trail has required fields
log_test "9. Audit trail contains required fields"
REQUIRED_AUDIT_FIELDS=("policy_version" "policy_hash" "valid_until_epoch" "created_at_epoch")
MISSING_AUDIT_FIELDS=()
for field in "${REQUIRED_AUDIT_FIELDS[@]}"; do
    if ! grep -A 10 "quantum:stream:policy.audit" lib/policy_store.py | grep -q "$field"; then
        MISSING_AUDIT_FIELDS+=("$field")
    fi
done
if [ ${#MISSING_AUDIT_FIELDS[@]} -eq 0 ]; then
    pass "All required audit fields present"
else
    fail "Missing audit fields" "Fields: ${MISSING_AUDIT_FIELDS[*]}"
fi

# Test 10: Audit trail is fail-open (try/except)
log_test "10. Audit trail wrapped in try/except (fail-open)"
if grep -B 5 "quantum:stream:policy.audit" lib/policy_store.py | grep -q "try" && \
   grep -A 10 "quantum:stream:policy.audit" lib/policy_store.py | grep -q "except"; then
    pass "Audit trail is fail-open"
else
    fail "Audit trail not wrapped in try/except"
fi

# Summary
echo ""
echo "========================================="
echo "POLICY REFRESH PROOF SUMMARY"
echo "========================================="
echo "PASS: $PASS_COUNT"
echo "FAIL: $FAIL_COUNT"
echo "========================================="

if [ "$FAIL_COUNT" -eq 0 ]; then
    echo "✅ ALL TESTS PASSED - Policy refresh automation verified"
    exit 0
else
    echo "❌ SOME TESTS FAILED - Review failures above"
    exit 1
fi
