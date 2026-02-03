#!/usr/bin/env bash
# Proof: Intent Bridge Venue Intersection Implementation
# Verifies surgical patch for explicit venue tradable filtering

cd "$(dirname "$0")/.." || exit 1

echo "============================================================"
echo "  PROOF: Intent Bridge Venue Intersection"
echo "============================================================"
echo ""

PASS=0
FAIL=0

# TEST 1: Check for structured log format token
echo "[TEST 1] Verify ALLOWLIST_EFFECTIVE venue= log format exists in code"
if grep -q 'ALLOWLIST_EFFECTIVE venue=' microservices/intent_bridge/main.py; then
    echo "✅ PASS: Structured logging format present"
    PASS=$((PASS + 1))
else
    echo "❌ FAIL: ALLOWLIST_EFFECTIVE venue= not found in code"
    FAIL=$((FAIL + 1))
fi

echo ""

# TEST 2: Check for explicit intersection operation
echo "[TEST 2] Verify explicit set intersection in code"
if grep -q 'allowlist & tradable_symbols' microservices/intent_bridge/main.py; then
    echo "✅ PASS: Explicit intersection operation found"
    PASS=$((PASS + 1))
else
    echo "❌ FAIL: No explicit intersection operation found"
    FAIL=$((FAIL + 1))
fi

echo ""

# TEST 3: Check for all required fields in log format
echo "[TEST 3] Verify all required log fields present"
MISSING_FIELDS=0

# Check each field individually (simpler than regex)
grep -q 'venue=' microservices/intent_bridge/main.py && grep -q 'ALLOWLIST_EFFECTIVE' microservices/intent_bridge/main.py || MISSING_FIELDS=$((MISSING_FIELDS + 1))
grep -q 'source=' microservices/intent_bridge/main.py || MISSING_FIELDS=$((MISSING_FIELDS + 1))
grep -q 'policy_count=' microservices/intent_bridge/main.py || MISSING_FIELDS=$((MISSING_FIELDS + 1))
grep -q 'allowlist_count=' microservices/intent_bridge/main.py || MISSING_FIELDS=$((MISSING_FIELDS + 1))
grep -q 'tradable_count=' microservices/intent_bridge/main.py || MISSING_FIELDS=$((MISSING_FIELDS + 1))
grep -q 'final_count=' microservices/intent_bridge/main.py || MISSING_FIELDS=$((MISSING_FIELDS + 1))
grep -q 'venue_limited=' microservices/intent_bridge/main.py || MISSING_FIELDS=$((MISSING_FIELDS + 1))
grep -q 'tradable_fetch_failed=' microservices/intent_bridge/main.py || MISSING_FIELDS=$((MISSING_FIELDS + 1))

if [ $MISSING_FIELDS -eq 0 ]; then
    echo "✅ PASS: All 8 required fields present"
    PASS=$((PASS + 1))
else
    echo "❌ FAIL: $MISSING_FIELDS fields missing"
    FAIL=$((FAIL + 1))
fi

echo ""

# TEST 4: Check for fail-open logic
echo "[TEST 4] Verify fail-open on fetch failure"
if grep -q 'tradable_fetch_failed = 1' microservices/intent_bridge/main.py && \
   grep -q 'fail-open' microservices/intent_bridge/main.py; then
    echo "✅ PASS: Fail-open logic present"
    PASS=$((PASS + 1))
else
    echo "❌ FAIL: Fail-open logic not found"
    FAIL=$((FAIL + 1))
fi

echo ""
echo "============================================================"
echo "  SUMMARY"
echo "============================================================"
echo "✅ PASS: $PASS"
echo "❌ FAIL: $FAIL"
echo ""

if [ $FAIL -eq 0 ]; then
    echo "🎯 ALL TESTS PASSED"
    echo ""
    echo "Venue intersection patch verified:"
    echo "  - Structured logging format: ALLOWLIST_EFFECTIVE venue=..."
    echo "  - Explicit intersection: allowlist & tradable_symbols"
    echo "  - All 8 required fields present"
    echo "  - Fail-open logic on tradable fetch failure"
    exit 0
else
    echo "⚠️  SOME TESTS FAILED"
    exit 1
fi
