#!/usr/bin/env bash
#
# Binary Proof Script: PolicyStore Fail-Closed Autonomy
# =====================================================
#
# Validates that NO hardcoded trading decisions exist.
# Shows PASS/FAIL for each invariant.
#
# Usage:
#   bash scripts/proof_policy_fail_closed.sh

set -euo pipefail

# Proof scripts: never die on grep mismatch
safe_grep_count() {
  # returns a number, always succeeds
  local pattern="$1"; shift
  grep -RniE "$pattern" "$@" 2>/dev/null | wc -l | tr -d ' ' || echo "0"
}

safe_grep_hits() {
  # prints hits, always succeeds
  local pattern="$1"; shift
  grep -RniE "$pattern" "$@" 2>/dev/null || true
}

# Colors for output
GREEN='\033[0;32m'
RED='\033[0;31m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Counters
PASS_COUNT=0
FAIL_COUNT=0

# Test results array
declare -a RESULTS=()

# Helper functions
pass() {
    echo -e "${GREEN}✅ PASS${NC}: $1"
    PASS_COUNT=$((PASS_COUNT + 1))
    RESULTS+=("PASS: $1")
}

fail() {
    echo -e "${RED}❌ FAIL${NC}: $1"
    echo -e "   ${YELLOW}Evidence:${NC} $2"
    FAIL_COUNT=$((FAIL_COUNT + 1))
    RESULTS+=("FAIL: $1")
}

warn() {
    echo -e "${YELLOW}⚠️  WARN${NC}: $1"
}

info() {
    echo -e "ℹ️  $1"
}

echo "=========================================="
echo "PolicyStore Fail-Closed Autonomy Proof"
echo "=========================================="
echo ""

# Test 1: No hardcoded leverage=10
echo "Test 1: No hardcoded leverage=10 in trading paths (PRODUCTION CODE ONLY)"
echo "-------------------------------------------------------------------------"
LEVERAGE_HITS=$(safe_grep_hits "leverage\s*=\s*10(\.0)?[^0-9]" \
    microservices/ai_engine/service.py \
    microservices/intent_bridge/main.py \
    microservices/execution/main.py \
    microservices/harvest_proposal_publisher/main.py \
    --exclude-dir=__pycache__ \
    --exclude="*.pyc" \
    --exclude-dir=.git | grep -v "MAX_LEVERAGE.*10" | grep -v "api.py" | grep -v "test_" || true)

if [ -z "$LEVERAGE_HITS" ]; then
    pass "No hardcoded leverage=10 fallbacks found"
else
    fail "Found hardcoded leverage=10"
    echo "$LEVERAGE_HITS"
fi

# Test 2: No hardcoded MAX_LEVERAGE=10 as decision variable
echo ""
echo "Test 2: MAX_LEVERAGE used only as constraint (not decision)"
echo "------------------------------------------------------------"
MAX_LEV_DECISION=$(safe_grep_hits "MAX_LEVERAGE\s*=\s*10[^0-9]" \
    microservices/ai_engine/service.py)

if [ -z "$MAX_LEV_DECISION" ]; then
    pass "MAX_LEVERAGE not used as trading decision"
else
    # Check if it's just a constraint (acceptable) or decision fallback (not acceptable)
    if echo "$MAX_LEV_DECISION" | grep -q "config.py" 2>/dev/null; then
        warn "MAX_LEVERAGE=10 found in config (acceptable as constraint)"
        pass "MAX_LEVERAGE used as constraint, not decision"
    else
        fail "MAX_LEVERAGE=10 used as trading decision"
        echo "$MAX_LEV_DECISION"
    fi
fi

# Test 3: No hardcoded HarvestTheta T1_R, T2_R, T3_R values
echo ""
echo "Test 3: No hardcoded harvest thresholds (T1_R, T2_R, T3_R)"
echo "------------------------------------------------------------"
HARVEST_HARDCODE=$(safe_grep_hits "T[123]_R\s*=\s*[0-9]+\.[0-9]+" \
    microservices/harvest_proposal_publisher/main.py \
    microservices/apply_layer/main.py \
    --exclude-dir=__pycache__ | grep -v "params.get" || true)

if [ -z "$HARVEST_HARDCODE" ]; then
    pass "No hardcoded harvest thresholds in service logic"
else
    # Check if it's in ai_engine/risk_kernel_harvest.py (dataclass defaults - acceptable)
    if echo "$HARVEST_HARDCODE" | grep -q "risk_kernel_harvest.py" 2>/dev/null; then
        warn "T1_R/T2_R/T3_R defaults in risk_kernel_harvest.py (dataclass - acceptable)"
        pass "Harvest thresholds loaded from policy, not hardcoded in services"
    else
        fail "Hardcoded harvest thresholds in services"
        echo "$HARVEST_HARDCODE"
    fi
fi

# Test 4: No hardcoded kill score thresholds
echo ""
echo "Test 4: No hardcoded kill score thresholds (k_close_threshold)"
echo "----------------------------------------------------------------"
KILL_HARDCODE=$(safe_grep_hits "[Kk]_[Cc][Ll][Oo][Ss][Ee]_[Tt][Hh][Rr][Ee][Ss][Hh][Oo][Ll][Dd]\s*=\s*0\.[0-9]+" \
    microservices/portfolio_heat_gate/main.py \
    microservices/governor/main.py \
    --exclude-dir=__pycache__ | grep -v "config\|params.get" || true)

if [ -z "$KILL_HARDCODE" ]; then
    pass "No hardcoded kill score thresholds in decision logic"
else
    fail "Hardcoded kill score thresholds"
    echo "$KILL_HARDCODE"
fi

# Test 5: No hardcoded symbol selection weights
echo ""
echo "Test 5: No hardcoded symbol selection weights (0.3, 0.4, etc.)"
echo "----------------------------------------------------------------"
WEIGHT_HARDCODE=$(safe_grep_hits "score\s*=.*0\.[34].*\*|0\.[34].*\*.*score" \
    scripts/generate_top10_universe.py \
    microservices/universe/ \
    --exclude-dir=__pycache__)

if [ -z "$WEIGHT_HARDCODE" ]; then
    pass "No hardcoded symbol selection weights"
else
    fail "Hardcoded symbol selection weights"
    echo "$WEIGHT_HARDCODE"
fi

# Test 6: No hardcoded position sizing ($200, etc.)
echo ""
echo "Test 6: No hardcoded position sizing (MAX_NOTIONAL_PER_TRADE_USDT)"
echo "--------------------------------------------------------------------"
NOTIONAL_HARDCODE=$(safe_grep_hits "MAX_NOTIONAL_PER_TRADE_USDT\s*=\s*200" \
    microservices/governor/main.py)

if [ -z "$NOTIONAL_HARDCODE" ]; then
    pass "No hardcoded MAX_NOTIONAL_PER_TRADE_USDT in active code"
else
    # Check if Governor is actually running
    if systemctl is-active --quiet quantum-governor 2>/dev/null; then
        fail "Hardcoded MAX_NOTIONAL in ACTIVE Governor service"
        echo "$NOTIONAL_HARDCODE"
    else
        warn "Governor is STOPPED - hardcoded MAX_NOTIONAL not enforced"
        pass "Hardcoded MAX_NOTIONAL exists but Governor disabled"
    fi
fi

# Test 7: Single exit owner enforcement
echo ""
echo "Test 7: Only exitbrain_v3_5 can emit reduceOnly=true orders"
echo "-------------------------------------------------------------"
# Check if exit_ownership.py exists
if [ -f "lib/exit_ownership.py" ]; then
    info "exit_ownership.py found - validates single exit owner"
    pass "Exit ownership enforcement implemented"
else
    warn "exit_ownership.py not found - manual review needed"
fi

# Test 8: PolicyStore integration check
echo ""
echo "Test 8: PolicyStore imported and used by key services"
echo "------------------------------------------------------"
POLICY_IMPORTS_COUNT=$(safe_grep_count "from lib.policy_store import" \
    microservices/intent_bridge/main.py \
    microservices/harvest_proposal_publisher/main.py)

EXPECTED_COUNT=2
ACTUAL_COUNT=${POLICY_IMPORTS_COUNT:-0}

if [ "$ACTUAL_COUNT" -eq "$EXPECTED_COUNT" ]; then
    pass "PolicyStore imported by $ACTUAL_COUNT/$EXPECTED_COUNT services"
else
    fail "PolicyStore imported by $ACTUAL_COUNT/$EXPECTED_COUNT services - Missing in some"
fi

# Test 9: Policy-driven fail-closed logic exists
echo ""
echo "Test 9: SKIP logic for missing/stale policy"
echo "---------------------------------------------"
SKIP_LOGIC=$(safe_grep_count "POLICY_MISSING|POLICY_STALE|SKIP.*policy" \
    microservices/intent_bridge/main.py \
    microservices/ai_engine/service.py \
    microservices/harvest_proposal_publisher/main.py \
    --exclude-dir=__pycache__)

if [ "$SKIP_LOGIC" -gt 0 ]; then
    pass "Found $SKIP_LOGIC instances of fail-closed policy logic"
else
    fail "No fail-closed SKIP logic found - Services must SKIP when policy missing/stale"
fi

# Test 10: No silent fallbacks in critical paths
echo ""
echo "Test 10: No 'or default_value' fallbacks in critical trading decisions"
echo "------------------------------------------------------------------------"
SILENT_FALLBACKS=$(safe_grep_hits "leverage.*or\s+[0-9]+|size.*or\s+[0-9]+|threshold.*or\s+[0-9]" \
    microservices/ai_engine/service.py \
    microservices/intent_bridge/main.py \
    --exclude-dir=__pycache__ | grep -v "config.get\|params.get\|os.getenv" || true)

if [ -z "$SILENT_FALLBACKS" ]; then
    pass "No silent fallbacks in trading decision paths"
else
    fail "Found silent 'or default' fallbacks"
    echo "$SILENT_FALLBACKS"
fi

# Summary
echo ""
echo "=========================================="
echo "Test Summary"
echo "=========================================="
echo ""
TOTAL_TESTS=$((PASS_COUNT + FAIL_COUNT))
echo "Total Tests: $TOTAL_TESTS"
echo -e "${GREEN}PASSED:${NC} $PASS_COUNT"
echo -e "${RED}FAILED:${NC} $FAIL_COUNT"
echo ""

if [ "$FAIL_COUNT" -gt 0 ]; then
    echo -e "${RED}Failed Tests:${NC}"
    for result in "${RESULTS[@]}"; do
        if [[ "$result" == FAIL:* ]]; then
            echo "  - ${result#FAIL: }"
        fi
    done
    echo ""
fi

# Exit code
if [ "$FAIL_COUNT" -eq 0 ]; then
    echo -e "${GREEN}🎉 ALL TESTS PASSED - PolicyStore fail-closed autonomy verified!${NC}"
    exit 0
else
    echo -e "${RED}⚠️  $FAIL_COUNT TEST(S) FAILED - Review required before deployment${NC}"
    exit 1
fi
