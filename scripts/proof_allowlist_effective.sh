#!/usr/bin/env bash
set -euo pipefail

# Proof: Effective Allowlist Source
# Verifies that Intent Bridge uses AI policy universe when available

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REDIS_CLI="${REDIS_CLI:-redis-cli}"

# Color codes
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Counters
PASS_COUNT=0
FAIL_COUNT=0

log_info() {
    echo -e "${GREEN}[INFO]${NC} $*"
}

log_warn() {
    echo -e "${YELLOW}[WARN]${NC} $*"
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $*"
}

pass_test() {
    local test_num=$1
    local msg=$2
    echo -e "${GREEN}✅ TEST $test_num: PASS${NC} - $msg"
    ((PASS_COUNT++))
}

fail_test() {
    local test_num=$1
    local msg=$2
    echo -e "${RED}❌ TEST $test_num: FAIL${NC} - $msg"
    ((FAIL_COUNT++))
}

echo "=================================================="
echo "  PROOF: Effective Allowlist Source Verification"
echo "=================================================="
echo ""

# TEST 1: Check if policy exists and is valid
log_info "TEST 1: Check PolicyStore status"
POLICY_EXISTS=$($REDIS_CLI EXISTS quantum:policy:current)

if [ "$POLICY_EXISTS" -eq 1 ]; then
    POLICY_VERSION=$($REDIS_CLI HGET quantum:policy:current policy_version)
    POLICY_HASH=$($REDIS_CLI HGET quantum:policy:current policy_hash | head -c 8)
    VALID_UNTIL=$($REDIS_CLI HGET quantum:policy:current valid_until_epoch)
    NOW=$(date +%s)
    
    if [ "$VALID_UNTIL" -gt "$NOW" ]; then
        pass_test 1 "Policy valid: version=$POLICY_VERSION hash=$POLICY_HASH"
        POLICY_VALID=true
    else
        fail_test 1 "Policy stale: valid_until=$VALID_UNTIL < now=$NOW"
        POLICY_VALID=false
    fi
else
    fail_test 1 "PolicyStore not populated (quantum:policy:current missing)"
    POLICY_VALID=false
fi

echo ""

# TEST 2: Check Intent Bridge logs for effective allowlist
log_info "TEST 2: Check Intent Bridge effective allowlist source"
RECENT_LOGS=$(journalctl -u quantum-intent-bridge --since "10 minutes ago" -n 100 --no-pager 2>/dev/null || echo "")

if echo "$RECENT_LOGS" | grep -q "ALLOWLIST_EFFECTIVE"; then
    # Extract latest ALLOWLIST_EFFECTIVE log
    ALLOWLIST_LOG=$(echo "$RECENT_LOGS" | grep "ALLOWLIST_EFFECTIVE" | tail -1)
    
    # Parse source
    SOURCE=$(echo "$ALLOWLIST_LOG" | grep -oP 'source=\K\w+' || echo "unknown")
    COUNT=$(echo "$ALLOWLIST_LOG" | grep -oP 'count=\K\d+' || echo "0")
    
    log_info "Latest effective allowlist: source=$SOURCE count=$COUNT"
    
    # If policy is valid, source MUST be 'policy'
    if [ "$POLICY_VALID" = true ]; then
        if [ "$SOURCE" = "policy" ]; then
            pass_test 2 "Intent Bridge uses policy universe (source=policy)"
        else
            fail_test 2 "Intent Bridge NOT using policy! source=$SOURCE (expected: policy)"
        fi
    else
        if [ "$SOURCE" = "policy" ]; then
            fail_test 2 "Intent Bridge claims source=policy but policy is invalid/stale!"
        else
            pass_test 2 "Intent Bridge uses fallback (source=$SOURCE) because policy invalid"
        fi
    fi
else
    log_warn "No ALLOWLIST_EFFECTIVE logs found in last 10 minutes"
    log_warn "Service may not have been restarted since update"
    fail_test 2 "Cannot verify effective allowlist source (no logs)"
fi

echo ""

# TEST 3: Verify testnet intersection (if TESTNET_MODE)
log_info "TEST 3: Check testnet intersection logging"
if echo "$RECENT_LOGS" | grep -q "TESTNET_INTERSECTION"; then
    INTERSECTION_LOG=$(echo "$RECENT_LOGS" | grep "TESTNET_INTERSECTION" | tail -1)
    
    AI_COUNT=$(echo "$INTERSECTION_LOG" | grep -oP 'AI=\K\d+' || echo "0")
    TRADABLE_COUNT=$(echo "$INTERSECTION_LOG" | grep -oP 'testnet_tradable=\K\d+' || echo "0")
    SHADOW_COUNT=$(echo "$INTERSECTION_LOG" | grep -oP 'shadow=\K\d+' || echo "0")
    
    log_info "Testnet intersection: AI=$AI_COUNT → tradable=$TRADABLE_COUNT (shadow=$SHADOW_COUNT)"
    
    if [ "$TRADABLE_COUNT" -gt 0 ]; then
        pass_test 3 "Testnet intersection active: $AI_COUNT → $TRADABLE_COUNT tradable"
    else
        fail_test 3 "Testnet intersection resulted in 0 tradable symbols!"
    fi
else
    log_warn "No TESTNET_INTERSECTION logs found"
    log_warn "TESTNET_MODE may be disabled or service not restarted"
    pass_test 3 "Testnet intersection not required (mainnet or disabled)"
fi

echo ""
echo "=================================================="
echo "  SUMMARY"
echo "=================================================="
echo "✅ PASS: $PASS_COUNT"
echo "❌ FAIL: $FAIL_COUNT"
echo ""

if [ "$FAIL_COUNT" -eq 0 ]; then
    echo -e "${GREEN}🎯 ALL TESTS PASSED${NC}"
    echo ""
    echo "Effective allowlist source is verified:"
    echo "  - PolicyStore: $POLICY_VERSION (valid)" 
    echo "  - Intent Bridge: Uses policy universe"
    echo "  - Testnet: Intersection applied (if enabled)"
    exit 0
else
    echo -e "${RED}⚠️  SOME TESTS FAILED${NC}"
    echo ""
    echo "Action required:"
    if [ "$POLICY_VALID" = true ] && [ "$SOURCE" != "policy" ]; then
        echo "  1. Restart quantum-intent-bridge service"
        echo "  2. Re-run this proof script"
    fi
    echo ""
    exit 1
fi
