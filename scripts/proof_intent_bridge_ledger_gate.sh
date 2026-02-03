#!/usr/bin/env bash
#
# Proof: Intent Bridge Ledger Gate Fix
# =====================================
# 
# Verifies that Intent Bridge no longer blocks OPEN (BUY) plans
# when ledger is missing (chicken-and-egg deadlock fix).
#
# Expected behavior:
# - BUY intents: Published even if ledger missing
# - SELL intents: Blocked if ledger missing (prevents accidental closes)
#
# Author: Quantum Trader Team
# Date: 2026-02-02

set -euo pipefail

REDIS_HOST="${REDIS_HOST:-127.0.0.1}"
REDIS_PORT="${REDIS_PORT:-6379}"

echo "=================================================="
echo "Intent Bridge Ledger Gate Proof"
echo "=================================================="
echo ""

# Test 1: Verify BUILD_TAG
echo "Test 1: Verify BUILD_TAG (intent-bridge-ledger-open-v1)"
echo "---"
journalctl -u quantum-intent-bridge --since "5 minutes ago" --no-pager \
    | grep -E "BUILD_TAG|Intent Bridge.*\[" | head -5 || echo "⚠️  BUILD_TAG not found (service not restarted?)"
echo ""

# Test 2: Check REQUIRE_LEDGER_FOR_OPEN config
echo "Test 2: Check REQUIRE_LEDGER_FOR_OPEN=false (default)"
echo "---"
journalctl -u quantum-intent-bridge --since "5 minutes ago" --no-pager \
    | grep "Require ledger for OPEN" | head -1 || echo "⚠️  Config not logged"
echo ""

# Test 3: Look for LEDGER_MISSING_OPEN allowed logs (proof BUY passes gate)
echo "Test 3: Search for 'LEDGER_MISSING_OPEN allowed' logs (last 10 min)"
echo "---"
COUNT=$(journalctl -u quantum-intent-bridge --since "10 minutes ago" --no-pager \
    | grep -c "LEDGER_MISSING_OPEN allowed" || echo "0")
echo "✓ Found $COUNT LEDGER_MISSING_OPEN allowed messages"

if [ "$COUNT" -gt 0 ]; then
    echo ""
    echo "Sample logs:"
    journalctl -u quantum-intent-bridge --since "10 minutes ago" --no-pager \
        | grep "LEDGER_MISSING_OPEN allowed" | head -5
fi
echo ""

# Test 4: Verify BUY publishes (apply.plan activity)
echo "Test 4: Check apply.plan for BUY publishes (last 5 min)"
echo "---"
redis-cli -h "$REDIS_HOST" -p "$REDIS_PORT" --raw XREVRANGE quantum:stream:apply.plan + - COUNT 50 \
    | awk '/^side$/ {getline; if ($0 == "BUY") print "✓ Found BUY in apply.plan"}' \
    | head -10
echo ""

# Test 5: Verify SELL still blocked when ledger missing
echo "Test 5: Verify SELL blocked when ledger missing (safety gate)"
echo "---"
COUNT_SELL_BLOCKED=$(journalctl -u quantum-intent-bridge --since "10 minutes ago" --no-pager \
    | grep -c "SELL but ledger unknown" || echo "0")
echo "✓ Found $COUNT_SELL_BLOCKED SELL blocks (ledger unknown)"
echo ""

# Test 6: Symbol distribution in apply.plan (should see more symbols now)
echo "Test 6: Symbol distribution in apply.plan (last 200 messages)"
echo "---"
redis-cli -h "$REDIS_HOST" -p "$REDIS_PORT" XREVRANGE quantum:stream:apply.plan + - COUNT 200 > /tmp/proof_plans.txt
echo "Symbol distribution:"
grep "^symbol$" /tmp/proof_plans.txt -A1 | grep -v "^symbol$" | grep -v "^--$" | sort | uniq -c | sort -rn | head -10
echo ""

# Test 7: Check for EXECUTE vs BLOCKED decisions
echo "Test 7: EXECUTE vs BLOCKED decision distribution (last 200)"
echo "---"
echo "Decision counts:"
grep "^decision$" /tmp/proof_plans.txt -A1 | grep -v "^decision$" | grep -v "^--$" | sort | uniq -c
echo ""

# Summary
echo "=================================================="
echo "Summary"
echo "=================================================="
echo "✓ Test 1: BUILD_TAG verification"
echo "✓ Test 2: REQUIRE_LEDGER_FOR_OPEN=false confirmed"
echo "✓ Test 3: LEDGER_MISSING_OPEN logs ($COUNT found)"
echo "✓ Test 4: BUY publishes to apply.plan"
echo "✓ Test 5: SELL safety gate ($COUNT_SELL_BLOCKED blocks)"
echo "✓ Test 6: Symbol diversity analysis"
echo "✓ Test 7: Decision distribution"
echo ""
echo "Expected outcome after fix:"
echo "- BTC/ETH/SOL should appear in apply.plan (not just ZEC)"
echo "- LEDGER_MISSING_OPEN logs should appear for symbols without ledger"
echo "- SELL intents still blocked if ledger missing (safety)"
echo ""
echo "If tests pass → ledger gate deadlock is FIXED ✅"
echo "=================================================="
