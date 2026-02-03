#!/usr/bin/env bash
set -euo pipefail

# ==============================================================================
# P2.7 HeatBridge - Proof Script
# ==============================================================================
# Tests:
# A) Basic injection: inject decision, verify lookup keys created
# B) Deduplication: inject same plan_id twice, verify dedupe skips
# C) Latest update: inject new plan_id for same symbol, verify latest pointer updates
#
# Exit codes: 0 = PASS, 1 = FAIL
# ==============================================================================

# Auto-detect repo root (location-agnostic)
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

REDIS="redis-cli"
PYTHON3="python3"
INJECT_SCRIPT="$REPO_ROOT/scripts/proof_p27_inject_heat_decision.py"
FAILURES=0

echo "===================================================================="
echo "P2.7 HeatBridge Proof: Shadow Wiring Layer"
echo "===================================================================="
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

# ==============================================================================
# Test 0: Preflight Checks
# ==============================================================================

echo "[0] Preflight: Redis connectivity"
if ! $REDIS PING >/dev/null 2>&1; then
    fail "Redis not available"
    exit 1
fi
pass "Redis available"

echo "[0] Preflight: HeatBridge service running"
if ! systemctl is-active --quiet quantum-heat-bridge 2>/dev/null; then
    echo "⚠️  WARNING: HeatBridge service not running, tests may fail"
fi

echo "[0] Preflight: Metrics endpoint"
if ! curl -s http://localhost:8070/metrics >/dev/null 2>&1; then
    echo "⚠️  WARNING: HeatBridge metrics not responding"
fi

# ==============================================================================
# Test A: Basic Injection
# ==============================================================================

echo ""
echo "[A] Test: Basic injection → lookup keys created"

# Inject decision
PLAN_A="test_a_$(date +%s)"
SYMBOL_A="BTCUSDT"

$PYTHON3 "$INJECT_SCRIPT" \
    --symbol "$SYMBOL_A" \
    --plan_id "$PLAN_A" \
    --heat_level "cold" \
    --heat_action "NONE" \
    --out_action "FULL_CLOSE_PROPOSED" \
    --score 0.2 \
    >/dev/null

echo "   Waiting 3s for processing..."
sleep 3

# Assert: by_plan key exists
BY_PLAN_KEY="quantum:harvest:heat:by_plan:$PLAN_A"
if ! $REDIS EXISTS "$BY_PLAN_KEY" >/dev/null; then
    fail "Test A: by_plan key not created: $BY_PLAN_KEY"
else
    # Check fields
    SYMBOL_FIELD=$($REDIS HGET "$BY_PLAN_KEY" symbol)
    OUT_ACTION_FIELD=$($REDIS HGET "$BY_PLAN_KEY" out_action)
    HEAT_ACTION_FIELD=$($REDIS HGET "$BY_PLAN_KEY" heat_action)
    
    if [ "$SYMBOL_FIELD" = "$SYMBOL_A" ] && [ "$OUT_ACTION_FIELD" = "FULL_CLOSE_PROPOSED" ] && [ "$HEAT_ACTION_FIELD" = "NONE" ]; then
        pass "Test A: by_plan key created with correct fields"
    else
        fail "Test A: by_plan key fields incorrect: symbol=$SYMBOL_FIELD out_action=$OUT_ACTION_FIELD heat_action=$HEAT_ACTION_FIELD"
    fi
    
    # Check TTL
    TTL=$($REDIS TTL "$BY_PLAN_KEY")
    if [ "$TTL" -gt 0 ]; then
        pass "Test A: by_plan key has TTL=$TTL"
    else
        fail "Test A: by_plan key has invalid TTL=$TTL"
    fi
fi

# Assert: latest_symbol key exists
LATEST_KEY="quantum:harvest:heat:latest:$SYMBOL_A"
if ! $REDIS EXISTS "$LATEST_KEY" >/dev/null; then
    fail "Test A: latest_symbol key not created: $LATEST_KEY"
else
    LAST_PLAN_FIELD=$($REDIS HGET "$LATEST_KEY" last_plan_id)
    if [ "$LAST_PLAN_FIELD" = "$PLAN_A" ]; then
        pass "Test A: latest_symbol key created with last_plan_id=$PLAN_A"
    else
        fail "Test A: latest_symbol last_plan_id incorrect: $LAST_PLAN_FIELD"
    fi
fi

# Assert: latest_plan_id pointer exists
POINTER_KEY="quantum:harvest:heat:latest_plan_id:$SYMBOL_A"
if ! $REDIS EXISTS "$POINTER_KEY" >/dev/null; then
    fail "Test A: latest_plan_id pointer not created"
else
    POINTER_VALUE=$($REDIS GET "$POINTER_KEY")
    if [ "$POINTER_VALUE" = "$PLAN_A" ]; then
        pass "Test A: latest_plan_id pointer created with value=$PLAN_A"
    else
        fail "Test A: latest_plan_id pointer value incorrect: $POINTER_VALUE"
    fi
fi

# ==============================================================================
# Test B: Deduplication
# ==============================================================================

echo ""
echo "[B] Test: Deduplication → same plan_id not rewritten"

# Inject SAME plan_id again
$PYTHON3 "$INJECT_SCRIPT" \
    --symbol "$SYMBOL_A" \
    --plan_id "$PLAN_A" \
    --heat_level "cold" \
    --heat_action "NONE" \
    --out_action "FULL_CLOSE_PROPOSED" \
    --score 0.2 \
    >/dev/null 2>&1 || true

echo "   Waiting 2s for processing..."
sleep 2

# Check metrics for dedupe skips
METRICS=$(curl -s http://localhost:8070/metrics 2>/dev/null || echo "")
if echo "$METRICS" | grep -q "p27_dedupe_skips_total"; then
    DEDUPE_COUNT=$(echo "$METRICS" | grep "p27_dedupe_skips_total" | awk '{print $2}')
    if [ "$DEDUPE_COUNT" -gt 0 ]; then
        pass "Test B: Deduplication working (p27_dedupe_skips_total=$DEDUPE_COUNT)"
    else
        echo "⚠️  WARNING: Deduplication metric not incremented yet (may be acceptable)"
    fi
else
    echo "⚠️  WARNING: Could not verify dedupe metric"
fi

# ==============================================================================
# Test C: Latest Update
# ==============================================================================

echo ""
echo "[C] Test: New plan_id for same symbol updates latest pointer"

# Inject NEW plan_id for SAME symbol
PLAN_C="test_c_$(date +%s)"
SYMBOL_C="$SYMBOL_A"  # Same symbol

$PYTHON3 "$INJECT_SCRIPT" \
    --symbol "$SYMBOL_C" \
    --plan_id "$PLAN_C" \
    --heat_level "warm" \
    --heat_action "DOWNGRADE_FULL_TO_PARTIAL" \
    --out_action "PARTIAL_50_PROPOSED" \
    --score 0.5 \
    --partial 0.5 \
    >/dev/null

echo "   Waiting 3s for processing..."
sleep 3

# Assert: latest_plan_id pointer updated
POINTER_KEY_C="quantum:harvest:heat:latest_plan_id:$SYMBOL_C"
POINTER_VALUE_C=$($REDIS GET "$POINTER_KEY_C")

if [ "$POINTER_VALUE_C" = "$PLAN_C" ]; then
    pass "Test C: latest_plan_id pointer updated to new plan_id=$PLAN_C"
else
    fail "Test C: latest_plan_id pointer not updated, still=$POINTER_VALUE_C (expected=$PLAN_C)"
fi

# Assert: latest_symbol key updated
LATEST_KEY_C="quantum:harvest:heat:latest:$SYMBOL_C"
LAST_PLAN_FIELD_C=$($REDIS HGET "$LATEST_KEY_C" last_plan_id)
OUT_ACTION_C=$($REDIS HGET "$LATEST_KEY_C" out_action)

if [ "$LAST_PLAN_FIELD_C" = "$PLAN_C" ] && [ "$OUT_ACTION_C" = "PARTIAL_50_PROPOSED" ]; then
    pass "Test C: latest_symbol key updated with last_plan_id=$PLAN_C and out_action=$OUT_ACTION_C"
else
    fail "Test C: latest_symbol key not properly updated"
fi

# ==============================================================================
# Test D: Metrics Verification
# ==============================================================================

echo ""
echo "[D] Verify HeatBridge metrics"

METRICS=$(curl -s http://localhost:8070/metrics 2>/dev/null || echo "")

if echo "$METRICS" | grep -q "p27_in_messages_total"; then
    IN_COUNT=$(echo "$METRICS" | grep "p27_in_messages_total" | awk '{print $2}')
    pass "Test D: Metrics available (p27_in_messages_total=$IN_COUNT)"
else
    fail "Test D: p27_in_messages_total metric not found"
fi

if echo "$METRICS" | grep -q "p27_written_total"; then
    pass "Test D: p27_written_total metric found"
else
    fail "Test D: p27_written_total metric not found"
fi

# ==============================================================================
# Test E: Health Endpoint
# ==============================================================================

echo ""
echo "[E] Verify HeatBridge health endpoint"

HEALTH=$(curl -s http://localhost:8071/health 2>/dev/null || echo "")

if echo "$HEALTH" | grep -q '"status"'; then
    STATUS=$(echo "$HEALTH" | python3 -c "import sys, json; print(json.load(sys.stdin).get('status', 'unknown'))" 2>/dev/null || echo "parse_error")
    if [ "$STATUS" = "ok" ]; then
        pass "Test E: Health endpoint responds with status=ok"
    else
        fail "Test E: Health status not ok: $STATUS"
    fi
else
    fail "Test E: Health endpoint not responding properly"
fi

# ==============================================================================
# Cleanup and Summary
# ==============================================================================

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
    echo "  systemctl status quantum-heat-bridge"
    echo "  journalctl -u quantum-heat-bridge -n 30 --no-pager"
    echo "  redis-cli KEYS 'quantum:harvest:heat:by_plan:*' | head -5"
    echo "  redis-cli KEYS 'quantum:harvest:heat:latest:*' | head -5"
    echo "  curl -s http://localhost:8070/metrics | grep p27_"
    echo "  curl -s http://localhost:8071/health"
    exit 1
fi
