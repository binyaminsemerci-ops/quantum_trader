#!/bin/bash
# Governor E2E Test - Automated proof of both BLOCK and PASS modes
# Usage: bash ops/tests/test_governor_e2e.sh
# Exit codes: 0=success, 1=failure

set -euo pipefail

# Configuration (from EventBus analysis)
STREAM_NAME="quantum:stream:trade.intent"
REDIS_CLI="redis-cli"
SERVICE_NAME="quantum-execution.service"

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

echo "=========================================="
echo "  GOVERNOR E2E TEST"
echo "=========================================="
echo ""

# Check prerequisites
echo "[CHECK] Verifying prerequisites..."
if ! command -v redis-cli &> /dev/null; then
    echo -e "${RED}FAIL: redis-cli not found${NC}"
    exit 1
fi

if ! systemctl is-active --quiet "$SERVICE_NAME"; then
    echo -e "${RED}FAIL: $SERVICE_NAME is not running${NC}"
    exit 1
fi

# Verify Redis connection
if ! $REDIS_CLI PING &> /dev/null; then
    echo -e "${RED}FAIL: Redis not responding${NC}"
    exit 1
fi

echo -e "${GREEN}✓ Prerequisites OK${NC}"
echo ""

# Set safe keys
echo "[SETUP] Initializing safe keys..."
$REDIS_CLI SET quantum:mode TESTNET > /dev/null
$REDIS_CLI SET quantum:governor:execution ENABLED > /dev/null
echo -e "${GREEN}✓ Safe keys set (mode=TESTNET, governor=ENABLED)${NC}"
echo ""

# Function to inject EventBus envelope format
inject_envelope() {
    local symbol=$1
    local side=$2
    local correlation_id="test-$(uuidgen 2>/dev/null || echo "$(date +%s)-$$")"
    local timestamp=$(date -u +%Y-%m-%dT%H:%M:%S)
    
    # Create JSON payload (proper envelope structure)
    local payload="{\"symbol\":\"$symbol\",\"side\":\"$side\",\"position_size_usd\":10.0,\"leverage\":1.0,\"entry_price\":0.32,\"stop_loss\":0.31,\"take_profit\":0.33,\"confidence\":0.75,\"timestamp\":\"$timestamp\",\"model\":\"test\",\"meta_strategy\":\"e2e_test\"}"
    
    # XADD with proper envelope fields
    $REDIS_CLI XADD "$STREAM_NAME" "*" \
        event_type "trade.intent" \
        payload "$payload" \
        correlation_id "$correlation_id" \
        timestamp "$timestamp" \
        source "ops-e2e-test" \
        trace_id "" > /dev/null
    
    echo "$correlation_id"
}

# ========================================
# TEST 1: BLOCKED MODE (kill=1)
# ========================================
echo "=========================================="
echo "TEST 1: GOVERNOR BLOCKED (kill=1)"
echo "=========================================="

$REDIS_CLI SET quantum:kill 1 > /dev/null
echo "[INJECT] Setting kill=1 (KILL MODE)"

CORR_ID=$(inject_envelope "OPUSDT" "BUY")
echo "[INJECT] Event injected: correlation_id=$CORR_ID"
echo "[WAIT] Waiting 3 seconds for processing..."
sleep 3

# Check logs for BLOCKED message
if journalctl -u "$SERVICE_NAME" --since "10 seconds ago" --no-pager | grep -q "🛑 BLOCKED"; then
    echo -e "${GREEN}✓ TEST 1 PASSED: Governor BLOCKED trade${NC}"
    echo ""
else
    echo -e "${RED}✗ TEST 1 FAILED: Expected 🛑 BLOCKED, not found in logs${NC}"
    echo ""
    echo "Recent logs:"
    journalctl -u "$SERVICE_NAME" --since "10 seconds ago" --no-pager | tail -20
    exit 1
fi

# ========================================
# TEST 2: PASSED MODE (kill=0)
# ========================================
echo "=========================================="
echo "TEST 2: GOVERNOR PASSED (kill=0)"
echo "=========================================="

$REDIS_CLI SET quantum:kill 0 > /dev/null
echo -e "${YELLOW}[CAUTION] Setting kill=0 (GO MODE)${NC}"

CORR_ID=$(inject_envelope "OPUSDT" "BUY")
echo "[INJECT] Event injected: correlation_id=$CORR_ID"
echo "[WAIT] Waiting 4 seconds for processing..."
sleep 4

# Check logs for PASSED message
if journalctl -u "$SERVICE_NAME" --since "15 seconds ago" --no-pager | grep -q "✅ PASSED"; then
    echo -e "${GREEN}✓ TEST 2 PASSED: Governor PASSED trade${NC}"
    
    # Verify execution occurred (in PAPER mode)
    if journalctl -u "$SERVICE_NAME" --since "15 seconds ago" --no-pager | grep -qE "Executing|PAPER ORDER"; then
        echo -e "${GREEN}✓ Order execution confirmed (PAPER mode)${NC}"
    else
        echo -e "${YELLOW}⚠ PASSED but no execution logged (may be OK if risk check failed)${NC}"
    fi
    echo ""
else
    echo -e "${RED}✗ TEST 2 FAILED: Expected ✅ PASSED, not found in logs${NC}"
    echo ""
    echo "Recent logs:"
    journalctl -u "$SERVICE_NAME" --since "15 seconds ago" --no-pager | tail -20
    
    # Restore safe state before failing
    $REDIS_CLI SET quantum:kill 1 > /dev/null
    exit 1
fi

# ========================================
# RESTORE SAFE STATE
# ========================================
echo "=========================================="
echo "RESTORING SAFE STATE"
echo "=========================================="

$REDIS_CLI SET quantum:kill 1 > /dev/null
echo -e "${GREEN}✓ kill=1 restored (KILL MODE)${NC}"
echo ""

# Verify final state
KILL_STATE=$($REDIS_CLI GET quantum:kill)
MODE=$($REDIS_CLI GET quantum:mode)
GOVERNOR=$($REDIS_CLI GET quantum:governor:execution)

echo "Final Redis State:"
echo "  quantum:kill = $KILL_STATE"
echo "  quantum:mode = $MODE"
echo "  quantum:governor:execution = $GOVERNOR"
echo ""

# ========================================
# SUMMARY
# ========================================
echo "=========================================="
echo -e "${GREEN}ALL TESTS PASSED ✓${NC}"
echo "=========================================="
echo "✓ TEST 1: Governor BLOCKED with kill=1"
echo "✓ TEST 2: Governor PASSED with kill=0"
echo "✓ Safe state restored (kill=1)"
echo ""
echo "Governor Gate is WORKING CORRECTLY!"
echo ""

exit 0
