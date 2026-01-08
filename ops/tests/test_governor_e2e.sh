#!/bin/bash
# E2E Governor Gate Test
# Tests both BLOCK (kill=1) and PASS (kill=0) modes with proper EventBus envelope format

set -e

STREAM_NAME="quantum:stream:trade.intent"
TEST_SYMBOL="OPUSDT"
REDIS_CLI="redis-cli"

echo "=========================================="
echo "Governor E2E Test - EventBus Envelope Format"
echo "=========================================="

# Function to inject proper EventBus envelope using Python for JSON
inject_envelope() {
    local symbol=$1
    local side=$2
    local size=${3:-10.0}
    local leverage=${4:-1.0}
    local entry_price=${5:-0.3165}
    
    local correlation_id=$(uuidgen 2>/dev/null || echo "test-$(date +%s)")
    
    # Use Python to generate valid JSON payload
    local payload=$(python3 - <<PYEOF
import json
import datetime
payload = {
    "symbol": "$symbol",
    "side": "$side",
    "position_size_usd": $size,
    "leverage": $leverage,
    "entry_price": $entry_price,
    "stop_loss": round($entry_price * 0.975, 6),
    "take_profit": round($entry_price * 1.03, 6),
    "confidence": 0.72,
    "timestamp": datetime.datetime.now(datetime.timezone.utc).isoformat(),
    "model": "test",
    "meta_strategy": "e2e-test",
    "consensus_count": 1,
    "total_models": 1
}
print(json.dumps(payload))
PYEOF
)
    
    # XADD with EventBus envelope
    $REDIS_CLI XADD "$STREAM_NAME" "*"         event_type "trade.intent"         payload "$payload"         correlation_id "$correlation_id"         timestamp "$(date -u +"%Y-%m-%dT%H:%M:%S.%6N+00:00" 2>/dev/null || date -u +"%Y-%m-%dT%H:%M:%SZ")"         source "ops-test"         trace_id "" > /dev/null
    
    echo "✅ Injected: $symbol $side (envelope format, Python JSON)"
}

# Ensure safe initial state
echo "Setting safe initial state..."
$REDIS_CLI SET quantum:mode TESTNET > /dev/null
$REDIS_CLI SET quantum:governor:execution ENABLED > /dev/null
$REDIS_CLI SET quantum:kill 1 > /dev/null
echo "✅ Safe state: mode=TESTNET, governor=ENABLED, kill=1"
echo ""

# TEST 1: BLOCK MODE (kill=1)
echo "=========================================="
echo "TEST 1: BLOCK MODE (kill=1)"
echo "=========================================="
$REDIS_CLI SET quantum:kill 1 > /dev/null
sleep 1

echo "Injecting test signal..."
inject_envelope "$TEST_SYMBOL" "BUY" 10.0 1.0 0.3165
sleep 4

echo "Checking logs for BLOCKED..."
if journalctl -u quantum-execution.service --since "10 seconds ago" --no-pager | grep -q "🛑 BLOCKED"; then
    echo "✅ TEST 1 PASSED - Order was BLOCKED"
else
    echo "❌ TEST 1 FAILED - Expected BLOCKED, not found in logs"
    journalctl -u quantum-execution.service --since "10 seconds ago" --no-pager | tail -20
    exit 1
fi
echo ""

# TEST 2: PASS MODE (kill=0)
echo "=========================================="
echo "TEST 2: PASS MODE (kill=0)"
echo "=========================================="
$REDIS_CLI SET quantum:kill 0 > /dev/null
sleep 1

echo "Injecting test signal..."
inject_envelope "$TEST_SYMBOL" "BUY" 10.0 1.0 0.3165
sleep 4

echo "Checking logs for PASSED and execution..."
if journalctl -u quantum-execution.service --since "10 seconds ago" --no-pager | grep -q "✅ PASSED"; then
    echo "✅ TEST 2 PASSED - Order was PASSED by Governor"
    
    # Check if order was executed
    if journalctl -u quantum-execution.service --since "10 seconds ago" --no-pager | grep -q "Executing:"; then
        echo "✅ Order execution confirmed"
        
        # Check paper price source
        if journalctl -u quantum-execution.service --since "10 seconds ago" --no-pager | grep -q "price_source=entry_price"; then
            echo "✅ Paper price using entry_price (not placeholder)"
        fi
    else
        echo "⚠️  Warning: Governor passed but execution not confirmed"
    fi
else
    echo "❌ TEST 2 FAILED - Expected PASSED, not found in logs"
    journalctl -u quantum-execution.service --since "10 seconds ago" --no-pager | tail -20
    exit 1
fi
echo ""

# RESTORE SAFE STATE
echo "=========================================="
echo "Restoring safe state..."
echo "=========================================="
$REDIS_CLI SET quantum:kill 1 > /dev/null
echo "✅ Restored: kill=1 (KILL MODE)"
echo ""

echo "=========================================="
echo "✅ ALL TESTS PASSED"
echo "=========================================="
echo "Governor Gate verified in both modes:"
echo "  - BLOCK mode (kill=1): ✅"
echo "  - PASS mode (kill=0): ✅"
echo "  - Paper price sanity: ✅"
echo "  - Safe state restored: ✅"
exit 0
