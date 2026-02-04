#!/bin/bash
# Verify ExitBrain v3.5 Adaptive Leverage Activation
# Created: 2025-12-24

echo "🔍 VERIFYING EXITBRAIN v3.5 ADAPTIVE LEVERAGE ACTIVATION"
echo "=================================================================="

# Test 1: Check for v3.5 log entries
echo ""
echo "📊 Test 1: Searching for ExitBrain v3.5 activity in logs..."
echo "-----------------------------------------------------------"
docker logs --tail 500 quantum_backend 2>&1 | grep -iE 'ExitBrain v3.5|compute_adaptive_levels|v35_integration|Adaptive Levels Calculated' | head -10

if [ $? -eq 0 ]; then
    echo "✅ FOUND v3.5 activity"
else
    echo "❌ NO v3.5 activity found"
fi

# Test 2: Check for leverage values != 1
echo ""
echo "📊 Test 2: Searching for adaptive leverage values (should be 5-80)..."
echo "----------------------------------------------------------------------"
docker logs --tail 500 quantum_backend 2>&1 | grep -iE 'leverage.*[5-9][0-9]|leverage.*[1-7][0-9]x|tp1.*%|harvest_scheme' | head -10

if [ $? -eq 0 ]; then
    echo "✅ FOUND adaptive leverage logs"
else
    echo "❌ NO adaptive leverage logs found"
fi

# Test 3: Check for ILF metadata
echo ""
echo "📊 Test 3: Searching for ILF metadata..."
echo "----------------------------------------"
docker logs --tail 500 quantum_backend 2>&1 | grep -iE 'ILF|ilf_metadata|volatility_factor|atr_value|exchange_divergence' | head -10

if [ $? -eq 0 ]; then
    echo "✅ FOUND ILF metadata"
else
    echo "❌ NO ILF metadata found"
fi

# Test 4: Check if exitbrain.adaptive_levels stream exists
echo ""
echo "📊 Test 4: Checking exitbrain.adaptive_levels stream..."
echo "-------------------------------------------------------"
STREAM_LEN=$(docker exec quantum_redis redis-cli XLEN quantum:stream:exitbrain.adaptive_levels 2>&1)

if [[ "$STREAM_LEN" =~ ^[0-9]+$ ]]; then
    echo "✅ Stream exists with $STREAM_LEN events"
else
    echo "⚠️  Stream does not exist or error: $STREAM_LEN"
fi

# Test 5: Check recent trade.intent events
echo ""
echo "📊 Test 5: Checking recent trade.intent events..."
echo "--------------------------------------------------"
docker exec quantum_redis redis-cli XREVRANGE quantum:stream:trade.intent + - COUNT 3 2>&1 | head -30

# Test 6: Extract leverage values from recent events
echo ""
echo "📊 Test 6: Extracting leverage values from events..."
echo "----------------------------------------------------"
docker exec quantum_redis redis-cli XREVRANGE quantum:stream:trade.intent + - COUNT 5 2>&1 | grep -oE '"leverage"[[:space:]]*:[[:space:]]*[0-9.]+' | head -10

# Test 7: Check subscriber startup
echo ""
echo "📊 Test 7: Checking TradeIntentSubscriber startup..."
echo "----------------------------------------------------"
docker logs --tail 500 quantum_backend 2>&1 | grep -iE 'TradeIntentSubscriber|trade_intent_subscriber|Phase 3.5' | head -5

if [ $? -eq 0 ]; then
    echo "✅ Subscriber startup logs found"
else
    echo "❌ NO subscriber startup logs"
fi

# Test 8: Count total log occurrences
echo ""
echo "📊 Test 8: Log occurrence counts..."
echo "-----------------------------------"
echo "  'leverage' mentions: $(docker logs --tail 500 quantum_backend 2>&1 | grep -ic 'leverage')"
echo "  'v3.5' mentions: $(docker logs --tail 500 quantum_backend 2>&1 | grep -ic 'v3.5')"
echo "  'adaptive' mentions: $(docker logs --tail 500 quantum_backend 2>&1 | grep -ic 'adaptive')"
echo "  'ILF' mentions: $(docker logs --tail 500 quantum_backend 2>&1 | grep -ic 'ilf')"
echo "  'compute_adaptive_levels' mentions: $(docker logs --tail 500 quantum_backend 2>&1 | grep -ic 'compute_adaptive_levels')"

echo ""
echo "=================================================================="
echo "🏁 VERIFICATION COMPLETE"
echo ""
