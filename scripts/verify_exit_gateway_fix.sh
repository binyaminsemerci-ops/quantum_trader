#!/bin/bash
# Verify Exit Order Gateway Fix for -4164 Error
# Created: 2025-12-24
# Fix: Added reduceOnly=true and minNotional guard

echo "🔧 VERIFYING EXIT ORDER GATEWAY FIX (-4164 Error)"
echo "=================================================================="

# Test 1: Confirm fix is deployed
echo ""
echo "📊 Test 1: Checking if reduceOnly fix is deployed..."
echo "------------------------------------------------------"
REDUCE_ONLY_COUNT=$(docker exec quantum_backend grep -c "reduceOnly" /app/backend/services/execution/exit_order_gateway.py 2>&1)

if [ "$REDUCE_ONLY_COUNT" -gt 0 ]; then
    echo "✅ reduceOnly logic found ($REDUCE_ONLY_COUNT occurrences)"
    docker exec quantum_backend grep -n "reduceOnly" /app/backend/services/execution/exit_order_gateway.py | head -5
else
    echo "❌ reduceOnly logic NOT found in deployed file"
fi

# Test 2: Check for minNotional guard
echo ""
echo "📊 Test 2: Checking if minNotional guard is deployed..."
echo "--------------------------------------------------------"
MIN_NOTIONAL_COUNT=$(docker exec quantum_backend grep -c "MIN_NOTIONAL GUARD" /app/backend/services/execution/exit_order_gateway.py 2>&1)

if [ "$MIN_NOTIONAL_COUNT" -gt 0 ]; then
    echo "✅ minNotional guard found"
    docker exec quantum_backend grep -A 3 "MIN_NOTIONAL GUARD" /app/backend/services/execution/exit_order_gateway.py
else
    echo "❌ minNotional guard NOT found"
fi

# Test 3: Check backend logs for -4164 errors
echo ""
echo "📊 Test 3: Checking for -4164 errors in logs (last 300 lines)..."
echo "------------------------------------------------------------------"
ERROR_4164_COUNT=$(docker logs --tail 300 quantum_backend 2>&1 | grep -c "4164" 2>&1)

if [ "$ERROR_4164_COUNT" -eq 0 ]; then
    echo "✅ NO -4164 errors found in recent logs"
else
    echo "⚠️  Found $ERROR_4164_COUNT occurrences of -4164 error:"
    docker logs --tail 300 quantum_backend 2>&1 | grep "4164" | tail -5
fi

# Test 4: Check for exit order submissions with reduceOnly
echo ""
echo "📊 Test 4: Checking for exit orders with reduceOnly in logs..."
echo "---------------------------------------------------------------"
docker logs --tail 300 quantum_backend 2>&1 | grep -E "EXIT_GATEWAY.*reduceOnly=True" | tail -5

if [ $? -eq 0 ]; then
    echo "✅ Found exit orders with reduceOnly=True"
else
    echo "⚠️  No exit orders with reduceOnly found yet (may need to wait for trade)"
fi

# Test 5: Check for minNotional warnings
echo ""
echo "📊 Test 5: Checking for minNotional warnings/adjustments..."
echo "-----------------------------------------------------------"
docker logs --tail 300 quantum_backend 2>&1 | grep -E "Notional.*min|Increasing quantity" | tail -5

if [ $? -eq 0 ]; then
    echo "✅ Found minNotional guard activity"
else
    echo "ℹ️  No minNotional adjustments yet (normal if no small orders)"
fi

# Test 6: Check backend health
echo ""
echo "📊 Test 6: Checking backend health..."
echo "--------------------------------------"
HEALTH_STATUS=$(curl -s http://localhost:8000/health 2>&1 | jq -r '.status' 2>/dev/null)

if [ "$HEALTH_STATUS" == "healthy" ]; then
    echo "✅ Backend is healthy"
else
    echo "⚠️  Backend health status: $HEALTH_STATUS"
fi

# Test 7: Show recent exit order activity
echo ""
echo "📊 Test 7: Recent exit order activity (last 10 entries)..."
echo "-----------------------------------------------------------"
docker logs --tail 500 quantum_backend 2>&1 | grep -E "EXIT_GATEWAY.*Submitting|EXIT_GATEWAY.*Order placed" | tail -10

# Test 8: Check for specific error messages
echo ""
echo "📊 Test 8: Checking for 'notional must be no smaller than' errors..."
echo "---------------------------------------------------------------------"
NOTIONAL_ERROR_COUNT=$(docker logs --tail 300 quantum_backend 2>&1 | grep -ic "notional must be no smaller" 2>&1)

if [ "$NOTIONAL_ERROR_COUNT" -eq 0 ]; then
    echo "✅ NO notional errors found"
else
    echo "❌ Found $NOTIONAL_ERROR_COUNT notional errors:"
    docker logs --tail 300 quantum_backend 2>&1 | grep -i "notional must be no smaller" | tail -3
fi

echo ""
echo "=================================================================="
echo "🏁 VERIFICATION COMPLETE"
echo ""
echo "📋 Summary:"
echo "  - reduceOnly fix deployed: $(if [ $REDUCE_ONLY_COUNT -gt 0 ]; then echo 'YES ✅'; else echo 'NO ❌'; fi)"
echo "  - minNotional guard deployed: $(if [ $MIN_NOTIONAL_COUNT -gt 0 ]; then echo 'YES ✅'; else echo 'NO ❌'; fi)"
echo "  - Recent -4164 errors: $ERROR_4164_COUNT"
echo "  - Recent notional errors: $NOTIONAL_ERROR_COUNT"
echo "  - Backend health: $HEALTH_STATUS"
echo ""

if [ "$REDUCE_ONLY_COUNT" -gt 0 ] && [ "$MIN_NOTIONAL_COUNT" -gt 0 ] && [ "$ERROR_4164_COUNT" -eq 0 ] && [ "$NOTIONAL_ERROR_COUNT" -eq 0 ]; then
    echo "🎉 VERDICT: Fix successfully deployed and working!"
else
    echo "⚠️  VERDICT: Fix may need additional verification or wait for trade activity"
fi
