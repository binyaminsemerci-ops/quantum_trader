#!/bin/bash
# Verify Trade.Intent Consumer Auto-Start
# Tests that consumer will restart automatically

echo "🔍 VERIFYING TRADE.INTENT CONSUMER AUTO-START"
echo "=============================================="

# Test 1: Check container restart policy
echo ""
echo "📋 Test 1: Container Restart Policy"
echo "------------------------------------"
RESTART_POLICY=$(docker inspect quantum_backend --format='{{.HostConfig.RestartPolicy.Name}}' 2>/dev/null)

if [ "$RESTART_POLICY" = "always" ]; then
    echo "✅ PASS: restart policy is 'always'"
else
    echo "❌ FAIL: restart policy is '$RESTART_POLICY' (expected 'always')"
fi

# Test 2: Check container is running
echo ""
echo "📋 Test 2: Container Running Status"
echo "------------------------------------"
if docker ps | grep -q quantum_backend; then
    echo "✅ PASS: quantum_backend is running"
    UPTIME=$(docker inspect quantum_backend --format='{{.State.StartedAt}}')
    echo "   Started: $UPTIME"
else
    echo "❌ FAIL: quantum_backend is not running"
fi

# Test 3: Check backend health
echo ""
echo "📋 Test 3: Backend Health Endpoint"
echo "-----------------------------------"
HEALTH=$(curl -s http://localhost:8000/health 2>&1)
if echo "$HEALTH" | grep -q "healthy\|OK"; then
    echo "✅ PASS: Backend health endpoint responding"
else
    echo "⚠️  WARN: Backend may still be starting"
fi

# Test 4: Check subscriber initialization
echo ""
echo "📋 Test 4: Subscriber Initialization"
echo "-------------------------------------"
if docker logs quantum_backend 2>&1 | grep -q "TradeIntentSubscriber\|Phase 3.5"; then
    echo "✅ PASS: TradeIntentSubscriber found in logs"
    LAST_LOG=$(docker logs quantum_backend 2>&1 | grep "TradeIntentSubscriber\|trade_intent" | tail -1)
    echo "   Last log: ${LAST_LOG:0:120}..."
else
    echo "❌ FAIL: TradeIntentSubscriber not found in logs"
fi

# Test 5: Check consumer group exists
echo ""
echo "📋 Test 5: Redis Consumer Group"
echo "--------------------------------"
if docker exec quantum_redis redis-cli XINFO GROUPS quantum:stream:trade.intent 2>/dev/null | grep -q "quantum:group:execution:trade.intent"; then
    echo "✅ PASS: Consumer group exists"
    
    # Check lag
    LAG=$(docker exec quantum_redis redis-cli XINFO GROUPS quantum:stream:trade.intent 2>/dev/null | grep "lag" | head -1 | awk '{print $2}')
    echo "   Current lag: $LAG events"
    
    # Check consumers
    CONSUMERS=$(docker exec quantum_redis redis-cli XINFO CONSUMERS quantum:stream:trade.intent quantum:group:execution:trade.intent 2>/dev/null | grep "name" | wc -l)
    echo "   Active consumers: $CONSUMERS"
else
    echo "⚠️  WARN: Consumer group not yet created (normal on first start)"
fi

# Test 6: Simulate restart (optional, destructive)
echo ""
echo "📋 Test 6: Restart Simulation (Optional)"
echo "----------------------------------------"
read -p "Do you want to test auto-restart by stopping the container? (y/N): " -n 1 -r
echo
if [[ $REPLY =~ ^[Yy]$ ]]; then
    echo "🔄 Stopping quantum_backend..."
    docker stop quantum_backend
    
    echo "⏳ Waiting 5 seconds for Docker to auto-restart..."
    sleep 5
    
    if docker ps | grep -q quantum_backend; then
        echo "✅ PASS: Container auto-restarted successfully!"
        
        # Wait for it to fully start
        echo "⏳ Waiting 10 seconds for subscriber to initialize..."
        sleep 10
        
        if docker logs quantum_backend 2>&1 | tail -50 | grep -q "TradeIntentSubscriber"; then
            echo "✅ PASS: Subscriber re-initialized after restart"
        else
            echo "⚠️  WARN: Subscriber may still be initializing"
        fi
    else
        echo "❌ FAIL: Container did not auto-restart"
    fi
else
    echo "⏭️  Skipped restart test"
fi

# Test 7: Check system service (if using systemd)
echo ""
echo "📋 Test 7: System Service Integration"
echo "--------------------------------------"
if systemctl is-enabled docker 2>/dev/null | grep -q "enabled"; then
    echo "✅ PASS: Docker service is enabled (will start on boot)"
else
    echo "⚠️  WARN: Docker service status unknown"
fi

# Summary
echo ""
echo "=============================================="
echo "📊 SUMMARY"
echo "=============================================="
echo ""

PASS_COUNT=0
FAIL_COUNT=0

[ "$RESTART_POLICY" = "always" ] && ((PASS_COUNT++)) || ((FAIL_COUNT++))
docker ps | grep -q quantum_backend && ((PASS_COUNT++)) || ((FAIL_COUNT++))
docker logs quantum_backend 2>&1 | grep -q "TradeIntentSubscriber" && ((PASS_COUNT++)) || ((FAIL_COUNT++))

echo "Tests passed: $PASS_COUNT"
echo "Tests failed: $FAIL_COUNT"
echo ""

if [ $FAIL_COUNT -eq 0 ]; then
    echo "🎉 ALL TESTS PASSED!"
    echo ""
    echo "✅ Trade.Intent Consumer will auto-start on reboot"
    echo "✅ Container has restart: always policy"
    echo "✅ Subscriber is initialized and ready"
    echo ""
    echo "🔄 To test reboot behavior:"
    echo "   sudo reboot"
    echo "   # After reboot, run:"
    echo "   docker ps | grep quantum_backend"
    echo "   docker logs quantum_backend | grep TradeIntentSubscriber"
else
    echo "⚠️  SOME TESTS FAILED"
    echo ""
    echo "🔧 Troubleshooting:"
    echo "   1. Ensure services are started:"
    echo "      ./scripts/start-trade-intent-consumer.sh"
    echo ""
    echo "   2. Check logs:"
    echo "      docker logs quantum_backend"
    echo ""
    echo "   3. Verify compose file:"
    echo "      docker compose -f docker-compose.trade-intent-consumer.yml config"
fi

echo ""
echo "📝 Quick Commands:"
echo "  Check status:  docker ps | grep quantum_backend"
echo "  View logs:     docker logs -f quantum_backend"
echo "  Check lag:     docker exec quantum_redis redis-cli XINFO GROUPS quantum:stream:trade.intent"
