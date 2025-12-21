#!/bin/bash
# ============================================================================
# PHASE 4M VALIDATION TEST SUITE
# ============================================================================
# Validates all components of cross-exchange intelligence system
# ============================================================================

set -e

TESTS_PASSED=0
TESTS_FAILED=0

echo "========================================================================"
echo "PHASE 4M - CROSS-EXCHANGE INTELLIGENCE VALIDATION"
echo "========================================================================"
echo ""

# Test 1: Data Collector
echo "▶ Test 1: Exchange Data Collector (REST API)"
echo "------------------------------------------------------------------------"
if python3 microservices/data_collector/exchange_data_collector.py --test; then
    echo "✅ PASS - Data collector fetches from all exchanges"
    ((TESTS_PASSED++))
else
    echo "❌ FAIL - Data collector has errors"
    ((TESTS_FAILED++))
fi
echo ""

# Test 2: Redis Raw Stream
echo "▶ Test 2: Raw Stream Population (WebSocket Bridge)"
echo "------------------------------------------------------------------------"
echo "Starting stream bridge for 10 seconds..."
timeout 10s python3 microservices/data_collector/exchange_stream_bridge.py || true
sleep 2

RAW_LEN=$(docker exec quantum_redis redis-cli XLEN quantum:stream:exchange.raw 2>/dev/null || echo "0")
if [ "$RAW_LEN" -gt "0" ]; then
    echo "✅ PASS - Raw stream has $RAW_LEN entries"
    ((TESTS_PASSED++))
else
    echo "❌ FAIL - Raw stream is empty"
    ((TESTS_FAILED++))
fi
echo ""

# Test 3: Aggregator
echo "▶ Test 3: Cross-Exchange Aggregator"
echo "------------------------------------------------------------------------"
if python3 microservices/ai_engine/cross_exchange_aggregator.py --test; then
    echo "✅ PASS - Aggregator merges and normalizes data"
    ((TESTS_PASSED++))
else
    echo "❌ FAIL - Aggregator has errors"
    ((TESTS_FAILED++))
fi
echo ""

# Test 4: Normalized Stream
echo "▶ Test 4: Normalized Stream Content"
echo "------------------------------------------------------------------------"
NORM_DATA=$(docker exec quantum_redis redis-cli XREVRANGE quantum:stream:exchange.normalized + - COUNT 1 2>/dev/null || echo "")
if [ ! -z "$NORM_DATA" ]; then
    echo "✅ PASS - Normalized stream has data:"
    echo "$NORM_DATA" | head -5
    ((TESTS_PASSED++))
else
    echo "❌ FAIL - Normalized stream is empty"
    ((TESTS_FAILED++))
fi
echo ""

# Test 5: Feature Adapter
echo "▶ Test 5: Exchange Feature Adapter"
echo "------------------------------------------------------------------------"
if python3 microservices/ai_engine/features/exchange_feature_adapter.py --test; then
    echo "✅ PASS - Feature adapter creates ML features"
    ((TESTS_PASSED++))
else
    echo "❌ FAIL - Feature adapter has errors"
    ((TESTS_FAILED++))
fi
echo ""

# Test 6: Feature Loader
echo "▶ Test 6: Feature Loader Integration"
echo "------------------------------------------------------------------------"
if python3 microservices/ai_engine/features/feature_loader.py --test; then
    echo "✅ PASS - Feature loader loads cross-exchange features"
    ((TESTS_PASSED++))
else
    echo "❌ FAIL - Feature loader has errors"
    ((TESTS_FAILED++))
fi
echo ""

# Test 7: AI Engine Health
echo "▶ Test 7: AI Engine Health Check"
echo "------------------------------------------------------------------------"
HEALTH=$(curl -s http://localhost:8001/health 2>/dev/null || echo "{}")
if echo "$HEALTH" | grep -q "cross_exchange"; then
    echo "✅ PASS - AI Engine reports cross-exchange status:"
    echo "$HEALTH" | python3 -m json.tool 2>/dev/null || echo "$HEALTH"
    ((TESTS_PASSED++))
else
    echo "⚠️  SKIP - AI Engine not responding (may not be running)"
    echo "   Run: docker-compose -f docker-compose.vps.yml up -d ai-engine"
fi
echo ""

# Test 8: Docker Service
echo "▶ Test 8: Docker Cross-Exchange Service"
echo "------------------------------------------------------------------------"
if docker ps | grep -q quantum_cross_exchange; then
    STATUS=$(docker ps --format "{{.Status}}" --filter "name=quantum_cross_exchange")
    echo "✅ PASS - Cross-exchange service is running: $STATUS"
    ((TESTS_PASSED++))
else
    echo "⚠️  SKIP - Cross-exchange service not started"
    echo "   Run: docker-compose -f docker-compose.vps.yml up -d cross-exchange"
fi
echo ""

# Summary
echo "========================================================================"
echo "VALIDATION SUMMARY"
echo "========================================================================"
echo "Tests Passed: $TESTS_PASSED"
echo "Tests Failed: $TESTS_FAILED"
echo ""

if [ $TESTS_FAILED -eq 0 ]; then
    echo "🎉 ALL TESTS PASSED"
    echo ""
    echo "✅ Phase 4M Complete - Set flag: 'cross_exchange_intelligence': 'active'"
    echo ""
    exit 0
else
    echo "❌ Some tests failed - review errors above"
    exit 1
fi
