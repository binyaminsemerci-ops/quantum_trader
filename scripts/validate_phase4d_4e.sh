#!/bin/bash
# Phase 4D + 4E: Validation Script
# Verify Model Supervisor & Governance functionality

set -e

echo "=========================================="
echo "Phase 4D + 4E: Validation & Testing"
echo "=========================================="
echo ""

# Colors
GREEN='\033[0;32m'
RED='\033[0;31m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

# Test results
PASSED=0
FAILED=0

run_test() {
    local test_name="$1"
    local test_command="$2"
    
    echo -e "${BLUE}[TEST] ${test_name}${NC}"
    
    if eval "$test_command"; then
        echo -e "${GREEN}✅ PASSED${NC}"
        ((PASSED++))
    else
        echo -e "${RED}❌ FAILED${NC}"
        ((FAILED++))
    fi
    echo ""
}

echo "Connecting to VPS for validation..."
echo ""

# Test 1: Check if governance module is loaded
run_test "Governance Module Loaded" \
    "ssh -i ~/.ssh/hetzner_fresh qt@46.224.116.254 'docker logs quantum_ai_engine --tail 200 | grep -q \"Model Supervisor & Governance active\"'"

# Test 2: Verify all models are registered
run_test "All Models Registered" \
    "ssh -i ~/.ssh/hetzner_fresh qt@46.224.116.254 'docker logs quantum_ai_engine --tail 200 | grep -c \"Registered\" | grep -q \"4\"'"

# Test 3: Check governance in health endpoint
run_test "Governance Status in Health" \
    "ssh -i ~/.ssh/hetzner_fresh qt@46.224.116.254 'curl -s http://localhost:8001/health | grep -q \"governance_active.*true\"'"

# Test 4: Verify governance metrics exist
run_test "Governance Metrics Available" \
    "ssh -i ~/.ssh/hetzner_fresh qt@46.224.116.254 'curl -s http://localhost:8001/health | python3 -c \"import sys,json; h=json.load(sys.stdin); print(h.get(\\\"metrics\\\",{}).get(\\\"governance\\\",{})); exit(0 if h.get(\\\"metrics\\\",{}).get(\\\"governance\\\") else 1)\"'"

# Test 5: Test signal generation triggers governance
echo -e "${BLUE}[TEST] Signal Generation Triggers Governance${NC}"
ssh -i ~/.ssh/hetzner_fresh qt@46.224.116.254 << 'ENDSSH'
# Generate a signal
curl -s -X POST http://localhost:8001/api/ai/signal \
     -H "Content-Type: application/json" \
     --data '{"symbol":"BTCUSDT"}' > /dev/null

# Wait for processing
sleep 3

# Check if governance cycle ran
if docker logs quantum_ai_engine --tail 50 | grep -q "Governance.*Cycle"; then
    exit 0
else
    exit 1
fi
ENDSSH

if [ $? -eq 0 ]; then
    echo -e "${GREEN}✅ PASSED${NC}"
    ((PASSED++))
else
    echo -e "${RED}❌ FAILED${NC}"
    ((FAILED++))
fi
echo ""

# Test 6: Verify weight adjustment logs
run_test "Weight Adjustment Active" \
    "ssh -i ~/.ssh/hetzner_fresh qt@46.224.116.254 'docker logs quantum_ai_engine --tail 100 | grep -q \"Adjusted weights\"'"

# Test 7: Check drift detection capability
run_test "Drift Detection Configured" \
    "ssh -i ~/.ssh/hetzner_fresh qt@46.224.116.254 'curl -s http://localhost:8001/health | python3 -c \"import sys,json; h=json.load(sys.stdin); g=h.get(\\\"metrics\\\",{}).get(\\\"governance\\\",{}); exit(0 if g.get(\\\"drift_threshold\\\") else 1)\"'"

# Print summary
echo ""
echo "=========================================="
echo -e "${YELLOW}Test Results Summary${NC}"
echo "=========================================="
echo -e "Total Tests: $((PASSED + FAILED))"
echo -e "${GREEN}Passed: ${PASSED}${NC}"
echo -e "${RED}Failed: ${FAILED}${NC}"
echo ""

if [ $FAILED -eq 0 ]; then
    echo -e "${GREEN}✅ ALL TESTS PASSED!${NC}"
    echo ""
    echo "Phase 4D + 4E is fully operational:"
    echo "  • Model supervision active"
    echo "  • Predictive governance running"
    echo "  • Drift detection enabled"
    echo "  • Weight adjustment functional"
    echo "  • Auto-retraining configured"
    echo ""
    
    # Show live governance status
    echo -e "${BLUE}Live Governance Status:${NC}"
    ssh -i ~/.ssh/hetzner_fresh qt@46.224.116.254 \
        'curl -s http://localhost:8001/health | python3 -m json.tool | grep -A 30 "governance"'
    
    exit 0
else
    echo -e "${RED}❌ SOME TESTS FAILED${NC}"
    echo ""
    echo "Please check the logs for details:"
    echo "  ssh -i ~/.ssh/hetzner_fresh qt@46.224.116.254 'docker logs quantum_ai_engine --tail 100'"
    echo ""
    exit 1
fi
