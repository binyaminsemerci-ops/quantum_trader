#!/bin/bash
# Phase 4P Validation Script - Linux/WSL Version
# Tests Adaptive Exposure Balancer implementation

set -e

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
CYAN='\033[0;36m'
NC='\033[0m' # No Color

# Test counters
TOTAL_TESTS=0
PASSED_TESTS=0

echo -e "${CYAN}================================================================${NC}"
echo -e "${CYAN}PHASE 4P VALIDATION - ADAPTIVE EXPOSURE BALANCER${NC}"
echo -e "${CYAN}================================================================${NC}"
echo ""

# Test function
test_component() {
    local name="$1"
    local test_cmd="$2"
    local category="${3:-General}"
    
    TOTAL_TESTS=$((TOTAL_TESTS + 1))
    echo -n "[$TOTAL_TESTS] Testing: $name... "
    
    if eval "$test_cmd" &>/dev/null; then
        echo -e "${GREEN}✓ PASS${NC}"
        PASSED_TESTS=$((PASSED_TESTS + 1))
        return 0
    else
        echo -e "${RED}✗ FAIL${NC}"
        return 1
    fi
}

echo -e "${YELLOW}Category: Core Module${NC}"
echo "---------------------------------------------"

test_component "exposure_balancer.py exists" \
    "[ -f microservices/exposure_balancer/exposure_balancer.py ]" \
    "Core Module"

test_component "ExposureBalancer class defined" \
    "grep -q 'class ExposureBalancer' microservices/exposure_balancer/exposure_balancer.py" \
    "Core Module"

test_component "Risk assessment methods present" \
    "grep -q 'def assess_risk' microservices/exposure_balancer/exposure_balancer.py && grep -q 'def execute_action' microservices/exposure_balancer/exposure_balancer.py" \
    "Core Module"

test_component "Priority-based action system" \
    "grep -q 'priority: int' microservices/exposure_balancer/exposure_balancer.py && grep -q 'priority == 1' microservices/exposure_balancer/exposure_balancer.py" \
    "Core Module"

test_component "Redis integration configured" \
    "grep -q 'quantum:stream:exposure.alerts' microservices/exposure_balancer/exposure_balancer.py && grep -q 'quantum:stream:executor.commands' microservices/exposure_balancer/exposure_balancer.py" \
    "Core Module"

echo ""
echo -e "${YELLOW}Category: Docker Setup${NC}"
echo "---------------------------------------------"

test_component "Dockerfile exists" \
    "[ -f microservices/exposure_balancer/Dockerfile ]" \
    "Docker"

test_component "service.py exists" \
    "[ -f microservices/exposure_balancer/service.py ]" \
    "Docker"

test_component "Background service loop implemented" \
    "grep -q 'def run_loop' microservices/exposure_balancer/service.py && grep -q 'rebalance()' microservices/exposure_balancer/service.py" \
    "Docker"

test_component "__init__.py present" \
    "[ -f microservices/exposure_balancer/__init__.py ]" \
    "Docker"

echo ""
echo -e "${YELLOW}Category: Integration${NC}"
echo "---------------------------------------------"

test_component "AI Engine health endpoint updated" \
    "grep -q 'exposure_balancer' microservices/ai_engine/service.py" \
    "Integration"

test_component "docker-compose.vps.yml updated" \
    "grep -q 'exposure-balancer:' docker-compose.vps.yml && grep -q 'EXPOSURE_BALANCER_ENABLED' docker-compose.vps.yml" \
    "Integration"

test_component "Environment variables configured" \
    "grep -q 'MAX_MARGIN_UTIL' docker-compose.vps.yml && grep -q 'REBALANCE_INTERVAL' docker-compose.vps.yml" \
    "Integration"

echo ""
echo -e "${YELLOW}Category: Phase Integration${NC}"
echo "---------------------------------------------"

test_component "Phase 4M+ integration (divergence)" \
    "grep -q 'quantum:cross:divergence' microservices/exposure_balancer/exposure_balancer.py" \
    "Integration"

test_component "Phase 4O+ integration (confidence)" \
    "grep -q 'quantum:meta:confidence' microservices/exposure_balancer/exposure_balancer.py" \
    "Integration"

test_component "Auto executor command interface" \
    "grep -q 'quantum:stream:executor.commands' microservices/exposure_balancer/exposure_balancer.py" \
    "Integration"

echo ""
echo -e "${YELLOW}Category: Risk Assessment Logic${NC}"
echo "---------------------------------------------"

test_component "Margin overload check (priority 1)" \
    "grep -q 'margin_utilization > self.max_margin_util' microservices/exposure_balancer/exposure_balancer.py && grep -q 'priority=1' microservices/exposure_balancer/exposure_balancer.py" \
    "Logic"

test_component "Symbol overexposure check (priority 2)" \
    "grep -q 'exposure > self.max_symbol_exposure' microservices/exposure_balancer/exposure_balancer.py && grep -q 'priority=2' microservices/exposure_balancer/exposure_balancer.py" \
    "Logic"

test_component "Diversification check" \
    "grep -q 'symbol_count < self.min_diversification' microservices/exposure_balancer/exposure_balancer.py" \
    "Logic"

test_component "Divergence check" \
    "grep -q 'cross_divergence > self.divergence_threshold' microservices/exposure_balancer/exposure_balancer.py" \
    "Logic"

test_component "Alert system implemented" \
    "grep -q 'def _send_alert' microservices/exposure_balancer/exposure_balancer.py && grep -q 'quantum:stream:exposure.alerts' microservices/exposure_balancer/exposure_balancer.py" \
    "Logic"

# Summary
echo ""
echo -e "${CYAN}================================================================${NC}"
echo -e "${CYAN}VALIDATION SUMMARY${NC}"
echo -e "${CYAN}================================================================${NC}"
echo ""
echo "Total Tests: $TOTAL_TESTS"
echo -e "${GREEN}Passed:      $PASSED_TESTS${NC}"
FAILED_TESTS=$((TOTAL_TESTS - PASSED_TESTS))
if [ $FAILED_TESTS -eq 0 ]; then
    echo -e "${GREEN}Failed:      $FAILED_TESTS${NC}"
else
    echo -e "${RED}Failed:      $FAILED_TESTS${NC}"
fi
echo ""

SUCCESS_RATE=$(echo "scale=1; ($PASSED_TESTS / $TOTAL_TESTS) * 100" | bc)
if [ "$SUCCESS_RATE" == "100.0" ]; then
    echo -e "${GREEN}Success Rate: $SUCCESS_RATE%${NC}"
elif (( $(echo "$SUCCESS_RATE >= 80" | bc -l) )); then
    echo -e "${YELLOW}Success Rate: $SUCCESS_RATE%${NC}"
else
    echo -e "${RED}Success Rate: $SUCCESS_RATE%${NC}"
fi
echo ""

# Final verdict
if [ $PASSED_TESTS -eq $TOTAL_TESTS ]; then
    echo -e "${GREEN}✓ ALL TESTS PASSED - Phase 4P Ready for Deployment!${NC}"
    exit 0
else
    echo -e "${RED}✗ SOME TESTS FAILED - Review above results${NC}"
    exit 1
fi
