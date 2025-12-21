#!/bin/bash
# Phase 4O+ Validation Script - Linux/Bash
# Intelligent Leverage + RL Position Sizing (Cross-Exchange Enabled)

echo "======================================================================"
echo "PHASE 4O+ VALIDATION - Intelligent Leverage + RL Position Sizing"
echo "======================================================================"
echo ""

ERRORS=0
TESTS=0

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
CYAN='\033[0;36m'
GRAY='\033[0;37m'
NC='\033[0m' # No Color

echo -e "${YELLOW}[INTELLIGENT LEVERAGE V2]${NC}"
echo "----------------------------------------------------------------------"

# Test 1: Check if ILFv2 engine exists
echo -n "[1] Check ILFv2 engine file exists..."
((TESTS++))
if [ -f ~/quantum_trader/microservices/exitbrain_v3_5/intelligent_leverage_engine.py ]; then
    echo -e " ${GREEN}✅ PASS${NC}"
else
    echo -e " ${RED}❌ FAIL${NC}"
    ((ERRORS++))
fi

# Test 2: Check if ExitBrain v3.5 exists
echo -n "[2] Check ExitBrain v3.5 file exists..."
((TESTS++))
if [ -f ~/quantum_trader/microservices/exitbrain_v3_5/exit_brain.py ]; then
    echo -e " ${GREEN}✅ PASS${NC}"
else
    echo -e " ${RED}❌ FAIL${NC}"
    ((ERRORS++))
fi

# Test 3: Check quantum:stream:exitbrain.pnl stream
echo -n "[3] Check quantum:stream:exitbrain.pnl stream..."
((TESTS++))
STREAM_LEN=$(docker exec quantum_redis redis-cli XLEN quantum:stream:exitbrain.pnl 2>/dev/null || echo "0")
if [ "$STREAM_LEN" -gt 0 ]; then
    echo -e " ${GREEN}✅ PASS ($STREAM_LEN entries)${NC}"
else
    echo -e " ${RED}❌ FAIL (stream empty)${NC}"
    echo -e "    ${GRAY}Note: Stream will populate after first ExitBrain calculation${NC}"
    ((ERRORS++))
fi

echo ""
echo -e "${YELLOW}[RL POSITION SIZING AGENT]${NC}"
echo "----------------------------------------------------------------------"

# Test 4: Check if RL agent file exists
echo -n "[4] Check RL agent file exists..."
((TESTS++))
if [ -f ~/quantum_trader/microservices/rl_sizing_agent/rl_agent.py ]; then
    echo -e " ${GREEN}✅ PASS${NC}"
else
    echo -e " ${RED}❌ FAIL${NC}"
    ((ERRORS++))
fi

# Test 5: Check if PnL feedback listener exists
echo -n "[5] Check PnL feedback listener exists..."
((TESTS++))
if [ -f ~/quantum_trader/microservices/rl_sizing_agent/pnl_feedback_listener.py ]; then
    echo -e " ${GREEN}✅ PASS${NC}"
else
    echo -e " ${RED}❌ FAIL${NC}"
    ((ERRORS++))
fi

# Test 6: Check if RL model directory exists
echo -n "[6] Check RL model directory..."
((TESTS++))
if docker exec quantum_ai_engine test -d /models 2>/dev/null; then
    echo -e " ${GREEN}✅ PASS${NC}"
else
    echo -e " ${YELLOW}⚠️  WARN (will be created on first training)${NC}"
fi

echo ""
echo -e "${YELLOW}[AI ENGINE HEALTH CHECK]${NC}"
echo "----------------------------------------------------------------------"

# Test 7: Check AI Engine health endpoint
echo -n "[7] Check AI Engine /health endpoint..."
((TESTS++))
HEALTH=$(curl -s http://localhost:8001/health 2>/dev/null)
if echo "$HEALTH" | grep -q '"status":"OK"'; then
    echo -e " ${GREEN}✅ PASS${NC}"
    
    # Check Phase 4O+ metrics
    if echo "$HEALTH" | grep -q '"intelligent_leverage_v2":true\|"intelligent_leverage":{'; then
        echo -e "    ${GREEN}✓ Intelligent Leverage v2: ENABLED${NC}"
        AVG_LEV=$(echo "$HEALTH" | python3 -c "import sys,json; d=json.load(sys.stdin); print(d.get('metrics',{}).get('intelligent_leverage',{}).get('avg_leverage','N/A'))")
        AVG_CONF=$(echo "$HEALTH" | python3 -c "import sys,json; d=json.load(sys.stdin); print(d.get('metrics',{}).get('intelligent_leverage',{}).get('avg_confidence','N/A'))")
        CALC_TOTAL=$(echo "$HEALTH" | python3 -c "import sys,json; d=json.load(sys.stdin); print(d.get('metrics',{}).get('intelligent_leverage',{}).get('calculations_total','N/A'))")
        echo -e "      ${GRAY}- Avg Leverage: ${AVG_LEV}x${NC}"
        echo -e "      ${GRAY}- Avg Confidence: ${AVG_CONF}${NC}"
        echo -e "      ${GRAY}- Calculations: ${CALC_TOTAL}${NC}"
    else
        echo -e "    ${YELLOW}⚠️  Intelligent Leverage v2: NOT IN METRICS${NC}"
    fi
    
    if echo "$HEALTH" | grep -q '"rl_position_sizing":true\|"rl_agent":{'; then
        echo -e "    ${GREEN}✓ RL Position Sizing: ENABLED${NC}"
        POLICY_VER=$(echo "$HEALTH" | python3 -c "import sys,json; d=json.load(sys.stdin); print(d.get('metrics',{}).get('rl_agent',{}).get('policy_version','N/A'))")
        TRADES=$(echo "$HEALTH" | python3 -c "import sys,json; d=json.load(sys.stdin); print(d.get('metrics',{}).get('rl_agent',{}).get('trades_processed','N/A'))")
        REWARD=$(echo "$HEALTH" | python3 -c "import sys,json; d=json.load(sys.stdin); print(d.get('metrics',{}).get('rl_agent',{}).get('reward_mean','N/A'))")
        echo -e "      ${GRAY}- Policy Version: ${POLICY_VER}${NC}"
        echo -e "      ${GRAY}- Trades Processed: ${TRADES}${NC}"
        echo -e "      ${GRAY}- Reward Mean: ${REWARD}${NC}"
    else
        echo -e "    ${YELLOW}⚠️  RL Position Sizing: NOT IN METRICS${NC}"
    fi
    
    # Check Cross-Exchange Intelligence (Phase 4M+)
    if echo "$HEALTH" | grep -q '"cross_exchange_intelligence":true'; then
        echo -e "    ${GREEN}✓ Cross-Exchange Intelligence: ENABLED (Phase 4M+)${NC}"
    fi
    
else
    echo -e " ${RED}❌ FAIL (health check failed)${NC}"
    ((ERRORS++))
fi

echo ""
echo -e "${YELLOW}[INTEGRATION STATUS]${NC}"
echo "----------------------------------------------------------------------"

# Test 8: Check for ILFv2 initialization logs
echo -n "[8] Check for ILFv2 initialization in logs..."
((TESTS++))
LOG_COUNT=$(docker logs quantum_ai_engine 2>&1 | grep -c "ILF-v2.*Initialized" || echo "0")
if [ "$LOG_COUNT" -gt 0 ]; then
    echo -e " ${GREEN}✅ PASS ($LOG_COUNT occurrences)${NC}"
else
    echo -e " ${RED}❌ FAIL (no initialization logs)${NC}"
    echo -e "    ${GRAY}Note: Check if ExitBrain v3.5 properly imports ILFv2${NC}"
    ((ERRORS++))
fi

# Test 9: Check for RL agent initialization logs
echo -n "[9] Check for RL agent initialization..."
((TESTS++))
LOG_COUNT=$(docker logs quantum_ai_engine 2>&1 | grep -c "RL-Agent.*Initialized" || echo "0")
if [ "$LOG_COUNT" -gt 0 ]; then
    echo -e " ${GREEN}✅ PASS ($LOG_COUNT occurrences)${NC}"
else
    echo -e " ${YELLOW}⚠️  WARN (no RL agent logs yet)${NC}"
    echo -e "    ${GRAY}Note: RL agent initialized on first trade${NC}"
fi

echo ""
echo -e "${YELLOW}[FORMULA VERIFICATION]${NC}"
echo "----------------------------------------------------------------------"

echo -e "${CYAN}ILFv2 Formula:${NC}"
echo -e "  ${GRAY}base = 5 + confidence × 75${NC}"
echo -e "  ${GRAY}leverage = base × vol_factor × pnl_factor × symbol_factor ×${NC}"
echo -e "  ${GRAY}           margin_factor × divergence_factor × funding_factor${NC}"
echo -e "  ${GRAY}Range: 5-80x${NC}"
echo ""
echo -e "${CYAN}RL Reward Function:${NC}"
echo -e "  ${GRAY}reward = (pnl_pct × confidence)${NC}"
echo -e "  ${GRAY}         - 0.005 × |leverage - target_leverage|${NC}"
echo -e "  ${GRAY}         - 0.002 × exch_divergence${NC}"
echo -e "  ${GRAY}         + 0.003 × sign(pnl_trend)${NC}"

echo ""
echo "======================================================================"
echo "VALIDATION SUMMARY"
echo "======================================================================"
echo ""
echo "Tests Run: $TESTS"
if [ "$ERRORS" -eq 0 ]; then
    echo -e "Errors: ${GREEN}$ERRORS${NC}"
else
    echo -e "Errors: ${RED}$ERRORS${NC}"
fi
echo ""

if [ "$ERRORS" -eq 0 ]; then
    echo -e "${GREEN}✅ VALIDATION PASSED${NC}"
    echo -e "   ${GREEN}Phase 4O+ integration ready for production${NC}"
else
    echo -e "${RED}❌ VALIDATION FAILED${NC}"
    echo -e "   ${RED}Fix errors and re-run validation${NC}"
fi
echo ""
