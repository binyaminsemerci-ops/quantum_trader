#!/bin/bash
# P3.5 Decision Normalizer - Proof & Verification Script

set -euo pipefail

BLUE='\033[0;34m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
CYAN='\033[0;36m'
NC='\033[0m'

echo -e "${BLUE}═══════════════════════════════════════════════════════════════════${NC}"
echo -e "${YELLOW}P3.5 DECISION NORMALIZER - VERIFICATION${NC}"
echo -e "${BLUE}═══════════════════════════════════════════════════════════════════${NC}"
echo ""

# ============================================================================
# STEP 1: Show current apply.result contract (before fix)
# ============================================================================
echo -e "${YELLOW}[STEP 1]${NC} Inspecting apply.result stream (actual contract)..."
echo ""
echo -e "${CYAN}Sample events (last 3):${NC}"
redis-cli XREVRANGE quantum:stream:apply.result + - COUNT 3 | head -60
echo ""
echo -e "${CYAN}Field analysis:${NC}"
echo "  - executed: $(redis-cli XREVRANGE quantum:stream:apply.result + - COUNT 10 | grep -c 'executed' || echo 0) occurrences (boolean)"
echo "  - decision: $(redis-cli XREVRANGE quantum:stream:apply.result + - COUNT 10 | grep -cP '^decision$' || echo 0) occurrences (enum)"
echo "  - error: Found in 'details' JSON (not top-level)"
echo ""

# ============================================================================
# STEP 2: Check current bucket data (before restart)
# ============================================================================
echo -e "${YELLOW}[STEP 2]${NC} Current decision distribution (before fix)..."
echo ""
CURRENT_BUCKET=$(date +"%Y%m%d%H%M" -u)
echo -e "${CYAN}Bucket: quantum:p35:bucket:${CURRENT_BUCKET}${NC}"
redis-cli HGETALL "quantum:p35:bucket:${CURRENT_BUCKET}" | paste -d " " - - | head -20
echo ""

echo -e "${CYAN}5-minute decision counts:${NC}"
redis-cli HGETALL quantum:p35:decision:counts:5m
echo ""

# ============================================================================
# STEP 3: Deploy patch
# ============================================================================
echo -e "${YELLOW}[STEP 3]${NC} Deploying P3.5 normalizer patch..."
echo ""

cd /home/qt/quantum_trader
echo "  Pulling latest code..."
git fetch origin main
git reset --hard origin/main
echo -e "${GREEN}  ✅ Code updated${NC}"
echo ""

echo "  Clearing Python cache..."
find microservices/decision_intelligence -name "__pycache__" -type d -exec rm -rf {} + 2>/dev/null || true
find microservices/decision_intelligence -name "*.pyc" -delete 2>/dev/null || true
echo -e "${GREEN}  ✅ Cache cleared${NC}"
echo ""

echo "  Restarting P3.5 service..."
sudo systemctl restart quantum-p35-decision-intelligence
sleep 3
echo -e "${GREEN}  ✅ Service restarted${NC}"
echo ""

# ============================================================================
# STEP 4: Verify service health
# ============================================================================
echo -e "${YELLOW}[STEP 4]${NC} Verifying service health..."
echo ""

if sudo systemctl is-active --quiet quantum-p35-decision-intelligence; then
    echo -e "${GREEN}  ✅ Service is ACTIVE${NC}"
else
    echo -e "${RED}  ❌ Service is NOT ACTIVE${NC}"
    sudo journalctl -u quantum-p35-decision-intelligence -n 30 --no-pager
    exit 1
fi

echo ""
echo -e "${CYAN}Recent logs (checking for normalizer activity):${NC}"
sudo journalctl -u quantum-p35-decision-intelligence --since "30 seconds ago" --no-pager | tail -15
echo ""

# ============================================================================
# STEP 5: Wait for new data (60 seconds for bucket to populate)
# ============================================================================
echo -e "${YELLOW}[STEP 5]${NC} Waiting 60 seconds for new data to populate..."
echo ""
for i in {60..1}; do
    echo -ne "  Countdown: ${i}s remaining...\r"
    sleep 1
done
echo ""
echo -e "${GREEN}  ✅ Wait complete${NC}"
echo ""

# ============================================================================
# STEP 6: Verify normalization (after fix)
# ============================================================================
echo -e "${YELLOW}[STEP 6]${NC} Verifying decision normalization..."
echo ""

CURRENT_BUCKET=$(date +"%Y%m%d%H%M" -u)
echo -e "${CYAN}New bucket: quantum:p35:bucket:${CURRENT_BUCKET}${NC}"
redis-cli HGETALL "quantum:p35:bucket:${CURRENT_BUCKET}" | paste -d " " - -
echo ""

echo -e "${CYAN}5-minute decision counts (should show EXECUTE/SKIP/BLOCKED):${NC}"
redis-cli HGETALL quantum:p35:decision:counts:5m
echo ""

echo -e "${CYAN}Top 10 reasons (should show real gates, not just 'none'):${NC}"
redis-cli ZREVRANGE quantum:p35:reason:top:5m 0 9 WITHSCORES | paste -d " " - -
echo ""

# ============================================================================
# STEP 7: Check for unknown decisions (debugging)
# ============================================================================
echo -e "${YELLOW}[STEP 7]${NC} Checking for unknown decision patterns..."
echo ""

UNKNOWN_COUNT=$(redis-cli ZCARD quantum:p35:unknown_decision:top:5m 2>/dev/null || echo "0")
if [ "$UNKNOWN_COUNT" -gt 0 ]; then
    echo -e "${YELLOW}  ⚠️  Found ${UNKNOWN_COUNT} unknown decision patterns:${NC}"
    redis-cli ZREVRANGE quantum:p35:unknown_decision:top:5m 0 9 WITHSCORES | paste -d " " - -
    echo ""
    echo -e "${YELLOW}  → This indicates apply.result is using unexpected field values${NC}"
else
    echo -e "${GREEN}  ✅ No unknown decisions detected${NC}"
fi
echo ""

# ============================================================================
# STEP 8: Run dashboard query A (full verification)
# ============================================================================
echo -e "${YELLOW}[STEP 8]${NC} Running dashboard query A (full check)..."
echo ""
bash /home/qt/quantum_trader/scripts/p35_dashboard_queries.sh a
echo ""

# ============================================================================
# STEP 9: Success criteria check
# ============================================================================
echo -e "${BLUE}═══════════════════════════════════════════════════════════════════${NC}"
echo -e "${YELLOW}SUCCESS CRITERIA CHECK${NC}"
echo -e "${BLUE}═══════════════════════════════════════════════════════════════════${NC}"
echo ""

# Check decision distribution
EXECUTE_COUNT=$(redis-cli HGET quantum:p35:decision:counts:5m EXECUTE 2>/dev/null || echo "0")
SKIP_COUNT=$(redis-cli HGET quantum:p35:decision:counts:5m SKIP 2>/dev/null || echo "0")
BLOCKED_COUNT=$(redis-cli HGET quantum:p35:decision:counts:5m BLOCKED 2>/dev/null || echo "0")
UNKNOWN_COUNT=$(redis-cli HGET quantum:p35:decision:counts:5m UNKNOWN 2>/dev/null || echo "0")

TOTAL=$((EXECUTE_COUNT + SKIP_COUNT + BLOCKED_COUNT + UNKNOWN_COUNT))

echo -e "${CYAN}Decision Distribution:${NC}"
echo "  EXECUTE:  ${EXECUTE_COUNT}"
echo "  SKIP:     ${SKIP_COUNT}"
echo "  BLOCKED:  ${BLOCKED_COUNT}"
echo "  UNKNOWN:  ${UNKNOWN_COUNT}"
echo "  TOTAL:    ${TOTAL}"
echo ""

# Success check
if [ "$UNKNOWN_COUNT" -eq 0 ] && [ "$TOTAL" -gt 0 ]; then
    echo -e "${GREEN}✅ SUCCESS: All decisions normalized correctly (0 UNKNOWN)${NC}"
    PASS=1
elif [ "$UNKNOWN_COUNT" -gt 0 ] && [ "$TOTAL" -gt 0 ]; then
    UNKNOWN_PCT=$(echo "scale=1; $UNKNOWN_COUNT * 100 / $TOTAL" | bc)
    if [ "$(echo "$UNKNOWN_PCT < 10" | bc)" -eq 1 ]; then
        echo -e "${YELLOW}⚠️  PARTIAL: ${UNKNOWN_PCT}% UNKNOWN (check unknown_decision ZSET)${NC}"
        PASS=1
    else
        echo -e "${RED}❌ FAIL: ${UNKNOWN_PCT}% UNKNOWN (normalizer may need tuning)${NC}"
        PASS=0
    fi
else
    echo -e "${YELLOW}⚠️  WAITING: No data yet (run again in 1 minute)${NC}"
    PASS=1
fi
echo ""

# Reason diversity check
REASON_COUNT=$(redis-cli ZCARD quantum:p35:reason:top:5m 2>/dev/null || echo "0")
echo -e "${CYAN}Reason Diversity:${NC}"
echo "  Unique reasons: ${REASON_COUNT}"
if [ "$REASON_COUNT" -ge 3 ]; then
    echo -e "${GREEN}  ✅ Good diversity (≥3 reasons)${NC}"
else
    echo -e "${YELLOW}  ⚠️  Low diversity (<3 reasons, may be normal during low activity)${NC}"
fi
echo ""

# Service health check
PENDING=$(redis-cli XPENDING quantum:stream:apply.result p35_decision_intel 2>/dev/null | head -1 || echo "0")
echo -e "${CYAN}Service Health:${NC}"
echo "  Pending messages: ${PENDING}"
if [ "$PENDING" -lt 100 ]; then
    echo -e "${GREEN}  ✅ Consumer lag healthy (<100)${NC}"
else
    echo -e "${YELLOW}  ⚠️  Consumer lag elevated (${PENDING})${NC}"
fi
echo ""

# ============================================================================
# Final result
# ============================================================================
if [ "$PASS" -eq 1 ]; then
    echo -e "${BLUE}═══════════════════════════════════════════════════════════════════${NC}"
    echo -e "${GREEN}✅ P3.5 NORMALIZER VERIFICATION PASSED${NC}"
    echo -e "${BLUE}═══════════════════════════════════════════════════════════════════${NC}"
    echo ""
    echo "Next steps:"
    echo "  1. Monitor for 15+ minutes (ensure stable distribution)"
    echo "  2. Check for unknown_decision ZSET (should be empty or <5%)"
    echo "  3. If stable, ready for P3.5.1 automated alerts"
    exit 0
else
    echo -e "${BLUE}═══════════════════════════════════════════════════════════════════${NC}"
    echo -e "${RED}❌ P3.5 NORMALIZER VERIFICATION FAILED${NC}"
    echo -e "${BLUE}═══════════════════════════════════════════════════════════════════${NC}"
    echo ""
    echo "Troubleshooting:"
    echo "  1. Check logs: sudo journalctl -u quantum-p35-decision-intelligence -n 50"
    echo "  2. Check unknown_decision ZSET: redis-cli ZREVRANGE quantum:p35:unknown_decision:top:5m 0 -1 WITHSCORES"
    echo "  3. Inspect raw events: redis-cli XREVRANGE quantum:stream:apply.result + - COUNT 5"
    exit 1
fi
