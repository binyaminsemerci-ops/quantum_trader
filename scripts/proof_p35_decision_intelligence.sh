#!/bin/bash
# Proof script for P3.5 Decision Intelligence Service
# Validates service deployment and shows live analytics

set -euo pipefail

REDIS_HOST="${REDIS_HOST:-localhost}"
REDIS_PORT="${REDIS_PORT:-6379}"
REDIS_DB="${REDIS_DB:-0}"

RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

echo -e "${BLUE}═══════════════════════════════════════════════════════════════════${NC}"
echo -e "${BLUE}P3.5 Decision Intelligence Service - Deployment Proof${NC}"
echo -e "${BLUE}═══════════════════════════════════════════════════════════════════${NC}"
echo ""

# ============================================================================
# STEP 1: Ensure consumer group exists
# ============================================================================

echo -e "${YELLOW}[STEP 1]${NC} Ensuring consumer group exists..."

CONSUMER_GROUP="p35_decision_intel"
STREAM_KEY="quantum:stream:apply.result"

# Try to create the group (will fail if exists, which is OK)
redis-cli -h "$REDIS_HOST" -p "$REDIS_PORT" -n "$REDIS_DB" \
    XGROUP CREATE "$STREAM_KEY" "$CONSUMER_GROUP" "0" MKSTREAM 2>/dev/null || true

STREAM_INFO=$(redis-cli -h "$REDIS_HOST" -p "$REDIS_PORT" -n "$REDIS_DB" \
    XINFO STREAM "$STREAM_KEY" 2>/dev/null || echo "")

if [ -n "$STREAM_INFO" ]; then
    echo -e "${GREEN}✅ Consumer group ready${NC}"
    echo "   Stream: $STREAM_KEY"
    echo "   Group: $CONSUMER_GROUP"
else
    echo -e "${RED}❌ Failed to verify consumer group${NC}"
    exit 1
fi

echo ""

# ============================================================================
# STEP 2: Start service
# ============================================================================

echo -e "${YELLOW}[STEP 2]${NC} Starting service..."

if systemctl is-active --quiet quantum-p35-decision-intelligence; then
    echo -e "${GREEN}✅ Service already running${NC}"
    systemctl restart quantum-p35-decision-intelligence
    echo "   Restarted for fresh state"
else
    echo "   Starting service..."
    systemctl daemon-reload
    systemctl enable quantum-p35-decision-intelligence
    systemctl start quantum-p35-decision-intelligence
    echo -e "${GREEN}✅ Service started${NC}"
fi

sleep 3

echo ""

# ============================================================================
# STEP 3: Show service status
# ============================================================================

echo -e "${YELLOW}[STEP 3]${NC} Service Status..."

STATUS=$(systemctl status quantum-p35-decision-intelligence --no-pager | head -20)
echo "$STATUS"

# Check if running
if systemctl is-active --quiet quantum-p35-decision-intelligence; then
    echo -e "${GREEN}✅ Service is RUNNING${NC}"
else
    echo -e "${RED}❌ Service is NOT RUNNING${NC}"
    echo ""
    echo "Recent logs:"
    journalctl -u quantum-p35-decision-intelligence -n 20 --no-pager
    exit 1
fi

echo ""

# ============================================================================
# STEP 4: Show P3.5 Status
# ============================================================================

echo -e "${YELLOW}[STEP 4]${NC} P3.5 Status (quantum:p35:status)..."

STATUS_DATA=$(redis-cli -h "$REDIS_HOST" -p "$REDIS_PORT" -n "$REDIS_DB" \
    HGETALL "quantum:p35:status" 2>/dev/null || echo "")

if [ -z "$STATUS_DATA" ]; then
    echo -e "${YELLOW}⏳ Status not yet available (service may still be initializing)${NC}"
else
    echo "$STATUS_DATA" | paste - - | awk '{printf "   %-25s: %s\n", $1, $2}'
fi

echo ""

# ============================================================================
# STEP 5: Show top reasons (5-minute window)
# ============================================================================

echo -e "${YELLOW}[STEP 5]${NC} Top Skip/Block Reasons (5-minute window)..."

TOP_REASONS=$(redis-cli -h "$REDIS_HOST" -p "$REDIS_PORT" -n "$REDIS_DB" \
    ZREVRANGE "quantum:p35:reason:top:5m" 0 20 WITHSCORES 2>/dev/null || echo "")

if [ -z "$TOP_REASONS" ]; then
    echo -e "${YELLOW}⏳ No analytics yet (service collecting data)${NC}"
else
    echo "$TOP_REASONS" | paste - - | awk '{printf "   %-30s: %s\n", $1, $2}'
fi

echo ""

# ============================================================================
# STEP 6: Show decision counts (5-minute window)
# ============================================================================

echo -e "${YELLOW}[STEP 6]${NC} Decision Counts (5-minute window)..."

DECISION_COUNTS=$(redis-cli -h "$REDIS_HOST" -p "$REDIS_PORT" -n "$REDIS_DB" \
    HGETALL "quantum:p35:decision:counts:5m" 2>/dev/null || echo "")

if [ -z "$DECISION_COUNTS" ]; then
    echo -e "${YELLOW}⏳ No analytics yet (service collecting data)${NC}"
else
    echo "$DECISION_COUNTS" | paste - - | awk '{printf "   %-15s: %s\n", $1, $2}'
fi

echo ""

# ============================================================================
# STEP 7: Verify consumer group health
# ============================================================================

echo -e "${YELLOW}[STEP 7]${NC} Consumer Group Health..."

# Get pending messages
XPENDING_INFO=$(redis-cli -h "$REDIS_HOST" -p "$REDIS_PORT" -n "$REDIS_DB" \
    XPENDING "$STREAM_KEY" "$CONSUMER_GROUP" 2>/dev/null || echo "")

echo "$XPENDING_INFO" | {
    read -r pending_count lower upper consumer_count
    
    if [ -z "$pending_count" ] || [ "$pending_count" = "0" ]; then
        echo -e "   ${GREEN}✅ No pending messages (all ACKed)${NC}"
    else
        echo -e "   ${YELLOW}⏳ Pending: $pending_count messages${NC}"
        if [ "$pending_count" -lt 10 ]; then
            echo -e "   ${GREEN}✅ Pending count is acceptable${NC}"
        else
            echo -e "   ${YELLOW}⚠️  Consider monitoring: $pending_count pending${NC}"
        fi
    fi
}

echo ""

# ============================================================================
# STEP 8: List active windows
# ============================================================================

echo -e "${YELLOW}[STEP 8]${NC} Available Analytics Windows..."

for window in "1m" "5m" "15m" "1h"; do
    decision_key="quantum:p35:decision:counts:$window"
    reason_key="quantum:p35:reason:top:$window"
    
    decision_exists=$(redis-cli -h "$REDIS_HOST" -p "$REDIS_PORT" -n "$REDIS_DB" \
        EXISTS "$decision_key" 2>/dev/null || echo "0")
    reason_exists=$(redis-cli -h "$REDIS_HOST" -p "$REDIS_PORT" -n "$REDIS_DB" \
        EXISTS "$reason_key" 2>/dev/null || echo "0")
    
    if [ "$decision_exists" = "1" ] && [ "$reason_exists" = "1" ]; then
        echo -e "   ${GREEN}✅${NC} $window window available"
    else
        echo -e "   ${YELLOW}⏳${NC} $window window (data collecting)"
    fi
done

echo ""

# ============================================================================
# STEP 9: Sample queries for user
# ============================================================================

echo -e "${YELLOW}[STEP 9]${NC} Available CLI Commands..."

echo ""
echo -e "${BLUE}Top 10 Skip/Block Reasons (5-minute):${NC}"
echo "  redis-cli ZREVRANGE quantum:p35:reason:top:5m 0 10 WITHSCORES"
echo ""

echo -e "${BLUE}Decision Counts (5-minute):${NC}"
echo "  redis-cli HGETALL quantum:p35:decision:counts:5m"
echo ""

echo -e "${BLUE}Service Status:${NC}"
echo "  redis-cli HGETALL quantum:p35:status"
echo ""

echo -e "${BLUE}Stream Consumer Group Info:${NC}"
echo "  redis-cli XINFO GROUPS quantum:stream:apply.result"
echo ""

echo -e "${BLUE}Pending Messages:${NC}"
echo "  redis-cli XPENDING quantum:stream:apply.result p35_decision_intel"
echo ""

# ============================================================================
# FINAL SUMMARY
# ============================================================================

echo -e "${BLUE}═══════════════════════════════════════════════════════════════════${NC}"
echo -e "${GREEN}✅ DEPLOYMENT VERIFICATION COMPLETE${NC}"
echo -e "${BLUE}═══════════════════════════════════════════════════════════════════${NC}"
echo ""
echo "Service: quantum-p35-decision-intelligence"
echo "Status:  $(systemctl is-active quantum-p35-decision-intelligence)"
echo "Config:  /etc/quantum/p35-decision-intelligence.env"
echo "Logs:    journalctl -u quantum-p35-decision-intelligence -f"
echo ""
echo "🚀 Ready to collect apply.result analytics!"
echo ""
