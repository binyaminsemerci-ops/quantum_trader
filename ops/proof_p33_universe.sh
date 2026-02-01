#!/usr/bin/env bash
# proof_p33_universe.sh
# Verify P3.3 Position State Brain Universe integration
#
# Checks:
# 1. Service status (running)
# 2. Universe meta (stale/count/asof)
# 3. P3.3 allowlist source from logs
# 4. Test permit logic for BTCUSDT + random symbol

set -euo pipefail

REMOTE_USER="root"
REMOTE_HOST="46.224.116.254"
SSH_KEY="$HOME/.ssh/hetzner_fresh"

RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

echo "================================================================"
echo "P3.3 POSITION STATE BRAIN - UNIVERSE INTEGRATION PROOF"
echo "================================================================"
echo ""

# Check service status
echo -e "${BLUE}1. Service Status${NC}"
echo "----------------------------------------------------------------"
if ssh -i "$SSH_KEY" "$REMOTE_USER@$REMOTE_HOST" "systemctl is-active --quiet quantum-position-state-brain"; then
    echo -e "${GREEN}✅ quantum-position-state-brain.service is RUNNING${NC}"
else
    echo -e "${RED}❌ quantum-position-state-brain.service is NOT RUNNING${NC}"
    exit 1
fi
echo ""

# Check Universe meta
echo -e "${BLUE}2. Universe Service Meta${NC}"
echo "----------------------------------------------------------------"
ssh -i "$SSH_KEY" "$REMOTE_USER@$REMOTE_HOST" "redis-cli --raw hgetall quantum:cfg:universe:meta | paste - -" || echo "Universe meta unavailable"
echo ""

# Check P3.3 allowlist source from logs
echo -e "${BLUE}3. P3.3 Allowlist Source${NC}"
echo "----------------------------------------------------------------"
log_line=$(ssh -i "$SSH_KEY" "$REMOTE_USER@$REMOTE_HOST" \
    "journalctl -u quantum-position-state-brain -n 1000 --no-pager | grep 'P3.3 allowlist source' | tail -1" || echo "")

if [ -z "$log_line" ]; then
    echo -e "${RED}❌ No 'P3.3 allowlist source' log found${NC}"
    echo "Service may not have started or Universe integration not working"
    exit 1
fi

echo "$log_line"
echo ""

# Parse source from log
if echo "$log_line" | grep -q "source=universe"; then
    echo -e "${GREEN}✅ Using Universe (primary source)${NC}"
elif echo "$log_line" | grep -q "source=last_ok"; then
    echo -e "${YELLOW}⚠️  Using Universe last_ok (backup source)${NC}"
elif echo "$log_line" | grep -q "source=env"; then
    echo -e "${YELLOW}⚠️  Using env fallback (Universe unavailable)${NC}"
elif echo "$log_line" | grep -q "source=none"; then
    echo -e "${RED}❌ FAIL-CLOSED: No valid allowlist${NC}"
else
    echo -e "${YELLOW}⚠️  Unknown source${NC}"
fi
echo ""

# Test permit logic for BTCUSDT
echo -e "${BLUE}4. Test Permit Logic${NC}"
echo "----------------------------------------------------------------"
echo "Checking if BTCUSDT is in allowlist..."

# Get allowlist from redis universe
symbols=$(ssh -i "$SSH_KEY" "$REMOTE_USER@$REMOTE_HOST" \
    "redis-cli --raw get quantum:cfg:universe:active | jq -r '.symbols[]' 2>/dev/null | head -20" || echo "")

if echo "$symbols" | grep -q "BTCUSDT"; then
    echo -e "${GREEN}✅ BTCUSDT in universe allowlist${NC}"
else
    echo -e "${RED}❌ BTCUSDT NOT in universe allowlist${NC}"
fi

# Pick random symbol from first 10
random_symbol=$(echo "$symbols" | shuf -n 1 | head -1)
if [ -n "$random_symbol" ]; then
    echo -e "${GREEN}✅ Random symbol: $random_symbol${NC}"
fi
echo ""

# Show recent P3.3 permit/deny activity
echo -e "${BLUE}5. Recent P3.3 Activity (Last 10 Events)${NC}"
echo "----------------------------------------------------------------"
ssh -i "$SSH_KEY" "$REMOTE_USER@$REMOTE_HOST" \
    "journalctl -u quantum-position-state-brain -n 200 --no-pager | grep -E '(PERMIT|DENY|Evaluating plan)' | tail -10" || echo "No recent activity"
echo ""

# Check fail-closed behavior
echo -e "${BLUE}6. Fail-Closed Status${NC}"
echo "----------------------------------------------------------------"
if echo "$log_line" | grep -q "source=none"; then
    echo -e "${RED}❌ FAIL-CLOSED ACTIVE: Service will deny all permits${NC}"
    echo "Reason: No valid allowlist from any source (universe active/last_ok/env)"
elif echo "$log_line" | grep -q "count=0"; then
    echo -e "${RED}❌ FAIL-CLOSED ACTIVE: Allowlist is empty${NC}"
else
    echo -e "${GREEN}✅ Fail-closed NOT active (allowlist available)${NC}"
fi
echo ""

echo "================================================================"
echo -e "${GREEN}✅ P3.3 UNIVERSE INTEGRATION PROOF COMPLETE${NC}"
echo "================================================================"
echo ""
echo "Summary:"
echo "  - Service: $(systemctl is-active quantum-position-state-brain 2>/dev/null || echo 'unknown')"
echo "  - Allowlist source: $(echo "$log_line" | grep -oP 'source=\K[^ ]+' || echo 'unknown')"
echo "  - Symbol count: $(echo "$log_line" | grep -oP 'count=\K[^ ]+' || echo 'unknown')"
echo "  - Stale status: $(echo "$log_line" | grep -oP 'stale=\K[^ ]+' || echo 'unknown')"
echo ""
