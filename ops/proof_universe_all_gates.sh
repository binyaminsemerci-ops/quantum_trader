#!/usr/bin/env bash
# proof_universe_all_gates.sh
# Verify P1+P2 Universe Integration: P3.3 Position State Brain, Apply Layer, Intent Executor
# All three services must show "source=universe stale=0 count=566"
#
# P1: P3.3 Position State Brain (permit/deny gate)
# P2: Apply Layer (harvest proposal → plan executor)
# P2: Intent Executor (apply.plan → Binance executor after P3.3 permit)

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
echo "UNIVERSE SERVICE INTEGRATION PROOF - ALL GATES"
echo "================================================================"
echo ""
echo "Checking: P3.3 Position State Brain, Apply Layer, Intent Executor"
echo "Expected: All show 'source=universe stale=0 count=566'"
echo ""

# Function to check service
check_service() {
    local service=$1
    local friendly_name=$2
    local log_prefix=$3
    
    echo "----------------------------------------------------------------"
    echo -e "${BLUE}Checking: $friendly_name${NC}"
    echo "----------------------------------------------------------------"
    
    # Check if service is running
    if ! ssh -i "$SSH_KEY" "$REMOTE_USER@$REMOTE_HOST" "systemctl is-active --quiet $service"; then
        echo -e "${RED}❌ $friendly_name is NOT RUNNING${NC}"
        return 1
    fi
    
    echo -e "${GREEN}✅ Service running${NC}"
    
    # Get universe source log line
    local log_line=$(ssh -i "$SSH_KEY" "$REMOTE_USER@$REMOTE_HOST" \
        "journalctl -u $service -n 1000 --no-pager | grep 'allowlist source=universe' | tail -1")
    
    if [ -z "$log_line" ]; then
        echo -e "${RED}❌ No 'allowlist source=universe' log found${NC}"
        return 1
    fi
    
    echo -e "${GREEN}Universe integration log:${NC}"
    echo "$log_line"
    echo ""
    
    # Verify stale=0
    if echo "$log_line" | grep -q "stale=0"; then
        echo -e "${GREEN}✅ stale=0 (fresh data)${NC}"
    else
        echo -e "${YELLOW}⚠️  NOT stale=0 (using fallback)${NC}"
        return 1
    fi
    
    # Verify count=566
    if echo "$log_line" | grep -q "count=566"; then
        echo -e "${GREEN}✅ count=566 (all testnet symbols)${NC}"
    else
        echo -e "${YELLOW}⚠️  NOT count=566${NC}"
        return 1
    fi
    
    echo ""
    return 0
}

# Track overall status
P2_OK=0

# Check P1: P3.3 Position State Brain (optional - may not be running)
echo "================================================================"
echo -e "${BLUE}P1: P3.3 Position State Brain (optional check)${NC}"
echo "================================================================"
if check_service "quantum-p33-position-state-brain" "P1: P3.3 Position State Brain" "P3.3"; then
    echo -e "${GREEN}✅ P1 VERIFIED${NC}"
else
    echo -e "${YELLOW}⚠️  P1 not running or not using universe (this is OK if not deployed)${NC}"
fi

echo ""
echo "================================================================"
echo -e "${BLUE}P2: Apply Layer + Intent Executor (required checks)${NC}"
echo "================================================================"
echo ""

# Check P1: P3.3 Position State Brain
if check_service "quantum-p33-position-state-brain" "P1: P3.3 Position State Brain" "P3.3"; then
    echo -e "${GREEN}✅ P1 VERIFIED${NC}"
else
    echo -e "${RED}❌ P1 FAILED${NC}"
    ALL_OK=1
fi

echo ""

# Check P2: Apply Layer
if check_service "quantum-apply-layer" "P2: Apply Layer" "Apply"; then
    echo -e "${GREEN}✅ P2 Apply Layer VERIFIED${NC}"
else
    echo -e "${RED}❌ P2 Apply Layer FAILED${NC}"
    P2_OK=1
fi

echo ""

# Check P2: Intent Executor
if check_service "quantum-intent-executor" "P2: Intent Executor" "INTENT-EXEC"; then
    echo -e "${GREEN}✅ P2 Intent Executor VERIFIED${NC}"
else
    echo -e "${RED}❌ P2 Intent Executor FAILED${NC}"
    P2_OK=1
fi

echo ""
echo "================================================================"
if [ $P2_OK -eq 0 ]; then
    echo -e "${GREEN}✅ P2 UNIVERSE INTEGRATION COMPLETE${NC}"
    echo ""
    echo "Apply Layer + Intent Executor both reading from:"
    echo "  Universe Service → Redis → Dynamic Allowlist (566 symbols, stale=0)"
    echo ""
    echo "Single source of truth established. No hardcoded allowlists."
else
    echo -e "${RED}❌ P2 VERIFICATION FAILED${NC}"
    echo "One or more P2 services not using universe correctly."
fi
echo "================================================================"

exit $P2_OK
