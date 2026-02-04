#!/bin/bash
################################################################################
# TESTNET CLEAN SLATE HELPER
#
# Safely prepare testnet for clean P2.9 metrics by:
# 1. Activating ESS (stop new trades)
# 2. Showing open positions
# 3. Waiting for manual position closure
# 4. Deactivating ESS (resume trading)
# 5. Displaying P2.9 baseline metrics
#
# NO DESTRUCTIVE ACTIONS - operator must manually close positions.
#
# Usage: bash scripts/testnet_clean_slate.sh
################################################################################

set -e

REPO_ROOT="/home/qt/quantum_trader"
ESS_SCRIPT="$REPO_ROOT/ops/ess_controller.sh"
REDIS_CLI="redis-cli"
TIMEOUT_SECONDS=600  # 10 minutes

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

echo -e "${BLUE}========================================${NC}"
echo -e "${BLUE}TESTNET CLEAN SLATE HELPER${NC}"
echo -e "${BLUE}========================================${NC}"
echo ""

# ============================================================================
# STEP 1: Activate ESS
# ============================================================================
echo -e "${YELLOW}[STEP 1] Activating ESS (Emergency Stop System)...${NC}"

if [ ! -f "$ESS_SCRIPT" ]; then
    echo -e "${RED}ERROR: ESS controller not found at $ESS_SCRIPT${NC}"
    exit 1
fi

bash "$ESS_SCRIPT" activate

echo -e "${GREEN}✓ ESS activated - no new trades will execute${NC}"
echo ""

# ============================================================================
# STEP 2: Show current open positions
# ============================================================================
echo -e "${YELLOW}[STEP 2] Current open positions from Redis...${NC}"
echo ""

# Get all position snapshot keys using SCAN (safe for large keyspaces)
POSITION_KEYS=$($REDIS_CLI --scan --pattern "quantum:position:snapshot:*")

if [ -z "$POSITION_KEYS" ]; then
    echo -e "${GREEN}No position snapshot keys found${NC}"
    POSITION_COUNT=0
else
    # Count only active positions (qty != 0)
    POSITION_COUNT=0
    ACTIVE_POSITIONS=""
    
    for key in $POSITION_KEYS; do
        qty=$($REDIS_CLI HGET "$key" qty 2>/dev/null || echo "0")
        # Check if qty is non-zero (active position)
        if [ -n "$qty" ] && [ "$qty" != "0" ] && [ "$qty" != "0.0" ]; then
            POSITION_COUNT=$((POSITION_COUNT + 1))
            ACTIVE_POSITIONS="$ACTIVE_POSITIONS $key"
        fi
    done
    
    if [ $POSITION_COUNT -eq 0 ]; then
        echo -e "${GREEN}No active positions (all qty=0)${NC}"
    else
        echo -e "${BLUE}Found $POSITION_COUNT active position(s):${NC}"
        echo ""
        
        # Display each active position
        for key in $ACTIVE_POSITIONS; do
            symbol=$(echo "$key" | sed 's/quantum:position:snapshot://')
            echo -e "${BLUE}=== $symbol ===${NC}"
            $REDIS_CLI HGETALL "$key" | paste - - | while read field value; do
                echo "  $field: $value"
            done
            echo ""
        done
    fi
fi

# ============================================================================
# STEP 3: Wait for manual position closure
# ============================================================================
if [ $POSITION_COUNT -gt 0 ]; then
    echo -e "${YELLOW}[STEP 3] Waiting for manual position closure...${NC}"
    echo ""
    echo -e "${RED}┌─────────────────────────────────────────────────────────┐${NC}"
    echo -e "${RED}│ ACTION REQUIRED: Close all open positions manually     │${NC}"
    echo -e "${RED}│                                                         │${NC}"
    echo -e "${RED}│ Options:                                                │${NC}"
    echo -e "${RED}│ 1. Use Binance Testnet UI to close positions           │${NC}"
    echo -e "${RED}│ 2. Use close_all_positions.py script                   │${NC}"
    echo -e "${RED}│ 3. Let existing positions expire/close naturally        │${NC}"
    echo -e "${RED}│                                                         │${NC}"
    echo -e "${RED}│ This script will poll every 10 seconds...              │${NC}"
    echo -e "${RED}│ Timeout: 10 minutes                                     │${NC}"
    echo -e "${RED}└─────────────────────────────────────────────────────────┘${NC}"
    echo ""
    
    START_TIME=$(date +%s)
    
    while true; do
        # Check current position count (only active positions with qty != 0)
        CURRENT_KEYS=$($REDIS_CLI --scan --pattern "quantum:position:snapshot:*" 2>/dev/null || echo "")
        CURRENT_COUNT=0
        
        if [ -n "$CURRENT_KEYS" ]; then
            # Count only positions with non-zero qty
            for key in $CURRENT_KEYS; do
                qty=$($REDIS_CLI HGET "$key" qty 2>/dev/null || echo "0")
                if [ -n "$qty" ] && [ "$qty" != "0" ] && [ "$qty" != "0.0" ]; then
                    CURRENT_COUNT=$((CURRENT_COUNT + 1))
                fi
            done
        fi
        
        # Check timeout
        CURRENT_TIME=$(date +%s)
        ELAPSED=$((CURRENT_TIME - START_TIME))
        
        if [ $CURRENT_COUNT -eq 0 ]; then
            echo -e "${GREEN}✓ All positions closed!${NC}"
            break
        fi
        
        if [ $ELAPSED -gt $TIMEOUT_SECONDS ]; then
            echo -e "${RED}✗ TIMEOUT: Still $CURRENT_COUNT active position(s) after 10 minutes${NC}"
            echo -e "${YELLOW}Remaining active positions:${NC}"
            for key in $CURRENT_KEYS; do
                qty=$($REDIS_CLI HGET "$key" qty 2>/dev/null || echo "0")
                if [ -n "$qty" ] && [ "$qty" != "0" ] && [ "$qty" != "0.0" ]; then
                    symbol=$(echo "$key" | sed 's/quantum:position:snapshot://')
                    echo "  - $symbol (qty=$qty)"
                fi
            done
            echo ""
            echo -e "${YELLOW}You can:${NC}"
            echo "  1. Continue manually closing positions"
            echo "  2. Run this script again later"
            echo "  3. Proceed anyway (metrics will include legacy positions)"
            echo ""
            read -p "Deactivate ESS and exit? (y/n): " -n 1 -r
            echo
            if [[ $REPLY =~ ^[Yy]$ ]]; then
                echo -e "${YELLOW}Deactivating ESS...${NC}"
                bash "$ESS_SCRIPT" deactivate
                echo -e "${GREEN}✓ ESS deactivated${NC}"
                echo -e "${YELLOW}Exiting with open positions remaining${NC}"
                exit 1
            else
                echo "Continuing to wait..."
            fi
        fi
        
        # Status update
        REMAINING=$((TIMEOUT_SECONDS - ELAPSED))
        echo -e "${BLUE}[$(date +'%H:%M:%S')] Waiting... $CURRENT_COUNT position(s) open, timeout in ${REMAINING}s${NC}"
        sleep 10
    done
    
    echo ""
else
    echo -e "${GREEN}[STEP 3] No positions to close - proceeding${NC}"
    echo ""
fi

# ============================================================================
# STEP 4: Deactivate ESS
# ============================================================================
echo -e "${YELLOW}[STEP 4] Deactivating ESS (resume trading)...${NC}"

bash "$ESS_SCRIPT" deactivate

echo -e "${GREEN}✓ ESS deactivated - system ready for clean slate trading${NC}"
echo ""

# ============================================================================
# STEP 5: Display baseline metrics and recent logs
# ============================================================================
echo -e "${YELLOW}[STEP 5] P2.9 Baseline Metrics & Recent Activity${NC}"
echo ""

echo -e "${BLUE}=== Governor P2.9 Metrics ===${NC}"
curl -s localhost:8044/metrics | grep -E "gov_p29_(checked|block|missing|stale)_total|gov_testnet_p29_enabled" | grep -v "^#"
echo ""

echo -e "${BLUE}=== Recent Testnet P2.9 Gate Activity (last 20 lines) ===${NC}"
journalctl -u quantum-governor --since "10 minutes ago" --no-pager | grep -i "testnet.*p2\.9\|p2\.9.*testnet" | tail -20
echo ""

echo -e "${GREEN}========================================${NC}"
echo -e "${GREEN}CLEAN SLATE PREPARATION COMPLETE${NC}"
echo -e "${GREEN}========================================${NC}"
echo ""
echo -e "${BLUE}Next Steps:${NC}"
echo "  1. Monitor new position openings with P2.9 enforcement"
echo "  2. Watch metrics: curl localhost:8044/metrics | grep gov_p29"
echo "  3. Follow logs: journalctl -u quantum-governor -f | grep P2.9"
echo ""
echo -e "${YELLOW}NOTE: Current metrics reflect pre-clean-slate activity.${NC}"
echo -e "${YELLOW}      New metrics will accumulate as fresh positions open.${NC}"
echo ""

exit 0
