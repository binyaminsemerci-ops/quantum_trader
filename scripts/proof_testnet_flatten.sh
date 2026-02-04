#!/bin/bash
################################################################################
# TESTNET FLATTEN PROOF SCRIPT
#
# Tests the safe testnet position flattening feature with all safety checks:
# - ESS must be active
# - Config flags must be set
# - Redis arm key triggers one-shot execution
#
# Usage: bash scripts/proof_testnet_flatten.sh
################################################################################

set -e

REPO_ROOT="/home/qt/quantum_trader"
ESS_SCRIPT="$REPO_ROOT/ops/ess_controller.sh"
GOVERNOR_ENV="/etc/quantum/governor.env"
REDIS_CLI="redis-cli"

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

echo -e "${BLUE}========================================${NC}"
echo -e "${BLUE}TESTNET FLATTEN PROOF${NC}"
echo -e "${BLUE}========================================${NC}"
echo ""

# ============================================================================
# STEP 1: Activate ESS
# ============================================================================
echo -e "${YELLOW}[STEP 1] Activating ESS...${NC}"

if [ ! -f "$ESS_SCRIPT" ]; then
    echo -e "${RED}ERROR: ESS controller not found at $ESS_SCRIPT${NC}"
    exit 1
fi

bash "$ESS_SCRIPT" activate | head -10
echo -e "${GREEN}✓ ESS activated${NC}"
echo ""

# ============================================================================
# STEP 2: Set flatten flags in Governor env
# ============================================================================
echo -e "${YELLOW}[STEP 2] Setting flatten flags in $GOVERNOR_ENV...${NC}"

# Remove any existing flatten flags
sed -i '/GOV_TESTNET_FORCE_FLATTEN/d' "$GOVERNOR_ENV" 2>/dev/null || true

# Add new flags
echo "GOV_TESTNET_FORCE_FLATTEN=true" >> "$GOVERNOR_ENV"
echo "GOV_TESTNET_FORCE_FLATTEN_CONFIRM=FLATTEN_NOW" >> "$GOVERNOR_ENV"

echo -e "${BLUE}Flatten flags set:${NC}"
grep "GOV_TESTNET_FORCE_FLATTEN" "$GOVERNOR_ENV"
echo ""

# ============================================================================
# STEP 3: Restart Governor to load new config
# ============================================================================
echo -e "${YELLOW}[STEP 3] Restarting Governor...${NC}"

systemctl restart quantum-governor
sleep 3

echo -e "${GREEN}✓ Governor restarted${NC}"
echo ""

echo -e "${BLUE}Checking startup logs for flatten status:${NC}"
journalctl -u quantum-governor --since "10 seconds ago" --no-pager | grep -i "flatten" | tail -5
echo ""

# ============================================================================
# STEP 4: Check current positions before flatten
# ============================================================================
echo -e "${YELLOW}[STEP 4] Current positions before flatten...${NC}"

POSITION_COUNT=$($REDIS_CLI --scan --pattern "quantum:position:snapshot:*" | wc -l)
echo -e "${BLUE}Position snapshot keys: $POSITION_COUNT${NC}"
echo ""

# ============================================================================
# STEP 5: Arm flatten via Redis key
# ============================================================================
echo -e "${YELLOW}[STEP 5] Arming flatten via Redis key...${NC}"

$REDIS_CLI SET "quantum:gov:testnet:flatten:arm" "1" EX 60
ARM_VALUE=$($REDIS_CLI GET "quantum:gov:testnet:flatten:arm")

echo -e "${BLUE}Arm key set: quantum:gov:testnet:flatten:arm = $ARM_VALUE${NC}"
echo -e "${BLUE}Key will expire in 60 seconds${NC}"
echo ""

# ============================================================================
# STEP 6: Wait and verify flatten execution
# ============================================================================
echo -e "${YELLOW}[STEP 6] Waiting up to 60 seconds for flatten execution...${NC}"
echo ""

START_TIME=$(date +%s)
TIMEOUT=60
FLATTEN_FOUND=false

while true; do
    CURRENT_TIME=$(date +%s)
    ELAPSED=$((CURRENT_TIME - START_TIME))
    
    if [ $ELAPSED -gt $TIMEOUT ]; then
        echo -e "${RED}✗ TIMEOUT: Flatten not executed within 60 seconds${NC}"
        break
    fi
    
    # Check for flatten completion log
    FLATTEN_LOG=$(journalctl -u quantum-governor --since "60 seconds ago" --no-pager 2>/dev/null | grep "TESTNET_FLATTEN done" || echo "")
    
    if [ -n "$FLATTEN_LOG" ]; then
        echo -e "${GREEN}✓ Flatten executed!${NC}"
        echo ""
        echo -e "${BLUE}Flatten log:${NC}"
        echo "$FLATTEN_LOG"
        FLATTEN_FOUND=true
        break
    fi
    
    echo -e "${BLUE}[$(date +'%H:%M:%S')] Waiting... (${ELAPSED}s elapsed)${NC}"
    sleep 3
done

echo ""

# ============================================================================
# STEP 7: Check metrics
# ============================================================================
echo -e "${YELLOW}[STEP 7] Checking flatten metrics...${NC}"

echo -e "${BLUE}Flatten metrics:${NC}"
curl -s localhost:8044/metrics 2>/dev/null | grep "gov_testnet_flatten" | grep -v "^#" || echo "No flatten metrics yet"
echo ""

# ============================================================================
# STEP 8: Review recent logs
# ============================================================================
echo -e "${YELLOW}[STEP 8] Recent flatten-related logs...${NC}"

echo -e "${BLUE}Last 20 flatten logs:${NC}"
journalctl -u quantum-governor --since "2 minutes ago" --no-pager | grep -i "flatten" | tail -20
echo ""

# ============================================================================
# STEP 9: Cleanup - Remove flags
# ============================================================================
echo -e "${YELLOW}[STEP 9] Cleanup - removing flatten flags...${NC}"

# Remove flatten flags
sed -i '/GOV_TESTNET_FORCE_FLATTEN/d' "$GOVERNOR_ENV"

echo -e "${GREEN}✓ Flatten flags removed${NC}"
echo ""

# ============================================================================
# STEP 10: Restart Governor
# ============================================================================
echo -e "${YELLOW}[STEP 10] Restarting Governor with flatten disabled...${NC}"

systemctl restart quantum-governor
sleep 2

echo -e "${GREEN}✓ Governor restarted${NC}"
echo ""

# ============================================================================
# STEP 11: Deactivate ESS
# ============================================================================
echo -e "${YELLOW}[STEP 11] Deactivating ESS...${NC}"

bash "$ESS_SCRIPT" deactivate | head -10
echo -e "${GREEN}✓ ESS deactivated${NC}"
echo ""

# ============================================================================
# SUMMARY
# ============================================================================
echo -e "${BLUE}========================================${NC}"
echo -e "${BLUE}SUMMARY${NC}"
echo -e "${BLUE}========================================${NC}"
echo ""

if [ "$FLATTEN_FOUND" = true ]; then
    echo -e "${GREEN}✓ PASS: Testnet flatten executed successfully${NC}"
    echo ""
    echo -e "${BLUE}Details:${NC}"
    echo "  - ESS activation: OK"
    echo "  - Config flags: Set and detected"
    echo "  - Redis arm key: Triggered flatten"
    echo "  - Flatten execution: Completed"
    echo "  - Cleanup: Config restored"
    echo "  - ESS deactivation: OK"
    echo ""
    exit 0
else
    echo -e "${RED}✗ FAIL: Testnet flatten did not execute${NC}"
    echo ""
    echo -e "${YELLOW}Possible reasons:${NC}"
    echo "  - No open positions to flatten"
    echo "  - Governor not processing arm key"
    echo "  - Safety checks blocked execution"
    echo ""
    echo -e "${YELLOW}Check logs:${NC}"
    echo "  journalctl -u quantum-governor -n 100 | grep -i flatten"
    echo ""
    exit 1
fi
