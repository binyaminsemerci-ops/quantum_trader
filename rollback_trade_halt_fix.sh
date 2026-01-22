#!/bin/bash
#
# ROLLBACK SCRIPT: Quantum Trader Trade Halt Fix
# Restores system to pre-fix state if regression detected
# Date: 2026-01-17
# Usage: bash rollback_trade_halt_fix.sh [--dry-run]
#

set -e

DRY_RUN=${1:-""}
SSH_HOST="root@46.224.116.254"
SSH_KEY="~/.ssh/hetzner_fresh"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

echo -e "${YELLOW}╔════════════════════════════════════════════════════════════╗${NC}"
echo -e "${YELLOW}║  ROLLBACK: Quantum Trader Trade Halt Fix                   ║${NC}"
echo -e "${YELLOW}║  Restores: AI Engine + Router to pre-fix state             ║${NC}"
echo -e "${YELLOW}╚════════════════════════════════════════════════════════════╝${NC}"
echo

if [ "$DRY_RUN" = "--dry-run" ]; then
    echo -e "${YELLOW}[DRY RUN MODE] - No changes will be made${NC}"
    echo
fi

# Step 1: Find backup files
echo -e "${YELLOW}[1/5] Locating backup files...${NC}"
ROUTER_BACKUP=$(ssh -i "$SSH_KEY" "$SSH_HOST" "ls -1 /usr/local/bin/ai_strategy_router.py.backup_* 2>/dev/null | head -1")

if [ -z "$ROUTER_BACKUP" ]; then
    echo -e "${RED}❌ ERROR: No router backup found${NC}"
    exit 1
fi

echo -e "${GREEN}✅ Found router backup: $ROUTER_BACKUP${NC}"
echo

# Step 2: Save current stream state (before rollback)
echo -e "${YELLOW}[2/5] Capturing current state (for comparison)...${NC}"
BEFORE_DECISION=$(ssh -i "$SSH_KEY" "$SSH_HOST" "export PATH=/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin; redis-cli XLEN quantum:stream:ai.decision.made")
BEFORE_INTENT=$(ssh -i "$SSH_KEY" "$SSH_HOST" "export PATH=/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin; redis-cli XLEN quantum:stream:trade.intent")
BEFORE_EXECUTION=$(ssh -i "$SSH_KEY" "$SSH_HOST" "export PATH=/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin; redis-cli XLEN quantum:stream:execution.result")

echo "  Current stream state:"
echo "    ai.decision.made: $BEFORE_DECISION"
echo "    trade.intent: $BEFORE_INTENT"
echo "    execution.result: $BEFORE_EXECUTION"
echo

# Step 3: Stop services
echo -e "${YELLOW}[3/5] Stopping services...${NC}"
if [ "$DRY_RUN" != "--dry-run" ]; then
    ssh -i "$SSH_KEY" "$SSH_HOST" "systemctl stop quantum-ai-strategy-router quantum-ai-engine quantum-execution" || true
    sleep 2
    echo -e "${GREEN}✅ Services stopped${NC}"
else
    echo -e "${YELLOW}[DRY RUN] Would stop: quantum-ai-strategy-router quantum-ai-engine quantum-execution${NC}"
fi
echo

# Step 4: Restore files
echo -e "${YELLOW}[4/5] Restoring files from backup...${NC}"
if [ "$DRY_RUN" != "--dry-run" ]; then
    ssh -i "$SSH_KEY" "$SSH_HOST" "cp $ROUTER_BACKUP /usr/local/bin/ai_strategy_router.py && echo 'Restored router from backup'"
    echo -e "${GREEN}✅ Router restored from: $ROUTER_BACKUP${NC}"
    
    # AI Engine will restart with fresh state automatically (no backup needed)
    echo -e "${GREEN}✅ AI Engine will reinit on restart${NC}"
else
    echo -e "${YELLOW}[DRY RUN] Would restore router from: $ROUTER_BACKUP${NC}"
    echo -e "${YELLOW}[DRY RUN] Would reinit AI Engine on restart${NC}"
fi
echo

# Step 5: Restart services
echo -e "${YELLOW}[5/5] Restarting services...${NC}"
if [ "$DRY_RUN" != "--dry-run" ]; then
    ssh -i "$SSH_KEY" "$SSH_HOST" "systemctl start quantum-ai-engine"
    sleep 10
    ssh -i "$SSH_KEY" "$SSH_HOST" "systemctl start quantum-ai-strategy-router quantum-execution"
    sleep 5
    
    # Verify
    AI_STATUS=$(ssh -i "$SSH_KEY" "$SSH_HOST" "systemctl is-active quantum-ai-engine")
    ROUTER_STATUS=$(ssh -i "$SSH_KEY" "$SSH_HOST" "systemctl is-active quantum-ai-strategy-router")
    EXEC_STATUS=$(ssh -i "$SSH_KEY" "$SSH_HOST" "systemctl is-active quantum-execution")
    
    echo -e "${GREEN}✅ Services restarted${NC}"
    echo "  AI Engine: $AI_STATUS"
    echo "  Router: $ROUTER_STATUS"
    echo "  Execution: $EXEC_STATUS"
    
    # Wait for streams to process
    sleep 5
    
    AFTER_DECISION=$(ssh -i "$SSH_KEY" "$SSH_HOST" "export PATH=/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin; redis-cli XLEN quantum:stream:ai.decision.made")
    AFTER_INTENT=$(ssh -i "$SSH_KEY" "$SSH_HOST" "export PATH=/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin; redis-cli XLEN quantum:stream:trade.intent")
    
    echo
    echo "Stream state after rollback:"
    echo "  ai.decision.made: $BEFORE_DECISION → $AFTER_DECISION"
    echo "  trade.intent: $BEFORE_INTENT → $AFTER_INTENT"
    
else
    echo -e "${YELLOW}[DRY RUN] Would start: quantum-ai-engine${NC}"
    echo -e "${YELLOW}[DRY RUN] Would start: quantum-ai-strategy-router quantum-execution${NC}"
    echo -e "${YELLOW}[DRY RUN] Would wait 15 seconds for service startup${NC}"
fi
echo

echo -e "${YELLOW}╔════════════════════════════════════════════════════════════╗${NC}"
if [ "$DRY_RUN" = "--dry-run" ]; then
    echo -e "${YELLOW}║  ROLLBACK SIMULATION COMPLETE                            ║${NC}"
    echo -e "${YELLOW}║  Run without --dry-run to execute                        ║${NC}"
else
    echo -e "${GREEN}║  ROLLBACK COMPLETE                                       ║${NC}"
    echo -e "${GREEN}║  System restored to pre-fix state                        ║${NC}"
fi
echo -e "${YELLOW}╚════════════════════════════════════════════════════════════╝${NC}"
