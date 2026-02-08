#!/bin/bash
#
# Meta-Agent V2 + Arbiter Deployment Script
# Trinnvis deployment med validering
#
# Usage: bash deploy_meta_v2_arbiter.sh
#

set -e  # Exit on error

CYAN='\033[0;36m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m' # No Color

VPS_HOST="root@46.224.116.254"
VPS_DIR="/home/qt/quantum_trader"
SERVICE_NAME="quantum-ai-engine"

echo -e "${CYAN}╔════════════════════════════════════════════════════════════════╗"
echo -e "║  META-AGENT V2 + ARBITER DEPLOYMENT                            ║"
echo -e "║  Trinnvis deployment med validering                            ║"
echo -e "╚════════════════════════════════════════════════════════════════╝${NC}"

# ============================================================================
# STEP 1: Upload Files
# ============================================================================
echo -e "\n${CYAN}[STEP 1/5] Uploading files to VPS...${NC}"

echo -e "${YELLOW}→ Uploading Arbiter Agent...${NC}"
scp -i ~/.ssh/hetzner_fresh \
    /mnt/c/quantum_trader/ai_engine/agents/arbiter_agent.py \
    ${VPS_HOST}:${VPS_DIR}/ai_engine/agents/ || exit 1

echo -e "${YELLOW}→ Uploading Meta-Agent V2 (refactored)...${NC}"
scp -i ~/.ssh/hetzner_fresh \
    /mnt/c/quantum_trader/ai_engine/agents/meta_agent_v2.py \
    ${VPS_HOST}:${VPS_DIR}/ai_engine/agents/meta_agent_v2.py || exit 1

echo -e "${YELLOW}→ Uploading Ensemble Manager (modified)...${NC}"
scp -i ~/.ssh/hetzner_fresh \
    /mnt/c/quantum_trader/ai_engine/ensemble_manager.py \
    ${VPS_HOST}:${VPS_DIR}/ai_engine/ || exit 1

echo -e "${YELLOW}→ Uploading Integration Test...${NC}"
scp -i ~/.ssh/hetzner_fresh \
    /mnt/c/quantum_trader/test_meta_v2_arbiter_integration.py \
    ${VPS_HOST}:${VPS_DIR}/ || exit 1

echo -e "${GREEN}✓ Files uploaded successfully${NC}"

# ============================================================================
# STEP 2: Run Integration Tests
# ============================================================================
echo -e "\n${CYAN}[STEP 2/5] Running integration tests on VPS...${NC}"

ssh -i ~/.ssh/hetzner_fresh ${VPS_HOST} << 'ENDSSH'
cd /home/qt/quantum_trader
/opt/quantum/venvs/ai-engine/bin/python test_meta_v2_arbiter_integration.py
ENDSSH

if [ $? -eq 0 ]; then
    echo -e "${GREEN}✓ Integration tests passed${NC}"
else
    echo -e "${RED}✗ Integration tests failed - ABORTING DEPLOYMENT${NC}"
    exit 1
fi

# ============================================================================
# STEP 3: Backup Current Configuration
# ============================================================================
echo -e "\n${CYAN}[STEP 3/5] Creating backup...${NC}"

ssh -i ~/.ssh/hetzner_fresh ${VPS_HOST} << 'ENDSSH'
# Backup service file
sudo cp /etc/systemd/system/quantum-ai-engine.service \
       /etc/systemd/system/quantum-ai-engine.service.backup.$(date +%Y%m%d_%H%M%S)

# Backup code files
cd /home/qt/quantum_trader
cp ai_engine/ensemble_manager.py ai_engine/ensemble_manager.py.backup.$(date +%Y%m%d_%H%M%S)
if [ -f ai_engine/agents/meta_agent_v2.py ]; then
    cp ai_engine/agents/meta_agent_v2.py ai_engine/agents/meta_agent_v2.py.backup.$(date +%Y%m%d_%H%M%S)
fi

echo "✓ Backups created"
ENDSSH

echo -e "${GREEN}✓ Backups created${NC}"

# ============================================================================
# STEP 4a: Enable Meta-V2 (Arbiter OFF)
# ============================================================================
echo -e "\n${CYAN}[STEP 4a/5] PHASE 1: Enabling Meta-V2 (Arbiter OFF)...${NC}"

ssh -i ~/.ssh/hetzner_fresh ${VPS_HOST} << 'ENDSSH'
# Update service file
sudo cp /etc/systemd/system/quantum-ai-engine.service /tmp/quantum-ai-engine.service.tmp

# Remove old Meta/Arbiter env vars if present
sudo sed -i '/Environment="META_AGENT_ENABLED/d' /tmp/quantum-ai-engine.service.tmp
sudo sed -i '/Environment="ARBITER_ENABLED/d' /tmp/quantum-ai-engine.service.tmp
sudo sed -i '/Environment="ARBITER_THRESHOLD/d' /tmp/quantum-ai-engine.service.tmp

# Add new env vars (Meta ON, Arbiter OFF)
sudo sed -i '/\[Service\]/a Environment="META_AGENT_ENABLED=true"' /tmp/quantum-ai-engine.service.tmp
sudo sed -i '/\[Service\]/a Environment="ARBITER_ENABLED=false"' /tmp/quantum-ai-engine.service.tmp
sudo sed -i '/\[Service\]/a Environment="ARBITER_THRESHOLD=0.70"' /tmp/quantum-ai-engine.service.tmp

# Apply changes
sudo mv /tmp/quantum-ai-engine.service.tmp /etc/systemd/system/quantum-ai-engine.service

# Reload and restart
sudo systemctl daemon-reload
sudo systemctl restart quantum-ai-engine

echo "✓ Service restarted with Meta-V2 enabled"
ENDSSH

echo -e "${GREEN}✓ Meta-V2 enabled (Arbiter OFF)${NC}"

# ============================================================================
# STEP 4b: Monitor Phase 1 (5 minutes)
# ============================================================================
echo -e "\n${CYAN}[STEP 4b/5] PHASE 1 MONITORING: Watching Meta-V2 behavior...${NC}"
echo -e "${YELLOW}→ Looking for DEFER/ESCALATE patterns (watching for 30 seconds)${NC}"

ssh -i ~/.ssh/hetzner_fresh ${VPS_HOST} << 'ENDSSH'
echo "Monitoring Meta-V2 for 30 seconds..."
timeout 30 journalctl -u quantum-ai-engine -f | grep --line-buffered -E 'META-V2|Meta-V2-Policy|DEFER|ESCALATE' || true

echo ""
echo "=== PHASE 1 STATUS CHECK ==="
sleep 2

# Check service status
systemctl is-active quantum-ai-engine

# Count Meta decisions
TOTAL=$(journalctl -u quantum-ai-engine --since "2 minutes ago" | grep -c 'Meta-V2-Policy' || echo 0)
DEFER=$(journalctl -u quantum-ai-engine --since "2 minutes ago" | grep -c 'DEFER' || echo 0)
ESCALATE=$(journalctl -u quantum-ai-engine --since "2 minutes ago" | grep -c 'ESCALATE' || echo 0)

echo "Meta-V2 decisions (last 2 min): Total=$TOTAL, DEFER=$DEFER, ESCALATE=$ESCALATE"

# Check for errors
ERRORS=$(journalctl -u quantum-ai-engine --since "2 minutes ago" | grep -ic 'error' || echo 0)
echo "Errors (last 2 min): $ERRORS"

if [ "$ERRORS" -gt 5 ]; then
    echo "⚠️  WARNING: High error count detected"
fi
ENDSSH

echo -e "\n${YELLOW}═══════════════════════════════════════════════════════════${NC}"
echo -e "${YELLOW}PHASE 1 COMPLETE: Meta-V2 is running (policy layer only)${NC}"
echo -e "${YELLOW}═══════════════════════════════════════════════════════════${NC}"

# Ask for confirmation before enabling Arbiter
echo -e "\n${CYAN}Press Enter to continue to PHASE 2 (Enable Arbiter) or Ctrl+C to stop:${NC}"
read

# ============================================================================
# STEP 5: Enable Arbiter (PHASE 2)
# ============================================================================
echo -e "\n${CYAN}[STEP 5/5] PHASE 2: Enabling Arbiter...${NC}"

ssh -i ~/.ssh/hetzner_fresh ${VPS_HOST} << 'ENDSSH'
# Update service file
sudo cp /etc/systemd/system/quantum-ai-engine.service /tmp/quantum-ai-engine.service.tmp

# Change Arbiter to enabled
sudo sed -i 's/ARBITER_ENABLED=false/ARBITER_ENABLED=true/' /tmp/quantum-ai-engine.service.tmp

# Apply changes
sudo mv /tmp/quantum-ai-engine.service.tmp /etc/systemd/system/quantum-ai-engine.service

# Reload and restart
sudo systemctl daemon-reload
sudo systemctl restart quantum-ai-engine

echo "✓ Service restarted with Arbiter enabled"
ENDSSH

echo -e "${GREEN}✓ Arbiter enabled${NC}"

# Monitor Phase 2
echo -e "\n${CYAN}[PHASE 2 MONITORING] Watching full decision hierarchy...${NC}"
echo -e "${YELLOW}→ Looking for Meta-V2 → Arbiter flow (watching for 30 seconds)${NC}"

ssh -i ~/.ssh/hetzner_fresh ${VPS_HOST} << 'ENDSSH'
echo "Monitoring full decision hierarchy for 30 seconds..."
timeout 30 journalctl -u quantum-ai-engine -f | grep --line-buffered -E 'Meta-V2|Arbiter|ESCALATE|OVERRIDE|DEFER' || true

echo ""
echo "=== PHASE 2 STATUS CHECK ==="
sleep 2

# Count decisions
ESCALATE=$(journalctl -u quantum-ai-engine --since "2 minutes ago" | grep -c 'ESCALATE' || echo 0)
ARBITER_INVOKED=$(journalctl -u quantum-ai-engine --since "2 minutes ago" | grep -c 'Arbiter.*INVOKED' || echo 0)
ARBITER_OVERRIDE=$(journalctl -u quantum-ai-engine --since "2 minutes ago" | grep -c 'Arbiter.*OVERRIDE' || echo 0)
ARBITER_DEFER=$(journalctl -u quantum-ai-engine --since "2 minutes ago" | grep -c 'Arbiter.*DEFER' || echo 0)

echo "Meta-V2 escalations: $ESCALATE"
echo "Arbiter invoked: $ARBITER_INVOKED"
echo "Arbiter overrides: $ARBITER_OVERRIDE"
echo "Arbiter defers: $ARBITER_DEFER"

# Check for errors
ERRORS=$(journalctl -u quantum-ai-engine --since "2 minutes ago" | grep -ic 'error' || echo 0)
echo "Errors (last 2 min): $ERRORS"

if [ "$ERRORS" -gt 5 ]; then
    echo "⚠️  WARNING: High error count detected"
fi
ENDSSH

# ============================================================================
# DEPLOYMENT COMPLETE
# ============================================================================
echo -e "\n${GREEN}╔════════════════════════════════════════════════════════════════╗${NC}"
echo -e "${GREEN}║  DEPLOYMENT COMPLETE ✓                                         ║${NC}"
echo -e "${GREEN}╚════════════════════════════════════════════════════════════════╝${NC}"

echo -e "\n${CYAN}Current Configuration:${NC}"
echo -e "  • Meta-Agent V2: ${GREEN}ENABLED${NC} (policy layer)"
echo -e "  • Arbiter Agent #5: ${GREEN}ENABLED${NC} (market understanding)"
echo -e "  • Base Ensemble: ${GREEN}ACTIVE${NC} (always fallback)"

echo -e "\n${CYAN}Monitoring Commands:${NC}"
echo -e "  ${YELLOW}# Real-time monitoring${NC}"
echo -e "  ssh -i ~/.ssh/hetzner_fresh ${VPS_HOST} 'journalctl -u ${SERVICE_NAME} -f | grep -E \"Meta-V2|Arbiter\"'"
echo -e ""
echo -e "  ${YELLOW}# Last 20 decisions${NC}"
echo -e "  ssh -i ~/.ssh/hetzner_fresh ${VPS_HOST} 'journalctl -u ${SERVICE_NAME} --since \"10 minutes ago\" | grep -E \"ENSEMBLE.*BTCUSDT|Meta-V2|Arbiter\" | tail -20'"

echo -e "\n${CYAN}Rollback (if needed):${NC}"
echo -e "  ${YELLOW}# Disable Arbiter only${NC}"
echo -e "  ssh -i ~/.ssh/hetzner_fresh ${VPS_HOST} 'sudo sed -i \"s/ARBITER_ENABLED=true/ARBITER_ENABLED=false/\" /etc/systemd/system/quantum-ai-engine.service && sudo systemctl daemon-reload && sudo systemctl restart quantum-ai-engine'"
echo -e ""
echo -e "  ${YELLOW}# Disable both Meta-V2 and Arbiter${NC}"
echo -e "  ssh -i ~/.ssh/hetzner_fresh ${VPS_HOST} 'sudo sed -i \"s/META_AGENT_ENABLED=true/META_AGENT_ENABLED=false/\" /etc/systemd/system/quantum-ai-engine.service && sudo sed -i \"s/ARBITER_ENABLED=true/ARBITER_ENABLED=false/\" /etc/systemd/system/quantum-ai-engine.service && sudo systemctl daemon-reload && sudo systemctl restart quantum-ai-engine'"

echo -e "\n${GREEN}✓ System is live with 3-layer decision hierarchy!${NC}"
echo -e "${CYAN}Watch logs for the next 30-60 minutes to validate behavior.${NC}\n"
