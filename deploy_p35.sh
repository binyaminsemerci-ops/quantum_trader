#!/bin/bash
# P3.5 Decision Intelligence Service - VPS Deployment Script
# Run this on the VPS to deploy P3.5

set -euo pipefail

BLUE='\033[0;34m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m'

echo -e "${BLUE}═══════════════════════════════════════════════════════════════════${NC}"
echo -e "${BLUE}P3.5 Decision Intelligence Service - VPS Deployment${NC}"
echo -e "${BLUE}═══════════════════════════════════════════════════════════════════${NC}"
echo ""

cd /home/qt/quantum_trader

# ============================================================================
# STEP 1: Pull latest from git
# ============================================================================
echo -e "${YELLOW}[STEP 1]${NC} Pulling latest code..."
git fetch origin main
git reset --hard origin/main
echo -e "${GREEN}✅ Code updated${NC}"
echo ""

# ============================================================================
# STEP 2: Copy configuration
# ============================================================================
echo -e "${YELLOW}[STEP 2]${NC} Installing configuration..."

if [ ! -d "/etc/quantum" ]; then
    sudo mkdir -p /etc/quantum
    sudo chown qt:qt /etc/quantum
fi

sudo cp etc/quantum/p35-decision-intelligence.env /etc/quantum/
sudo chown qt:qt /etc/quantum/p35-decision-intelligence.env
echo -e "${GREEN}✅ Configuration installed${NC}"
echo "   Path: /etc/quantum/p35-decision-intelligence.env"
echo ""

# ============================================================================
# STEP 3: Copy systemd unit
# ============================================================================
echo -e "${YELLOW}[STEP 3]${NC} Installing systemd unit..."

sudo cp etc/systemd/system/quantum-p35-decision-intelligence.service /etc/systemd/system/
sudo systemctl daemon-reload
echo -e "${GREEN}✅ Systemd unit installed${NC}"
echo "   Path: /etc/systemd/system/quantum-p35-decision-intelligence.service"
echo ""

# ============================================================================
# STEP 4: Clear Python cache
# ============================================================================
echo -e "${YELLOW}[STEP 4]${NC} Clearing Python cache..."
find microservices/decision_intelligence -name "__pycache__" -type d -exec rm -rf {} + 2>/dev/null || true
find microservices/decision_intelligence -name "*.pyc" -delete 2>/dev/null || true
echo -e "${GREEN}✅ Cache cleared${NC}"
echo ""

# ============================================================================
# STEP 5: Enable and start service
# ============================================================================
echo -e "${YELLOW}[STEP 5]${NC} Enabling and starting service..."

# Idempotent: enable + start (or restart if already running)
if sudo systemctl is-active --quiet quantum-p35-decision-intelligence; then
    echo "   Service already running, restarting..."
    sudo systemctl restart quantum-p35-decision-intelligence
else
    sudo systemctl enable --now quantum-p35-decision-intelligence
fi
echo -e "${GREEN}✅ Service enabled and started${NC}"

sleep 2
if sudo systemctl is-active --quiet quantum-p35-decision-intelligence; then
    echo -e "${GREEN}✅ Service is RUNNING${NC}"
else
    echo -e "${RED}❌ Service is NOT RUNNING${NC}"
    echo "Logs:"
    sudo journalctl -u quantum-p35-decision-intelligence -n 30 --no-pager
    exit 1
fi
echo ""

# ============================================================================
# STEP 6: Run proof script
# ============================================================================
echo -e "${YELLOW}[STEP 6]${NC} Running deployment proof..."
echo ""

bash scripts/proof_p35_decision_intelligence.sh

echo ""
echo -e "${BLUE}═══════════════════════════════════════════════════════════════════${NC}"
echo -e "${GREEN}✅ P3.5 DEPLOYMENT COMPLETE${NC}"
echo -e "${BLUE}═══════════════════════════════════════════════════════════════════${NC}"
echo ""
echo "Service: quantum-p35-decision-intelligence"
echo "Status:  $(sudo systemctl is-active quantum-p35-decision-intelligence)"
echo "Config:  /etc/quantum/p35-decision-intelligence.env"
echo "Logs:    journalctl -u quantum-p35-decision-intelligence -f"
echo ""
echo "Next: Monitor analytics with:"
echo "  redis-cli HGETALL quantum:p35:status"
echo "  redis-cli ZREVRANGE quantum:p35:reason:top:5m 0 20 WITHSCORES"
echo ""
