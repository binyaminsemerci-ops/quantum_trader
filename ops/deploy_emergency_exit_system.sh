#!/bin/bash
# =============================================================================
# Deploy Emergency Exit Worker & Watchdog System
# =============================================================================
# This script deploys the emergency exit infrastructure:
# 1. Emergency Exit Worker (closes all positions on panic_close)
# 2. Exit Brain Watchdog (monitors Exit Brain, triggers panic_close if failed)
# =============================================================================

set -e

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

echo -e "${YELLOW}========================================${NC}"
echo -e "${YELLOW} DEPLOYING EMERGENCY EXIT SYSTEM${NC}"
echo -e "${YELLOW}========================================${NC}"

# Paths
QT_HOME="/home/qt/quantum_trader"
SYSTEMD_DIR="/etc/systemd/system"

# Check if running as root
if [ "$EUID" -ne 0 ]; then
    echo -e "${RED}Please run as root${NC}"
    exit 1
fi

# Check paths exist
if [ ! -d "$QT_HOME" ]; then
    echo -e "${RED}ERROR: $QT_HOME does not exist${NC}"
    exit 1
fi

echo -e "\n${GREEN}[1/5] Copying Emergency Exit Worker files...${NC}"
# Files should already be in place from git pull
ls -la $QT_HOME/services/emergency_exit_worker/
ls -la $QT_HOME/services/exit_brain/exit_brain_watchdog.py

echo -e "\n${GREEN}[2/5] Installing systemd services...${NC}"
cp $QT_HOME/ops/systemd/quantum-emergency-exit-worker.service $SYSTEMD_DIR/
cp $QT_HOME/ops/systemd/quantum-exit-brain-watchdog.service $SYSTEMD_DIR/
chmod 644 $SYSTEMD_DIR/quantum-emergency-exit-worker.service
chmod 644 $SYSTEMD_DIR/quantum-exit-brain-watchdog.service

echo -e "\n${GREEN}[3/5] Reloading systemd...${NC}"
systemctl daemon-reload

echo -e "\n${GREEN}[4/5] Enabling services...${NC}"
systemctl enable quantum-emergency-exit-worker
systemctl enable quantum-exit-brain-watchdog

echo -e "\n${GREEN}[5/5] Starting services...${NC}"
systemctl start quantum-emergency-exit-worker
systemctl start quantum-exit-brain-watchdog

echo -e "\n${YELLOW}========================================${NC}"
echo -e "${GREEN} DEPLOYMENT COMPLETE${NC}"
echo -e "${YELLOW}========================================${NC}"

echo -e "\n${GREEN}Service Status:${NC}"
echo "---"
systemctl is-active quantum-emergency-exit-worker || true
systemctl is-active quantum-exit-brain-watchdog || true

echo -e "\n${GREEN}Recent logs:${NC}"
echo "--- Emergency Exit Worker ---"
journalctl -u quantum-emergency-exit-worker --no-pager -n 10
echo ""
echo "--- Exit Brain Watchdog ---"
journalctl -u quantum-exit-brain-watchdog --no-pager -n 10

echo -e "\n${YELLOW}========================================${NC}"
echo -e "${GREEN} VERIFY DEPLOYMENT${NC}"
echo -e "${YELLOW}========================================${NC}"
echo -e "
Run these commands to verify:

  # Check services running
  systemctl status quantum-emergency-exit-worker
  systemctl status quantum-exit-brain-watchdog
  
  # Check Redis streams created
  redis-cli XINFO STREAM quantum:stream:system.panic_close
  redis-cli XINFO STREAM quantum:stream:exit_brain.heartbeat
  
  # Test panic_close (ONLY ON TESTNET!)
  # redis-cli XADD quantum:stream:system.panic_close '*' source ops reason 'TEST' timestamp \$(date +%s)
"

echo -e "${GREEN}Done!${NC}"
