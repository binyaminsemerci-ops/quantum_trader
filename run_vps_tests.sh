#!/bin/bash
###############################################################################
# Kjør Comprehensive System Test på VPS (Bash version)
###############################################################################

VPS_IP="46.224.116.254"
VPS_USER="qt"
SSH_KEY="$HOME/.ssh/hetzner_fresh"

# Colors
BLUE='\033[0;34m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m'

echo -e "\n${BLUE}═══════════════════════════════════════════════════${NC}"
echo -e "${BLUE}  QUANTUM TRADER VPS TEST${NC}"
echo -e "${BLUE}═══════════════════════════════════════════════════${NC}\n"

echo -e "${BLUE}[INFO]${NC} VPS: $VPS_USER@$VPS_IP"
echo -e "${BLUE}[INFO]${NC} SSH Key: $SSH_KEY\n"

# Test SSH connection
echo -e "${YELLOW}[1/4] Testing SSH connection...${NC}"
if ssh -i "$SSH_KEY" -o ConnectTimeout=5 -o StrictHostKeyChecking=no "$VPS_USER@$VPS_IP" "echo OK" &>/dev/null; then
    echo -e "  ${GREEN}✓ SSH connection successful${NC}\n"
else
    echo -e "  ${RED}✗ SSH connection failed!${NC}\n"
    echo -e "${YELLOW}Troubleshooting steps:${NC}"
    echo -e "${YELLOW}1. Check if SSH key exists: ls $SSH_KEY${NC}"
    echo -e "${YELLOW}2. Check key permissions: chmod 600 $SSH_KEY${NC}"
    echo -e "${YELLOW}3. Test manual connection: ssh -i $SSH_KEY $VPS_USER@$VPS_IP${NC}"
    exit 1
fi

# Copy test script to VPS
echo -e "${YELLOW}[2/4] Copying test script to VPS...${NC}"
if scp -i "$SSH_KEY" -o StrictHostKeyChecking=no ./comprehensive_system_test.sh "$VPS_USER@$VPS_IP:/home/qt/comprehensive_system_test.sh" &>/dev/null; then
    echo -e "  ${GREEN}✓ Script copied successfully${NC}\n"
else
    echo -e "  ${RED}✗ Failed to copy script${NC}\n"
    exit 1
fi

# Make script executable
echo -e "${YELLOW}[3/4] Making script executable...${NC}"
if ssh -i "$SSH_KEY" "$VPS_USER@$VPS_IP" "chmod +x /home/qt/comprehensive_system_test.sh" &>/dev/null; then
    echo -e "  ${GREEN}✓ Script is executable${NC}\n"
else
    echo -e "  ${RED}✗ Failed to set permissions${NC}\n"
    exit 1
fi

# Run the test
echo -e "${YELLOW}[4/4] Running comprehensive system test on VPS...${NC}"
echo -e "${BLUE}═══════════════════════════════════════════════════${NC}\n"

ssh -i "$SSH_KEY" "$VPS_USER@$VPS_IP" "cd /home/qt/quantum_trader && bash /home/qt/comprehensive_system_test.sh"

echo -e "\n${BLUE}═══════════════════════════════════════════════════${NC}"
echo -e "${BLUE}  TEST COMPLETE${NC}"
echo -e "${BLUE}═══════════════════════════════════════════════════${NC}\n"
