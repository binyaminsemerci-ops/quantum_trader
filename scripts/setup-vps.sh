#!/bin/bash
# Quantum Trader - VPS Setup Script
# Purpose: Initial setup on fresh VPS
# Platform: Ubuntu 22.04+ / Debian 11+
# Created: 2025-12-16

set -e

# Colors
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m'

echo -e "${GREEN}🚀 Quantum Trader - VPS Setup${NC}"
echo "================================"

# 1️⃣ System Update
echo -e "${YELLOW}1️⃣ Updating system...${NC}"
sudo apt update
sudo apt upgrade -y

# 2️⃣ Install Podman
echo -e "${YELLOW}2️⃣ Installing Podman...${NC}"
sudo apt install -y podman

# 3️⃣ Install Python and pip
echo -e "${YELLOW}3️⃣ Installing Python tools...${NC}"
sudo apt install -y python3 python3-pip python3-venv

# 4️⃣ Install podman-compose
echo -e "${YELLOW}4️⃣ Installing podman-compose...${NC}"
pip3 install podman-compose

# 5️⃣ Install essential tools
echo -e "${YELLOW}5️⃣ Installing essential tools...${NC}"
sudo apt install -y git curl jq htop

# 6️⃣ Setup firewall (UFW)
echo -e "${YELLOW}6️⃣ Configuring firewall...${NC}"
sudo apt install -y ufw
sudo ufw allow ssh
sudo ufw allow 8000/tcp  # Backend API
sudo ufw allow 8001/tcp  # AI Engine API
sudo ufw --force enable

# 7️⃣ Verify installations
echo ""
echo -e "${GREEN}✅ Verifying installations:${NC}"
echo "Podman: $(podman --version)"
echo "Python: $(python3 --version)"
echo "pip3: $(pip3 --version)"
echo "podman-compose: $(podman-compose --version 2>/dev/null || echo 'installed')"

echo ""
echo -e "${GREEN}✅ VPS Setup Complete!${NC}"
echo ""
echo "Next steps:"
echo "  1. Clone repo: git clone https://github.com/binyaminsemerci-ops/quantum_trader.git ~/quantum_trader"
echo "  2. Add .env file with API credentials"
echo "  3. Start services: cd ~/quantum_trader && ./scripts/start-wsl-podman.sh"
