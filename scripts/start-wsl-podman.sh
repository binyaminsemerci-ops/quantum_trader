#!/bin/bash
# Quantum Trader - WSL Podman Startup Script
# Purpose: Start Redis + AI-Engine using podman-compose in WSL
# Created: 2025-12-16

set -e  # Exit on error

# Colors for output
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m' # No Color

echo -e "${GREEN}🚀 Starting Quantum Trader in WSL with Podman${NC}"
echo "================================================"

# 1️⃣ Verify we're in WSL
if ! grep -qi microsoft /proc/version; then
    echo -e "${RED}❌ Error: This script must run in WSL${NC}"
    exit 1
fi
echo -e "${GREEN}✅ Running in WSL${NC}"

# 2️⃣ Change to correct directory
cd ~/quantum_trader || {
    echo -e "${RED}❌ Error: ~/quantum_trader not found${NC}"
    exit 1
}
echo -e "${GREEN}✅ Working directory: $(pwd)${NC}"

# 3️⃣ Verify podman-compose is installed
if ! command -v podman-compose &> /dev/null; then
    echo -e "${RED}❌ Error: podman-compose not found${NC}"
    echo "Install with: pip3 install podman-compose"
    exit 1
fi
echo -e "${GREEN}✅ podman-compose: $(podman-compose --version)${NC}"

# 4️⃣ Verify docker-compose.wsl.yml exists
if [ ! -f docker-compose.wsl.yml ]; then
    echo -e "${RED}❌ Error: docker-compose.wsl.yml not found${NC}"
    exit 1
fi
echo -e "${GREEN}✅ Configuration file found${NC}"

# 5️⃣ Stop any existing containers
echo -e "${YELLOW}🛑 Stopping existing containers...${NC}"
podman-compose -f docker-compose.wsl.yml down 2>/dev/null || true

# 6️⃣ Start services
echo -e "${GREEN}🚀 Starting Redis + AI-Engine...${NC}"
podman-compose -f docker-compose.wsl.yml up -d redis ai-engine

# 7️⃣ Wait for services to be healthy
echo -e "${YELLOW}⏳ Waiting for services to be healthy...${NC}"
sleep 5

# 8️⃣ Verify containers are running
echo ""
echo "Container Status:"
podman ps --filter "name=quantum" --format "table {{.Names}}\t{{.Status}}\t{{.Ports}}"

echo ""
echo -e "${GREEN}✅ Services started successfully!${NC}"
echo ""
echo "Next steps:"
echo "  - Check logs: podman logs quantum_ai_engine"
echo "  - Test health: curl http://localhost:8001/health"
echo "  - View all: podman ps"
