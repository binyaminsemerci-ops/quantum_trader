#!/bin/bash
#===============================================================================
# AI ENGINE WSL STARTUP SCRIPT
# Start ai_engine microservice in WSL with proper environment
#===============================================================================

set -e  # Exit on error

# Colors
GREEN='\033[0;32m'
CYAN='\033[0;36m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m' # No Color

echo -e "${CYAN}╔════════════════════════════════════════════╗${NC}"
echo -e "${CYAN}║   AI ENGINE WSL STARTUP                    ║${NC}"
echo -e "${CYAN}╚════════════════════════════════════════════╝${NC}"

# Kill existing instances
echo -e "\n${YELLOW}→ Stopping existing instances...${NC}"
pkill -9 -f "uvicorn.*ai_engine" 2>/dev/null || true
sleep 1

# Navigate to project
cd ~/quantum_trader || { echo -e "${RED}✗ Project directory not found${NC}"; exit 1; }

# Sync latest code from Windows
echo -e "${YELLOW}→ Syncing latest code from Windows...${NC}"
if [ -f "/mnt/c/quantum_trader/microservices/ai_engine/service.py" ]; then
    cp /mnt/c/quantum_trader/microservices/ai_engine/service.py ~/quantum_trader/microservices/ai_engine/service.py
    echo -e "${GREEN}✓ service.py synced${NC}"
    
    # Clean Python cache to force reload
    find ~/quantum_trader/microservices/ai_engine -name __pycache__ -type d -exec rm -rf {} + 2>/dev/null || true
    find ~/quantum_trader/microservices/ai_engine -name "*.pyc" -delete 2>/dev/null || true
    echo -e "${GREEN}✓ Python cache cleaned${NC}"
else
    echo -e "${YELLOW}⚠ Windows mount not found, using existing WSL code${NC}"
fi

# Activate venv
echo -e "${YELLOW}→ Activating Python 3.11 environment...${NC}"
source .venv/bin/activate || { echo -e "${RED}✗ Failed to activate venv${NC}"; exit 1; }

# Verify Python version
PYTHON_VERSION=$(python --version 2>&1 | awk '{print $2}')
echo -e "${GREEN}✓ Python ${PYTHON_VERSION}${NC}"

# Set environment variables
export PYTHONPATH=$HOME/quantum_trader
export REDIS_HOST=localhost
export REDIS_PORT=6379
export LOG_LEVEL=INFO

# Memory optimization for RTX 3060
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:512
export OMP_NUM_THREADS=2
export MKL_NUM_THREADS=2
export NUMEXPR_NUM_THREADS=2

echo -e "${GREEN}✓ Environment configured${NC}"

# Verify Redis is running
echo -e "${YELLOW}→ Checking Redis...${NC}"
if podman ps | grep -q quantum_redis; then
    echo -e "${GREEN}✓ Redis container running${NC}"
else
    echo -e "${RED}✗ Redis not running! Start with: podman-compose up -d redis${NC}"
    exit 1
fi

# Create logs directory
mkdir -p logs

# Start service
echo -e "${CYAN}\n→ Starting AI Engine on port 8001...${NC}"
uvicorn microservices.ai_engine.main:app \
    --host 0.0.0.0 \
    --port 8001 \
    --log-level info \
    "$@"  # Pass any additional arguments
