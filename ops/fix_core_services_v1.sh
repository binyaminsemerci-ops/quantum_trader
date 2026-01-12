#!/bin/bash
# ============================================================================
# Tier 1 Core Loop Deployment Script
# ============================================================================
# Deploys:
# - Risk Safety Service (port 8003)
# - Execution Service (port 8002)
# - Position Monitor (port 8004)
#
# Requirements:
# - Redis running on localhost:6379
# - Python 3.10+ with virtualenv at /opt/quantum/venvs/ai-engine
# - Existing GovernerAgent in ai_engine/agents/
#
# Usage:
#   sudo bash ops/fix_core_services_v1.sh
#
# Author: Quantum Trader Team
# Date: 2026-01-12
# ============================================================================

set -e  # Exit on error

# Colors
GREEN='\033[0;32m'
BLUE='\033[0;34m'
RED='\033[0;31m'
YELLOW='\033[1;33m'
NC='\033[0m'  # No Color

echo -e "${BLUE}============================================${NC}"
echo -e "${BLUE}  TIER 1 CORE LOOP DEPLOYMENT${NC}"
echo -e "${BLUE}============================================${NC}"

# ============================================================================
# STEP 1: INSTALL DEPENDENCIES
# ============================================================================

echo -e "\n${YELLOW}[1/8]${NC} Installing Python dependencies..."

cd /home/qt/quantum_trader
source /opt/quantum/venvs/ai-engine/bin/activate

pip install --quiet --upgrade redis fastapi uvicorn pydantic aiofiles

echo -e "${GREEN}✅ Dependencies installed${NC}"

# ============================================================================
# STEP 2: CREATE LOG DIRECTORY
# ============================================================================

echo -e "\n${YELLOW}[2/8]${NC} Setting up log directory..."

mkdir -p /var/log/quantum
chown -R qt:qt /var/log/quantum

echo -e "${GREEN}✅ Log directory ready${NC}"

# ============================================================================
# STEP 3: CLEAR PYTHON CACHE
# ============================================================================

echo -e "\n${YELLOW}[3/8]${NC} Clearing Python cache..."

find /home/qt/quantum_trader -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true

echo -e "${GREEN}✅ Cache cleared${NC}"

# ============================================================================
# STEP 4: CREATE SYSTEMD SERVICES
# ============================================================================

echo -e "\n${YELLOW}[4/8]${NC} Creating systemd service files..."

# Risk Safety Service
cat > /etc/systemd/system/quantum-risk-safety.service << 'EOFSERVICE'
[Unit]
Description=Quantum Trader - Risk Safety Service
Documentation=file:///home/qt/quantum_trader/AI_V5_ARCHITECTURE.md
After=network.target redis.service
Wants=redis.service

[Service]
Type=simple
User=qt
Group=qt
WorkingDirectory=/home/qt/quantum_trader
Environment="PYTHONPATH=/home/qt/quantum_trader"
Environment="REDIS_URL=redis://localhost:6379"
Environment="SERVICE_PORT=8003"
ExecStart=/opt/quantum/venvs/ai-engine/bin/python3 services/risk_safety_service.py
Restart=always
RestartSec=10
StandardOutput=append:/var/log/quantum/risk-safety.log
StandardError=append:/var/log/quantum/risk-safety.log

# Security
NoNewPrivileges=true
PrivateTmp=true

[Install]
WantedBy=multi-user.target
EOFSERVICE

# Execution Service
cat > /etc/systemd/system/quantum-execution.service << 'EOFSERVICE'
[Unit]
Description=Quantum Trader - Execution Service
Documentation=file:///home/qt/quantum_trader/AI_V5_ARCHITECTURE.md
After=network.target redis.service
Wants=redis.service

[Service]
Type=simple
User=qt
Group=qt
WorkingDirectory=/home/qt/quantum_trader
Environment="PYTHONPATH=/home/qt/quantum_trader"
Environment="REDIS_URL=redis://localhost:6379"
Environment="SERVICE_PORT=8002"
ExecStart=/opt/quantum/venvs/ai-engine/bin/python3 services/execution_service.py
Restart=always
RestartSec=10
StandardOutput=append:/var/log/quantum/execution.log
StandardError=append:/var/log/quantum/execution.log

# Security
NoNewPrivileges=true
PrivateTmp=true

[Install]
WantedBy=multi-user.target
EOFSERVICE

# Position Monitor Service
cat > /etc/systemd/system/quantum-position-monitor.service << 'EOFSERVICE'
[Unit]
Description=Quantum Trader - Position Monitor
Documentation=file:///home/qt/quantum_trader/AI_V5_ARCHITECTURE.md
After=network.target redis.service
Wants=redis.service

[Service]
Type=simple
User=qt
Group=qt
WorkingDirectory=/home/qt/quantum_trader
Environment="PYTHONPATH=/home/qt/quantum_trader"
Environment="REDIS_URL=redis://localhost:6379"
Environment="SERVICE_PORT=8004"
ExecStart=/opt/quantum/venvs/ai-engine/bin/python3 services/position_monitor.py
Restart=always
RestartSec=10
StandardOutput=append:/var/log/quantum/position-monitor.log
StandardError=append:/var/log/quantum/position-monitor.log

# Security
NoNewPrivileges=true
PrivateTmp=true

[Install]
WantedBy=multi-user.target
EOFSERVICE

echo -e "${GREEN}✅ Service files created${NC}"

# ============================================================================
# STEP 5: RELOAD SYSTEMD
# ============================================================================

echo -e "\n${YELLOW}[5/8]${NC} Reloading systemd daemon..."

systemctl daemon-reload

echo -e "${GREEN}✅ Systemd reloaded${NC}"

# ============================================================================
# STEP 6: ENABLE SERVICES
# ============================================================================

echo -e "\n${YELLOW}[6/8]${NC} Enabling services..."

systemctl enable quantum-risk-safety.service
systemctl enable quantum-execution.service
systemctl enable quantum-position-monitor.service

echo -e "${GREEN}✅ Services enabled${NC}"

# ============================================================================
# STEP 7: START SERVICES
# ============================================================================

echo -e "\n${YELLOW}[7/8]${NC} Starting services..."

# Start in order (risk → execution → monitor)
systemctl restart quantum-risk-safety.service
sleep 3

systemctl restart quantum-execution.service
sleep 3

systemctl restart quantum-position-monitor.service
sleep 5

echo -e "${GREEN}✅ Services started${NC}"

# ============================================================================
# STEP 8: VALIDATE DEPLOYMENT
# ============================================================================

echo -e "\n${YELLOW}[8/8]${NC} Validating deployment..."

# Check services
SERVICES=("quantum-risk-safety" "quantum-execution" "quantum-position-monitor")
ALL_OK=true

for service in "${SERVICES[@]}"; do
    if systemctl is-active --quiet "$service.service"; then
        echo -e "  ${GREEN}✅${NC} $service: ACTIVE"
    else
        echo -e "  ${RED}❌${NC} $service: INACTIVE"
        ALL_OK=false
    fi
done

# Check ports
echo ""
echo "Port status:"
for port in 8002 8003 8004; do
    if netstat -tuln | grep -q ":$port "; then
        echo -e "  ${GREEN}✅${NC} Port $port: LISTENING"
    else
        echo -e "  ${YELLOW}⚠${NC}  Port $port: NOT LISTENING"
    fi
done

# ============================================================================
# FINAL REPORT
# ============================================================================

echo ""
echo -e "${BLUE}============================================${NC}"

if [ "$ALL_OK" = true ]; then
    echo -e "${GREEN}✅ DEPLOYMENT SUCCESSFUL ✅${NC}"
    echo ""
    echo "Services running:"
    echo "  • Risk Safety:    http://localhost:8003/health"
    echo "  • Execution:      http://localhost:8002/health"
    echo "  • Position Monitor: http://localhost:8004/health"
    echo ""
    echo "Logs:"
    echo "  • tail -f /var/log/quantum/risk-safety.log"
    echo "  • tail -f /var/log/quantum/execution.log"
    echo "  • tail -f /var/log/quantum/position-monitor.log"
    echo ""
    echo "Next steps:"
    echo "  1. Run validation: python3 ops/validate_core_loop.py"
    echo "  2. Monitor logs for errors"
    echo "  3. Test integration: python3 -m pytest tests/test_core_loop.py"
else
    echo -e "${RED}❌ DEPLOYMENT FAILED ❌${NC}"
    echo ""
    echo "Check logs for errors:"
    echo "  • sudo journalctl -u quantum-risk-safety.service -n 50"
    echo "  • sudo journalctl -u quantum-execution.service -n 50"
    echo "  • sudo journalctl -u quantum-position-monitor.service -n 50"
fi

echo -e "${BLUE}============================================${NC}"

# Return exit code
if [ "$ALL_OK" = true ]; then
    exit 0
else
    exit 1
fi
