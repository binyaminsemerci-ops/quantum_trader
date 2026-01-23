#!/bin/bash
# P3.2 Governor Hardening - VPS Deployment Script
# Idempotent deployment with proof

set -e

echo "=== P3.2 Governor Hardening Deployment ==="
echo "Started: $(date)"

# Colors
GREEN='\033[0;32m'
RED='\033[0;31m'
YELLOW='\033[1;33m'
NC='\033[0m'

# Step 1: Pull latest code
echo -e "\n${YELLOW}Step 1: Pulling latest code...${NC}"
cd /root/quantum_trader
git pull

# Step 2: Sync to runtime directory
echo -e "\n${YELLOW}Step 2: Syncing to runtime directory...${NC}"
rsync -av --exclude='.git' --exclude='__pycache__' --exclude='*.pyc' \
    /root/quantum_trader/ /home/qt/quantum_trader/

# Step 3: Install Governor environment config
echo -e "\n${YELLOW}Step 3: Installing Governor config...${NC}"
if [ ! -f /etc/quantum/governor.env ]; then
    echo -e "${YELLOW}Creating /etc/quantum/governor.env${NC}"
    cat > /etc/quantum/governor.env <<EOF
# P3.2 Governor Service Configuration
REDIS_HOST=localhost
REDIS_PORT=6379
GOVERNOR_METRICS_PORT=8044

# Rate limits
MAX_CLOSE_PER_HOUR=3
MAX_CLOSE_PER_5MIN=2
MAX_REDUCE_NOTIONAL_PER_DAY_USD=5000
MAX_REDUCE_QTY_PER_DAY=0.02

# Kill score gates
KILL_SCORE_GATE_CRITICAL=0.8
KILL_SCORE_GATE_RISK_INCREASE=0.6

# Log level
LOG_LEVEL=INFO
EOF
else
    echo -e "${GREEN}✓ /etc/quantum/governor.env already exists${NC}"
fi

# Copy testnet credentials from Apply Layer config
if [ -f /etc/quantum/testnet.env ]; then
    echo -e "${GREEN}✓ Copying Binance testnet credentials from testnet.env${NC}"
    grep "BINANCE_TESTNET_API_KEY=" /etc/quantum/testnet.env >> /etc/quantum/governor.env || true
    grep "BINANCE_TESTNET_API_SECRET=" /etc/quantum/testnet.env >> /etc/quantum/governor.env || true
else
    echo -e "${RED}✗ WARNING: /etc/quantum/testnet.env not found - Governor will use fallback limits${NC}"
fi

# Step 4: Install systemd service
echo -e "\n${YELLOW}Step 4: Installing systemd service...${NC}"
cat > /etc/systemd/system/quantum-governor.service <<EOF
[Unit]
Description=Quantum Trading P3.2 Governor Service
After=network.target redis.service
Requires=redis.service

[Service]
Type=simple
User=root
WorkingDirectory=/home/qt/quantum_trader
EnvironmentFile=/etc/quantum/governor.env
ExecStart=/usr/bin/python3 /home/qt/quantum_trader/microservices/governor/main.py
Restart=always
RestartSec=10
StandardOutput=journal
StandardError=journal

# Resource limits
MemoryLimit=1G
CPUQuota=50%

[Install]
WantedBy=multi-user.target
EOF

# Step 5: Enable and start Governor service
echo -e "\n${YELLOW}Step 5: Starting Governor service...${NC}"
systemctl daemon-reload
systemctl enable quantum-governor
systemctl restart quantum-governor
sleep 3

# Verify Governor is running
if systemctl is-active --quiet quantum-governor; then
    echo -e "${GREEN}✓ Governor service is active${NC}"
else
    echo -e "${RED}✗ Governor service failed to start${NC}"
    journalctl -u quantum-governor -n 30 --no-pager
    exit 1
fi

# Step 6: Restart Apply Layer (to pick up any changes)
echo -e "\n${YELLOW}Step 6: Restarting Apply Layer...${NC}"
systemctl restart quantum-apply-layer
sleep 3

if systemctl is-active --quiet quantum-apply-layer; then
    echo -e "${GREEN}✓ Apply Layer service is active${NC}"
else
    echo -e "${RED}✗ Apply Layer service failed to start${NC}"
    journalctl -u quantum-apply-layer -n 30 --no-pager
    exit 1
fi

# Step 7: Run proof script
echo -e "\n${YELLOW}Step 7: Running Governor proof tests...${NC}"
bash /home/qt/quantum_trader/ops/p32_proof_governor.sh > /home/qt/quantum_trader/docs/P3_2_VPS_PROOF.txt 2>&1

# Display proof summary
echo -e "\n${GREEN}=== PROOF SUMMARY ===${NC}"
tail -50 /home/qt/quantum_trader/docs/P3_2_VPS_PROOF.txt

echo -e "\n${GREEN}=== DEPLOYMENT COMPLETE ===${NC}"
echo "Governor service: $(systemctl is-active quantum-governor)"
echo "Apply Layer service: $(systemctl is-active quantum-apply-layer)"
echo "Proof saved to: /home/qt/quantum_trader/docs/P3_2_VPS_PROOF.txt"
echo "Completed: $(date)"
