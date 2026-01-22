#!/bin/bash
# Deploy P0.5 MarketState Metrics Publisher to VPS

set -e

echo "================================================================================"
echo "P0.5 MarketState Metrics Publisher — Deployment"
echo "================================================================================"
echo ""

# Step 1: Create directories
echo "STEP 1: Creating directories..."
mkdir -p /home/qt/quantum_trader/microservices/market_state_publisher
mkdir -p /etc/quantum
echo "✅ Directories created"
echo ""

# Step 2: Copy config file (if not exists)
echo "STEP 2: Setting up configuration..."
if [ ! -f /etc/quantum/marketstate.env ]; then
    cat > /etc/quantum/marketstate.env <<'EOF'
# P0.5 MarketState Metrics Publisher Configuration
MARKETSTATE_SYMBOLS=BTCUSDT,ETHUSDT,SOLUSDT
MARKETSTATE_PUBLISH_INTERVAL=60
MARKETSTATE_REDIS_HOST=localhost
MARKETSTATE_REDIS_PORT=6379
MARKETSTATE_WINDOW_SIZE=300
MARKETSTATE_SOURCE=candles
MARKETSTATE_LOG_LEVEL=INFO
EOF
    echo "✅ Config file created at /etc/quantum/marketstate.env"
else
    echo "⚠️  Config file already exists, skipping"
fi
echo ""

# Step 3: Install systemd unit file
echo "STEP 3: Installing systemd service..."
if [ -f /home/qt/quantum_trader/deployment/systemd/quantum-marketstate.service ]; then
    cp /home/qt/quantum_trader/deployment/systemd/quantum-marketstate.service /etc/systemd/system/
    echo "✅ Service file installed"
else
    echo "❌ Service file not found in repo, creating manually..."
    cat > /etc/systemd/system/quantum-marketstate.service <<'EOF'
[Unit]
Description=Quantum Trader - MarketState Metrics Publisher (P0.5)
After=network-online.target redis.service
Wants=network-online.target

[Service]
Type=simple
User=root
WorkingDirectory=/home/qt/quantum_trader
EnvironmentFile=/etc/quantum/marketstate.env
ExecStart=/opt/quantum/venvs/ai-engine/bin/python3 \
    /home/qt/quantum_trader/microservices/market_state_publisher/main.py
Restart=always
RestartSec=10
StandardOutput=journal
StandardError=journal
SyslogIdentifier=quantum-marketstate
MemoryMax=512M
CPUQuota=50%

[Install]
WantedBy=multi-user.target
EOF
fi
echo ""

# Step 4: Check Python files exist
echo "STEP 4: Verifying Python files..."
if [ ! -f /home/qt/quantum_trader/microservices/market_state_publisher/main.py ]; then
    echo "❌ main.py not found!"
    echo "Please deploy files first:"
    echo "  scp -i ~/.ssh/hetzner_fresh microservices/market_state_publisher/main.py root@46.224.116.254:/home/qt/quantum_trader/microservices/market_state_publisher/"
    exit 1
fi
echo "✅ Python files present"
echo ""

# Step 5: Reload systemd
echo "STEP 5: Reloading systemd..."
systemctl daemon-reload
echo "✅ Systemd reloaded"
echo ""

# Step 6: Enable and start service
echo "STEP 6: Starting service..."
systemctl enable quantum-marketstate.service
systemctl restart quantum-marketstate.service
echo "✅ Service started"
echo ""

# Step 7: Wait and check status
echo "STEP 7: Checking service status..."
sleep 2
systemctl status quantum-marketstate.service --no-pager || true
echo ""

# Step 8: Show recent logs
echo "STEP 8: Recent logs..."
journalctl -u quantum-marketstate -n 20 --no-pager
echo ""

echo "================================================================================"
echo "Deployment complete!"
echo "================================================================================"
echo ""
echo "Next steps:"
echo "  1. Monitor logs:    journalctl -u quantum-marketstate -f"
echo "  2. Verify metrics:  bash /home/qt/quantum_trader/ops/verify_marketstate_metrics.sh"
echo "  3. Check Redis:     redis-cli HGETALL quantum:marketstate:BTCUSDT"
echo ""
