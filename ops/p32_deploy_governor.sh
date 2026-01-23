#!/bin/bash
set -euo pipefail

# P3.2 Governor - Idempotent VPS Deployment

BOLD="\033[1m"
GREEN="\033[0;32m"
RED="\033[0;31m"
NC="\033[0m"

echo -e "${BOLD}=== P3.2 GOVERNOR DEPLOYMENT ===${NC}"
echo "Timestamp: $(date -u +"%Y-%m-%d %H:%M:%S UTC")"
echo ""

# ============================================================================
# STEP 1: SYNC CODE
# ============================================================================
echo -e "${BOLD}[1/6] Sync code from /root → /home/qt${NC}"

if ! rsync -av --delete \
    --exclude='.git' \
    --exclude='__pycache__' \
    --exclude='*.pyc' \
    /root/quantum_trader/ \
    /home/qt/quantum_trader/; then
    echo -e "${RED}❌ Failed to sync code${NC}"
    exit 1
fi

echo "✅ Code synced"
echo ""

# ============================================================================
# STEP 2: INSTALL CONFIG
# ============================================================================
echo -e "${BOLD}[2/6] Install Governor config${NC}"

if [ ! -f /etc/quantum/governor.env ]; then
    echo "Installing config for first time..."
    if ! cp /root/quantum_trader/deployment/config/governor.env /etc/quantum/governor.env; then
        echo -e "${RED}❌ Failed to install config${NC}"
        exit 1
    fi
    echo "✅ Config installed: /etc/quantum/governor.env"
else
    echo "✅ Config already exists: /etc/quantum/governor.env (not overwriting)"
fi
echo ""

# ============================================================================
# STEP 3: INSTALL SYSTEMD UNIT
# ============================================================================
echo -e "${BOLD}[3/6] Install systemd service${NC}"

if ! cp /root/quantum_trader/deployment/systemd/quantum-governor.service /etc/systemd/system/; then
    echo -e "${RED}❌ Failed to install systemd unit${NC}"
    exit 1
fi

if ! systemctl daemon-reload; then
    echo -e "${RED}❌ Failed to reload systemd${NC}"
    exit 1
fi

echo "✅ Systemd unit installed"
echo ""

# ============================================================================
# STEP 4: CHECK PYTHON DEPENDENCIES
# ============================================================================
echo -e "${BOLD}[4/6] Check Python dependencies${NC}"

MISSING_DEPS=()

if ! python3 -c "import redis" 2>/dev/null; then
    MISSING_DEPS+=("redis")
fi

if ! python3 -c "import prometheus_client" 2>/dev/null; then
    MISSING_DEPS+=("prometheus-client")
fi

if [ ${#MISSING_DEPS[@]} -gt 0 ]; then
    echo "Installing missing dependencies: ${MISSING_DEPS[*]}"
    if ! pip3 install "${MISSING_DEPS[@]}"; then
        echo -e "${RED}❌ Failed to install dependencies${NC}"
        exit 1
    fi
    echo "✅ Dependencies installed"
else
    echo "✅ All dependencies present"
fi
echo ""

# ============================================================================
# STEP 5: START/RESTART SERVICE
# ============================================================================
echo -e "${BOLD}[5/6] Start Governor service${NC}"

if systemctl is-active --quiet quantum-governor; then
    echo "Service already running, restarting..."
    if ! systemctl restart quantum-governor; then
        echo -e "${RED}❌ Failed to restart service${NC}"
        exit 1
    fi
    echo "✅ Service restarted"
else
    echo "Service not running, starting..."
    if ! systemctl start quantum-governor; then
        echo -e "${RED}❌ Failed to start service${NC}"
        exit 1
    fi
    echo "✅ Service started"
fi

if ! systemctl enable quantum-governor 2>/dev/null; then
    echo "⚠️  Warning: Could not enable service (may already be enabled)"
fi

sleep 3

if systemctl is-active --quiet quantum-governor; then
    echo "✅ Service is active"
else
    echo -e "${RED}❌ Service failed to start${NC}"
    echo "Checking logs:"
    journalctl -u quantum-governor -n 20 --no-pager
    exit 1
fi
echo ""

# ============================================================================
# STEP 6: VERIFY METRICS ENDPOINT
# ============================================================================
echo -e "${BOLD}[6/6] Verify metrics endpoint${NC}"

sleep 2

if curl -s http://localhost:8044/metrics > /dev/null 2>&1; then
    echo "✅ Metrics endpoint responding on port 8044"
    METRIC_COUNT=$(curl -s http://localhost:8044/metrics | grep -c "^quantum_govern" || echo "0")
    echo "   Found $METRIC_COUNT Governor metrics"
else
    echo -e "${RED}❌ Metrics endpoint not responding${NC}"
    exit 1
fi
echo ""

# ============================================================================
# DEPLOYMENT SUMMARY
# ============================================================================
echo -e "${BOLD}=== DEPLOYMENT COMPLETE ===${NC}"
echo -e "${GREEN}✅ P3.2 Governor deployed successfully${NC}"
echo ""
echo "Service status:"
systemctl status quantum-governor --no-pager -l | head -15
echo ""
echo "Config: /etc/quantum/governor.env"
echo "Logs: journalctl -u quantum-governor -f"
echo "Metrics: http://localhost:8044/metrics"
echo ""
echo -e "${GREEN}Governor is now enforcing rate limits and safety gates!${NC}"

exit 0
