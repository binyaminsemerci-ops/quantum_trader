#!/bin/bash
# Activate ExitBrain v3.5 on VPS - SYSTEMD VERSION

set -e

echo "🚀 ExitBrain v3.5 Activation Script (Systemd)"
echo "=============================================="
echo ""

# 1️⃣ Enable ExitBrain v3.5 via environment variable
echo "📝 Step 1: Enabling ExitBrain v3.5..."
ENV_FILE="/opt/quantum/.env"
if ! grep -q "EXIT_BRAIN_V35_ENABLED" "$ENV_FILE"; then
    echo "Adding EXIT_BRAIN_V35_ENABLED=true to $ENV_FILE..."
    echo "EXIT_BRAIN_V35_ENABLED=true" >> "$ENV_FILE"
    echo "✅ Environment variable added"
else
    echo "✅ EXIT_BRAIN_V35_ENABLED already in $ENV_FILE"
fi

# 2️⃣ Reload systemd configuration
echo ""
echo "🔧 Step 2: Reloading systemd configuration..."
systemctl daemon-reload

# 3️⃣ Restart position-monitor
echo ""
echo "🔄 Step 3: Restarting position-monitor service..."
systemctl restart quantum-position-monitor.service

# 4️⃣ Wait for startup
echo ""
echo "⏳ Step 4: Waiting for service startup..."
sleep 5

# 5️⃣ Validate ExitBrain v3.5 activation
echo ""
echo "🔍 Step 5: Validating ExitBrain v3.5..."
echo "========================================"

# Check for v3.5 initialization logs
echo ""
echo "📊 Checking initialization logs..."
if journalctl -u quantum-position-monitor.service -n 50 --no-pager | grep -q "EXIT_BRAIN_V3.5.*ACTIVE"; then
    echo "✅ ExitBrain v3.5 ACTIVE confirmed!"
    journalctl -u quantum-position-monitor.service -n 50 --no-pager | grep "EXIT_BRAIN_V3.5"
else
    echo "⚠️  ExitBrain v3.5 not found in logs - checking availability..."
    journalctl -u quantum-position-monitor.service -n 100 --no-pager | grep -i "exit.*brain" | head -20
fi

# Check service status
echo ""
echo "📦 Service status:"
systemctl status quantum-position-monitor.service --no-pager | head -10

echo ""
echo "✅ Deployment complete!"
echo ""
echo "To verify ExitBrain v3.5 is processing positions:"
echo "  journalctl -u quantum-position-monitor.service -f | grep EXIT_BRAIN_V3.5"
echo ""
echo "To test with injected position:"
echo "  bash test_exitbrain_v3_5_integration.sh"
