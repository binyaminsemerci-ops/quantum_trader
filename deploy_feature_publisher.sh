#!/bin/bash
# Deploy Feature Publisher Service to VPS

set -e

VPS="root@46.224.116.254"
SSH_KEY="~/.ssh/hetzner_fresh"
PROJECT_DIR="/home/qt/quantum_trader"

echo "============================================================================"
echo "Deploying Feature Publisher Service (PATH 2.3D Bridge)"
echo "============================================================================"
echo ""

# Step 1: Upload service file
echo "[1/5] Uploading feature_publisher_service.py..."
scp -i $SSH_KEY ai_engine/services/feature_publisher_service.py $VPS:$PROJECT_DIR/ai_engine/services/
echo "✅ Service file uploaded"
echo ""

# Step 2: Upload systemd service
echo "[2/5] Uploading systemd service definition..."
scp -i $SSH_KEY quantum-feature-publisher.service $VPS:/tmp/
ssh -i $SSH_KEY $VPS "sudo mv /tmp/quantum-feature-publisher.service /etc/systemd/system/"
echo "✅ Systemd service uploaded"
echo ""

# Step 3: Reload systemd
echo "[3/5] Reloading systemd..."
ssh -i $SSH_KEY $VPS "sudo systemctl daemon-reload"
echo "✅ Systemd reloaded"
echo ""

# Step 4: Enable and start service
echo "[4/5] Enabling and starting service..."
ssh -i $SSH_KEY $VPS "sudo systemctl enable quantum-feature-publisher.service"
ssh -i $SSH_KEY $VPS "sudo systemctl restart quantum-feature-publisher.service"
sleep 3
echo "✅ Service started"
echo ""

# Step 5: Verify service status
echo "[5/5] Verifying service status..."
ssh -i $SSH_KEY $VPS "systemctl status quantum-feature-publisher.service --no-pager -n 20"
echo ""

echo "============================================================================"
echo "✅ DEPLOYMENT COMPLETE"
echo "============================================================================"
echo ""
echo "Monitor service:"
echo "  systemctl status quantum-feature-publisher.service"
echo "  journalctl -u quantum-feature-publisher.service -f"
echo ""
echo "Check feature stream:"
echo "  redis-cli XLEN quantum:stream:features"
echo "  redis-cli XREVRANGE quantum:stream:features + - COUNT 5"
echo ""
echo "Expected flow:"
echo "  market.tick → feature_publisher → quantum:stream:features →"
echo "  → ensemble_predictor → quantum:stream:signal.score"
echo ""
