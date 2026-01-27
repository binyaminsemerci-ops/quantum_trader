#!/bin/bash
# MetricPack Builder v1 - VPS Deployment Script
# Execute as root on 46.224.116.254

set -e

echo "=== MetricPack Builder v1 Deployment ==="
echo "Date: $(date)"
echo ""

# Step 1: Update repo
echo "[1/7] Updating repository..."
cd /home/qt/quantum_trader
git pull origin main

# Step 2: Install Python dependencies
echo "[2/7] Installing dependencies..."
su - qt -c "cd /home/qt/quantum_trader && source venv/bin/activate && pip install -r microservices/metricpack_builder/requirements.txt"

# Step 3: Deploy config
echo "[3/7] Deploying configuration..."
cp /home/qt/quantum_trader/deploy/config/metricpack-builder.env /etc/quantum/metricpack-builder.env
chown root:root /etc/quantum/metricpack-builder.env
chmod 644 /etc/quantum/metricpack-builder.env

# Step 4: Install systemd service
echo "[4/7] Installing systemd service..."
cp /home/qt/quantum_trader/deploy/systemd/quantum-metricpack-builder.service /etc/systemd/system/
systemctl daemon-reload

# Step 5: Enable and start service
echo "[5/7] Starting service..."
systemctl enable quantum-metricpack-builder
systemctl restart quantum-metricpack-builder

# Wait for service to initialize
sleep 3

# Step 6: Verify service
echo "[6/7] Verifying service..."
systemctl status quantum-metricpack-builder --no-pager || true

# Step 7: Test endpoints
echo "[7/7] Testing endpoints..."
echo ""
echo "Health endpoint:"
curl -s http://localhost:8051/health | python3 -m json.tool

echo ""
echo "Metrics endpoint (sample):"
curl -s http://localhost:8051/metrics | head -30

echo ""
echo "=== Deployment Complete ==="
echo ""
echo "Next steps:"
echo "1. Add Prometheus scrape target:"
echo "   sudo bash -c 'cat /home/qt/quantum_trader/deploy/config/prometheus_metricpack_builder.yml >> /etc/prometheus/prometheus.yml'"
echo "   sudo systemctl reload prometheus"
echo ""
echo "2. Check logs:"
echo "   journalctl -u quantum-metricpack-builder -f"
echo ""
echo "3. Verify metrics in Prometheus:"
echo "   curl -s 'http://localhost:9091/api/v1/query?query=quantum_exit_events_processed_total' | jq"
