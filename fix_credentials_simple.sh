#!/bin/bash
set -euo pipefail

echo "=== SIMPLE FIX: Use SetCredential Instead ==="
echo ""

# SystemD 247+ supports SetCredential which passes decrypted content as env vars
# But our systemd 255 supports it - let's use simpler EnvironmentFile approach

echo "1) Remove old drop-in..."
rm -f /etc/systemd/system/quantum-position-monitor.service.d/credentials.conf
echo ""

echo "2) Create environment file from decrypted credentials..."
# Decrypt directly to environment file (root-only readable)
install -d -m 700 /etc/quantum/position-monitor-secrets
systemd-creds decrypt /etc/quantum/creds/BINANCE_API_KEY.cred > /tmp/key.txt
systemd-creds decrypt /etc/quantum/creds/BINANCE_API_SECRET.cred > /tmp/secret.txt

cat > /etc/quantum/position-monitor-secrets/binance.env << EOF
BINANCE_API_KEY=$(cat /tmp/key.txt)
BINANCE_API_SECRET=$(cat /tmp/secret.txt)
EOF

shred -u /tmp/key.txt /tmp/secret.txt
chmod 600 /etc/quantum/position-monitor-secrets/binance.env
chown qt:qt /etc/quantum/position-monitor-secrets/binance.env

echo "✅ Created /etc/quantum/position-monitor-secrets/binance.env (600, qt:qt)"
ls -la /etc/quantum/position-monitor-secrets/
echo ""

echo "3) Create drop-in to load environment file..."
mkdir -p /etc/systemd/system/quantum-position-monitor.service.d
cat > /etc/systemd/system/quantum-position-monitor.service.d/credentials.conf << 'EOF'
[Service]
# Load Binance credentials from protected environment file
EnvironmentFile=-/etc/quantum/position-monitor-secrets/binance.env
EOF

echo "✅ Drop-in created:"
cat /etc/systemd/system/quantum-position-monitor.service.d/credentials.conf
echo ""

echo "4) Reload and restart..."
systemctl daemon-reload
systemctl restart quantum-position-monitor.service
sleep 5

echo ""
if systemctl is-active --quiet quantum-position-monitor.service; then
    echo "✅ Service ACTIVE"
else
    echo "❌ Service FAILED"
    systemctl status quantum-position-monitor.service --no-pager -l | tail -20
    exit 1
fi

echo ""
echo "5) Test API access..."
cd /home/qt/quantum_trader
timeout 15s python3 check_positions_tpsl.py 2>&1 | head -30

echo ""
echo "=== VERDICT ==="
if timeout 10s python3 check_positions_tpsl.py 2>&1 | grep -q "API Secret required"; then
    echo "❌ FAIL: API credentials not accessible"
else
    echo "✅ PASS: API credentials working"
    echo ""
    echo "Credentials stored in: /etc/quantum/position-monitor-secrets/binance.env (qt:qt, 600)"
    echo "Encrypted backup: /etc/quantum/creds/*.cred (root:root, 600)"
fi
