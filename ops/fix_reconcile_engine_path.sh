#!/bin/bash
# Fix reconcile-engine service to use correct production path
# /root/quantum_trader is old; production is /home/qt/quantum_trader

set -e

echo "=== Fix quantum-reconcile-engine service path ==="

# Backup original
cp /etc/systemd/system/quantum-reconcile-engine.service \
   /etc/systemd/system/quantum-reconcile-engine.service.bak.$(date +%s)

# Write corrected service file
cat > /etc/systemd/system/quantum-reconcile-engine.service << 'SVCEOF'
[Unit]
Description=Quantum Trader - P3.4 Position Reconciliation Engine
After=network.target redis.service

[Service]
Type=simple
User=qt
Group=qt
WorkingDirectory=/home/qt/quantum_trader
EnvironmentFile=/etc/quantum/apply-layer.env
EnvironmentFile=/etc/quantum/reconcile-engine.env
Environment="PYTHONUNBUFFERED=1"
Environment="PYTHONPATH=/home/qt/quantum_trader"
ExecStart=/usr/bin/python3 /home/qt/quantum_trader/microservices/reconcile_engine/main.py
Restart=always
RestartSec=10

[Install]
WantedBy=multi-user.target
SVCEOF

echo "Service file updated."

# Reload + restart
systemctl daemon-reload
systemctl restart quantum-reconcile-engine
sleep 3
systemctl status quantum-reconcile-engine --no-pager -l | head -15

echo "Done."
