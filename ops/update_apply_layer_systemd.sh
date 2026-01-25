#!/bin/bash
# Update Apply Layer systemd unit with hardening

SERVICE_FILE="/etc/systemd/system/quantum-apply-layer.service"

cat > "$SERVICE_FILE" << 'EOF'
[Unit]
Description=Quantum Trader - Apply Layer (P3)
After=network.target redis.service
Wants=redis.service

[Service]
Type=simple
User=qt
Group=qt
WorkingDirectory=/home/qt/quantum_trader
EnvironmentFile=/etc/quantum/apply-layer.env
Environment=PYTHONUNBUFFERED=1

# Free port 8043 before starting (prevents "address already in use")
ExecStartPre=/bin/sh -c 'fuser -k 8043/tcp 2>/dev/null || true'

ExecStart=/usr/bin/python3 -u microservices/apply_layer/main.py

# Restart behavior
Restart=always
RestartSec=2

# Ensure all child processes are killed on stop
KillMode=control-group
TimeoutStopSec=15
KillSignal=SIGTERM

# Logging
StandardOutput=journal
StandardError=journal
SyslogIdentifier=quantum-apply-layer

[Install]
WantedBy=multi-user.target
EOF

echo "✓ Updated $SERVICE_FILE"
echo
echo "Reloading systemd..."
systemctl daemon-reload

echo "✓ Systemd reloaded"
echo
echo "Restart service to apply changes:"
echo "  systemctl restart quantum-apply-layer"
