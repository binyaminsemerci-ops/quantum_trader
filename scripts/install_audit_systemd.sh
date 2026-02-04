#!/bin/bash
# Quantum Trader V3 - Audit Logger Systemd Service Installer
# Creates systemd service for daily audit summary

cat > /tmp/quantum-audit.service <<'EOF'
[Unit]
Description=Quantum Trader V3 Daily Audit Summary
After=network.target

[Service]
Type=oneshot
User=root
WorkingDirectory=/home/qt/quantum_trader/tools
ExecStart=/usr/bin/python3 /home/qt/quantum_trader/tools/audit_logger.py daily_summary
StandardOutput=append:/var/log/quantum_audit.log
StandardError=append:/var/log/quantum_audit.log

[Install]
WantedBy=multi-user.target
EOF

cat > /tmp/quantum-audit.timer <<'EOF'
[Unit]
Description=Quantum Trader V3 Daily Audit Summary Timer
Requires=quantum-audit.service

[Timer]
OnCalendar=daily
Persistent=true
Unit=quantum-audit.service

[Install]
WantedBy=timers.target
EOF

# Install systemd files
sudo mv /tmp/quantum-audit.service /etc/systemd/system/
sudo mv /tmp/quantum-audit.timer /etc/systemd/system/
sudo chmod 644 /etc/systemd/system/quantum-audit.service
sudo chmod 644 /etc/systemd/system/quantum-audit.timer

# Reload systemd and enable timer
sudo systemctl daemon-reload
sudo systemctl enable quantum-audit.timer
sudo systemctl start quantum-audit.timer

echo "✅ Systemd timer for daily audit summary installed"
echo "   - Service: quantum-audit.service"
echo "   - Timer: quantum-audit.timer"
echo "   - Schedule: Daily at midnight"
echo ""
echo "Check timer status:"
echo "   sudo systemctl status quantum-audit.timer"
echo ""
echo "Check service status:"
echo "   sudo systemctl status quantum-audit.service"
echo ""
echo "View logs:"
echo "   sudo journalctl -u quantum-audit.service"
echo "   sudo tail -f /var/log/quantum_audit.log"
echo ""
echo "Test immediately:"
echo "   sudo systemctl start quantum-audit.service"
