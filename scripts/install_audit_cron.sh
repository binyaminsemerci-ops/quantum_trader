#!/bin/bash
# Quantum Trader V3 - Daily Audit Summary Cron Job Installer
# Installs cron job to send daily audit summaries at midnight UTC

cat > /tmp/quantum_audit_summary <<'EOF'
# Quantum Trader V3 - Daily Audit Summary
# Runs at midnight UTC every day
0 0 * * * root /usr/bin/python3 /home/qt/quantum_trader/tools/audit_logger.py daily_summary >> /var/log/quantum_audit.log 2>&1
EOF

# Install cron job
sudo mv /tmp/quantum_audit_summary /etc/cron.d/quantum_audit_summary
sudo chmod 644 /etc/cron.d/quantum_audit_summary
sudo chown root:root /etc/cron.d/quantum_audit_summary

# Restart cron to apply changes
sudo systemctl restart cron 2>/dev/null || sudo service cron restart 2>/dev/null

echo "✅ Daily audit summary cron job installed"
echo "   - Runs at: 00:00 UTC daily"
echo "   - Script: /home/qt/quantum_trader/tools/audit_logger.py daily_summary"
echo "   - Log file: /var/log/quantum_audit.log"
echo ""
echo "To test immediately:"
echo "   sudo python3 /home/qt/quantum_trader/tools/audit_logger.py daily_summary"
echo ""
echo "To view cron log:"
echo "   sudo tail -f /var/log/quantum_audit.log"
