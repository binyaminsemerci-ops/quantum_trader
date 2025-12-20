#!/bin/bash
# Quantum Trader V3 - User-Level Cron Job Installer (No Sudo Required)
# Installs cron job in user's crontab to send daily audit summaries

# Add cron job to user's crontab
(crontab -l 2>/dev/null | grep -v "quantum_trader/tools/audit_logger.py"; echo "0 0 * * * /usr/bin/python3 /home/qt/quantum_trader/tools/audit_logger.py daily_summary >> /home/qt/quantum_trader/status/audit_summary.log 2>&1") | crontab -

echo "✅ Daily audit summary cron job installed (user-level)"
echo "   - Runs at: 00:00 UTC daily"
echo "   - Script: /home/qt/quantum_trader/tools/audit_logger.py daily_summary"
echo "   - Log file: /home/qt/quantum_trader/status/audit_summary.log"
echo ""
echo "To verify installation:"
echo "   crontab -l | grep audit_logger"
echo ""
echo "To test immediately:"
echo "   python3 /home/qt/quantum_trader/tools/audit_logger.py daily_summary"
echo ""
echo "To view cron log:"
echo "   tail -f /home/qt/quantum_trader/status/audit_summary.log"
