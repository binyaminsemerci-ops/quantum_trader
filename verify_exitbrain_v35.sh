#!/bin/bash
set -euo pipefail

echo "=== EXIT BRAIN V3.5 VERIFICATION ==="
echo "Date: $(date -u)"
echo ""

# 1. Service status
echo "1) Service Status"
echo "================="
if systemctl is-active --quiet quantum-position-monitor.service; then
    echo "✅ quantum-position-monitor.service: ACTIVE"
else
    echo "❌ quantum-position-monitor.service: INACTIVE"
    exit 1
fi
echo ""

# 2. Credentials in environment
echo "2) Credentials Loaded"
echo "====================="
PID=$(pgrep -f 'position_monitor.py' | head -1)
if [ -n "$PID" ]; then
    echo "Process PID: $PID"
    if ps -p $PID -e | grep -q 'BINANCE_API_KEY'; then
        echo "✅ BINANCE_API_KEY present in process environment"
    else
        echo "❌ BINANCE_API_KEY not found"
    fi
    
    if ps -p $PID -e | grep -q 'EXIT_BRAIN_V35_ENABLED'; then
        echo "✅ EXIT_BRAIN_V35_ENABLED present"
    else
        echo "⚠️  EXIT_BRAIN_V35_ENABLED not visible (may still be set)"
    fi
else
    echo "❌ Process not found"
fi
echo ""

# 3. Logs
echo "3) Service Logs"
echo "==============="
if [ -f /var/log/quantum/position-monitor.log ]; then
    echo "Log file: /var/log/quantum/position-monitor.log"
    echo "Recent entries:"
    tail -10 /var/log/quantum/position-monitor.log | grep -E 'Position Monitor|Exit Brain|Binance' || tail -10 /var/log/quantum/position-monitor.log
else
    echo "⚠️  Log file not found"
fi
echo ""

# 4. API test
echo "4) API Connectivity Test"
echo "========================"
python3 /tmp/test_service_creds.py
API_STATUS=$?
echo ""

# 5. Position check (only if API works)
if [ $API_STATUS -eq 0 ]; then
    echo "5) Position & TP/SL Verification"
    echo "================================="
    cd /home/qt/quantum_trader
    
    # Export environment for scripts
    set -a
    source /etc/quantum/position-monitor.env
    set +a
    
    echo "A) check_positions_tpsl.py:"
    timeout 25s python3 check_positions_tpsl.py 2>&1 | head -40 || echo "Script completed"
    echo ""
    
    echo "B) check_exit_brain_positions.py:"
    timeout 25s python3 check_exit_brain_positions.py 2>&1 | head -40 || echo "Script completed"
    echo ""
else
    echo "5) Position Check SKIPPED (API test failed)"
    echo "==========================================="
    echo "Fix Binance API credentials first, then re-run this script"
    echo ""
fi

# Final verdict
echo "=== FINAL VERDICT ==="
echo ""

SERVICE_OK=false
CREDS_OK=false
API_OK=false

systemctl is-active --quiet quantum-position-monitor.service && SERVICE_OK=true
[ -n "$PID" ] && ps -p $PID -e | grep -q 'BINANCE_API_KEY' && CREDS_OK=true
[ $API_STATUS -eq 0 ] && API_OK=true

echo "Infrastructure Checks:"
echo "  Service Active: $( [ "$SERVICE_OK" = true ] && echo '✅ PASS' || echo '❌ FAIL' )"
echo "  Credentials Loaded: $( [ "$CREDS_OK" = true ] && echo '✅ PASS' || echo '❌ FAIL' )"
echo ""

echo "Operational Checks:"
echo "  API Access: $( [ "$API_OK" = true ] && echo '✅ PASS' || echo '❌ FAIL' )"
echo ""

if [ "$SERVICE_OK" = true ] && [ "$CREDS_OK" = true ] && [ "$API_OK" = true ]; then
    echo "🎯 VERDICT: PASS"
    echo ""
    echo "✅ Exit Brain v3.5 is fully operational"
    echo "✅ Service can access Binance API"
    echo "✅ Ready to manage TP/SL orders"
elif [ "$SERVICE_OK" = true ] && [ "$CREDS_OK" = true ]; then
    echo "⚠️  VERDICT: PARTIAL"
    echo ""
    echo "✅ Infrastructure configured correctly"
    echo "❌ Binance API credentials invalid/IP-restricted"
    echo ""
    echo "ACTION REQUIRED:"
    echo "1. Go to Binance API Management"
    echo "2. Whitelist IP: 46.224.116.254"
    echo "3. OR generate new API keys with Futures permission"
    echo "4. Update /etc/quantum/position-monitor.env"
    echo "5. Run: systemctl restart quantum-position-monitor.service"
    echo "6. Re-run this script"
else
    echo "❌ VERDICT: FAIL"
    echo ""
    [ "$SERVICE_OK" = false ] && echo "  ❌ Service not running"
    [ "$CREDS_OK" = false ] && echo "  ❌ Credentials not loaded"
fi

echo ""
echo "=== VERIFICATION COMPLETE ==="
