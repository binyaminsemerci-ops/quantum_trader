#!/bin/bash
# healthcheck_systemd_quantum.sh
# Performs read-only health check of quantum systemd services
# Usage: bash scripts/ops/healthcheck_systemd_quantum.sh

set -euo pipefail

echo "=== Quantum Systemd Health Check ==="
echo "Date: $(date -u '+%Y-%m-%d %H:%M:%S UTC')"
echo

# 1. Check for failed units
echo "1. Failed Units:"
FAILED_COUNT=$(systemctl --failed --no-pager --no-legend | wc -l)
if [ "$FAILED_COUNT" -eq 0 ]; then
    echo "   ✅ No failed units"
else
    echo "   ❌ $FAILED_COUNT failed units:"
    systemctl --failed --no-pager
fi
echo

# 2. Count running quantum services
echo "2. Running Quantum Services:"
RUNNING_COUNT=$(systemctl list-units 'quantum*.service' --state=running --no-pager --no-legend | wc -l)
if [ "$RUNNING_COUNT" -eq 12 ]; then
    echo "   ✅ All expected services running ($RUNNING_COUNT/12)"
else
    echo "   ⚠️  $RUNNING_COUNT services running (expected: 12)"
fi
echo

# 3. Check for bad states
echo "3. Services in Bad States:"
BAD_STATES=$(systemctl list-units 'quantum*.service' --all --no-pager --no-legend | grep -E 'activating|failed' || true)
if [ -z "$BAD_STATES" ]; then
    echo "   ✅ No services in activating/failed state"
else
    echo "   ❌ Services with issues:"
    echo "$BAD_STATES"
fi
echo

# 4. Verify quantum_redis alias
echo "4. quantum_redis Hostname:"
if getent hosts quantum_redis | grep -q 127.0.0.1; then
    echo "   ✅ quantum_redis resolves to 127.0.0.1"
else
    echo "   ❌ quantum_redis does not resolve correctly"
fi
echo

# 5. Check key services uptime
echo "5. Key Services Uptime:"
for service in quantum-backend quantum-portfolio-intelligence quantum-execution quantum-rl-monitor; do
    if systemctl is-active --quiet ${service}.service; then
        UPTIME=$(systemctl show ${service}.service --property=ActiveEnterTimestamp --value)
        echo "   ✅ ${service}: $UPTIME"
    else
        echo "   ❌ ${service}: NOT RUNNING"
    fi
done
echo

# 6. Test Redis connectivity
echo "6. Redis Connection:"
if redis-cli -h 127.0.0.1 PING 2>/dev/null | grep -q PONG; then
    echo "   ✅ Redis responding on 127.0.0.1:6379"
else
    echo "   ❌ Redis not responding"
fi
echo

# Summary
echo "=== SUMMARY ==="
if [ "$FAILED_COUNT" -eq 0 ] && [ "$RUNNING_COUNT" -eq 12 ] && [ -z "$BAD_STATES" ]; then
    echo "✅ System is healthy"
    exit 0
else
    echo "⚠️  System has issues - see details above"
    exit 1
fi
