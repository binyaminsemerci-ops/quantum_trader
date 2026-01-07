#!/bin/bash
# Quantum Trader Systemd Health Check
# Usage: ./healthcheck.sh

set -e

echo "========================================="
echo "QUANTUM TRADER SYSTEMD HEALTH CHECK"
echo "Date: $(date '+%Y-%m-%d %H:%M:%S')"
echo "========================================="
echo

# 1. Failed units
echo "1. FAILED UNITS:"
if systemctl --failed --no-pager | grep -q "0 loaded units listed"; then
    echo "   ✅ No failed units"
else
    echo "   ❌ Failed units detected:"
    systemctl --failed --no-pager
fi
echo

# 2. Running count
echo "2. RUNNING QUANTUM SERVICES:"
RUNNING_COUNT=$(systemctl list-units "quantum*.service" --state=running --no-legend | wc -l)
echo "   Count: $RUNNING_COUNT/12 expected"
if [ "$RUNNING_COUNT" -eq 12 ]; then
    echo "   ✅ All expected services running"
else
    echo "   ⚠️  Expected 12, got $RUNNING_COUNT"
    echo "   Running services:"
    systemctl list-units "quantum*.service" --state=running --no-legend | awk '{print "     - " $1}'
fi
echo

# 3. Bad states check
echo "3. BAD STATES CHECK (activating/failed):"
BAD_STATES=$(systemctl list-units "quantum*.service" --all --no-pager | grep -E "activating|failed" | grep -v "not-found inactive dead" || true)
if [ -z "$BAD_STATES" ]; then
    echo "   ✅ No bad states (activating/failed)"
else
    echo "   ❌ Bad states detected:"
    echo "$BAD_STATES" | sed 's/^/     /'
fi
echo

# 4. Quantum_redis hostname
echo "4. QUANTUM_REDIS HOSTNAME:"
if getent hosts quantum_redis | grep -q "127.0.0.1"; then
    echo "   ✅ quantum_redis resolves to 127.0.0.1"
else
    echo "   ❌ quantum_redis not resolving correctly"
    echo "   Add to /etc/hosts: 127.0.0.1 quantum_redis"
fi
echo

# 5. Key service status
echo "5. KEY SERVICES STATUS:"
KEY_SERVICES=(
    "quantum-execution"
    "quantum-portfolio-intelligence"
    "quantum-rl-monitor"
    "quantum-strategy-brain"
)

for service in "${KEY_SERVICES[@]}"; do
    if systemctl is-active --quiet "$service.service"; then
        UPTIME=$(systemctl show "$service.service" -p ActiveEnterTimestamp --value)
        echo "   ✅ $service (up since: $(echo $UPTIME | awk '{print $2, $3}'))"
    else
        echo "   ❌ $service (not running)"
    fi
done
echo

# 6. Redis connectivity
echo "6. REDIS CONNECTIVITY:"
if redis-cli -h 127.0.0.1 -p 6379 PING 2>/dev/null | grep -q "PONG"; then
    echo "   ✅ Redis responding on 127.0.0.1:6379"
else
    echo "   ❌ Redis not responding"
    echo "   Check: systemctl status redis-server.service"
fi
echo

# Summary
echo "========================================="
echo "SUMMARY:"
if [ "$RUNNING_COUNT" -eq 12 ] && [ -z "$BAD_STATES" ]; then
    echo "✅ System is healthy"
    exit 0
else
    echo "⚠️  Issues detected - review output above"
    exit 1
fi
