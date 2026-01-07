#!/bin/bash
# verify_quantum_redis_alias.sh
# Verifies quantum_redis hostname alias configuration
# Usage: bash scripts/ops/verify_quantum_redis_alias.sh

set -euo pipefail

echo "=== quantum_redis Alias Verification ==="
echo

# Check /etc/hosts entry
echo "1. /etc/hosts Entry:"
if grep -q '^127\.0\.0\.1.*quantum_redis' /etc/hosts; then
    LINE_NUM=$(grep -n '^127\.0\.0\.1.*quantum_redis' /etc/hosts | cut -d: -f1)
    echo "   ✅ Found at line $LINE_NUM:"
    grep '^127\.0\.0\.1.*quantum_redis' /etc/hosts | sed 's/^/      /'
else
    echo "   ❌ NOT FOUND in /etc/hosts"
    echo "   Fix: Add '127.0.0.1 quantum_redis' to /etc/hosts"
    exit 1
fi
echo

# Test hostname resolution
echo "2. Hostname Resolution:"
if RESOLVED=$(getent hosts quantum_redis 2>/dev/null); then
    echo "   ✅ quantum_redis resolves:"
    echo "      $RESOLVED"
    if echo "$RESOLVED" | grep -q '^127\.0\.0\.1'; then
        echo "   ✅ Resolves to localhost (correct)"
    else
        echo "   ⚠️  Resolves to unexpected address"
    fi
else
    echo "   ❌ quantum_redis does NOT resolve"
    exit 1
fi
echo

# Test Redis connection via alias
echo "3. Redis Connection via Alias:"
if redis-cli -h quantum_redis PING 2>/dev/null | grep -q PONG; then
    echo "   ✅ Redis responds via quantum_redis hostname"
else
    echo "   ❌ Redis does not respond via quantum_redis"
    echo "   (Check if Redis is running: systemctl status redis-server)"
    exit 1
fi
echo

# Check portfolio-intelligence service
echo "4. Portfolio-Intelligence Service:"
if systemctl is-active --quiet quantum-portfolio-intelligence.service; then
    echo "   ✅ quantum-portfolio-intelligence.service is active"
    UPTIME=$(systemctl show quantum-portfolio-intelligence.service --property=ActiveEnterTimestamp --value)
    echo "      Active since: $UPTIME"
else
    echo "   ❌ quantum-portfolio-intelligence.service is NOT active"
    exit 1
fi
echo

echo "=== RESULT ==="
echo "✅ quantum_redis alias is correctly configured and working"
exit 0
