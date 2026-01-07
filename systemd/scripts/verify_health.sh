#!/bin/bash
# Verify Quantum Trader system health
# Can be run as any user

set -euo pipefail

FAILED=0

check_service() {
    local service=$1
    local port=$2
    
    printf "%-35s" "$service"
    
    if systemctl is-active --quiet "$service"; then
        printf " ✅ running  "
        
        if [ -n "$port" ]; then
            if timeout 2 bash -c "cat < /dev/null > /dev/tcp/127.0.0.1/$port" 2>/dev/null; then
                printf " ✅ port $port"
            else
                printf " ⚠️ port $port unreachable"
                FAILED=1
            fi
        fi
        echo ""
    else
        printf " ❌ NOT RUNNING\n"
        FAILED=1
    fi
}

echo "🔍 Quantum Trader Health Check"
echo "=============================="
echo ""

# Core infrastructure
echo "📦 INFRASTRUCTURE:"
check_service "quantum-redis.service" "6379"
echo ""

# Model servers (CRITICAL)
echo "🧠 MODEL SERVERS:"
check_service "quantum-ai-engine.service" "8001"
check_service "quantum-rl-sizer.service" ""
check_service "quantum-strategy-ops.service" ""
echo ""

# Brain services
echo "🧠 BRAIN SERVICES:"
check_service "quantum-ceo-brain.service" "8010"
check_service "quantum-strategy-brain.service" "8011"
check_service "quantum-risk-brain.service" "8012"
echo ""

# Critical AI clients
echo "🤖 CRITICAL AI CLIENTS:"
check_service "quantum-execution.service" "8002"
check_service "quantum-exposure-balancer.service" ""
check_service "quantum-position-monitor.service" ""
check_service "quantum-trade-intent-consumer.service" ""
echo ""

# HTTP health checks
echo "🏥 HTTP HEALTH CHECKS:"
check_http() {
    local name=$1
    local port=$2
    printf "%-35s" "$name"
    
    if curl -sf "http://127.0.0.1:$port/health" -m 3 &>/dev/null; then
        echo " ✅ healthy"
    else
        echo " ❌ unhealthy"
        FAILED=1
    fi
}

check_http "ai-engine" "8001"
check_http "execution" "8002"
check_http "ceo-brain" "8010"
check_http "strategy-brain" "8011"
check_http "risk-brain" "8012"

# Redis connectivity
echo ""
echo "🔗 REDIS CONNECTIVITY:"
if redis-cli -h 127.0.0.1 -p 6379 ping &>/dev/null; then
    echo "   ✅ Redis responds to PING"
else
    echo "   ❌ Redis not responding"
    FAILED=1
fi

# Memory usage
echo ""
echo "💾 MEMORY USAGE:"
free -h | grep -E 'Mem:|Swap:'

echo ""
echo "📊 TOP 10 MEMORY CONSUMERS:"
ps aux --sort=-%mem | head -11 | awk '{printf "%-20s %6s %6s %s\n", $11, $4"%", $3"%", $2}'

# Disk usage
echo ""
echo "💿 DISK USAGE:"
df -h /opt/quantum /data/quantum /var/log/quantum 2>/dev/null || df -h /

# Check for recent crashes
echo ""
echo "🔥 RECENT CRASHES (last 1 hour):"
CRASHES=$(journalctl -u 'quantum-*' --since "1 hour ago" | grep -i "failed\|error\|crash" | wc -l)
if [ "$CRASHES" -eq 0 ]; then
    echo "   ✅ No crashes detected"
else
    echo "   ⚠️ Found $CRASHES error/crash entries in logs"
    FAILED=1
fi

# Final result
echo ""
echo "========================================"
if [ $FAILED -eq 0 ]; then
    echo "✅ ALL SYSTEMS OPERATIONAL"
    exit 0
else
    echo "❌ SOME SERVICES FAILED"
    echo ""
    echo "🔍 Check logs with:"
    echo "   journalctl -u quantum-<service> -n 50 --no-pager"
    exit 1
fi
