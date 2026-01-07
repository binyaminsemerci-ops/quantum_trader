#!/bin/bash
# Start all Quantum Trader services in correct order
# Run as root

set -euo pipefail

echo "🚦 Starting Quantum Trader Services"
echo "===================================="
echo ""

# STEP 1: Start Redis
echo "🔴 [1/6] Starting Redis..."
systemctl start quantum-redis.service
sleep 3

if systemctl is-active --quiet quantum-redis.service; then
    echo "   ✅ Redis is running"
else
    echo "   ❌ Redis failed to start"
    journalctl -u quantum-redis.service -n 20 --no-pager
    exit 1
fi

# Test Redis connectivity
if redis-cli -h 127.0.0.1 -p 6379 ping &>/dev/null; then
    echo "   ✅ Redis connectivity verified"
else
    echo "   ❌ Redis not responding"
    exit 1
fi

# STEP 2: Start Brains (lightweight, fast)
echo ""
echo "🧠 [2/6] Starting Brain services..."
systemctl start quantum-ceo-brain.service
systemctl start quantum-strategy-brain.service
systemctl start quantum-risk-brain.service
sleep 5

for brain in ceo-brain strategy-brain risk-brain; do
    if systemctl is-active --quiet quantum-$brain.service; then
        echo "   ✅ $brain running"
    else
        echo "   ⚠️ $brain not running (may be optional)"
    fi
done

# STEP 3: Start AI Engine (MODEL SERVER - slow)
echo ""
echo "🤖 [3/6] Starting AI Engine (Model Server)..."
echo "   ⏳ This may take 30-60 seconds (loading ML models)..."
systemctl start quantum-ai-engine.service
sleep 20

if systemctl is-active --quiet quantum-ai-engine.service; then
    echo "   ✅ AI Engine is running"
    
    # Wait for health check
    for i in {1..30}; do
        if curl -sf http://127.0.0.1:8001/health &>/dev/null; then
            echo "   ✅ AI Engine health check passed"
            break
        fi
        if [ $i -eq 30 ]; then
            echo "   ⚠️ AI Engine health check timeout (may still be loading models)"
        fi
        sleep 2
    done
else
    echo "   ❌ AI Engine failed to start"
    journalctl -u quantum-ai-engine.service -n 30 --no-pager
    exit 1
fi

# STEP 4: Start other model servers
echo ""
echo "🎯 [4/6] Starting RL Sizing and Strategy Ops..."
systemctl start quantum-rl-sizer.service
systemctl start quantum-strategy-ops.service
sleep 5

for svc in rl-sizer strategy-ops; do
    if systemctl is-active --quiet quantum-$svc.service; then
        echo "   ✅ $svc running"
    else
        echo "   ⚠️ $svc not running"
    fi
done

# STEP 5: Start execution service
echo ""
echo "⚙️ [5/6] Starting Execution service..."
systemctl start quantum-execution.service
sleep 5

if systemctl is-active --quiet quantum-execution.service; then
    echo "   ✅ Execution service running"
else
    echo "   ⚠️ Execution service not running"
fi

# STEP 6: Start remaining services via target
echo ""
echo "🌐 [6/6] Starting all remaining services..."
systemctl start quantum-trader.target
sleep 10

echo ""
echo "✅ All services started!"
echo ""
echo "📊 Service status:"
systemctl list-units 'quantum-*' --no-pager | grep -E 'quantum-|UNIT'

echo ""
echo "🔍 Run './verify_health.sh' to verify full system health"
