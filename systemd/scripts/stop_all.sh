#!/bin/bash
# Stop all Quantum Trader services
# Run as root

set -euo pipefail

echo "🛑 Stopping Quantum Trader Services"
echo "===================================="
echo ""

# Stop target (cascade stops all)
echo "🔴 Stopping all services via target..."
systemctl stop quantum-trader.target

# Force stop critical services
echo "🔴 Force stopping model servers..."
systemctl stop quantum-ai-engine.service || true
systemctl stop quantum-rl-sizer.service || true
systemctl stop quantum-strategy-ops.service || true

echo "🔴 Force stopping Redis..."
systemctl stop quantum-redis.service || true

# Wait for clean shutdown
sleep 3

# Verify all stopped
echo ""
echo "📊 Final status:"
systemctl list-units 'quantum-*' --all --no-pager | grep quantum

echo ""
echo "✅ All services stopped"
