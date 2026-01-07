#!/bin/bash
# Create system users for all Quantum Trader services
# Run as root

set -euo pipefail

echo "👤 Creating Quantum Trader System Users"
echo "========================================"

SERVICES=(
    "quantum-redis"
    "quantum-ai-engine"
    "quantum-cross-exchange"
    "quantum-market-publisher"
    "quantum-exposure-balancer"
    "quantum-portfolio-governance"
    "quantum-meta-regime"
    "quantum-portfolio-intelligence"
    "quantum-strategic-memory"
    "quantum-strategic-evolution"
    "quantum-position-monitor"
    "quantum-trade-intent-consumer"
    "quantum-ceo-brain"
    "quantum-strategy-brain"
    "quantum-risk-brain"
    "quantum-model-federation"
    "quantum-retraining-worker"
    "quantum-universe-os"
    "quantum-pil"
    "quantum-model-supervisor"
    "quantum-rl-sizer"
    "quantum-strategy-ops"
    "quantum-rl-feedback-v2"
    "quantum-rl-monitor"
    "quantum-binance-pnl-tracker"
    "quantum-risk-safety"
    "quantum-execution"
    "quantum-clm"
    "quantum-frontend"
    "quantum-quantumfond-frontend"
    "quantum-rl-dashboard"
    "quantum-nginx-proxy"
)

for service in "${SERVICES[@]}"; do
    if ! id "$service" &>/dev/null; then
        useradd --system --no-create-home --shell /usr/sbin/nologin "$service"
        echo "✅ Created user: $service"
    else
        echo "⚠️  User exists: $service"
    fi
done

# Create runtime directory for sockets
mkdir -p /run/quantum
chown quantum-redis:quantum-redis /run/quantum
chmod 755 /run/quantum

echo ""
echo "✅ All system users created"
