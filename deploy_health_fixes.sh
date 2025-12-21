#!/bin/bash
# Deploy Container Health Fixes to VPS
# Fixes: Risk Safety path + Meta Regime data stream

set -e

echo "🚀 DEPLOYING CONTAINER HEALTH FIXES"
echo "===================================="
echo ""
echo "📋 Changes:"
echo "   1. Risk Safety: Fixed Dockerfile CMD path"
echo "   2. Meta Regime: Read from quantum:stream:exchange.raw"
echo "   3. Portfolio Governance: No change (0 samples is normal)"
echo ""

cd /root/quantum_trader

echo "📥 Pulling latest code..."
git pull origin main

echo ""
echo "🛑 Stopping affected services..."
docker-compose -f docker-compose.vps.yml stop risk-safety meta-regime

echo ""
echo "🔨 Rebuilding containers..."
docker-compose -f docker-compose.vps.yml build risk-safety meta-regime

echo ""
echo "🚀 Starting services..."
docker-compose -f docker-compose.vps.yml up -d risk-safety meta-regime

echo ""
echo "⏳ Waiting 15 seconds for startup..."
sleep 15

echo ""
echo "🔍 VERIFICATION"
echo "=============="

echo ""
echo "1️⃣ Risk Safety Status:"
docker ps --filter "name=quantum_risk_safety" --format "table {{.Names}}\t{{.Status}}"

echo ""
echo "2️⃣ Meta Regime Status:"
docker ps --filter "name=quantum_meta_regime" --format "table {{.Names}}\t{{.Status}}"

echo ""
echo "3️⃣ Risk Safety Logs (last 5):"
docker logs --tail 5 quantum_risk_safety 2>&1

echo ""
echo "4️⃣ Meta Regime Logs (last 5):"
docker logs --tail 5 quantum_meta_regime 2>&1

echo ""
echo "5️⃣ Checking for market data processing..."
sleep 30
REGIME_LOGS=$(docker logs --tail 10 quantum_meta_regime 2>&1)
if echo "$REGIME_LOGS" | grep -q "regime_detected\|samples"; then
    echo "✅ Meta Regime processing data successfully!"
else
    echo "⚠️  Meta Regime still warming up..."
fi

echo ""
echo "✅ DEPLOYMENT COMPLETE"
echo ""
echo "📊 Full system check:"
docker ps --format "table {{.Names}}\t{{.Status}}" | grep -E "quantum_(risk_safety|meta_regime|portfolio_governance)"
