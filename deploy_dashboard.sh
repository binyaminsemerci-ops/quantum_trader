#!/bin/bash
# 🚀 Deploy Quantum Trader Dashboard to VPS
# Usage: ./deploy_dashboard.sh

set -e

echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "🚀 DEPLOYING QUANTUM TRADER DASHBOARD TO VPS"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

VPS_HOST="root@46.224.116.254"
VPS_PATH="/root/quantum_trader"
LOCAL_PATH="."

echo ""
echo "📦 Step 1: Syncing dashboard code to VPS..."
rsync -avz --progress \
    --exclude 'node_modules' \
    --exclude '.venv' \
    --exclude '__pycache__' \
    --exclude '*.pyc' \
    --exclude '.git' \
    --exclude 'dist' \
    dashboard_v4/ ${VPS_HOST}:${VPS_PATH}/dashboard_v4/

echo ""
echo "📦 Step 2: Syncing systemctl.yml..."
rsync -avz systemctl.yml ${VPS_HOST}:${VPS_PATH}/

echo ""
echo "🔧 Step 3: Building and starting dashboard on VPS..."
ssh ${VPS_HOST} << 'ENDSSH'
cd /root/quantum_trader

# Stop existing dashboard containers if any
echo "🛑 Stopping existing dashboard containers..."
systemctl --profile dashboard down || true

# Build and start dashboard
echo "🔨 Building dashboard images..."
systemctl --profile dashboard build dashboard-backend dashboard-frontend

echo "🚀 Starting dashboard services..."
systemctl --profile dashboard up -d dashboard-backend dashboard-frontend

# Wait for services to be healthy
echo "⏳ Waiting for services to be healthy..."
sleep 10

# Check health
echo ""
echo "🏥 Checking dashboard health..."
systemctl --profile dashboard ps

# Test backend endpoint
echo ""
echo "🧪 Testing backend health..."
curl -f http://localhost:8025/health || echo "⚠️ Backend not responding yet"

echo ""
echo "✅ Dashboard deployment complete!"
echo ""
echo "🌐 Access your dashboard at:"
echo "   - Backend:  http://46.224.116.254:8025"
echo "   - Frontend: http://46.224.116.254:8888"
echo "   - Domain:   http://quantumtrader.com:8888 (when DNS propagates)"

ENDSSH

echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "✅ DEPLOYMENT COMPLETE!"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

