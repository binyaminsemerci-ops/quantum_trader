#!/bin/bash
# 🚀 Quick Deploy Dashboard on VPS
# Run this script DIRECTLY on the VPS server
# ssh root@46.224.116.254
# cd /root/quantum_trader
# bash deploy_dashboard_vps.sh

set -e

echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "🚀 DEPLOYING DASHBOARD ON VPS"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

cd /root/quantum_trader

# Stop existing dashboard containers
echo ""
echo "🛑 Stopping existing dashboard containers..."
docker compose --profile dashboard down 2>/dev/null || true

# Build dashboard images
echo ""
echo "🔨 Building dashboard backend..."
docker compose --profile dashboard build dashboard-backend

echo ""
echo "🔨 Building dashboard frontend..."
docker compose --profile dashboard build dashboard-frontend

# Start dashboard services
echo ""
echo "🚀 Starting dashboard services..."
docker compose --profile dashboard up -d dashboard-backend dashboard-frontend

# Wait for services
echo ""
echo "⏳ Waiting for services to start (15 seconds)..."
sleep 15

# Check status
echo ""
echo "🏥 Dashboard Status:"
docker compose --profile dashboard ps

# Test backend
echo ""
echo "🧪 Testing backend health..."
curl -f http://localhost:8025/health && echo "" || echo "⚠️ Backend not responding"

# Test frontend
echo ""
echo "🧪 Testing frontend..."
curl -f http://localhost:8888 -I | head -n 1 || echo "⚠️ Frontend not responding"

echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "✅ DASHBOARD DEPLOYED!"
echo ""
echo "🌐 Access URLs:"
echo "   Backend:  http://46.224.116.254:8025/health"
echo "   Frontend: http://46.224.116.254:8888"
echo "   API Docs: http://46.224.116.254:8025/docs"
echo ""
echo "🔍 View logs:"
echo "   docker compose --profile dashboard logs -f dashboard-backend"
echo "   docker compose --profile dashboard logs -f dashboard-frontend"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
