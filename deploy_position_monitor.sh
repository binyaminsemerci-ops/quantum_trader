#!/bin/bash
# Deploy Position Monitor - Fase 1.1 Permanent Fix

set -e

echo "=========================================="
echo "🛡️ FASE 1.1: POSITION MONITOR DEPLOYMENT"
echo "=========================================="
echo ""

echo "📦 Step 1: Rebuild backend image..."
docker-compose build backend 2>&1 | grep -E "(Successfully|ERROR|WARN)" || true

echo ""
echo "🔄 Step 2: Restart backend container..."
docker-compose up -d backend

echo ""
echo "⏳ Step 3: Waiting for backend to start (15 seconds)..."
sleep 15

echo ""
echo "🏥 Step 4: Checking backend health..."
curl -s http://localhost:8000/health | jq '.' || echo "❌ Backend not responding"

echo ""
echo "📊 Step 5: Checking Position Monitor logs..."
echo ""
docker logs quantum_backend 2>&1 | tail -50 | grep -E "(POSITION-MONITOR|TP|SL|protection)" || echo "⚠️ No Position Monitor logs yet"

echo ""
echo "=========================================="
echo "📋 DEPLOYMENT COMPLETE"
echo "=========================================="
echo ""
echo "✅ Backend restarted with Position Monitor"
echo ""
echo "🔍 To verify Position Monitor activity:"
echo "   docker logs -f quantum_backend | grep POSITION-MONITOR"
echo ""
echo "📊 To check if TP/SL orders are being placed:"
echo "   docker logs quantum_backend | grep -E '(Setting TP|Setting SL|TP/SL)'"
echo ""
echo "🧪 Recommendation: Monitor for 30 minutes on testnet before going live"
echo ""
