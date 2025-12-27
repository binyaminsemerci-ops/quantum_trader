#!/bin/bash
# Phase 21 Deployment Script for VPS
# Deploys performance analytics module to production

set -e

echo "========================================="
echo "  PHASE 21 DEPLOYMENT TO VPS"
echo "========================================="

# Variables
VPS_IP="46.224.116.254"
SSH_KEY="~/.ssh/hetzner_fresh"
PROJECT_DIR="/home/qt/quantum_trader"

echo "📦 Step 1: Uploading Phase 21 files..."
scp -i $SSH_KEY \
    quantumfond_backend/routers/performance_router.py \
    quantumfond_backend/routers/export_router.py \
    quantumfond_backend/requirements.txt \
    quantumfond_backend/main.py \
    root@$VPS_IP:/tmp/phase21/

echo "✅ Files uploaded"

echo "📥 Step 2: Installing dependencies in backend container..."
ssh -i $SSH_KEY root@$VPS_IP << 'EOF'
    docker exec quantumfond_backend pip install pandas numpy weasyprint
EOF

echo "✅ Dependencies installed"

echo "🔄 Step 3: Copying files to production directory..."
ssh -i $SSH_KEY root@$VPS_IP << 'EOF'
    # Find actual backend directory
    BACKEND_DIR=$(docker inspect quantumfond_backend | grep -o '"Source": "[^"]*' | head -1 | cut -d'"' -f4)
    
    if [ -z "$BACKEND_DIR" ]; then
        echo "Backend uses built image, need to rebuild container"
        cd /home/qt/quantum_trader
        
        # Copy Phase 21 files to backend source
        cp /tmp/phase21/performance_router.py quantumfond_backend/routers/
        cp /tmp/phase21/export_router.py quantumfond_backend/routers/
        cp /tmp/phase21/requirements.txt quantumfond_backend/
        cp /tmp/phase21/main.py quantumfond_backend/
        
        # Rebuild and restart backend
        docker compose -f docker-compose.quantumfond.yml build quantumfond-backend
        docker compose -f docker-compose.quantumfond.yml up -d quantumfond-backend
    else
        # Volume-mounted, copy directly
        cp /tmp/phase21/* "$BACKEND_DIR/"
        docker restart quantumfond_backend
    fi
EOF

echo "✅ Production deployment complete"

echo "🧪 Step 4: Verifying deployment..."
sleep 5

# Test endpoints
echo "Testing /performance/metrics..."
curl -s http://$VPS_IP:8026/performance/metrics | jq '.metrics' || echo "⚠️ Metrics endpoint not responding"

echo "Testing /reports/export/json..."
curl -s -I http://$VPS_IP:8026/reports/export/json | grep "HTTP" || echo "⚠️ Export endpoint not responding"

echo ""
echo "========================================="
echo "  PHASE 21 DEPLOYMENT COMPLETE"
echo "========================================="
echo ""
echo "📍 Access points:"
echo "   Metrics: http://$VPS_IP:8026/performance/metrics"
echo "   Export:  http://$VPS_IP:8026/reports/export/{json|csv|pdf}"
echo ""
echo "📊 Frontend: Update app.quantumfond.com with performance_enhanced.tsx"
echo ""
