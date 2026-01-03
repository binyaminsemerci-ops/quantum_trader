#!/bin/bash
set -euo pipefail

# =============================================================================
# Deploy Grafana Integration to quantumfond.com Dashboard
# =============================================================================

echo "========================================="
echo "🚀 Deploying Grafana to quantumfond.com"
echo "========================================="

VPS_USER="root"
VPS_HOST="46.224.116.254"
VPS_KEY="~/.ssh/hetzner_fresh"
VPS_DIR="/home/qt/quantum_trader"

# Step 1: Push code to GitHub
echo ""
echo "📤 Step 1: Committing and pushing changes..."
git add docker-compose.monitoring.yml nginx/app.quantumfond.com.conf
git commit -m "feat: Integrate Grafana into quantumfond.com dashboard at /grafana path

- Configure Grafana for iframe embedding with anonymous viewer access
- Add reverse proxy at app.quantumfond.com/grafana
- Enable WebSocket support for live metrics updates
- Set root URL to https://app.quantumfond.com/grafana
- Allow embedding for dashboard integration
- Add 3 tabs to System page: Overview, Performance Metrics, Log Analysis
- Embed P1-C Performance Baseline dashboard (800px iframe)
- Embed P1-B Log Analysis dashboard (800px iframe)
- Add navigation links to open full Grafana interface

Access:
- Dashboard: https://app.quantumfond.com (System tab → Performance Metrics / Log Analysis)
- Admin: https://app.quantumfond.com/grafana (admin/quantum2026secure)
- Viewer: Anonymous access enabled for embedded dashboards
"
git push origin main

# Step 2: Pull on VPS
echo ""
echo "📥 Step 2: Pulling latest code on VPS..."
ssh -i $VPS_KEY $VPS_USER@$VPS_HOST << 'EOF'
cd /home/qt/quantum_trader
git pull origin main
EOF

# Step 3: Rebuild Grafana container
echo ""
echo "🔨 Step 3: Rebuilding Grafana with new config..."
ssh -i $VPS_KEY $VPS_USER@$VPS_HOST << 'EOF'
cd /home/qt/quantum_trader
docker compose -f docker-compose.monitoring.yml build grafana
EOF

# Step 4: Restart Grafana
echo ""
echo "♻️  Step 4: Restarting Grafana service..."
ssh -i $VPS_KEY $VPS_USER@$VPS_HOST << 'EOF'
cd /home/qt/quantum_trader
docker compose -f docker-compose.monitoring.yml up -d grafana
sleep 5
EOF

# Step 4.5: Rebuild and restart dashboard frontend
echo ""
echo "🎨 Step 4.5: Rebuilding dashboard frontend with Grafana tabs..."
ssh -i $VPS_KEY $VPS_USER@$VPS_HOST << 'EOF'
cd /home/qt/quantum_trader
docker compose --profile dashboard build dashboard-frontend
docker compose --profile dashboard up -d dashboard-frontend
sleep 3
EOF

# Step 5: Reload Nginx
echo ""
echo "🔄 Step 5: Reloading Nginx configuration..."
ssh -i $VPS_KEY $VPS_USER@$VPS_HOST << 'EOF'
nginx -t && nginx -s reload
EOF

# Step 6: Verify Grafana
echo ""
echo "✅ Step 6: Verifying Grafana..."
ssh -i $VPS_KEY $VPS_USER@$VPS_HOST << 'EOF'
echo "=== Grafana Container Status ==="
docker ps --filter name=quantum_grafana --format "table {{.Names}}\t{{.Status}}\t{{.Ports}}"

echo ""
echo "=== Grafana Health Check ==="
curl -s http://localhost:3001/api/health | jq .

echo ""
echo "=== Recent Grafana Logs ==="
docker logs quantum_grafana --tail 10 2>&1
EOF

echo ""
echo "========================================="
echo "✅ GRAFANA DEPLOYMENT COMPLETE!"
echo "========================================="
echo ""
echo "🌐 Access URLs:"
echo "   Admin:  https://app.quantumfond.com/grafana"
echo "   Login:  admin / quantum2026secure"
echo "   Viewer: Anonymous access enabled"
echo ""
echo "📊 Dashboard URLs (for iframe):"
echo "   P1-B Logs:       /grafana/d/p1b-logs"
echo "   P1-C Baseline:   /grafana/d/p1c-baseline"
echo "   P1-C Full:       /grafana/d/p1c-performance-baseline"
echo ""
echo "📝 Iframe Example:"
echo '   <iframe src="https://app.quantumfond.com/grafana/d/p1c-baseline?kiosk=tv" width="100%" height="600px" frameborder="0"></iframe>'
echo ""
echo "🔍 Test Access:"
echo "   curl -I https://app.quantumfond.com/grafana/"
echo ""
