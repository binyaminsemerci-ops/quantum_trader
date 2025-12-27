# 🚀 Deploy Quantum Trader Dashboard to VPS
# Usage: .\deploy_dashboard.ps1

$ErrorActionPreference = "Stop"

Write-Host "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━" -ForegroundColor Cyan
Write-Host "🚀 DEPLOYING QUANTUM TRADER DASHBOARD TO VPS" -ForegroundColor Green
Write-Host "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━" -ForegroundColor Cyan

$VPS_HOST = "root@46.224.116.254"
$VPS_PATH = "/root/quantum_trader"

Write-Host ""
Write-Host "📦 Step 1: Syncing dashboard code to VPS..." -ForegroundColor Yellow

# Sync dashboard code
scp -r dashboard_v4 ${VPS_HOST}:${VPS_PATH}/

Write-Host ""
Write-Host "📦 Step 2: Syncing docker-compose.yml..." -ForegroundColor Yellow
scp docker-compose.yml ${VPS_HOST}:${VPS_PATH}/

Write-Host ""
Write-Host "🔧 Step 3: Building and starting dashboard on VPS..." -ForegroundColor Yellow

ssh ${VPS_HOST} @'
cd /root/quantum_trader

# Stop existing dashboard containers if any
echo "🛑 Stopping existing dashboard containers..."
docker-compose --profile dashboard down || true

# Build and start dashboard
echo "🔨 Building dashboard images..."
docker-compose --profile dashboard build dashboard-backend dashboard-frontend

echo "🚀 Starting dashboard services..."
docker-compose --profile dashboard up -d dashboard-backend dashboard-frontend

# Wait for services to be healthy
echo "⏳ Waiting for services to be healthy..."
sleep 10

# Check health
echo ""
echo "🏥 Checking dashboard health..."
docker-compose --profile dashboard ps

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
'@

Write-Host ""
Write-Host "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━" -ForegroundColor Cyan
Write-Host "✅ DEPLOYMENT COMPLETE!" -ForegroundColor Green
Write-Host "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━" -ForegroundColor Cyan
