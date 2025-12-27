#!/usr/bin/pwsh
$ErrorActionPreference = "Stop"

Write-Host "`n🚀 Deploying RL Monitor Daemon to VPS..." -ForegroundColor Cyan

wsl ssh -i ~/.ssh/hetzner_fresh root@46.224.116.254 @'
set -e
cd /home/qt/quantum_trader
echo "📥 Pulling latest changes from GitHub..."
git pull origin main
echo "🔨 Building RL monitor container..."
docker compose -f docker-compose.vps.yml build rl-monitor
echo "▶️  Starting RL monitor service..."
docker compose -f docker-compose.vps.yml up -d rl-monitor
sleep 5
echo ""
echo "=== Container Status ==="
docker ps --format 'table {{.Names}}\t{{.Status}}' | grep -E 'rl_monitor|rl_sizing'
echo ""
echo "=== RL Monitor Logs ==="
docker logs quantum_rl_monitor --tail 15
echo ""
echo "✅ RL Monitoring Daemon deployed successfully!"
'@

Write-Host "`n✅ Deployment complete!" -ForegroundColor Green
