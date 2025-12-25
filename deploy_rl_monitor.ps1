# Exit any running Python sessions
if (Get-Process -Name python -ErrorAction SilentlyContinue) { Stop-Process -Name python -Force -ErrorAction SilentlyContinue }

Set-Location C:\quantum_trader

# Git operations
Write-Host "📦 Committing RL monitor daemon..." -ForegroundColor Cyan
git add -A
git commit -m "feat: Add RL monitoring daemon with visualization and Discord alerts"
git push origin main

Write-Host "✅ Committed to GitHub" -ForegroundColor Green
Write-Host "`n🚀 Deploying to VPS..." -ForegroundColor Cyan

# Deploy to VPS
wsl ssh -i ~/.ssh/hetzner_fresh root@46.224.116.254 @"
cd /home/qt/quantum_trader
git pull origin main
docker compose -f docker-compose.vps.yml build rl-monitor
docker compose -f docker-compose.vps.yml up -d rl-monitor
sleep 5
docker ps --format 'table {{.Names}}\t{{.Status}}' | grep rl_monitor
docker logs quantum_rl_monitor --tail 15
"@

Write-Host "`n✅ RL Monitor Daemon deployed!" -ForegroundColor Green
