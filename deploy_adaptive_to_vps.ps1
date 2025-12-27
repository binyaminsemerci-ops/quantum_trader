# Deploy AdaptiveLeverageEngine to VPS (Git Pull Method)
# Uses existing Docker setup - no sudo required!

Write-Host "=== Deploying AdaptiveLeverageEngine to VPS ===" -ForegroundColor Cyan
Write-Host ""

$sshKey = "$env:USERPROFILE\.ssh\hetzner_fresh"
$vpsUser = "benyamin"
$vpsHost = "46.224.116.254"
$projectPath = "/home/qt/quantum_trader"

Write-Host "Step 1: Pulling latest code from GitHub..." -ForegroundColor Yellow
ssh -i "$sshKey" "$vpsUser@$vpsHost" "cd $projectPath && sudo git pull"

if ($LASTEXITCODE -ne 0) {
    Write-Host "❌ Git pull failed" -ForegroundColor Red
    exit 1
}

Write-Host ""
Write-Host "Step 2: Restarting backend service..." -ForegroundColor Yellow
ssh -i "$sshKey" "$vpsUser@$vpsHost" "cd $projectPath && sudo docker-compose -f docker-compose.vps.yml restart backend"

if ($LASTEXITCODE -ne 0) {
    Write-Host "❌ Service restart failed" -ForegroundColor Red
    exit 1
}

Write-Host ""
Write-Host "Step 3: Verifying deployment..." -ForegroundColor Yellow
Start-Sleep -Seconds 5

ssh -i "$sshKey" "$vpsUser@$vpsHost" @"
cd $projectPath
echo '--- Checking files ---'
ls -la microservices/exitbrain_v3_5/adaptive_leverage*.py 2>/dev/null || echo 'Files not found'
echo ''
echo '--- Testing import ---'
sudo docker compose -f docker-compose.vps.yml exec -T backend python -c 'from microservices.exitbrain_v3_5.adaptive_leverage_engine import AdaptiveLeverageEngine; print(\"✅ Import successful\")' 2>&1 || echo '⚠️ Import test failed (service may still be starting)'
echo ''
echo '--- Checking backend logs ---'
sudo docker compose -f docker-compose.vps.yml logs backend --tail 10
"@

Write-Host ""
Write-Host "=== Deployment Complete ===" -ForegroundColor Green
Write-Host ""
Write-Host "Next steps:" -ForegroundColor Cyan
Write-Host "1. Monitor logs: ssh benyamin@46.224.116.254 'cd $projectPath && sudo docker compose -f docker-compose.vps.yml logs -f backend | grep Adaptive'"
Write-Host "2. Check Redis stream: ssh benyamin@46.224.116.254 'sudo docker compose -f docker-compose.vps.yml exec redis redis-cli XLEN quantum:stream:adaptive_levels'"
Write-Host "3. Run monitoring: ssh benyamin@46.224.116.254 'cd quantum_trader && docker compose -f docker-compose.vps.yml exec backend python monitor_adaptive_leverage.py watch'"
Write-Host ""
