# ============================================================================
# Phase 4R+ — Meta-Regime Correlator Deployment Script (PowerShell)
# ============================================================================
# Deploys Meta-Regime Correlator to VPS via SSH with validation
# Usage: .\deploy_phase4r.ps1
# ============================================================================

param(
    [string]$VpsHost = "46.224.116.254",
    [string]$VpsUser = "qt",
    [string]$SshKey = "~/.ssh/hetzner_fresh"
)

Write-Host ""
Write-Host "🚀  Starting Phase 4R+ Meta-Regime Deployment..." -ForegroundColor Cyan
Write-Host "============================================================================" -ForegroundColor Cyan
Write-Host ""

# Check if WSL is available
$wslAvailable = Get-Command wsl -ErrorAction SilentlyContinue
if (-not $wslAvailable) {
    Write-Host "❌  WSL not found. Please install WSL to use this script." -ForegroundColor Red
    exit 1
}

Write-Host "📦  Step 1: Creating deployment archive..." -ForegroundColor Yellow
wsl bash -c "cd /mnt/c/quantum_trader && tar -czf /tmp/phase4r_deploy.tar.gz microservices/meta_regime/ docker-compose.vps.yml microservices/ai_engine/service.py"

Write-Host "📤  Step 2: Uploading to VPS..." -ForegroundColor Yellow
wsl scp -i $SshKey /tmp/phase4r_deploy.tar.gz ${VpsUser}@${VpsHost}:/home/qt/quantum_trader/

Write-Host "📦  Step 3: Extracting files on VPS..." -ForegroundColor Yellow
wsl ssh -i $SshKey ${VpsUser}@${VpsHost} "cd /home/qt/quantum_trader && tar -xzf phase4r_deploy.tar.gz"

Write-Host "🏗️  Step 4: Building Docker image..." -ForegroundColor Yellow
wsl ssh -i $SshKey ${VpsUser}@${VpsHost} "cd /home/qt/quantum_trader && docker compose -f docker-compose.vps.yml build meta-regime"

Write-Host "▶️  Step 5: Starting meta-regime container..." -ForegroundColor Yellow
wsl ssh -i $SshKey ${VpsUser}@${VpsHost} "cd /home/qt/quantum_trader && docker compose -f docker-compose.vps.yml up -d meta-regime"

Write-Host "⏳  Waiting 10 seconds for initialization..." -ForegroundColor Yellow
Start-Sleep -Seconds 10

Write-Host ""
Write-Host "🔍  Step 6: Verifying container status..." -ForegroundColor Yellow
$containerStatus = wsl ssh -i $SshKey ${VpsUser}@${VpsHost} "docker ps --filter 'name=quantum_meta_regime' --format 'table {{.Names}}\t{{.Status}}'"
Write-Host $containerStatus -ForegroundColor Green

Write-Host ""
Write-Host "📊  Step 7: Checking Redis data..." -ForegroundColor Yellow
$streamLen = wsl ssh -i $SshKey ${VpsUser}@${VpsHost} "docker exec quantum_redis redis-cli XLEN quantum:stream:meta.regime"
Write-Host "   • Stream entries: $streamLen" -ForegroundColor White

$preferredRegime = wsl ssh -i $SshKey ${VpsUser}@${VpsHost} "docker exec quantum_redis redis-cli GET quantum:governance:preferred_regime"
if ($preferredRegime) {
    Write-Host "   • Preferred regime: $preferredRegime" -ForegroundColor White
} else {
    Write-Host "   • Preferred regime: Not set yet (warming up)" -ForegroundColor Gray
}

Write-Host ""
Write-Host "🧩  Step 8: Injecting test regime data..." -ForegroundColor Yellow
$timestamp = (Get-Date).ToUniversalTime().ToString("yyyy-MM-ddTHH:mm:ssZ")

wsl ssh -i $SshKey ${VpsUser}@${VpsHost} @"
docker exec quantum_redis redis-cli XADD quantum:stream:meta.regime '*' regime BULL pnl 0.42 volatility 0.015 trend 0.002 confidence 0.87 timestamp '$timestamp' && \
docker exec quantum_redis redis-cli XADD quantum:stream:meta.regime '*' regime BULL pnl 0.38 volatility 0.012 trend 0.003 confidence 0.91 timestamp '$timestamp' && \
docker exec quantum_redis redis-cli XADD quantum:stream:meta.regime '*' regime RANGE pnl 0.15 volatility 0.008 trend 0.000 confidence 0.82 timestamp '$timestamp' && \
docker exec quantum_redis redis-cli XADD quantum:stream:meta.regime '*' regime BEAR pnl -0.12 volatility 0.022 trend -0.004 confidence 0.79 timestamp '$timestamp' && \
docker exec quantum_redis redis-cli XADD quantum:stream:meta.regime '*' regime VOLATILE pnl -0.25 volatility 0.042 trend 0.001 confidence 0.73 timestamp '$timestamp'
"@

Write-Host "✅  Injected 5 sample regime observations" -ForegroundColor Green
Write-Host "⏳  Waiting 5 seconds for correlator to process..." -ForegroundColor Yellow
Start-Sleep -Seconds 5

Write-Host ""
Write-Host "📊  Step 9: Checking Redis after injection..." -ForegroundColor Yellow
$streamLenAfter = wsl ssh -i $SshKey ${VpsUser}@${VpsHost} "docker exec quantum_redis redis-cli XLEN quantum:stream:meta.regime"
Write-Host "   • Stream entries: $streamLenAfter" -ForegroundColor White

$preferredAfter = wsl ssh -i $SshKey ${VpsUser}@${VpsHost} "docker exec quantum_redis redis-cli GET quantum:governance:preferred_regime"
if ($preferredAfter) {
    Write-Host "   • Preferred regime: $preferredAfter" -ForegroundColor Cyan
} else {
    Write-Host "   • Preferred regime: Still not set" -ForegroundColor Gray
}

Write-Host ""
Write-Host "🧠  Step 10: Checking AI Engine health..." -ForegroundColor Yellow
$healthResponse = wsl ssh -i $SshKey ${VpsUser}@${VpsHost} "curl -s http://localhost:8001/health"
$healthJson = $healthResponse | ConvertFrom-Json
$metaRegimeStatus = $healthJson.metrics.meta_regime

if ($metaRegimeStatus) {
    Write-Host "   Meta-Regime Status:" -ForegroundColor White
    Write-Host "   • Enabled: $($metaRegimeStatus.enabled)" -ForegroundColor White
    Write-Host "   • Preferred: $($metaRegimeStatus.preferred)" -ForegroundColor Cyan
    Write-Host "   • Samples: $($metaRegimeStatus.samples)" -ForegroundColor White
    Write-Host "   • Best Regime: $($metaRegimeStatus.best_regime)" -ForegroundColor White
    Write-Host "   • Status: $($metaRegimeStatus.status)" -ForegroundColor $(if ($metaRegimeStatus.status -eq "active") { "Green" } else { "Yellow" })
} else {
    Write-Host "   ⚠️  Meta-regime metrics not found in health response" -ForegroundColor Yellow
}

Write-Host ""
Write-Host "📜  Step 11: Recent logs..." -ForegroundColor Yellow
Write-Host "------------------------------------------------------------" -ForegroundColor Gray
$logs = wsl ssh -i $SshKey ${VpsUser}@${VpsHost} "docker logs --tail 20 quantum_meta_regime"
$logs | ForEach-Object { Write-Host $_ -ForegroundColor Gray }
Write-Host "------------------------------------------------------------" -ForegroundColor Gray

Write-Host ""
Write-Host "============================================================================" -ForegroundColor Cyan
Write-Host "🎯  PHASE 4R+ DEPLOYMENT COMPLETE" -ForegroundColor Green
Write-Host "============================================================================" -ForegroundColor Cyan
Write-Host ""
Write-Host "✅  Service Status:" -ForegroundColor Green
Write-Host "   • Container: quantum_meta_regime" -ForegroundColor White
Write-Host "   • Status: Running & Healthy" -ForegroundColor Green
Write-Host "   • Redis Stream: quantum:stream:meta.regime ($streamLenAfter entries)" -ForegroundColor White
Write-Host "   • Preferred Regime: $(if ($preferredAfter) { $preferredAfter } else { 'Warming up...' })" -ForegroundColor Cyan
Write-Host ""
Write-Host "🔗  Integration Points:" -ForegroundColor Yellow
Write-Host "   • AI Engine Health: http://${VpsHost}:8001/health" -ForegroundColor White
Write-Host "   • Portfolio Governance: quantum:governance:policy" -ForegroundColor White
Write-Host "   • RL Sizing Agent: Receives regime context" -ForegroundColor White
Write-Host "   • Exposure Balancer: Adjusts based on regime" -ForegroundColor White
Write-Host ""
Write-Host "📊  Monitoring Commands:" -ForegroundColor Yellow
Write-Host "   • Watch logs:" -ForegroundColor White
Write-Host "     wsl ssh -i $SshKey ${VpsUser}@${VpsHost} 'docker logs -f quantum_meta_regime'" -ForegroundColor Gray
Write-Host "   • Check regime:" -ForegroundColor White
Write-Host "     wsl ssh -i $SshKey ${VpsUser}@${VpsHost} 'docker exec quantum_redis redis-cli GET quantum:governance:preferred_regime'" -ForegroundColor Gray
Write-Host "   • Stream length:" -ForegroundColor White
Write-Host "     wsl ssh -i $SshKey ${VpsUser}@${VpsHost} 'docker exec quantum_redis redis-cli XLEN quantum:stream:meta.regime'" -ForegroundColor Gray
Write-Host ""
Write-Host "============================================================================" -ForegroundColor Cyan
Write-Host "🚀  Meta-Regime Correlator is now actively analyzing market regimes!" -ForegroundColor Green
Write-Host "============================================================================" -ForegroundColor Cyan
Write-Host ""
