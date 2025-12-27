#!/usr/bin/env pwsh
<#
.SYNOPSIS
    Deploy Phase 4T Strategic Evolution Engine to VPS
.DESCRIPTION
    Connects to VPS via SSH and executes Phase 4T deployment
#>

$VpsHost = "46.224.116.254"
$VpsUser = "qt"
$SSH_KEY = "~/.ssh/hetzner_fresh"

Write-Host "🚀 Phase 4T VPS Deployment Initiating..." -ForegroundColor Cyan
Write-Host "Target: $VpsUser@$VpsHost" -ForegroundColor Yellow
Write-Host ""

# Test VPS connectivity
Write-Host "🔍 Testing VPS connectivity..." -ForegroundColor Yellow
$testConnection = wsl ssh -i $SSH_KEY -o ConnectTimeout=10 -o StrictHostKeyChecking=no "$VpsUser@$VpsHost" "echo 'Connection OK'" 2>&1

if ($testConnection -match "Connection OK") {
    Write-Host "✅ VPS connection successful" -ForegroundColor Green
} else {
    Write-Host "❌ Cannot connect to VPS" -ForegroundColor Red
    exit 1
}

Write-Host ""
Write-Host "📡 Executing Phase 4T deployment on VPS..." -ForegroundColor Cyan
Write-Host "-------------------------------------------------------" -ForegroundColor Gray

# Execute deployment on VPS
wsl ssh -i $SSH_KEY -o StrictHostKeyChecking=no "$VpsUser@$VpsHost" "cd /home/qt/quantum_trader && git fetch origin && git reset --hard origin/main && chmod +x deploy_phase4t.sh && ./deploy_phase4t.sh"

if ($LASTEXITCODE -eq 0) {
    Write-Host ""
    Write-Host "-------------------------------------------------------" -ForegroundColor Gray
    Write-Host "✅ Phase 4T deployment completed successfully!" -ForegroundColor Green
    Write-Host ""
    Write-Host "📊 To monitor the deployment:" -ForegroundColor Cyan
    Write-Host "  wsl ssh -i $SSH_KEY $VpsUser@$VpsHost 'docker logs -f quantum_strategic_evolution'" -ForegroundColor Gray
    Write-Host ""
    Write-Host "🔍 To check rankings:" -ForegroundColor Cyan
    Write-Host "  wsl ssh -i $SSH_KEY $VpsUser@$VpsHost 'docker exec redis redis-cli GET quantum:evolution:rankings'" -ForegroundColor Gray
    Write-Host ""
    Write-Host "🧠 To verify AI Engine health:" -ForegroundColor Cyan
    Write-Host "  wsl ssh -i $SSH_KEY $VpsUser@$VpsHost 'curl -s http://localhost:8001/health'" -ForegroundColor Gray
    Write-Host ""
} else {
    Write-Host ""
    Write-Host "-------------------------------------------------------" -ForegroundColor Gray
    Write-Host "❌ Deployment failed with exit code: $LASTEXITCODE" -ForegroundColor Red
    Write-Host ""
    Write-Host "🔍 To check logs:" -ForegroundColor Yellow
    Write-Host "  wsl ssh -i $SSH_KEY $VpsUser@$VpsHost 'docker logs --tail 50 quantum_strategic_evolution'" -ForegroundColor Gray
    exit 1
}
