#!/usr/bin/env pwsh
<#
.SYNOPSIS
    Manual Phase 4T deployment - copies files directly to VPS
#>

$VpsHost = "46.224.116.254"
$VpsUser = "qt"
$SSH_KEY = "~/.ssh/hetzner_fresh"

Write-Host "🚀 Phase 4T Manual Deployment" -ForegroundColor Cyan
Write-Host ""

# Step 1: Copy deployment script
Write-Host "📤 Step 1: Uploading deploy_phase4t.sh..." -ForegroundColor Yellow
wsl scp -i $SSH_KEY deploy_phase4t.sh "${VpsUser}@${VpsHost}:/tmp/"

# Step 2: Copy docker-compose file
Write-Host "📤 Step 2: Uploading docker-compose.vps.yml..." -ForegroundColor Yellow
wsl scp -i $SSH_KEY docker-compose.vps.yml "${VpsUser}@${VpsHost}:/tmp/"

# Step 3: Copy AI Engine service.py (with Phase 4T integration)
Write-Host "📤 Step 3: Uploading ai_engine/service.py..." -ForegroundColor Yellow
wsl ssh -i $SSH_KEY "${VpsUser}@${VpsHost}" "mkdir -p /tmp/microservices/ai_engine"
wsl scp -i $SSH_KEY microservices/ai_engine/service.py "${VpsUser}@${VpsHost}:/tmp/microservices/ai_engine/"

# Step 4: Copy strategic_evolution microservice directory
Write-Host "📤 Step 4: Uploading strategic_evolution microservice..." -ForegroundColor Yellow
wsl ssh -i $SSH_KEY "${VpsUser}@${VpsHost}" "mkdir -p /tmp/microservices/strategic_evolution"
wsl scp -i $SSH_KEY -r microservices/strategic_evolution/* "${VpsUser}@${VpsHost}:/tmp/microservices/strategic_evolution/"

# Step 5: Execute deployment
Write-Host "🚀 Step 5: Running deployment on VPS..." -ForegroundColor Cyan
wsl ssh -i $SSH_KEY "${VpsUser}@${VpsHost}" "cd /tmp && chmod +x deploy_phase4t.sh && ./deploy_phase4t.sh"

if ($LASTEXITCODE -eq 0) {
    Write-Host ""
    Write-Host "✅ Phase 4T deployed successfully!" -ForegroundColor Green
    Write-Host ""
    Write-Host "📊 Monitor with:" -ForegroundColor Cyan
    Write-Host "  wsl ssh -i $SSH_KEY ${VpsUser}@${VpsHost} 'docker logs -f quantum_strategic_evolution'" -ForegroundColor Gray
} else {
    Write-Host ""
    Write-Host "❌ Deployment failed" -ForegroundColor Red
}
