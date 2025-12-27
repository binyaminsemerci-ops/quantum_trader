#!/usr/bin/env pwsh
<#
.SYNOPSIS
    Manual Phase 4S+ deployment - copies files directly to VPS
#>

$VpsHost = "46.224.116.254"
$VpsUser = "qt"
$SSH_KEY = "~/.ssh/hetzner_fresh"

Write-Host "🚀 Phase 4S+ Manual Deployment" -ForegroundColor Cyan
Write-Host ""

# Step 1: Copy deployment script
Write-Host "📤 Step 1: Uploading deploy_phase4s.sh..." -ForegroundColor Yellow
wsl scp -i $SSH_KEY deploy_phase4s.sh "${VpsUser}@${VpsHost}:/tmp/"

# Step 2: Copy docker-compose file
Write-Host "📤 Step 2: Uploading docker-compose.vps.yml..." -ForegroundColor Yellow
wsl scp -i $SSH_KEY docker-compose.vps.yml "${VpsUser}@${VpsHost}:/tmp/"

# Step 3: Copy microservice directory
Write-Host "📤 Step 3: Uploading strategic_memory microservice..." -ForegroundColor Yellow
wsl ssh -i $SSH_KEY "${VpsUser}@${VpsHost}" "mkdir -p /tmp/microservices"
wsl scp -i $SSH_KEY -r microservices/strategic_memory "${VpsUser}@${VpsHost}:/tmp/microservices/"

# Step 4: Execute deployment
Write-Host "🚀 Step 4: Running deployment on VPS..." -ForegroundColor Cyan
wsl ssh -i $SSH_KEY "${VpsUser}@${VpsHost}" "cd /tmp && chmod +x deploy_phase4s.sh && ./deploy_phase4s.sh"

if ($LASTEXITCODE -eq 0) {
    Write-Host ""
    Write-Host "✅ Phase 4S+ deployed successfully!" -ForegroundColor Green
} else {
    Write-Host ""
    Write-Host "❌ Deployment failed" -ForegroundColor Red
}
