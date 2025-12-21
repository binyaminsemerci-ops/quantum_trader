#!/usr/bin/env pwsh
<#
.SYNOPSIS
    Deploy Phase 4S+ Strategic Memory Sync to VPS
.DESCRIPTION
    Connects to VPS via SSH and executes Phase 4S+ deployment
#>

param(
    [string]$VpsHost = "46.224.116.254",
    [string]$VpsUser = "qt",
    [string]$SshKeyPath = "$env:USERPROFILE\.ssh\hetzner_fresh"
)

Write-Host "🚀 Phase 4S+ VPS Deployment Initiating..." -ForegroundColor Cyan
Write-Host "Target: $VpsUser@$VpsHost" -ForegroundColor Yellow
Write-Host ""

# Check SSH key exists
if (-not (Test-Path $SshKeyPath)) {
    Write-Host "❌ SSH key not found at: $SshKeyPath" -ForegroundColor Red
    Write-Host "Please specify correct path with -SshKeyPath parameter" -ForegroundColor Yellow
    exit 1
}

Write-Host "✅ SSH key found: $SshKeyPath" -ForegroundColor Green
Write-Host ""

# Test VPS connectivity
Write-Host "🔍 Testing VPS connectivity..." -ForegroundColor Yellow
$wslSshKey = $SshKeyPath -replace '\\', '/' -replace 'C:', '/mnt/c' -replace 'Users', 'users'
$testConnection = wsl ssh -i $wslSshKey -o ConnectTimeout=10 -o StrictHostKeyChecking=no "$VpsUser@$VpsHost" "echo 'Connection OK'" 2>&1

if ($LASTEXITCODE -ne 0) {
    Write-Host "❌ Cannot connect to VPS: $VpsHost" -ForegroundColor Red
    Write-Host "Error: $testConnection" -ForegroundColor Red
    exit 1
}

Write-Host "✅ VPS connection successful" -ForegroundColor Green
Write-Host ""

# Execute deployment on VPS
Write-Host "📡 Executing Phase 4S+ deployment on VPS..." -ForegroundColor Cyan
Write-Host "-------------------------------------------------------" -ForegroundColor Gray

$deploymentScript = @'
#!/bin/bash
set -e

echo "🔄 Navigating to quantum_trader directory..."
cd /home/qt/quantum_trader

echo "📥 Pulling latest code from repository..."
git pull origin main

echo "🔧 Making deployment script executable..."
chmod +x deploy_phase4s.sh

echo "🚀 Launching Phase 4S+ deployment..."
./deploy_phase4s.sh
'@

# Execute deployment via SSH
$deploymentScript | wsl ssh -i $wslSshKey -o StrictHostKeyChecking=no "$VpsUser@$VpsHost" "bash -s"

if ($LASTEXITCODE -eq 0) {
    Write-Host ""
    Write-Host "-------------------------------------------------------" -ForegroundColor Gray
    Write-Host "✅ Phase 4S+ deployment completed successfully!" -ForegroundColor Green
    Write-Host ""
    Write-Host "📊 To monitor the deployment:" -ForegroundColor Cyan
    Write-Host "  wsl ssh -i $wslSshKey $VpsUser@$VpsHost 'docker logs -f quantum_strategic_memory'" -ForegroundColor Gray
    Write-Host ""
    Write-Host "🔍 To check feedback loop:" -ForegroundColor Cyan
    Write-Host "  wsl ssh -i $wslSshKey $VpsUser@$VpsHost 'docker exec redis redis-cli GET quantum:feedback:strategic_memory'" -ForegroundColor Gray
    Write-Host ""
    Write-Host "🧠 To verify AI Engine health:" -ForegroundColor Cyan
    Write-Host "  wsl ssh -i $wslSshKey $VpsUser@$VpsHost 'curl -s http://localhost:8001/health | jq .metrics.strategic_memory'" -ForegroundColor Gray
    Write-Host ""
} else {
    Write-Host ""
    Write-Host "-------------------------------------------------------" -ForegroundColor Gray
    Write-Host "❌ Deployment failed with exit code: $LASTEXITCODE" -ForegroundColor Red
    Write-Host ""
    Write-Host "🔍 To check logs:" -ForegroundColor Yellow
    Write-Host "  wsl ssh -i $wslSshKey $VpsUser@$VpsHost 'docker logs --tail 50 quantum_strategic_memory'" -ForegroundColor Gray
    exit 1
}
