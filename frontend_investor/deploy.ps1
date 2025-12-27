# QuantumFond Investor Portal - PowerShell Deployment Script
# Windows deployment wrapper for investor.quantumfond.com

Write-Host "🚀 QuantumFond Investor Portal Deployment (Windows)" -ForegroundColor Green
Write-Host "=========================================="

# Configuration
$VPS_HOST = "46.224.116.254"
$VPS_USER = "root"
$SSH_KEY = "~/.ssh/hetzner_fresh"
$REMOTE_DIR = "/home/qt/quantum_trader/frontend_investor"

Write-Host "📦 Step 1: Building Next.js application..." -ForegroundColor Cyan
npm install
if ($LASTEXITCODE -ne 0) {
    Write-Host "❌ npm install failed" -ForegroundColor Red
    exit 1
}

npm run build
if ($LASTEXITCODE -ne 0) {
    Write-Host "❌ Build failed" -ForegroundColor Red
    exit 1
}

Write-Host "📤 Step 2: Deploying via WSL..." -ForegroundColor Cyan
wsl bash -c "cd /mnt/c/quantum_trader/frontend_investor && bash deploy.sh"

Write-Host ""
Write-Host "✅ Deployment Complete!" -ForegroundColor Green
Write-Host "🌐 Investor Portal: https://investor.quantumfond.com" -ForegroundColor Yellow
Write-Host ""
Write-Host ">>> [Phase 22 Complete – Investor Portal Operational]" -ForegroundColor Green
