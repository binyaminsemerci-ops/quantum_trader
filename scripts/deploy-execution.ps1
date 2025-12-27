#!/usr/bin/env pwsh
<#
.SYNOPSIS
    Deploy Execution Service V2 to VPS

.DESCRIPTION
    Syncs source files, builds Docker image, and starts Execution Service
    
    Features:
    - Clean microservice (NO monolith dependencies)
    - PAPER trading mode by default
    - EventBus integration
    - RiskStub validation
    - Health checks

.PARAMETER Mode
    Execution mode: PAPER (default), TESTNET, or LIVE

.EXAMPLE
    .\deploy-execution.ps1
    .\deploy-execution.ps1 -Mode TESTNET
#>

param(
    [ValidateSet("PAPER", "TESTNET", "LIVE")]
    [string]$Mode = "PAPER"
)

$ErrorActionPreference = "Stop"

$VPS_HOST = "46.224.116.254"
$VPS_USER = "qt"
$REMOTE_DIR = "/home/qt/quantum_trader"

Write-Host "========================================" -ForegroundColor Cyan
Write-Host "EXECUTION SERVICE V2 DEPLOYMENT" -ForegroundColor Cyan
Write-Host "========================================" -ForegroundColor Cyan
Write-Host "Mode: $Mode" -ForegroundColor Yellow
Write-Host ""

# Step 1: Sync source files
Write-Host "[1/5] Syncing source files..." -ForegroundColor Cyan

$files = @(
    "microservices/execution/service_v2.py",
    "microservices/execution/main_v2.py",
    "microservices/execution/config.py",
    "microservices/execution/models.py",
    "microservices/execution/risk_stub.py",
    "microservices/execution/binance_adapter.py",
    "microservices/execution/Dockerfile.v2",
    "microservices/execution/requirements_v2.txt"
)

foreach ($file in $files) {
    Write-Host "  Uploading $file..." -ForegroundColor Gray
    scp $file "${VPS_USER}@${VPS_HOST}:${REMOTE_DIR}/$file"
    if ($LASTEXITCODE -ne 0) {
        Write-Host "❌ Failed to upload $file" -ForegroundColor Red
        exit 1
    }
}

Write-Host "✅ Files synced" -ForegroundColor Green
Write-Host ""

# Step 2: Sync docker-compose.services.yml
Write-Host "[2/5] Syncing Docker Compose config..." -ForegroundColor Cyan

scp docker-compose.services.yml "${VPS_USER}@${VPS_HOST}:${REMOTE_DIR}/docker-compose.services.yml"
if ($LASTEXITCODE -ne 0) {
    Write-Host "❌ Failed to upload docker-compose.services.yml" -ForegroundColor Red
    exit 1
}

Write-Host "✅ Docker Compose config synced" -ForegroundColor Green
Write-Host ""

# Step 3: Build Docker image
Write-Host "[3/5] Building Docker image..." -ForegroundColor Cyan

ssh "${VPS_USER}@${VPS_HOST}" @"
cd $REMOTE_DIR
docker compose -f docker-compose.services.yml build execution
"@

if ($LASTEXITCODE -ne 0) {
    Write-Host "❌ Docker build failed" -ForegroundColor Red
    exit 1
}

Write-Host "✅ Docker image built" -ForegroundColor Green
Write-Host ""

# Step 4: Start service
Write-Host "[4/5] Starting Execution Service..." -ForegroundColor Cyan

ssh "${VPS_USER}@${VPS_HOST}" @"
cd $REMOTE_DIR
export EXECUTION_MODE=$Mode
docker compose -f docker-compose.vps.yml -f docker-compose.services.yml up -d execution
"@

if ($LASTEXITCODE -ne 0) {
    Write-Host "❌ Failed to start service" -ForegroundColor Red
    exit 1
}

Write-Host "✅ Service started" -ForegroundColor Green
Write-Host ""

# Step 5: Wait and verify health
Write-Host "[5/5] Verifying health (waiting 15s)..." -ForegroundColor Cyan
Start-Sleep -Seconds 15

ssh "${VPS_USER}@${VPS_HOST}" @"
cd $REMOTE_DIR
echo ""
echo "=== CONTAINER STATUS ==="
docker ps --filter name=quantum_execution --format 'table {{.Names}}\t{{.Status}}\t{{.Ports}}'
echo ""
echo "=== HEALTH CHECK ==="
curl -s http://localhost:8002/health | python3 -m json.tool
echo ""
echo "=== RECENT LOGS ==="
docker logs quantum_execution --tail 30
"@

Write-Host ""
Write-Host "========================================" -ForegroundColor Cyan
Write-Host "DEPLOYMENT COMPLETE" -ForegroundColor Green
Write-Host "========================================" -ForegroundColor Cyan
Write-Host ""
Write-Host "Endpoints:" -ForegroundColor Yellow
Write-Host "  Health: http://localhost:8002/health" -ForegroundColor White
Write-Host "  Positions: http://localhost:8002/positions" -ForegroundColor White
Write-Host "  Trades: http://localhost:8002/trades" -ForegroundColor White
Write-Host ""
Write-Host "View logs: docker logs -f quantum_execution" -ForegroundColor Gray
Write-Host ""
