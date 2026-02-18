#!/usr/bin/env pwsh
#
# Portfolio Selection Layer - VPS Deployment Script
# Deploys portfolio_selector.py with correlation filtering
#
# Usage: .\deploy_portfolio_selector.ps1
#

$ErrorActionPreference = "Stop"

$VPS_IP = "46.224.116.254"
$VPS_USER = "root"
$SSH_KEY = "~/.ssh/hetzner_fresh"
$REMOTE_PATH = "/opt/quantum/microservices/ai_engine"
$SERVICE_NAME = "quantum-ai-engine"

Write-Host "============================================================" -ForegroundColor Cyan
Write-Host "PORTFOLIO SELECTION LAYER - VPS DEPLOYMENT" -ForegroundColor Cyan
Write-Host "============================================================" -ForegroundColor Cyan
Write-Host ""

# Step 1: Validate local files
Write-Host "[1/7] Validating local files..." -ForegroundColor Yellow

$files = @(
    "microservices\ai_engine\portfolio_selector.py",
    "microservices\ai_engine\service.py",
    "microservices\ai_engine\config.py"
)

foreach ($file in $files) {
    if (-not (Test-Path $file)) {
        Write-Host "❌ Missing file: $file" -ForegroundColor Red
        exit 1
    }
    Write-Host "  ✅ Found: $file" -ForegroundColor Green
}

Write-Host ""

# Step 2: Run Python syntax check
Write-Host "[2/7] Running Python syntax validation..." -ForegroundColor Yellow

foreach ($file in $files) {
    $result = python -m py_compile $file 2>&1
    if ($LASTEXITCODE -ne 0) {
        Write-Host "❌ Syntax error in $file" -ForegroundColor Red
        Write-Host $result
        exit 1
    }
}

Write-Host "  ✅ All files compile successfully" -ForegroundColor Green
Write-Host ""

# Step 3: Create backup on VPS
Write-Host "[3/7] Creating backup on VPS..." -ForegroundColor Yellow

wsl bash -c 'ssh -i ~/.ssh/hetzner_fresh root@46.224.116.254 "mkdir -p /opt/quantum/backups/portfolio_selector_`$(date +%Y%m%d_%H%M%S) && cp -r /opt/quantum/microservices/ai_engine/*.py /opt/quantum/backups/portfolio_selector_`$(date +%Y%m%d_%H%M%S)/ 2>/dev/null || true"'

if ($LASTEXITCODE -eq 0) {
    Write-Host "  ✅ Backup created" -ForegroundColor Green
} else {
    Write-Host "  ⚠️ Backup skipped (might be first deployment)" -ForegroundColor Yellow
}

Write-Host ""

# Step 4: Upload files to VPS
Write-Host "[4/7] Uploading files to VPS..." -ForegroundColor Yellow

# Convert Windows paths to WSL paths
$wslFiles = @(
    "/mnt/c/quantum_trader/microservices/ai_engine/portfolio_selector.py",
    "/mnt/c/quantum_trader/microservices/ai_engine/service.py",
    "/mnt/c/quantum_trader/microservices/ai_engine/config.py"
)

foreach ($wslFile in $wslFiles) {
    $fileName = Split-Path $wslFile -Leaf
    Write-Host "  Uploading $fileName..." -ForegroundColor Gray
    
    wsl bash -c "scp -i ~/.ssh/hetzner_fresh '$wslFile' root@46.224.116.254:/opt/quantum/microservices/ai_engine/"
    
    if ($LASTEXITCODE -ne 0) {
        Write-Host "❌ Failed to upload $fileName" -ForegroundColor Red
        exit 1
    }
}

Write-Host "  ✅ All files uploaded" -ForegroundColor Green
Write-Host ""

# Step 5: Configure environment variables
Write-Host "[5/7] Configuring environment variables..." -ForegroundColor Yellow

$envConfig = @"
# Check if already configured
if ! grep -q 'MAX_SYMBOL_CORRELATION' /etc/systemd/system/quantum-ai-engine.service 2>/dev/null; then
    # Add environment variables to service file
    sed -i '/\[Service\]/a Environment="MAX_SYMBOL_CORRELATION=0.80"' /etc/systemd/system/quantum-ai-engine.service
    echo 'Added MAX_SYMBOL_CORRELATION=0.80'
else
    echo 'MAX_SYMBOL_CORRELATION already configured'
fi

# Verify TOP_N_LIMIT exists (should be from Phase 7)
if ! grep -q 'TOP_N_LIMIT' /etc/systemd/system/quantum-ai-engine.service 2>/dev/null; then
    sed -i '/\[Service\]/a Environment="TOP_N_LIMIT=10"' /etc/systemd/system/quantum-ai-engine.service
    echo 'Added TOP_N_LIMIT=10'
else
    echo 'TOP_N_LIMIT already configured'
fi

# Verify TOP_N_BUFFER_INTERVAL_SEC exists
if ! grep -q 'TOP_N_BUFFER_INTERVAL_SEC' /etc/systemd/system/quantum-ai-engine.service 2>/dev/null; then
    sed -i '/\[Service\]/a Environment="TOP_N_BUFFER_INTERVAL_SEC=2.0"' /etc/systemd/system/quantum-ai-engine.service
    echo 'Added TOP_N_BUFFER_INTERVAL_SEC=2.0'
else
    echo 'TOP_N_BUFFER_INTERVAL_SEC already configured'
fi

# Show current config
echo ''
echo '=== Current Configuration ==='
grep -E 'TOP_N|MAX_SYMBOL' /etc/systemd/system/quantum-ai-engine.service || echo 'No config found'
"@

wsl bash -c "ssh -i ~/.ssh/hetzner_fresh root@46.224.116.254 '$envConfig'"

Write-Host "  ✅ Environment configured" -ForegroundColor Green
Write-Host ""

# Step 6: Reload and restart service
Write-Host "[6/7] Restarting AI Engine service..." -ForegroundColor Yellow

wsl bash -c "ssh -i ~/.ssh/hetzner_fresh root@46.224.116.254 'systemctl daemon-reload && systemctl restart quantum-ai-engine'"

if ($LASTEXITCODE -eq 0) {
    Write-Host "  ✅ Service restarted" -ForegroundColor Green
} else {
    Write-Host "❌ Failed to restart service" -ForegroundColor Red
    exit 1
}

Write-Host ""
Write-Host "Waiting 5 seconds for service startup..." -ForegroundColor Gray
Start-Sleep -Seconds 5
Write-Host ""

# Step 7: Verify deployment
Write-Host "[7/7] Verifying deployment..." -ForegroundColor Yellow

# Check service status
Write-Host "  Checking service status..." -ForegroundColor Gray
$serviceStatus = wsl bash -c "ssh -i ~/.ssh/hetzner_fresh root@46.224.116.254 'systemctl is-active quantum-ai-engine'"

if ($serviceStatus -match "active") {
    Write-Host "  ✅ Service is running" -ForegroundColor Green
} else {
    Write-Host "  ❌ Service is NOT running: $serviceStatus" -ForegroundColor Red
    Write-Host ""
    Write-Host "Fetching error logs..." -ForegroundColor Yellow
    wsl bash -c "ssh -i ~/.ssh/hetzner_fresh root@46.224.116.254 'journalctl -u quantum-ai-engine -n 30 --no-pager'"
    exit 1
}

# Check for Portfolio-Selector logs
Write-Host "  Checking Portfolio-Selector initialization..." -ForegroundColor Gray
$psLogs = wsl bash -c "ssh -i ~/.ssh/hetzner_fresh root@46.224.116.254 'journalctl -u quantum-ai-engine -n 100 --no-pager | grep Portfolio-Selector'"

if ($psLogs) {
    Write-Host "  ✅ Portfolio-Selector initialized" -ForegroundColor Green
    Write-Host ""
    Write-Host "Recent Portfolio-Selector logs:" -ForegroundColor Cyan
    Write-Host $psLogs -ForegroundColor White
} else {
    Write-Host "  ⚠️ No Portfolio-Selector logs found yet (might need more time)" -ForegroundColor Yellow
}

Write-Host ""
Write-Host "============================================================" -ForegroundColor Cyan
Write-Host "✅ DEPLOYMENT COMPLETE" -ForegroundColor Green
Write-Host "============================================================" -ForegroundColor Cyan
Write-Host ""
Write-Host "Configuration:" -ForegroundColor Yellow
Write-Host "  TOP_N_LIMIT: 10" -ForegroundColor White
Write-Host "  MAX_SYMBOL_CORRELATION: 0.80" -ForegroundColor White
Write-Host "  TOP_N_BUFFER_INTERVAL_SEC: 2.0" -ForegroundColor White
Write-Host ""
Write-Host "Monitoring Commands:" -ForegroundColor Yellow
Write-Host "  # Watch Portfolio-Selector activity:" -ForegroundColor Gray
Write-Host "  wsl ssh -i ~/.ssh/hetzner_fresh root@46.224.116.254 'journalctl -u quantum-ai-engine -f | grep Portfolio-Selector'" -ForegroundColor White
Write-Host ""
Write-Host "  # Check correlation filtering:" -ForegroundColor Gray
Write-Host "  wsl ssh -i ~/.ssh/hetzner_fresh root@46.224.116.254 'journalctl -u quantum-ai-engine -f | grep correlation'" -ForegroundColor White
Write-Host ""
Write-Host "  # Check selection statistics:" -ForegroundColor Gray
Write-Host "  wsl ssh -i ~/.ssh/hetzner_fresh root@46.224.116.254 'journalctl -u quantum-ai-engine -n 200 | grep \"Selection complete\"'" -ForegroundColor White
Write-Host ""
Write-Host "Rollback Command (if needed):" -ForegroundColor Yellow
Write-Host "  wsl ssh -i ~/.ssh/hetzner_fresh root@46.224.116.254 'export MAX_SYMBOL_CORRELATION=0.99 && systemctl restart quantum-ai-engine'" -ForegroundColor White
Write-Host ""
