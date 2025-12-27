# ============================================================================
# ESS Deployment Script for Staging Environment
# ============================================================================
# This script deploys the Emergency Stop System to staging with paper trading.
#
# Usage:
#   .\scripts\deploy-ess-staging.ps1
#
# Prerequisites:
#   - Python 3.11+
#   - Virtual environment activated
#   - .env file configured with ESS settings
#   - Redis running (for OpportunityRanker)
#
# Author: Quantum Trader Team
# Date: 2024-11-30
# ============================================================================

param(
    [switch]$SkipTests,
    [switch]$Verbose
)

$ErrorActionPreference = "Stop"

Write-Host "`n========================================" -ForegroundColor Cyan
Write-Host "ESS STAGING DEPLOYMENT" -ForegroundColor Cyan
Write-Host "========================================`n" -ForegroundColor Cyan

# Step 1: Validate environment
Write-Host "📋 Step 1: Validating environment..." -ForegroundColor Yellow

if (-not (Test-Path ".venv")) {
    Write-Host "❌ Virtual environment not found. Run: python -m venv .venv" -ForegroundColor Red
    exit 1
}

if (-not (Test-Path ".env")) {
    Write-Host "❌ .env file not found. Copy from .env.example" -ForegroundColor Red
    exit 1
}

Write-Host "✅ Environment validated" -ForegroundColor Green

# Step 2: Check Redis
Write-Host "`n📋 Step 2: Checking Redis..." -ForegroundColor Yellow

try {
    $redisCheck = docker ps --filter "name=redis" --filter "status=running" --format "{{.Names}}"
    if ($redisCheck) {
        Write-Host "✅ Redis is running: $redisCheck" -ForegroundColor Green
    } else {
        Write-Host "⚠️  Redis not running. Starting..." -ForegroundColor Yellow
        docker run -d --name redis -p 6379:6379 redis:7-alpine
        Start-Sleep -Seconds 3
        Write-Host "✅ Redis started" -ForegroundColor Green
    }
} catch {
    Write-Host "❌ Failed to check/start Redis: $_" -ForegroundColor Red
    exit 1
}

# Step 3: Verify ESS configuration
Write-Host "`n📋 Step 3: Verifying ESS configuration..." -ForegroundColor Yellow

$envContent = Get-Content .env -Raw

$requiredVars = @(
    "QT_ESS_ENABLED",
    "QT_ESS_MAX_DAILY_LOSS",
    "QT_ESS_MAX_EQUITY_DD"
)

$missingVars = @()
foreach ($var in $requiredVars) {
    if ($envContent -notmatch "$var=") {
        $missingVars += $var
    }
}

if ($missingVars.Count -gt 0) {
    Write-Host "⚠️  Missing ESS configuration:" -ForegroundColor Yellow
    foreach ($var in $missingVars) {
        Write-Host "   - $var" -ForegroundColor Yellow
    }
    Write-Host "   Copying from .env.ess..." -ForegroundColor Yellow
    
    if (Test-Path ".env.ess") {
        Get-Content .env.ess | Add-Content .env
        Write-Host "✅ ESS configuration added to .env" -ForegroundColor Green
    } else {
        Write-Host "❌ .env.ess not found. Please configure ESS manually." -ForegroundColor Red
        exit 1
    }
} else {
    Write-Host "✅ ESS configuration found" -ForegroundColor Green
}

# Step 4: Verify ESS module
Write-Host "`n📋 Step 4: Verifying ESS module..." -ForegroundColor Yellow

if (-not (Test-Path "backend/services/emergency_stop_system.py")) {
    Write-Host "❌ emergency_stop_system.py not found" -ForegroundColor Red
    exit 1
}

if (-not (Test-Path "backend/services/ess_alerters.py")) {
    Write-Host "❌ ess_alerters.py not found" -ForegroundColor Red
    exit 1
}

Write-Host "✅ ESS modules found" -ForegroundColor Green

# Step 5: Run ESS tests (unless skipped)
if (-not $SkipTests) {
    Write-Host "`n📋 Step 5: Running ESS tests..." -ForegroundColor Yellow
    
    & .venv\Scripts\Activate.ps1
    
    $testResult = python -m pytest backend/services/test_emergency_stop_system.py -v --tb=short 2>&1
    
    if ($LASTEXITCODE -eq 0) {
        Write-Host "✅ All ESS tests passed" -ForegroundColor Green
    } else {
        Write-Host "❌ ESS tests failed:" -ForegroundColor Red
        Write-Host $testResult -ForegroundColor Red
        
        $continue = Read-Host "`nContinue deployment anyway? (y/N)"
        if ($continue -ne "y") {
            exit 1
        }
    }
} else {
    Write-Host "`n📋 Step 5: Skipping tests (--SkipTests flag)" -ForegroundColor Yellow
}

# Step 6: Stop existing backend
Write-Host "`n📋 Step 6: Stopping existing backend..." -ForegroundColor Yellow

$pythonProcesses = Get-Process python -ErrorAction SilentlyContinue | Where-Object {
    $_.CommandLine -like "*uvicorn*" -or $_.CommandLine -like "*main.py*"
}

if ($pythonProcesses) {
    Write-Host "   Stopping $($pythonProcesses.Count) Python process(es)..." -ForegroundColor Yellow
    $pythonProcesses | Stop-Process -Force
    Start-Sleep -Seconds 2
    Write-Host "✅ Existing backend stopped" -ForegroundColor Green
} else {
    Write-Host "✅ No existing backend running" -ForegroundColor Green
}

# Step 7: Clear cache
Write-Host "`n📋 Step 7: Clearing Python cache..." -ForegroundColor Yellow

Get-ChildItem -Path backend -Recurse -Filter "__pycache__" | Remove-Item -Recurse -Force -ErrorAction SilentlyContinue
Get-ChildItem -Path backend -Recurse -Filter "*.pyc" | Remove-Item -Force -ErrorAction SilentlyContinue

Write-Host "✅ Cache cleared" -ForegroundColor Green

# Step 8: Start backend with ESS
Write-Host "`n📋 Step 8: Starting backend with ESS..." -ForegroundColor Yellow

& .venv\Scripts\Activate.ps1

# Set staging environment variables
$env:QT_ESS_ENABLED = "true"
$env:QT_EXECUTION_EXCHANGE = "paper"  # Paper trading for staging
$env:STAGING_MODE = "true"
$env:QT_EVENT_DRIVEN_MODE = "true"

Write-Host "   Environment: STAGING (paper trading)" -ForegroundColor Cyan
Write-Host "   ESS: ENABLED" -ForegroundColor Cyan
Write-Host "   Starting uvicorn..." -ForegroundColor Cyan

# Start backend in background
$backendJob = Start-Job -ScriptBlock {
    param($workDir)
    Set-Location $workDir
    & .venv\Scripts\Activate.ps1
    python -m uvicorn backend.main:app --host 0.0.0.0 --port 8000 2>&1
} -ArgumentList (Get-Location)

Write-Host "   Backend starting (Job ID: $($backendJob.Id))..." -ForegroundColor Yellow
Start-Sleep -Seconds 10

# Step 9: Verify backend health
Write-Host "`n📋 Step 9: Verifying backend health..." -ForegroundColor Yellow

$maxRetries = 10
$retryCount = 0
$healthOk = $false

while ($retryCount -lt $maxRetries) {
    try {
        $health = Invoke-RestMethod -Uri "http://localhost:8000/health" -Method GET -TimeoutSec 5
        Write-Host "✅ Backend is healthy" -ForegroundColor Green
        $healthOk = $true
        break
    } catch {
        $retryCount++
        Write-Host "   Attempt $retryCount/$maxRetries - waiting for backend..." -ForegroundColor Yellow
        Start-Sleep -Seconds 3
    }
}

if (-not $healthOk) {
    Write-Host "❌ Backend failed to start" -ForegroundColor Red
    Write-Host "`nJob output:" -ForegroundColor Yellow
    Receive-Job -Job $backendJob
    Stop-Job -Job $backendJob
    Remove-Job -Job $backendJob
    exit 1
}

# Step 10: Verify ESS status
Write-Host "`n📋 Step 10: Verifying ESS status..." -ForegroundColor Yellow

try {
    $essStatus = Invoke-RestMethod -Uri "http://localhost:8000/api/emergency/status" -Method GET -TimeoutSec 5
    
    if ($essStatus.available) {
        Write-Host "✅ ESS is available" -ForegroundColor Green
        Write-Host "   Active: $($essStatus.active)" -ForegroundColor Cyan
        Write-Host "   Status: $($essStatus.status)" -ForegroundColor Cyan
        
        if ($essStatus.active) {
            Write-Host "`n⚠️  WARNING: ESS is currently ACTIVE" -ForegroundColor Red
            Write-Host "   Reason: $($essStatus.reason)" -ForegroundColor Red
            Write-Host "   Triggered by: $($essStatus.triggered_by)" -ForegroundColor Red
            Write-Host "   Timestamp: $($essStatus.timestamp)" -ForegroundColor Red
        }
    } else {
        Write-Host "❌ ESS not available: $($essStatus.message)" -ForegroundColor Red
        exit 1
    }
} catch {
    Write-Host "❌ Failed to verify ESS status: $_" -ForegroundColor Red
    exit 1
}

# Step 11: Run smoke tests
Write-Host "`n📋 Step 11: Running smoke tests..." -ForegroundColor Yellow

$smokeTestsPassed = $true

# Test 1: Get health
try {
    $health = Invoke-RestMethod -Uri "http://localhost:8000/health" -Method GET -TimeoutSec 5
    Write-Host "   ✅ Health endpoint" -ForegroundColor Green
} catch {
    Write-Host "   ❌ Health endpoint failed" -ForegroundColor Red
    $smokeTestsPassed = $false
}

# Test 2: Get ESS status
try {
    $status = Invoke-RestMethod -Uri "http://localhost:8000/api/emergency/status" -Method GET -TimeoutSec 5
    Write-Host "   ✅ ESS status endpoint" -ForegroundColor Green
} catch {
    Write-Host "   ❌ ESS status endpoint failed" -ForegroundColor Red
    $smokeTestsPassed = $false
}

# Test 3: Check PolicyStore
try {
    $policy = Invoke-RestMethod -Uri "http://localhost:8000/api/policy" -Method GET -TimeoutSec 5
    Write-Host "   ✅ PolicyStore endpoint" -ForegroundColor Green
} catch {
    Write-Host "   ⚠️  PolicyStore endpoint unavailable (non-critical)" -ForegroundColor Yellow
}

if ($smokeTestsPassed) {
    Write-Host "`n✅ All smoke tests passed" -ForegroundColor Green
} else {
    Write-Host "`n❌ Some smoke tests failed" -ForegroundColor Red
}

# Final summary
Write-Host "`n========================================" -ForegroundColor Cyan
Write-Host "DEPLOYMENT SUMMARY" -ForegroundColor Cyan
Write-Host "========================================" -ForegroundColor Cyan
Write-Host "Backend URL:    http://localhost:8000" -ForegroundColor White
Write-Host "API Docs:       http://localhost:8000/api/docs" -ForegroundColor White
Write-Host "ESS Status:     http://localhost:8000/api/emergency/status" -ForegroundColor White
Write-Host "Environment:    STAGING (paper trading)" -ForegroundColor White
Write-Host "Job ID:         $($backendJob.Id)" -ForegroundColor White
Write-Host "`nTo view logs:   Receive-Job -Job $($backendJob.Id) -Keep" -ForegroundColor Yellow
Write-Host "To stop:        Stop-Job -Job $($backendJob.Id); Remove-Job -Job $($backendJob.Id)" -ForegroundColor Yellow
Write-Host "`n✅ ESS deployment to staging complete!" -ForegroundColor Green
Write-Host "`n⚠️  Remember to run quarterly drills: .\scripts\ess-drill.ps1" -ForegroundColor Yellow
Write-Host "========================================`n" -ForegroundColor Cyan
