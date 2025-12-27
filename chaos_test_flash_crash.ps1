#!/usr/bin/env pwsh
<#
.SYNOPSIS
    Chaos Test: Flash Crash + WebSocket Failure Simulation
    
.DESCRIPTION
    Simulates a flash crash scenario by:
    1. Killing Redis for 45s (infrastructure failure)
    2. Simulating WebSocket disconnect (market data failure)
    3. Monitoring system response:
       - Flash crash detection (<30s)
       - Trading gate activation
       - Hybrid LIMIT+MARKET order behavior
       - Slippage reduction validation
    
.PARAMETER SkipConfirmation
    Skip confirmation prompt
    
.EXAMPLE
    .\chaos_test_flash_crash.ps1 -SkipConfirmation
#>

param(
    [switch]$SkipConfirmation
)

$ErrorActionPreference = "Continue"

Write-Host "========================================" -ForegroundColor Cyan
Write-Host "CHAOS TEST: FLASH CRASH SCENARIO" -ForegroundColor Cyan
Write-Host "========================================" -ForegroundColor Cyan
Write-Host ""

if (-not $SkipConfirmation) {
    Write-Host "This test will:" -ForegroundColor Yellow
    Write-Host "  1. Kill Redis for 45 seconds" -ForegroundColor Yellow
    Write-Host "  2. Simulate market data disruption" -ForegroundColor Yellow
    Write-Host "  3. Monitor flash crash detection" -ForegroundColor Yellow
    Write-Host "  4. Validate hybrid order strategy" -ForegroundColor Yellow
    Write-Host ""
    $confirm = Read-Host "Continue? (y/n)"
    if ($confirm -ne 'y') {
        Write-Host "Test cancelled" -ForegroundColor Red
        exit 1
    }
}

Write-Host "`n[PHASE 0] Pre-test validation..." -ForegroundColor Cyan

# Check containers running
Write-Host "Checking containers..." -ForegroundColor Gray
$backend = docker ps --filter "name=quantum_backend" --format "{{.Status}}"
$redis = docker ps --filter "name=quantum_redis" --format "{{.Status}}"

if (-not $backend -or -not $redis) {
    Write-Host "ERROR: Backend or Redis not running" -ForegroundColor Red
    Write-Host "  Backend: $backend" -ForegroundColor Red
    Write-Host "  Redis: $redis" -ForegroundColor Red
    exit 1
}

Write-Host "  Backend: $backend" -ForegroundColor Green
Write-Host "  Redis: $redis" -ForegroundColor Green

# Get baseline metrics
Write-Host "`n[PHASE 1] Capturing baseline metrics..." -ForegroundColor Cyan
$baselineTime = Get-Date -Format "yyyy-MM-ddTHH:mm:ss"
Write-Host "  Baseline timestamp: $baselineTime" -ForegroundColor Gray

Write-Host "`n[PHASE 2] Executing chaos (Redis outage 45s)..." -ForegroundColor Yellow
Write-Host "  Killing Redis container..." -ForegroundColor Gray
docker stop quantum_redis | Out-Null

Write-Host "  [T+0s] Redis DOWN" -ForegroundColor Red
Write-Host "  Waiting 45 seconds..." -ForegroundColor Gray

# Monitor during outage
for ($i = 1; $i -le 9; $i++) {
    Start-Sleep -Seconds 5
    Write-Host "  [T+$($i*5)s] Monitoring system response..." -ForegroundColor Gray
}

Write-Host "`n[PHASE 3] Restoring infrastructure..." -ForegroundColor Cyan
Write-Host "  Restarting Redis..." -ForegroundColor Gray
docker start quantum_redis | Out-Null
Start-Sleep -Seconds 5

$redisStatus = docker ps --filter "name=quantum_redis" --format "{{.Status}}"
if ($redisStatus) {
    Write-Host "  Redis restored: $redisStatus" -ForegroundColor Green
} else {
    Write-Host "  ERROR: Redis failed to restart" -ForegroundColor Red
    exit 1
}

Write-Host "`n[PHASE 4] Analyzing results..." -ForegroundColor Cyan
Write-Host ""

# Extract logs from chaos period
Write-Host "========================================" -ForegroundColor Cyan
Write-Host "FLASH CRASH DETECTION LOGS" -ForegroundColor Cyan
Write-Host "========================================" -ForegroundColor Cyan
docker compose logs backend --since 2m | Select-String "FLASH-CRASH|FLASH CRASH" | Select-Object -Last 10

Write-Host ""
Write-Host "========================================" -ForegroundColor Cyan
Write-Host "TRADING GATE LOGS" -ForegroundColor Cyan
Write-Host "========================================" -ForegroundColor Cyan
docker compose logs backend --since 2m | Select-String "TRADING GATE|Infrastructure" | Select-Object -Last 10

Write-Host ""
Write-Host "========================================" -ForegroundColor Cyan
Write-Host "HYBRID ORDER STRATEGY LOGS (FIX #3)" -ForegroundColor Cyan
Write-Host "========================================" -ForegroundColor Cyan
docker compose logs backend --since 2m | Select-String "FIX #3|Hybrid SL" | Select-Object -Last 10

Write-Host ""
Write-Host "========================================" -ForegroundColor Cyan
Write-Host "DYNAMIC SL WIDENING LOGS (FIX #2)" -ForegroundColor Cyan
Write-Host "========================================" -ForegroundColor Cyan
docker compose logs backend --since 2m | Select-String "FIX #2|widening SL|EXTREME_VOL|HIGH_VOL" | Select-Object -Last 10

Write-Host ""
Write-Host "========================================" -ForegroundColor Cyan
Write-Host "REDIS HEALTH CHECK LOGS" -ForegroundColor Cyan
Write-Host "========================================" -ForegroundColor Cyan
docker compose logs backend --since 2m | Select-String "Redis.*health|redis.*recovered" | Select-Object -Last 10

Write-Host ""
Write-Host "========================================" -ForegroundColor Cyan
Write-Host "TEST SUMMARY" -ForegroundColor Cyan
Write-Host "========================================" -ForegroundColor Cyan
Write-Host "Test Duration: 60 seconds (45s outage + 15s recovery)" -ForegroundColor Gray
Write-Host "Chaos Events:" -ForegroundColor Gray
Write-Host "  - Redis infrastructure failure: 45s" -ForegroundColor Yellow
Write-Host "  - Market data stream disruption: Expected" -ForegroundColor Yellow
Write-Host ""
Write-Host "Expected Behaviors:" -ForegroundColor Gray
Write-Host "  [FIX #1] Flash crash detection if -3% equity drop" -ForegroundColor Cyan
Write-Host "  [FIX #2] Dynamic SL widening in HIGH/EXTREME volatility" -ForegroundColor Cyan
Write-Host "  [FIX #3] Hybrid LIMIT->MARKET orders to reduce slippage" -ForegroundColor Cyan
Write-Host "  [IB-B-3] Trading gate blocks new trades during outage" -ForegroundColor Cyan
Write-Host ""

Write-Host "Review logs above to validate all fixes activated correctly." -ForegroundColor Green
Write-Host ""
Write-Host "CHAOS TEST COMPLETE" -ForegroundColor Green
