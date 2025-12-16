#!/usr/bin/env pwsh
<#
.SYNOPSIS
    Quick-start script to resolve GO-LIVE blockers and prepare system for activation.

.DESCRIPTION
    This script:
    1. Starts backend service
    2. Waits for health endpoints to respond
    3. Verifies metrics endpoint
    4. Re-runs preflight checks
    5. Reports status

.EXAMPLE
    .\scripts\resolve_blockers.ps1
#>

$ErrorActionPreference = "Stop"

Write-Host ""
Write-Host "=" * 80 -ForegroundColor Cyan
Write-Host "QUANTUM TRADER v2.0 - BLOCKER RESOLUTION SCRIPT" -ForegroundColor Cyan
Write-Host "=" * 80 -ForegroundColor Cyan
Write-Host ""

# Configuration
$BackendHost = "localhost"
$BackendPort = 8000
$HealthTimeout = 60  # seconds
$MetricsTimeout = 30  # seconds

# Step 1: Check if backend already running
Write-Host "[STEP 1] Checking if backend service is already running..." -ForegroundColor Yellow
$existingProcess = Get-Process -Name "python*" -ErrorAction SilentlyContinue | Where-Object { 
    $_.CommandLine -like "*uvicorn*" -or $_.CommandLine -like "*backend.main*" 
}

if ($existingProcess) {
    Write-Host "  ⚠️  Backend service appears to be running (PID: $($existingProcess.Id))" -ForegroundColor Yellow
    Write-Host "  Checking if responsive..." -ForegroundColor Yellow
    
    try {
        $response = Invoke-WebRequest -Uri "http://${BackendHost}:${BackendPort}/health/live" -UseBasicParsing -TimeoutSec 5
        if ($response.StatusCode -eq 200) {
            Write-Host "  ✅ Backend service is healthy and responsive" -ForegroundColor Green
            $serviceStarted = $true
        } else {
            Write-Host "  ❌ Backend service not responding correctly (status: $($response.StatusCode))" -ForegroundColor Red
            Write-Host "  Killing existing process and restarting..." -ForegroundColor Yellow
            Stop-Process -Id $existingProcess.Id -Force
            Start-Sleep -Seconds 2
            $serviceStarted = $false
        }
    } catch {
        Write-Host "  ❌ Backend service not responding (error: $($_.Exception.Message))" -ForegroundColor Red
        Write-Host "  Killing existing process and restarting..." -ForegroundColor Yellow
        Stop-Process -Id $existingProcess.Id -Force
        Start-Sleep -Seconds 2
        $serviceStarted = $false
    }
} else {
    Write-Host "  ℹ️  No backend service currently running" -ForegroundColor Cyan
    $serviceStarted = $false
}

# Step 2: Start backend service if needed
if (-not $serviceStarted) {
    Write-Host ""
    Write-Host "[STEP 2] Starting backend service..." -ForegroundColor Yellow
    Write-Host "  Command: uvicorn backend.main:app --host $BackendHost --port $BackendPort" -ForegroundColor Gray
    
    # Start backend in new window
    $backendProcess = Start-Process -FilePath "python" -ArgumentList "-m", "uvicorn", "backend.main:app", "--host", $BackendHost, "--port", $BackendPort -PassThru -WindowStyle Normal
    
    if ($backendProcess) {
        Write-Host "  ✅ Backend service started (PID: $($backendProcess.Id))" -ForegroundColor Green
        Write-Host "  ℹ️  Service running in separate window" -ForegroundColor Cyan
    } else {
        Write-Host "  ❌ Failed to start backend service" -ForegroundColor Red
        exit 1
    }
} else {
    Write-Host ""
    Write-Host "[STEP 2] Backend service already running - skipping start" -ForegroundColor Green
}

# Step 3: Wait for health endpoints
Write-Host ""
Write-Host "[STEP 3] Waiting for health endpoints to respond..." -ForegroundColor Yellow
Write-Host "  Endpoint: http://${BackendHost}:${BackendPort}/health/live" -ForegroundColor Gray
Write-Host "  Timeout: ${HealthTimeout}s" -ForegroundColor Gray

$healthReady = $false
$elapsed = 0
$interval = 2  # seconds between checks

while (-not $healthReady -and $elapsed -lt $HealthTimeout) {
    try {
        $response = Invoke-WebRequest -Uri "http://${BackendHost}:${BackendPort}/health/live" -UseBasicParsing -TimeoutSec 5
        if ($response.StatusCode -eq 200) {
            Write-Host "  ✅ Liveness endpoint responding (${elapsed}s elapsed)" -ForegroundColor Green
            $healthReady = $true
        }
    } catch {
        Write-Host "  ⏳ Waiting for service... (${elapsed}s / ${HealthTimeout}s)" -ForegroundColor Gray
        Start-Sleep -Seconds $interval
        $elapsed += $interval
    }
}

if (-not $healthReady) {
    Write-Host "  ❌ Health endpoint did not respond within ${HealthTimeout}s" -ForegroundColor Red
    Write-Host "  Check backend service logs for errors" -ForegroundColor Yellow
    exit 1
}

# Step 4: Verify readiness endpoint
Write-Host ""
Write-Host "[STEP 4] Verifying readiness endpoint..." -ForegroundColor Yellow
Write-Host "  Endpoint: http://${BackendHost}:${BackendPort}/health/ready" -ForegroundColor Gray

try {
    $response = Invoke-WebRequest -Uri "http://${BackendHost}:${BackendPort}/health/ready" -UseBasicParsing -TimeoutSec 5
    if ($response.StatusCode -eq 200) {
        Write-Host "  ✅ Readiness endpoint responding" -ForegroundColor Green
    } else {
        Write-Host "  ⚠️  Readiness endpoint returned status $($response.StatusCode)" -ForegroundColor Yellow
    }
} catch {
    Write-Host "  ⚠️  Readiness endpoint not available: $($_.Exception.Message)" -ForegroundColor Yellow
    Write-Host "  This may be normal if endpoint not implemented" -ForegroundColor Gray
}

# Step 5: Verify metrics endpoint
Write-Host ""
Write-Host "[STEP 5] Verifying metrics endpoint..." -ForegroundColor Yellow
Write-Host "  Endpoint: http://${BackendHost}:${BackendPort}/metrics" -ForegroundColor Gray

$metricsReady = $false
$elapsed = 0

while (-not $metricsReady -and $elapsed -lt $MetricsTimeout) {
    try {
        $response = Invoke-WebRequest -Uri "http://${BackendHost}:${BackendPort}/metrics" -UseBasicParsing -TimeoutSec 5
        if ($response.StatusCode -eq 200) {
            Write-Host "  ✅ Metrics endpoint responding" -ForegroundColor Green
            
            # Check for required metrics
            $content = $response.Content
            $requiredMetrics = @(
                "risk_gate_decisions_total",
                "ess_triggers_total",
                "exchange_failover_events_total"
            )
            
            Write-Host "  Checking for required metrics..." -ForegroundColor Gray
            $allPresent = $true
            foreach ($metric in $requiredMetrics) {
                if ($content -like "*$metric*") {
                    Write-Host "    ✅ $metric" -ForegroundColor Green
                } else {
                    Write-Host "    ❌ $metric (missing)" -ForegroundColor Red
                    $allPresent = $false
                }
            }
            
            if ($allPresent) {
                Write-Host "  ✅ All required metrics present" -ForegroundColor Green
            } else {
                Write-Host "  ⚠️  Some metrics missing - observability may be limited" -ForegroundColor Yellow
            }
            
            $metricsReady = $true
        }
    } catch {
        Write-Host "  ⏳ Waiting for metrics... (${elapsed}s / ${MetricsTimeout}s)" -ForegroundColor Gray
        Start-Sleep -Seconds $interval
        $elapsed += $interval
    }
}

if (-not $metricsReady) {
    Write-Host "  ⚠️  Metrics endpoint did not respond within ${MetricsTimeout}s" -ForegroundColor Yellow
    Write-Host "  Preflight observability check may fail" -ForegroundColor Yellow
}

# Step 6: Re-run preflight checks
Write-Host ""
Write-Host "[STEP 6] Re-running preflight checks..." -ForegroundColor Yellow
Write-Host "  Command: python scripts/preflight_check.py" -ForegroundColor Gray
Write-Host ""

$preflightResult = & python scripts/preflight_check.py
$preflightExitCode = $LASTEXITCODE

Write-Host ""
if ($preflightExitCode -eq 0) {
    Write-Host "  ✅ Preflight checks PASSED" -ForegroundColor Green
} else {
    Write-Host "  ⚠️  Preflight checks FAILED (exit code: $preflightExitCode)" -ForegroundColor Yellow
    Write-Host "  Review output above for details" -ForegroundColor Gray
}

# Step 7: Summary
Write-Host ""
Write-Host "=" * 80 -ForegroundColor Cyan
Write-Host "BLOCKER RESOLUTION SUMMARY" -ForegroundColor Cyan
Write-Host "=" * 80 -ForegroundColor Cyan
Write-Host ""

Write-Host "Backend Service:" -ForegroundColor White
if ($serviceStarted -or $healthReady) {
    Write-Host "  ✅ Running and healthy" -ForegroundColor Green
    Write-Host "  URL: http://${BackendHost}:${BackendPort}" -ForegroundColor Gray
} else {
    Write-Host "  ❌ Not healthy" -ForegroundColor Red
}

Write-Host ""
Write-Host "Health Endpoints:" -ForegroundColor White
if ($healthReady) {
    Write-Host "  ✅ Responding" -ForegroundColor Green
} else {
    Write-Host "  ❌ Not responding" -ForegroundColor Red
}

Write-Host ""
Write-Host "Metrics Endpoint:" -ForegroundColor White
if ($metricsReady) {
    Write-Host "  ✅ Responding" -ForegroundColor Green
} else {
    Write-Host "  ⚠️  Not responding" -ForegroundColor Yellow
}

Write-Host ""
Write-Host "Preflight Checks:" -ForegroundColor White
if ($preflightExitCode -eq 0) {
    Write-Host "  ✅ PASSED (6/6)" -ForegroundColor Green
} else {
    Write-Host "  ❌ FAILED" -ForegroundColor Red
}

Write-Host ""
Write-Host "=" * 80 -ForegroundColor Cyan

if ($preflightExitCode -eq 0) {
    Write-Host ""
    Write-Host "✅ ALL BLOCKERS RESOLVED" -ForegroundColor Green
    Write-Host ""
    Write-Host "Next Steps:" -ForegroundColor White
    Write-Host "  1. Proceed to Phase 2: Config Freeze" -ForegroundColor Gray
    Write-Host "  2. Run: Review CONFIG_FREEZE_SNAPSHOT.yaml" -ForegroundColor Gray
    Write-Host "  3. Proceed to Phase 3: GO-LIVE Activation" -ForegroundColor Gray
    Write-Host ""
    exit 0
} else {
    Write-Host ""
    Write-Host "⚠️  BLOCKERS REMAIN" -ForegroundColor Yellow
    Write-Host ""
    Write-Host "Actions Required:" -ForegroundColor White
    Write-Host "  1. Check backend service logs for errors" -ForegroundColor Gray
    Write-Host "  2. Verify database/Redis connectivity" -ForegroundColor Gray
    Write-Host "  3. Re-run this script after resolving issues" -ForegroundColor Gray
    Write-Host ""
    exit 1
}
