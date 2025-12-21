#!/usr/bin/env pwsh
# PHASE 4M+ VALIDATION SCRIPT
# Validates Cross-Exchange Intelligence → ExitBrain v3 Integration

Write-Host "=" * 70 -ForegroundColor Cyan
Write-Host "PHASE 4M+ VALIDATION - Cross-Exchange → ExitBrain v3 Integration" -ForegroundColor Cyan
Write-Host "=" * 70 -ForegroundColor Cyan
Write-Host ""

$ErrorCount = 0
$WarningCount = 0
$TestCount = 0

function Test-Step {
    param(
        [string]$Name,
        [scriptblock]$Action,
        [scriptblock]$Validation
    )
    
    $script:TestCount++
    Write-Host "[$script:TestCount] $Name..." -ForegroundColor Yellow
    
    try {
        $result = & $Action
        $validated = & $Validation $result
        
        if ($validated) {
            Write-Host "    ✅ PASS" -ForegroundColor Green
            return $true
        } else {
            Write-Host "    ❌ FAIL" -ForegroundColor Red
            $script:ErrorCount++
            return $false
        }
    } catch {
        Write-Host "    ❌ ERROR: $_" -ForegroundColor Red
        $script:ErrorCount++
        return $false
    }
}

# ============================================================================
# TEST 1: Cross-Exchange Data Stream
# ============================================================================
Write-Host "`n[CROSS-EXCHANGE DATA VALIDATION]" -ForegroundColor Cyan
Write-Host "-" * 70

Test-Step "Check quantum:stream:exchange.raw exists" `
    { docker exec quantum_redis redis-cli EXISTS "quantum:stream:exchange.raw" } `
    { param($r) $r -eq "1" }

Test-Step "Check exchange.raw has data (> 100 entries)" `
    { [int](docker exec quantum_redis redis-cli XLEN "quantum:stream:exchange.raw") } `
    { param($r) $r -gt 100 }

Test-Step "Check quantum:stream:exchange.normalized stream created" `
    { docker exec quantum_redis redis-cli EXISTS "quantum:stream:exchange.normalized" } `
    { param($r) $r -eq "1" }

# ============================================================================
# TEST 2: ExitBrain Status Stream
# ============================================================================
Write-Host "`n[EXITBRAIN STATUS STREAM]" -ForegroundColor Cyan
Write-Host "-" * 70

Test-Step "Check quantum:stream:exitbrain.status stream exists" `
    { docker exec quantum_redis redis-cli EXISTS "quantum:stream:exitbrain.status" } `
    { param($r) $r -eq "1" }

Test-Step "Check exitbrain.status has recent data" `
    {
        $entries = docker exec quantum_redis redis-cli XREVRANGE "quantum:stream:exitbrain.status" + - COUNT 1
        if ($entries) {
            $entries | Out-String
        } else {
            throw "No entries"
        }
    } `
    { param($r) $r -ne $null }

# ============================================================================
# TEST 3: Cross-Exchange Adapter Health
# ============================================================================
Write-Host "`n[CROSS-EXCHANGE ADAPTER HEALTH]" -ForegroundColor Cyan
Write-Host "-" * 70

Test-Step "Check AI Engine /health endpoint" `
    {
        try {
            $response = Invoke-RestMethod -Uri "http://localhost:8001/health" -Method Get -TimeoutSec 5
            $response | ConvertTo-Json -Depth 10
        } catch {
            throw "Health endpoint not responding"
        }
    } `
    { param($r) $r -match "cross_exchange" }

# ============================================================================
# TEST 4: Cross-Exchange State Verification
# ============================================================================
Write-Host "`n[CROSS-EXCHANGE STATE]" -ForegroundColor Cyan
Write-Host "-" * 70

$latestState = docker exec quantum_redis redis-cli --raw XREVRANGE "quantum:stream:exchange.normalized" + - COUNT 1

if ($latestState) {
    Write-Host "✓ Latest normalized data:" -ForegroundColor Green
    Write-Host $latestState
} else {
    Write-Host "⚠️  No normalized stream data yet (aggregator may not be running)" -ForegroundColor Yellow
    $WarningCount++
}

# ============================================================================
# TEST 5: ExitBrain Integration Logs
# ============================================================================
Write-Host "`n[EXITBRAIN INTEGRATION LOGS]" -ForegroundColor Cyan
Write-Host "-" * 70

Test-Step "Check for cross-exchange adapter initialization" `
    {
        docker logs quantum_ai_engine 2>&1 | Select-String -Pattern "Cross-Exchange" | Select-Object -Last 5
    } `
    { param($r) $r.Count -gt 0 }

Test-Step "Check for ATR adjustments in logs" `
    {
        docker logs quantum_ai_engine 2>&1 | Select-String -Pattern "adjustments applied|ATR|volatility_factor" | Select-Object -Last 3
    } `
    { param($r) $r.Count -gt 0 }

# ============================================================================
# TEST 6: Alert Stream (Fail-Safe Monitoring)
# ============================================================================
Write-Host "`n[FAIL-SAFE MONITORING]" -ForegroundColor Cyan
Write-Host "-" * 70

$alerts = docker exec quantum_redis redis-cli XLEN "quantum:stream:exitbrain.alerts"

if ([int]$alerts -eq 0) {
    Write-Host "✓ No fallback alerts (system operating normally)" -ForegroundColor Green
} else {
    Write-Host "⚠️  Found $alerts alerts, checking recent:" -ForegroundColor Yellow
    docker exec quantum_redis redis-cli --raw XREVRANGE "quantum:stream:exitbrain.alerts" + - COUNT 3
    $WarningCount++
}

# ============================================================================
# SUMMARY
# ============================================================================
Write-Host "`n" + "=" * 70 -ForegroundColor Cyan
Write-Host "VALIDATION SUMMARY" -ForegroundColor Cyan
Write-Host "=" * 70 -ForegroundColor Cyan

Write-Host "`nTests Run: $TestCount" -ForegroundColor White
Write-Host "Errors: $ErrorCount" -ForegroundColor $(if ($ErrorCount -eq 0) { "Green" } else { "Red" })
Write-Host "Warnings: $WarningCount" -ForegroundColor $(if ($WarningCount -eq 0) { "Green" } else { "Yellow" })

if ($ErrorCount -eq 0) {
    Write-Host "`n✅ PHASE 4M+ INTEGRATION VALIDATED" -ForegroundColor Green
    Write-Host "   Cross-Exchange Intelligence → ExitBrain v3 is operational" -ForegroundColor Green
    exit 0
} else {
    Write-Host "`n❌ VALIDATION FAILED" -ForegroundColor Red
    Write-Host "   Fix errors and re-run validation" -ForegroundColor Red
    exit 1
}
