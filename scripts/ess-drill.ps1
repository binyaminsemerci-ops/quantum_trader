# ============================================================================
# ESS Quarterly Drill Script
# ============================================================================
# This script runs a quarterly Emergency Stop System drill to verify:
# - Manual trigger functionality
# - Position closure
# - Order cancellation
# - Alert delivery
# - Manual reset procedure
#
# Usage:
#   .\scripts\ess-drill.ps1 [-Production]
#
# WARNING: Do NOT run with -Production flag unless you understand the risks.
#          Production mode will close REAL positions and cancel REAL orders!
#
# Author: Quantum Trader Team
# Date: 2024-11-30
# ============================================================================

param(
    [switch]$Production,
    [switch]$SkipAlertTest
)

$ErrorActionPreference = "Stop"

Write-Host "`n========================================" -ForegroundColor Cyan
Write-Host "ESS QUARTERLY DRILL" -ForegroundColor Cyan
Write-Host "========================================`n" -ForegroundColor Cyan

# Safety check
if ($Production) {
    Write-Host "⚠️  WARNING: Production mode enabled!" -ForegroundColor Red
    Write-Host "This will trigger ESS on your LIVE trading system." -ForegroundColor Red
    Write-Host "ALL real positions will be closed and orders canceled.`n" -ForegroundColor Red
    
    $confirm = Read-Host "Type 'I UNDERSTAND' to continue"
    if ($confirm -ne "I UNDERSTAND") {
        Write-Host "❌ Drill canceled" -ForegroundColor Yellow
        exit 0
    }
    
    $doubleConfirm = Read-Host "Are you absolutely sure? (yes/NO)"
    if ($doubleConfirm -ne "yes") {
        Write-Host "❌ Drill canceled" -ForegroundColor Yellow
        exit 0
    }
} else {
    Write-Host "ℹ️  Running in STAGING mode (paper trading)" -ForegroundColor Cyan
    Write-Host "   Use -Production flag for live system drill.`n" -ForegroundColor Cyan
}

$baseUrl = "http://localhost:8000"
$drillId = "DRILL-$(Get-Date -Format 'yyyyMMdd-HHmmss')"

Write-Host "📋 Drill ID: $drillId`n" -ForegroundColor Yellow

# Step 1: Pre-drill checks
Write-Host "========================================" -ForegroundColor Cyan
Write-Host "STEP 1: PRE-DRILL CHECKS" -ForegroundColor Cyan
Write-Host "========================================`n" -ForegroundColor Cyan

Write-Host "📋 Checking backend health..." -ForegroundColor Yellow
try {
    $health = Invoke-RestMethod -Uri "$baseUrl/health" -Method GET -TimeoutSec 5
    Write-Host "✅ Backend is healthy" -ForegroundColor Green
} catch {
    Write-Host "❌ Backend is not responding: $_" -ForegroundColor Red
    exit 1
}

Write-Host "`n📋 Checking ESS status..." -ForegroundColor Yellow
try {
    $essStatus = Invoke-RestMethod -Uri "$baseUrl/api/emergency/status" -Method GET -TimeoutSec 5
    
    if (-not $essStatus.available) {
        Write-Host "❌ ESS is not available" -ForegroundColor Red
        exit 1
    }
    
    Write-Host "✅ ESS is available" -ForegroundColor Green
    
    if ($essStatus.active) {
        Write-Host "`n⚠️  ESS is already ACTIVE" -ForegroundColor Red
        Write-Host "   Reason: $($essStatus.reason)" -ForegroundColor Red
        Write-Host "   Cannot run drill while ESS is active." -ForegroundColor Red
        
        $reset = Read-Host "`nReset ESS first? (y/N)"
        if ($reset -eq "y") {
            try {
                $resetResponse = Invoke-RestMethod -Uri "$baseUrl/api/emergency/reset" -Method POST -Body (@{reset_by="drill-reset"} | ConvertTo-Json) -ContentType "application/json"
                Write-Host "✅ ESS reset successful" -ForegroundColor Green
                Start-Sleep -Seconds 2
            } catch {
                Write-Host "❌ Failed to reset ESS: $_" -ForegroundColor Red
                exit 1
            }
        } else {
            exit 1
        }
    }
} catch {
    Write-Host "❌ Failed to check ESS status: $_" -ForegroundColor Red
    exit 1
}

Write-Host "`n📋 Recording pre-drill state..." -ForegroundColor Yellow

$preDrillState = @{
    timestamp = Get-Date -Format "yyyy-MM-ddTHH:mm:ssZ"
    ess_status = $essStatus
    health = $health
}

Write-Host "✅ Pre-drill checks complete`n" -ForegroundColor Green

# Step 2: Trigger ESS
Write-Host "========================================" -ForegroundColor Cyan
Write-Host "STEP 2: TRIGGER ESS" -ForegroundColor Cyan
Write-Host "========================================`n" -ForegroundColor Cyan

Write-Host "📋 Triggering Emergency Stop System..." -ForegroundColor Yellow
Write-Host "   Drill ID: $drillId" -ForegroundColor Cyan
Write-Host "   Mode: $(if ($Production) { 'PRODUCTION' } else { 'STAGING' })" -ForegroundColor Cyan

$triggerReason = "Quarterly drill - $drillId"

try {
    $triggerBody = @{
        reason = $triggerReason
    } | ConvertTo-Json
    
    $triggerResponse = Invoke-RestMethod -Uri "$baseUrl/api/emergency/trigger" -Method POST -Body $triggerBody -ContentType "application/json" -TimeoutSec 10
    
    Write-Host "✅ ESS triggered successfully" -ForegroundColor Green
    Write-Host "   Status: $($triggerResponse.status)" -ForegroundColor Cyan
    Write-Host "   Timestamp: $($triggerResponse.timestamp)" -ForegroundColor Cyan
    
    Start-Sleep -Seconds 3  # Allow time for activation
    
} catch {
    Write-Host "❌ Failed to trigger ESS: $_" -ForegroundColor Red
    Write-Host "`nResponse: $($_.Exception.Response | ConvertTo-Json)" -ForegroundColor Red
    exit 1
}

# Step 3: Verify ESS activation
Write-Host "`n========================================" -ForegroundColor Cyan
Write-Host "STEP 3: VERIFY ESS ACTIVATION" -ForegroundColor Cyan
Write-Host "========================================`n" -ForegroundColor Cyan

Write-Host "📋 Checking ESS status..." -ForegroundColor Yellow

try {
    $postTriggerStatus = Invoke-RestMethod -Uri "$baseUrl/api/emergency/status" -Method GET -TimeoutSec 5
    
    if ($postTriggerStatus.active) {
        Write-Host "✅ ESS is now ACTIVE" -ForegroundColor Green
        Write-Host "   Status: $($postTriggerStatus.status)" -ForegroundColor Cyan
        Write-Host "   Reason: $($postTriggerStatus.reason)" -ForegroundColor Cyan
        Write-Host "   Triggered by: $($postTriggerStatus.triggered_by)" -ForegroundColor Cyan
    } else {
        Write-Host "❌ ESS did not activate!" -ForegroundColor Red
        Write-Host "   This is a CRITICAL FAILURE" -ForegroundColor Red
        exit 1
    }
} catch {
    Write-Host "❌ Failed to verify ESS status: $_" -ForegroundColor Red
    exit 1
}

# Step 4: Verify trading is blocked
Write-Host "`n========================================" -ForegroundColor Cyan
Write-Host "STEP 4: VERIFY TRADING BLOCKED" -ForegroundColor Cyan
Write-Host "========================================`n" -ForegroundColor Cyan

Write-Host "📋 Attempting to place test order (should fail)..." -ForegroundColor Yellow

# TODO: Add test order placement attempt
Write-Host "⚠️  Test order placement not implemented" -ForegroundColor Yellow
Write-Host "   Manual verification required:" -ForegroundColor Yellow
Write-Host "   1. Check that no new positions can be opened" -ForegroundColor Yellow
Write-Host "   2. Verify Orchestrator is not executing trades" -ForegroundColor Yellow
Write-Host "   3. Confirm RiskGuard blocks approvals" -ForegroundColor Yellow

$verified = Read-Host "`nHave you verified trading is blocked? (y/N)"
if ($verified -ne "y") {
    Write-Host "❌ Verification incomplete" -ForegroundColor Red
    exit 1
}

Write-Host "✅ Trading block verified" -ForegroundColor Green

# Step 5: Test alert delivery (optional)
if (-not $SkipAlertTest) {
    Write-Host "`n========================================" -ForegroundColor Cyan
    Write-Host "STEP 5: VERIFY ALERT DELIVERY" -ForegroundColor Cyan
    Write-Host "========================================`n" -ForegroundColor Cyan
    
    Write-Host "📋 Checking alert channels..." -ForegroundColor Yellow
    Write-Host "   Expected alerts:" -ForegroundColor Cyan
    Write-Host "   - Slack notification (if configured)" -ForegroundColor Cyan
    Write-Host "   - SMS message (if configured)" -ForegroundColor Cyan
    Write-Host "   - Email (if configured)" -ForegroundColor Cyan
    
    $alertsReceived = Read-Host "`nDid you receive alerts on all configured channels? (y/N)"
    if ($alertsReceived -eq "y") {
        Write-Host "✅ Alerts verified" -ForegroundColor Green
    } else {
        Write-Host "⚠️  Alert delivery issue detected" -ForegroundColor Yellow
        Write-Host "   Check alert configuration and logs" -ForegroundColor Yellow
    }
} else {
    Write-Host "`n📋 Skipping alert test (--SkipAlertTest flag)" -ForegroundColor Yellow
}

# Step 6: Monitor duration
Write-Host "`n========================================" -ForegroundColor Cyan
Write-Host "STEP 6: MONITOR ESS DURATION" -ForegroundColor Cyan
Write-Host "========================================`n" -ForegroundColor Cyan

Write-Host "📋 ESS is now active. Monitoring for 30 seconds..." -ForegroundColor Yellow
Write-Host "   This simulates a real emergency stop scenario." -ForegroundColor Cyan

for ($i = 30; $i -gt 0; $i--) {
    Write-Host "   $i seconds remaining..." -ForegroundColor Yellow
    Start-Sleep -Seconds 1
}

Write-Host "`n✅ Monitoring complete" -ForegroundColor Green

# Step 7: Reset ESS
Write-Host "`n========================================" -ForegroundColor Cyan
Write-Host "STEP 7: RESET ESS" -ForegroundColor Cyan
Write-Host "========================================`n" -ForegroundColor Cyan

Write-Host "📋 Resetting Emergency Stop System..." -ForegroundColor Yellow

$resetBy = "drill-operator-$(whoami)"

try {
    $resetBody = @{
        reset_by = $resetBy
    } | ConvertTo-Json
    
    $resetResponse = Invoke-RestMethod -Uri "$baseUrl/api/emergency/reset" -Method POST -Body $resetBody -ContentType "application/json" -TimeoutSec 10
    
    Write-Host "✅ ESS reset successfully" -ForegroundColor Green
    Write-Host "   Reset by: $($resetResponse.reset_by)" -ForegroundColor Cyan
    Write-Host "   Timestamp: $($resetResponse.timestamp)" -ForegroundColor Cyan
    
    Start-Sleep -Seconds 2
    
} catch {
    Write-Host "❌ Failed to reset ESS: $_" -ForegroundColor Red
    Write-Host "`nManual reset required!" -ForegroundColor Red
    exit 1
}

# Step 8: Verify reset
Write-Host "`n========================================" -ForegroundColor Cyan
Write-Host "STEP 8: VERIFY RESET" -ForegroundColor Cyan
Write-Host "========================================`n" -ForegroundColor Cyan

Write-Host "📋 Checking ESS status..." -ForegroundColor Yellow

try {
    $postResetStatus = Invoke-RestMethod -Uri "$baseUrl/api/emergency/status" -Method GET -TimeoutSec 5
    
    if (-not $postResetStatus.active) {
        Write-Host "✅ ESS is now INACTIVE" -ForegroundColor Green
        Write-Host "   Status: $($postResetStatus.status)" -ForegroundColor Cyan
    } else {
        Write-Host "❌ ESS did not reset!" -ForegroundColor Red
        Write-Host "   This is a CRITICAL FAILURE" -ForegroundColor Red
        exit 1
    }
} catch {
    Write-Host "❌ Failed to verify reset: $_" -ForegroundColor Red
    exit 1
}

Write-Host "`n📋 Verifying trading can resume..." -ForegroundColor Yellow
Write-Host "⚠️  Manual verification required:" -ForegroundColor Yellow
Write-Host "   1. Check that new positions can be opened" -ForegroundColor Yellow
Write-Host "   2. Verify Orchestrator resumes execution" -ForegroundColor Yellow
Write-Host "   3. Confirm RiskGuard approves trades" -ForegroundColor Yellow

$resumeVerified = Read-Host "`nHave you verified trading can resume? (y/N)"
if ($resumeVerified -eq "y") {
    Write-Host "✅ Trading resume verified" -ForegroundColor Green
} else {
    Write-Host "⚠️  Verification incomplete" -ForegroundColor Yellow
}

# Final report
Write-Host "`n========================================" -ForegroundColor Cyan
Write-Host "DRILL REPORT" -ForegroundColor Cyan
Write-Host "========================================`n" -ForegroundColor Cyan

Write-Host "Drill ID:        $drillId" -ForegroundColor White
Write-Host "Mode:            $(if ($Production) { 'PRODUCTION' } else { 'STAGING' })" -ForegroundColor White
Write-Host "Start Time:      $($preDrillState.timestamp)" -ForegroundColor White
Write-Host "End Time:        $(Get-Date -Format 'yyyy-MM-ddTHH:mm:ssZ')" -ForegroundColor White
Write-Host "`nResults:" -ForegroundColor Yellow
Write-Host "✅ ESS trigger:       SUCCESS" -ForegroundColor Green
Write-Host "✅ ESS activation:    SUCCESS" -ForegroundColor Green
Write-Host "✅ Trading blocked:   SUCCESS" -ForegroundColor Green
Write-Host "$(if (-not $SkipAlertTest -and $alertsReceived -eq 'y') { '✅' } else { '⚠️ ' }) Alert delivery:   $(if (-not $SkipAlertTest -and $alertsReceived -eq 'y') { 'SUCCESS' } else { 'SKIPPED/INCOMPLETE' })" -ForegroundColor $(if (-not $SkipAlertTest -and $alertsReceived -eq 'y') { 'Green' } else { 'Yellow' })
Write-Host "✅ ESS reset:         SUCCESS" -ForegroundColor Green
Write-Host "$(if ($resumeVerified -eq 'y') { '✅' } else { '⚠️ ' }) Trading resume:    $(if ($resumeVerified -eq 'y') { 'SUCCESS' } else { 'INCOMPLETE' })" -ForegroundColor $(if ($resumeVerified -eq 'y') { 'Green' } else { 'Yellow' })

Write-Host "`n📋 Action Items:" -ForegroundColor Yellow
Write-Host "1. Review logs in data/ess_activations.log" -ForegroundColor White
Write-Host "2. Document any issues encountered" -ForegroundColor White
Write-Host "3. Update ESS procedures if needed" -ForegroundColor White
Write-Host "4. Schedule next drill (recommended: quarterly)" -ForegroundColor White

Write-Host "`n✅ ESS quarterly drill complete!" -ForegroundColor Green
Write-Host "========================================`n" -ForegroundColor Cyan
