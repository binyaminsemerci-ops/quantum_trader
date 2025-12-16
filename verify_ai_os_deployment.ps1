# AI-OS Full Deployment Verification Script
# Checks all 9 subsystems after backend restart

Write-Host "`n" -NoNewline
Write-Host "═══════════════════════════════════════════════════════════════" -ForegroundColor Cyan
Write-Host "   AI-OS FULL DEPLOYMENT VERIFICATION" -ForegroundColor Cyan
Write-Host "═══════════════════════════════════════════════════════════════" -ForegroundColor Cyan
Write-Host ""

Write-Host "[1] Checking backend status..." -ForegroundColor Yellow

Write-Host ""
Write-Host "═══════════════════════════════════════════════════════════════" -ForegroundColor Cyan
Write-Host "   SUBSYSTEM STATUS (Initialization + Runtime Activity)" -ForegroundColor Cyan
Write-Host "═══════════════════════════════════════════════════════════════" -ForegroundColor Cyan

# Check for both initialization logs (from startup) and runtime activity logs (recent)
$subsystems = @{
    "Self-Healing" = @("Self-Healing.*Initialized", "Self-Healing.*check")
    "Model Supervisor" = @("Model Supervisor.*Initialized", "Model Supervisor.*observation")
    "Retraining" = @("Retraining.*Available", "Retraining.*Initialized")
    "PIL" = @("PIL.*Initialized", "\[PIL\]")
    "PBA" = @("PBA.*Available", "PBA.*Initialized", "Portfolio.*Balancer", "PORTFOLIO_BALANCER")
    "PAL" = @("PAL.*Initialized", "\[PAL\]", "Profit Amplification")
    "Dynamic TP/SL" = @("Dynamic TP/SL.*Initialized", "DYNAMIC TP/SL", "DynamicTPSL")
    "Universe OS" = @("Universe OS.*Initialized", "\[Universe OS\]", "UNIVERSE_OS", "selection engine")
    "AI-HFOS" = @("AI-HFOS.*Initialized", "\[AI-HFOS\].*Coordination", "HFOS.*complete")
}

$initialized = 0
$failed = 0

foreach ($name in $subsystems.Keys) {
    $patterns = $subsystems[$name]
    $found = $false
    
    # Try each pattern (initialization or runtime activity)
    foreach ($pattern in $patterns) {
        $result = docker logs quantum_backend 2>&1 | Select-String -Pattern $pattern | Select-Object -Last 5
        $result = docker logs quantum_backend 2>&1 | Select-String -Pattern $pattern | Select-Object -Last 5
        
        if ($result) {
            $found = $true
            break
        }
    }
    
    if ($found) {
        Write-Host "  ✅ $name" -ForegroundColor Green
        $initialized++
    } else {
        # Check if it failed
        $failPattern = "$name.*failed|$name.*error"
        $failResult = docker logs quantum_backend 2>&1 | Select-String -Pattern $failPattern | Select-Object -Last 2
        
        if ($failResult) {
            Write-Host "  ❌ $name (initialization failed)" -ForegroundColor Red
            $failed++
        } else {
            Write-Host "  ⚠️  $name (not found in logs)" -ForegroundColor Yellow
        }
    }
}

Write-Host ""
Write-Host "═══════════════════════════════════════════════════════════════" -ForegroundColor Cyan
Write-Host "   INTEGRATION STAGE" -ForegroundColor Cyan
Write-Host "═══════════════════════════════════════════════════════════════" -ForegroundColor Cyan

$stageResult = docker logs quantum_backend 2>&1 | Select-String -Pattern "Stage=|integration_stage" | Select-Object -Last 5
if ($stageResult) {
    if ($stageResult -match "Stage=AUTONOMY|AUTONOMY") {
        Write-Host "  ✅ Integration Stage: AUTONOMY (FULL DEPLOYMENT)" -ForegroundColor Green
    } elseif ($stageResult -match "Stage=OBSERVATION|OBSERVATION") {
        Write-Host "  ⚠️  Integration Stage: OBSERVATION (monitoring only)" -ForegroundColor Yellow
    } else {
        Write-Host "  ⚠️  Integration Stage: $($stageResult | Select-Object -Last 1)" -ForegroundColor Yellow
    }
} else {
    Write-Host "  ❌ Integration Stage: NOT FOUND" -ForegroundColor Red
}

Write-Host ""
Write-Host "═══════════════════════════════════════════════════════════════" -ForegroundColor Cyan
Write-Host "   RUNTIME ACTIVITY (Last 2 Minutes)" -ForegroundColor Cyan
Write-Host "═══════════════════════════════════════════════════════════════" -ForegroundColor Cyan

# Check for recent AI-OS activity
$aiHfosActivity = docker logs quantum_backend --since 2m 2>&1 | Select-String -Pattern "AI-HFOS.*Coordination complete" | Select-Object -Last 1
$universeActivity = docker logs quantum_backend --since 2m 2>&1 | Select-String -Pattern "Universe OS.*ENFORCED|selection engine" | Select-Object -Last 1
$pbaActivity = docker logs quantum_backend --since 2m 2>&1 | Select-String -Pattern "PORTFOLIO_BALANCER|Portfolio.*Balancer" | Select-Object -Last 1

if ($aiHfosActivity) {
    Write-Host "  ✅ AI-HFOS: Active (coordination cycles running)" -ForegroundColor Green
} else {
    Write-Host "  ⚠️  AI-HFOS: No recent activity" -ForegroundColor Yellow
}

if ($universeActivity) {
    Write-Host "  ✅ Universe OS: Active (symbol filtering)" -ForegroundColor Green
} else {
    Write-Host "  ⚠️  Universe OS: No recent activity" -ForegroundColor Yellow
}

if ($pbaActivity) {
    Write-Host "  ✅ Portfolio Balancer: Active (monitoring)" -ForegroundColor Green
} else {
    Write-Host "  ⚠️  Portfolio Balancer: No recent activity" -ForegroundColor Yellow
}

Write-Host ""
Write-Host "═══════════════════════════════════════════════════════════════" -ForegroundColor Cyan
Write-Host "   BACKEND STATUS" -ForegroundColor Cyan
Write-Host "═══════════════════════════════════════════════════════════════" -ForegroundColor Cyan

$uvicornRunning = docker logs quantum_backend 2>&1 | Select-String -Pattern "Uvicorn running" | Select-Object -Last 1
$startupFailed = docker logs quantum_backend --since 5m 2>&1 | Select-String -Pattern "Application startup failed"

if ($uvicornRunning) {
    Write-Host "  ✅ Backend: RUNNING" -ForegroundColor Green
    $backendUp = $true
} elseif ($startupFailed) {
    Write-Host "  ❌ Backend: STARTUP FAILED" -ForegroundColor Red
    $backendUp = $false
    Write-Host ""
    Write-Host "     Error details:" -ForegroundColor Red
    docker logs quantum_backend --tail 30 2>&1 | Select-String -Pattern "ERROR" | Select-Object -Last 3 | ForEach-Object {
        Write-Host "     $_" -ForegroundColor DarkRed
    }
} else {
    Write-Host "  ⚠️  Backend: STATUS UNKNOWN" -ForegroundColor Yellow
    $backendUp = $false
}

Write-Host ""
Write-Host "═══════════════════════════════════════════════════════════════" -ForegroundColor Cyan
Write-Host "   SUMMARY" -ForegroundColor Cyan
Write-Host "═══════════════════════════════════════════════════════════════" -ForegroundColor Cyan
Write-Host ""
Write-Host "  Subsystems Initialized: $initialized/9" -ForegroundColor $(if ($initialized -ge 7) { "Green" } elseif ($initialized -ge 5) { "Yellow" } else { "Red" })
Write-Host "  Subsystems Failed: $failed" -ForegroundColor $(if ($failed -eq 0) { "Green" } else { "Red" })

if ($backendUp -and $initialized -ge 7) {
    Write-Host ""
    Write-Host "  🎉 DEPLOYMENT STATUS: SUCCESS" -ForegroundColor Green
    Write-Host "  AI-OS is operational with $initialized subsystems active" -ForegroundColor Green
    Write-Host ""
    Write-Host "  Next steps:" -ForegroundColor Cyan
    Write-Host "    - Monitor AI-OS decision-making: docker logs quantum_backend -f | Select-String 'PIL|PAL|PBA'" -ForegroundColor Gray
    Write-Host "    - Check Dynamic TP/SL: docker logs quantum_backend -f | Select-String 'DYNAMIC TP/SL'" -ForegroundColor Gray
    Write-Host "    - Watch Self-Healing: docker logs quantum_backend -f | Select-String 'SELF-HEALING|directive'" -ForegroundColor Gray
} elseif ($backendUp) {
    Write-Host ""
    Write-Host "  ⚠️  DEPLOYMENT STATUS: PARTIAL" -ForegroundColor Yellow
    Write-Host "  Backend is running but some subsystems failed to initialize" -ForegroundColor Yellow
    Write-Host "  System will operate with reduced AI capabilities (fail-safe mode)" -ForegroundColor Yellow
} else {
    Write-Host ""
    Write-Host "  ❌ DEPLOYMENT STATUS: FAILED" -ForegroundColor Red
    Write-Host "  Backend failed to start - check logs above for details" -ForegroundColor Red
}

Write-Host ""
Write-Host "═══════════════════════════════════════════════════════════════" -ForegroundColor Cyan
Write-Host ""
