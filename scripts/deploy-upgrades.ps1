#!/usr/bin/env pwsh
# DEPLOYMENT SCRIPT - QUANTUM TRADER UPGRADES
# Date: 2025-11-23 02:30 UTC
# Features: Countertrend Short Filter + Model Supervisor (Observation Mode)

Write-Host ""
Write-Host "========================================" -ForegroundColor Cyan
Write-Host "  QUANTUM TRADER - UPGRADE DEPLOYMENT" -ForegroundColor Cyan
Write-Host "========================================" -ForegroundColor Cyan
Write-Host ""

# Step 1: Show current status
Write-Host "[1/5] Checking current backend status..." -ForegroundColor Yellow
docker-compose ps quantum_backend

# Step 2: Restart backend to load new code
Write-Host ""
Write-Host "[2/5] Restarting backend to load upgrades..." -ForegroundColor Yellow
docker-compose restart quantum_backend

Write-Host ""
Write-Host "Waiting 10 seconds for backend to initialize..." -ForegroundColor Gray
Start-Sleep -Seconds 10

# Step 3: Verify backend is running
Write-Host ""
Write-Host "[3/5] Verifying backend health..." -ForegroundColor Yellow
try {
    $health = Invoke-WebRequest -Uri "http://localhost:8000/health" -UseBasicParsing -TimeoutSec 5
    Write-Host "✅ Backend is healthy" -ForegroundColor Green
} catch {
    Write-Host "⚠️  Backend health check failed: $_" -ForegroundColor Red
    Write-Host "Check logs with: docker logs quantum_backend" -ForegroundColor Yellow
}

# Step 4: Check configuration
Write-Host ""
Write-Host "[4/5] Checking configuration..." -ForegroundColor Yellow
Write-Host ""
Write-Host "QT_COUNTERTREND_MIN_CONF:" -ForegroundColor Cyan
docker exec quantum_backend env 2>$null | Select-String "QT_COUNTERTREND_MIN_CONF"
if (-not $?) {
    Write-Host "  Not set (will use default: 0.55)" -ForegroundColor Gray
}

Write-Host ""
Write-Host "MODEL_SUPERVISOR_MODE:" -ForegroundColor Cyan
docker exec quantum_backend env 2>$null | Select-String "MODEL_SUPERVISOR_MODE"
if (-not $?) {
    Write-Host "  Not set (will use default: OBSERVE)" -ForegroundColor Gray
}

# Step 5: Show recent logs
Write-Host ""
Write-Host "[5/5] Recent logs (last 20 lines)..." -ForegroundColor Yellow
Write-Host ""
docker logs quantum_backend --tail 20

Write-Host ""
Write-Host "========================================" -ForegroundColor Green
Write-Host "  DEPLOYMENT COMPLETE!" -ForegroundColor Green
Write-Host "========================================" -ForegroundColor Green
Write-Host ""

# Instructions
Write-Host "NEXT STEPS:" -ForegroundColor Cyan
Write-Host ""
Write-Host "1. Monitor for new features:" -ForegroundColor White
Write-Host "   docker logs -f quantum_backend 2>&1 | Select-String 'SHORT_ALLOWED|SHORT_BLOCKED|MODEL_SUPERVISOR'" -ForegroundColor Gray
Write-Host ""
Write-Host "2. Check for trade execution:" -ForegroundColor White
Write-Host "   docker logs -f quantum_backend 2>&1 | Select-String 'Trade OPENED|orders_submitted'" -ForegroundColor Gray
Write-Host ""
Write-Host "3. Adjust threshold if needed (optional):" -ForegroundColor White
Write-Host "   Add to .env: QT_COUNTERTREND_MIN_CONF=0.55" -ForegroundColor Gray
Write-Host ""

# Expected behavior
Write-Host "EXPECTED BEHAVIOR:" -ForegroundColor Cyan
Write-Host ""
Write-Host "✅ HIGH confidence SHORT in uptrend:" -ForegroundColor Green
Write-Host "   → Will see: SHORT_ALLOWED_AGAINST_TREND_HIGH_CONF" -ForegroundColor Gray
Write-Host "   → Trade will execute" -ForegroundColor Gray
Write-Host ""
Write-Host "❌ LOW confidence SHORT in uptrend:" -ForegroundColor Yellow
Write-Host "   → Will see: SHORT_BLOCKED_AGAINST_TREND_LOW_CONF" -ForegroundColor Gray
Write-Host "   → Trade will be rejected (safety preserved)" -ForegroundColor Gray
Write-Host ""
Write-Host "🔍 Model bias detection:" -ForegroundColor Cyan
Write-Host "   → Will see: SHORT BIAS DETECTED (if >70% shorts in uptrend)" -ForegroundColor Gray
Write-Host "   → No action taken (observation only)" -ForegroundColor Gray
Write-Host ""

Write-Host "Press any key to exit..."
$null = $Host.UI.RawUI.ReadKey("NoEcho,IncludeKeyDown")
