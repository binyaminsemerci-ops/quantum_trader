#!/usr/bin/env pwsh
# Phase 4N Validation Script (PowerShell)
# Tests Adaptive Leverage Engine functionality

Write-Host "======================================================================"
Write-Host "PHASE 4N - ADAPTIVE LEVERAGE ENGINE VALIDATION"
Write-Host "======================================================================"
Write-Host ""

# Test 1: Verify module imports
Write-Host "Test 1: Module Import Test"
python -c "from microservices.exitbrain_v3_5.adaptive_leverage_engine import AdaptiveLeverageEngine; print('✅ Import successful')"

if ($LASTEXITCODE -ne 0) {
    Write-Host "❌ Import failed" -ForegroundColor Red
    exit 1
}

# Test 2: Run unit tests
Write-Host ""
Write-Host "Test 2: Unit Tests"
python .\microservices\exitbrain_v3_5\adaptive_leverage_engine.py --test

if ($LASTEXITCODE -ne 0) {
    Write-Host "❌ Unit tests failed" -ForegroundColor Red
    exit 1
}

# Test 3: Simulate leverage calculations
Write-Host ""
Write-Host "Test 3: Leverage Calculation Simulation"
python -c @"
from microservices.exitbrain_v3_5.adaptive_leverage_engine import AdaptiveLeverageEngine
e = AdaptiveLeverageEngine()
levels = e.compute_levels(50, 1.2)
print(f'✅ 50x leverage with 1.2x volatility:')
print(f'   TP1={levels[\"tp1\"]:.3%}, TP2={levels[\"tp2\"]:.3%}, TP3={levels[\"tp3\"]:.3%}')
print(f'   SL={levels[\"sl\"]:.3%}, LSF={levels[\"LSF\"]:.4f}')
valid = e.validate_levels(levels, 50)
print(f'   Valid: {valid}')
"@

if ($LASTEXITCODE -ne 0) {
    Write-Host "❌ Calculation simulation failed" -ForegroundColor Red
    exit 1
}

# Test 4: Integration check (if Redis available)
Write-Host ""
Write-Host "Test 4: Integration Check"
$redisAvailable = (docker ps --filter name=quantum_redis --format "{{.Names}}") -eq "quantum_redis"

if ($redisAvailable) {
    Write-Host "✅ Redis container running"
    
    # Check PnL stream
    $streamLen = docker exec quantum_redis redis-cli XLEN "quantum:stream:exitbrain.pnl"
    Write-Host "   PnL stream entries: $streamLen"
    
    # Check health endpoint (if AI Engine running)
    try {
        $health = Invoke-RestMethod -Uri "http://localhost:8001/health" -TimeoutSec 3 -ErrorAction SilentlyContinue
        if ($health.adaptive_leverage_enabled) {
            Write-Host "✅ Adaptive leverage enabled in AI Engine health"
        } else {
            Write-Host "⚠️  Adaptive leverage not enabled (expected if not deployed)" -ForegroundColor Yellow
        }
    } catch {
        Write-Host "⚠️  AI Engine not responding (OK if not started)" -ForegroundColor Yellow
    }
} else {
    Write-Host "⚠️  Redis not running (skipping integration test)" -ForegroundColor Yellow
}

Write-Host ""
Write-Host "======================================================================"
Write-Host "✅ PHASE 4N VALIDATION COMPLETE" -ForegroundColor Green
Write-Host "======================================================================"
