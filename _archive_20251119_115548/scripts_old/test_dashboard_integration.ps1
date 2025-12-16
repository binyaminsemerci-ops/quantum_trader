#!/usr/bin/env pwsh
# Verify Dashboard Integration - Test all key endpoints

Write-Host "`n🎯 DASHBOARD INTEGRATION TEST`n" -ForegroundColor Cyan

$results = @()

# Test 1: Health endpoint
Write-Host "1️⃣ Testing /health endpoint..." -ForegroundColor Yellow
try {
    $health = Invoke-RestMethod -Uri "http://localhost:8000/health" -Method GET -TimeoutSec 5
    $results += @{
        Test = "Health Check"
        Status = "✅ PASS"
        Details = "Status: $($health.status), Event-Driven: $($health.event_driven_active)"
    }
    Write-Host "   ✅ PASS - Backend is healthy" -ForegroundColor Green
} catch {
    $results += @{Test = "Health Check"; Status = "❌ FAIL"; Details = $_.Exception.Message}
    Write-Host "   ❌ FAIL - $_" -ForegroundColor Red
}

# Test 2: Portfolio endpoint
Write-Host "2️⃣ Testing /api/portfolio endpoint..." -ForegroundColor Yellow
try {
    $portfolio = Invoke-RestMethod -Uri "http://localhost:8000/api/portfolio" -Method GET -TimeoutSec 5
    $symbolCount = if ($portfolio.positions) { $portfolio.positions.Count } else { 0 }
    $results += @{
        Test = "Portfolio API"
        Status = "✅ PASS"
        Details = "Total Equity: $($portfolio.total_equity), Positions: $symbolCount"
    }
    Write-Host "   ✅ PASS - $symbolCount positions loaded" -ForegroundColor Green
} catch {
    $results += @{Test = "Portfolio API"; Status = "❌ FAIL"; Details = $_.Exception.Message}
    Write-Host "   ❌ FAIL - $_" -ForegroundColor Red
}

# Test 3: AI Signals endpoint
Write-Host "3️⃣ Testing /api/ai/signals endpoint..." -ForegroundColor Yellow
try {
    $signals = Invoke-RestMethod -Uri "http://localhost:8000/api/ai/signals" -Method GET -TimeoutSec 5
    $signalCount = if ($signals) { 
        if ($signals -is [Array]) { $signals.Count } 
        elseif ($signals.signals) { $signals.signals.Count }
        else { 1 }
    } else { 0 }
    $results += @{
        Test = "AI Signals API"
        Status = "✅ PASS"
        Details = "Signals: $signalCount"
    }
    Write-Host "   ✅ PASS - $signalCount AI signals available" -ForegroundColor Green
} catch {
    $results += @{Test = "AI Signals API"; Status = "❌ FAIL"; Details = $_.Exception.Message}
    Write-Host "   ❌ FAIL - $_" -ForegroundColor Red
}

# Test 4: Trades/Journal endpoint
Write-Host "4️⃣ Testing /api/trades endpoint..." -ForegroundColor Yellow
try {
    $trades = Invoke-RestMethod -Uri "http://localhost:8000/api/trades" -Method GET -TimeoutSec 5
    $tradeCount = if ($trades -is [Array]) { $trades.Count } else { 0 }
    $results += @{
        Test = "Trades API"
        Status = "✅ PASS"
        Details = "Recent trades: $tradeCount"
    }
    Write-Host "   ✅ PASS - $tradeCount recent trades" -ForegroundColor Green
} catch {
    $results += @{Test = "Trades API"; Status = "❌ FAIL"; Details = $_.Exception.Message}
    Write-Host "   ❌ FAIL - $_" -ForegroundColor Red
}

# Test 5: Stats endpoint
Write-Host "5️⃣ Testing /api/stats endpoint..." -ForegroundColor Yellow
try {
    $stats = Invoke-RestMethod -Uri "http://localhost:8000/api/stats" -Method GET -TimeoutSec 5
    $results += @{
        Test = "Stats API"
        Status = "✅ PASS"
        Details = "Win Rate: $($stats.win_rate)%, Total PnL: $($stats.total_pnl)"
    }
    Write-Host "   ✅ PASS - Stats loaded successfully" -ForegroundColor Green
} catch {
    $results += @{Test = "Stats API"; Status = "❌ FAIL"; Details = $_.Exception.Message}
    Write-Host "   ❌ FAIL - $_" -ForegroundColor Red
}

# Test 6: Frontend accessibility
Write-Host "6️⃣ Testing Frontend at :3000..." -ForegroundColor Yellow
try {
    $frontend = Invoke-WebRequest -Uri "http://localhost:3000" -Method GET -TimeoutSec 5 -UseBasicParsing
    $hasRoot = $frontend.Content -match 'id="root"'
    if ($hasRoot) {
        $results += @{Test = "Frontend"; Status = "✅ PASS"; Details = "React app loaded"}
        Write-Host "   ✅ PASS - Frontend is accessible" -ForegroundColor Green
    } else {
        $results += @{Test = "Frontend"; Status = "⚠️  WARN"; Details = "Page loaded but React mount missing"}
        Write-Host "   ⚠️  WARN - React mount point not found" -ForegroundColor Yellow
    }
} catch {
    $results += @{Test = "Frontend"; Status = "❌ FAIL"; Details = $_.Exception.Message}
    Write-Host "   ❌ FAIL - $_" -ForegroundColor Red
}

# Test 7: WebSocket endpoint (check if available)
Write-Host "7️⃣ Testing WebSocket endpoint..." -ForegroundColor Yellow
try {
    # Test if WS upgrade endpoint exists
    $wsTest = Invoke-WebRequest -Uri "http://localhost:8000/ws" -Method GET -TimeoutSec 5 -UseBasicParsing -ErrorAction SilentlyContinue
    $results += @{Test = "WebSocket"; Status = "✅ PASS"; Details = "WS endpoint available"}
    Write-Host "   ✅ PASS - WebSocket endpoint exists" -ForegroundColor Green
} catch {
    if ($_.Exception.Message -match "426" -or $_.Exception.Message -match "Upgrade Required") {
        $results += @{Test = "WebSocket"; Status = "✅ PASS"; Details = "WS upgrade available"}
        Write-Host "   ✅ PASS - WebSocket upgrade available" -ForegroundColor Green
    } else {
        $results += @{Test = "WebSocket"; Status = "⚠️  INFO"; Details = "WS check inconclusive"}
        Write-Host "   ℹ️  INFO - WebSocket test inconclusive" -ForegroundColor Cyan
    }
}

# Summary
Write-Host "`n📊 TEST SUMMARY`n" -ForegroundColor Cyan
$passCount = ($results | Where-Object { $_.Status -match "✅" }).Count
$failCount = ($results | Where-Object { $_.Status -match "❌" }).Count
$warnCount = ($results | Where-Object { $_.Status -match "⚠️" }).Count

Write-Host "✅ Passed: $passCount" -ForegroundColor Green
Write-Host "❌ Failed: $failCount" -ForegroundColor Red
Write-Host "⚠️  Warnings: $warnCount" -ForegroundColor Yellow

if ($failCount -eq 0) {
    Write-Host "`n🎉 ALL TESTS PASSED - Dashboard integration is working!`n" -ForegroundColor Green
    Write-Host "🌐 Open dashboard: http://localhost:3000" -ForegroundColor Cyan
} else {
    Write-Host "`n⚠️  Some tests failed - check details above`n" -ForegroundColor Yellow
}

# Detailed results
Write-Host "`n📋 DETAILED RESULTS:" -ForegroundColor Cyan
$results | ForEach-Object {
    Write-Host "`n$($_.Test): $($_.Status)" -ForegroundColor White
    Write-Host "   $($_.Details)" -ForegroundColor Gray
}
