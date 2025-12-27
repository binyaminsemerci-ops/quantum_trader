#!/usr/bin/env pwsh
# Quick system health check for autonomous trading

Write-Host "`n🔍 QUANTUM TRADER SYSTEM STATUS`n" -ForegroundColor Cyan

# 1. Check if backend is running
Write-Host "1️⃣ Backend Container Status:" -ForegroundColor Yellow
docker ps --filter "name=quantum_backend" --format "table {{.Names}}\t{{.Status}}\t{{.Ports}}"

# 2. Check recent AI activity (last 2 minutes)
Write-Host "`n2️⃣ Recent AI Activity (Last 2 min):" -ForegroundColor Yellow
$signals = docker logs quantum_backend --since 2m 2>&1 | Select-String "Strong signals:" | Select-Object -Last 3
if ($signals) {
    $signals | ForEach-Object { Write-Host "   $_" -ForegroundColor Green }
} else {
    Write-Host "   ⚠️  No signals in last 2 minutes" -ForegroundColor Yellow
}

# 3. Check recent orders (last 5 minutes)
Write-Host "`n3️⃣ Recent Order Activity (Last 5 min):" -ForegroundColor Yellow
$orders = docker logs quantum_backend --since 5m 2>&1 | Select-String "DRY-RUN|Futures order" | Select-Object -Last 5
if ($orders) {
    $orders | ForEach-Object { Write-Host "   $_" -ForegroundColor Green }
} else {
    Write-Host "   ℹ️  No orders submitted yet (waiting for strong signals)" -ForegroundColor Cyan
}

# 4. Check latest rebalancing result
Write-Host "`n4️⃣ Latest Rebalancing Result:" -ForegroundColor Yellow
$rebalance = docker logs quantum_backend --tail 200 2>&1 | Select-String "Rebalancing result" | Select-Object -Last 1
if ($rebalance) {
    Write-Host "   $rebalance" -ForegroundColor Green
}

# 5. Check for any errors (last 5 minutes)
Write-Host "`n5️⃣ Recent Errors (Last 5 min):" -ForegroundColor Yellow
$errors = docker logs quantum_backend --since 5m 2>&1 | Select-String "ERROR|Exception|Failed.*order" | Select-Object -Last 3
if ($errors) {
    $errors | ForEach-Object { Write-Host "   ⚠️  $_" -ForegroundColor Red }
} else {
    Write-Host "   ✅ No errors detected!" -ForegroundColor Green
}

# 6. Check configuration
Write-Host "`n6️⃣ Current Configuration:" -ForegroundColor Yellow
$config = docker exec quantum_backend env 2>&1 | Select-String "QT_CONFIDENCE_THRESHOLD|QT_MAX_NOTIONAL|STAGING_MODE|QT_ALLOWED_SYMBOLS"
$config | ForEach-Object { Write-Host "   $_" -ForegroundColor Cyan }

Write-Host "`n✅ Health check complete!`n" -ForegroundColor Green
Write-Host "💡 Tip: Run this script every few hours to monitor system health" -ForegroundColor Yellow
Write-Host "💡 To see live activity: docker logs quantum_backend -f`n" -ForegroundColor Yellow
