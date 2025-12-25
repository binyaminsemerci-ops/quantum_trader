#!/usr/bin/env pwsh
# Quantum Trader Production Monitoring Script
# Usage: .\monitor-production.ps1

param(
    [switch]$Continuous,
    [int]$RefreshSeconds = 5
)

$VPS_HOST = "root@46.224.116.254"
$SSH_KEY = "~/.ssh/hetzner_fresh"

function Get-ContainerStatus {
    Write-Host "`n=== 🐳 CONTAINER STATUS ===" -ForegroundColor Cyan
    ssh -i $SSH_KEY $VPS_HOST "docker ps --format 'table {{.Names}}\t{{.Status}}\t{{.Ports}}' | grep -E '(NAME|quantum)'"
}

function Get-ConsumerHealth {
    Write-Host "`n=== 🎯 CONSUMER HEALTH ===" -ForegroundColor Cyan
    $logs = ssh -i $SSH_KEY $VPS_HOST "docker logs --tail 5 quantum_trade_intent_consumer 2>&1"
    
    if ($logs -match "ERROR") {
        Write-Host "❌ ERRORS DETECTED" -ForegroundColor Red
        $logs | Select-String "ERROR" | Select-Object -First 3
    } elseif ($logs -match "Order submitted") {
        Write-Host "✅ Processing events and placing orders" -ForegroundColor Green
        $logs | Select-String "Order submitted" | Select-Object -Last 1
    } else {
        Write-Host "⏳ Waiting for events..." -ForegroundColor Yellow
    }
}

function Get-StreamInfo {
    Write-Host "`n=== 📊 REDIS STREAM INFO ===" -ForegroundColor Cyan
    $streamInfo = ssh -i $SSH_KEY $VPS_HOST "docker exec quantum_redis redis-cli XINFO STREAM quantum:stream:trade.intent | grep -E '(length|entries-added|groups)'"
    
    $length = ($streamInfo | Select-String "length" | Select-Object -First 1) -replace '.*\n', ''
    $entries = ($streamInfo | Select-String "entries-added" | Select-Object -First 1) -replace '.*\n', ''
    
    Write-Host "  Stream length: $length events" -ForegroundColor White
    Write-Host "  Total entries: $entries" -ForegroundColor White
}

function Get-TradingBotStatus {
    Write-Host "`n=== 🤖 TRADING BOT STATUS ===" -ForegroundColor Cyan
    $logs = ssh -i $SSH_KEY $VPS_HOST "docker logs --tail 10 quantum_trading_bot 2>&1 | grep -E '(Fallback signal|ERROR)'"
    
    if ($logs -match "ERROR") {
        Write-Host "❌ Bot has errors" -ForegroundColor Red
    } elseif ($logs -match "Fallback signal") {
        Write-Host "✅ Generating signals (fallback mode)" -ForegroundColor Green
        $logs | Select-String "Fallback signal" | Select-Object -Last 3
    }
}

function Get-RecentOrders {
    Write-Host "`n=== 📈 RECENT ORDERS (Last 5) ===" -ForegroundColor Cyan
    ssh -i $SSH_KEY $VPS_HOST "docker logs --tail 100 quantum_trade_intent_consumer 2>&1 | grep 'Order submitted' | tail -5"
}

function Show-Summary {
    Write-Host "`n=== 📋 SYSTEM SUMMARY ===" -ForegroundColor Cyan
    Write-Host "  Timestamp: $(Get-Date -Format 'yyyy-MM-dd HH:mm:ss')" -ForegroundColor Gray
    Write-Host "  Monitoring: VPS $VPS_HOST" -ForegroundColor Gray
    Write-Host ""
}

# Main monitoring loop
do {
    Clear-Host
    Write-Host "🎯 QUANTUM TRADER - PRODUCTION MONITOR" -ForegroundColor Green
    Write-Host "========================================`n" -ForegroundColor Green
    
    Show-Summary
    Get-ContainerStatus
    Get-ConsumerHealth
    Get-StreamInfo
    Get-TradingBotStatus
    Get-RecentOrders
    
    if ($Continuous) {
        Write-Host "`n⏳ Refreshing in $RefreshSeconds seconds... (Ctrl+C to stop)" -ForegroundColor Yellow
        Start-Sleep -Seconds $RefreshSeconds
    }
} while ($Continuous)

Write-Host "`n✅ Monitoring complete!" -ForegroundColor Green
