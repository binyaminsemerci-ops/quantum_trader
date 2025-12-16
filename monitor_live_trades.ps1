#!/usr/bin/env pwsh
# Monitor Live Trading Activity
# Shows: Signals, Order Placements, Position Protection, and P&L

Write-Host "`n🔴 LIVE TRADING MONITOR - CTRL+C to stop`n" -ForegroundColor Red
Write-Host "Configuration:" -ForegroundColor Yellow
Write-Host "  • Max Positions: 4" -ForegroundColor Cyan
Write-Host "  • Leverage: 20x" -ForegroundColor Cyan
Write-Host "  • Stop Loss: 2%" -ForegroundColor Cyan
Write-Host "  • Take Profit: 3%" -ForegroundColor Cyan
Write-Host "  • Order Type: STOP_LOSS (guaranteed execution)`n" -ForegroundColor Green

docker logs quantum_backend -f --tail 50 | Where-Object {
    $_ -match "ORDER PLACED|protected|Position check|SIGNAL|confidence|entry_price|STOP_LOSS|TP/SL"
}
