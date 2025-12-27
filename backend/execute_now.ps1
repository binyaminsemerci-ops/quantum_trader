# Quick manual execution trigger - run this while server is up
$Headers = @{ 'X-Admin-Token' = 'live-admin-token' }

Write-Host "=== Manual Execution Trigger ===" -ForegroundColor Cyan
Write-Host "Triggering execution cycle..." -ForegroundColor Yellow

try {
    $Response = Invoke-RestMethod -Method POST -Uri http://localhost:8000/scheduler/execution -Headers $Headers -TimeoutSec 30
    
    Write-Host "`n✓ Execution completed!" -ForegroundColor Green
    Write-Host "  Status: $($Response.execution.status)" -ForegroundColor Gray
    Write-Host "  Orders planned:   $($Response.execution.orders_planned)" -ForegroundColor Gray
    Write-Host "  Orders submitted: $($Response.execution.orders_submitted)" -ForegroundColor $(if ($Response.execution.orders_submitted -gt 0) { 'Green' } else { 'Yellow' })
    Write-Host "  Orders skipped:   $($Response.execution.orders_skipped)" -ForegroundColor Gray
    Write-Host "  Orders failed:    $($Response.execution.orders_failed)" -ForegroundColor $(if ($Response.execution.orders_failed -gt 0) { 'Red' } else { 'Gray' })
    Write-Host "  Gross exposure:   $([math]::Round($Response.execution.gross_exposure, 2)) USD" -ForegroundColor Gray
    Write-Host "  Positions synced: $($Response.execution.positions_synced)" -ForegroundColor Gray
    Write-Host "  Run ID: $($Response.execution.run_id)" -ForegroundColor DarkGray
    
    if ($Response.execution.orders_submitted -gt 0) {
        Write-Host "`n🚀 LIVE ORDERS SUBMITTED TO BINANCE FUTURES!" -ForegroundColor Green -BackgroundColor Black
        Write-Host "   Check backend terminal for order IDs" -ForegroundColor Cyan
    }
    elseif ($Response.execution.orders_planned -gt 0) {
        Write-Host "`n⚠ Orders planned but not submitted - check risk limits or strategy filters" -ForegroundColor Yellow
    }
    else {
        Write-Host "`n📊 No trading signals in current market conditions" -ForegroundColor Gray
    }
}
catch {
    Write-Host "`n✗ Execution failed: $($_.Exception.Message)" -ForegroundColor Red
    if ($_.Exception.Message -like "*Cannot connect*" -or $_.Exception.Message -like "*målmaskinen*") {
        Write-Host "   Server is not running! Start it with: .\start_live.ps1" -ForegroundColor Yellow
    }
}
