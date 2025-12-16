# Quick script to trigger execution and liquidity cycles
# Requires: backend running and API keys configured

$Headers = @{ 'X-Admin-Token' = 'live-admin-token' }

Write-Host "=== Triggering Liquidity Refresh ===" -ForegroundColor Cyan
try {
    $Liquidity = Invoke-RestMethod -Method POST -Uri http://localhost:8000/scheduler/liquidity -Headers $Headers
    Write-Host "✓ Liquidity: $($Liquidity.liquidity.status)" -ForegroundColor Green
    Write-Host "  Run ID: $($Liquidity.liquidity.run_id)" -ForegroundColor Gray
    Write-Host "  Selection size: $($Liquidity.liquidity.selection_size)" -ForegroundColor Gray
}
catch {
    Write-Host "✗ Liquidity failed: $($_.Exception.Message)" -ForegroundColor Red
}

Start-Sleep -Seconds 2

Write-Host "`n=== Triggering Execution Cycle ===" -ForegroundColor Cyan
try {
    $Execution = Invoke-RestMethod -Method POST -Uri http://localhost:8000/scheduler/execution -Headers $Headers
    Write-Host "✓ Execution: $($Execution.execution.status)" -ForegroundColor Green
    Write-Host "  Orders planned:   $($Execution.execution.orders_planned)" -ForegroundColor Gray
    Write-Host "  Orders submitted: $($Execution.execution.orders_submitted)" -ForegroundColor $(if ($Execution.execution.orders_submitted -gt 0) { 'Green' } else { 'Yellow' })
    Write-Host "  Orders skipped:   $($Execution.execution.orders_skipped)" -ForegroundColor Gray
    Write-Host "  Orders failed:    $($Execution.execution.orders_failed)" -ForegroundColor $(if ($Execution.execution.orders_failed -gt 0) { 'Red' } else { 'Gray' })
    Write-Host "  Gross exposure:   $([math]::Round($Execution.execution.gross_exposure, 2)) USD" -ForegroundColor Gray
    Write-Host "  Run ID: $($Execution.execution.run_id)" -ForegroundColor Gray
    
    if ($Execution.execution.orders_submitted -gt 0) {
        Write-Host "`n✓ LIVE ORDERS SUBMITTED - Check backend logs for order IDs" -ForegroundColor Green
    }
    else {
        Write-Host "`n⚠ No orders submitted (may be filtered by strategy/risk)" -ForegroundColor Yellow
    }
}
catch {
    Write-Host "✗ Execution failed: $($_.Exception.Message)" -ForegroundColor Red
}

Write-Host "`n=== Risk Guard Status ===" -ForegroundColor Cyan
try {
    $Risk = Invoke-RestMethod -Method GET -Uri http://localhost:8000/risk -Headers $Headers
    Write-Host "  Staging mode:           $($Risk.config.staging_mode)" -ForegroundColor $(if ($Risk.config.staging_mode) { 'Yellow' } else { 'Green' })
    Write-Host "  Kill switch:            $($Risk.config.kill_switch)" -ForegroundColor $(if ($Risk.config.kill_switch) { 'Red' } else { 'Green' })
    Write-Host "  Max notional per trade: $($Risk.config.max_notional_per_trade) USD" -ForegroundColor Gray
    Write-Host "  Max gross exposure:     $($Risk.config.max_gross_exposure) USD" -ForegroundColor Gray
    Write-Host "  Trade count today:      $($Risk.state.trade_count)" -ForegroundColor Gray
    Write-Host "  Daily loss:             $([math]::Round($Risk.state.daily_loss, 2)) USD" -ForegroundColor Gray
}
catch {
    Write-Host "✗ Risk status unavailable: $($_.Exception.Message)" -ForegroundColor Red
}
