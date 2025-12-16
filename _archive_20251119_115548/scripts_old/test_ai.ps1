# Test AI Integration
# Quick verification that AI components are working

$baseUrl = "http://localhost:8000"
$token = $env:QT_ADMIN_TOKEN
if (-not $token) {
    $token = "your-secret-admin-token"
    Write-Host "Warning: Using default admin token. Set QT_ADMIN_TOKEN env var for production." -ForegroundColor Yellow
}

$headers = @{
    "X-Admin-Token" = $token
    "Content-Type" = "application/json"
}

Write-Host "`n========================================" -ForegroundColor Cyan
Write-Host "Quantum Trader AI Integration Test" -ForegroundColor Cyan
Write-Host "========================================`n" -ForegroundColor Cyan

# Test 1: AI Status
Write-Host "[1/4] Testing AI Status..." -ForegroundColor Green
try {
    $response = Invoke-RestMethod -Uri "$baseUrl/ai/live-status" -Headers $headers -Method Get
    Write-Host "  Status: $($response.status)" -ForegroundColor White
    Write-Host "  Model loaded: $($response.model.loaded)" -ForegroundColor White
    Write-Host "  Model type: $($response.model.type)" -ForegroundColor White
    
    if ($response.predictions.total -gt 0) {
        $total = $response.predictions.total
        $buy = $response.predictions.buy_signals
        $sell = $response.predictions.sell_signals
        $hold = $response.predictions.hold_signals
        
        Write-Host "`n  Predictions: $total total" -ForegroundColor White
        Write-Host "    BUY:  $buy ($([math]::Round(100*$buy/$total, 1))%)" -ForegroundColor Green
        Write-Host "    SELL: $sell ($([math]::Round(100*$sell/$total, 1))%)" -ForegroundColor Red
        Write-Host "    HOLD: $hold ($([math]::Round(100*$hold/$total, 1))%)" -ForegroundColor Yellow
    }
    
    if ($response.recent_signals) {
        Write-Host "`n  Top signals:" -ForegroundColor White
        $response.recent_signals | Select-Object -First 3 | ForEach-Object {
            Write-Host "    $($_.symbol) - weight=$([math]::Round($_.weight, 3)) score=$([math]::Round($_.score, 2))" -ForegroundColor White
        }
    }
    Write-Host "  ✓ PASS`n" -ForegroundColor Green
} catch {
    Write-Host "  ✗ FAIL: $($_.Exception.Message)`n" -ForegroundColor Red
}

# Test 2: AI Predictions
Write-Host "[2/4] Testing AI Predictions..." -ForegroundColor Green
try {
    $payload = @{
        symbols = @("BTCUSDC", "ETHUSDC", "SOLUSDC")
    } | ConvertTo-Json
    
    $response = Invoke-RestMethod -Uri "$baseUrl/ai/predict" -Headers $headers -Method Post -Body $payload
    
    Write-Host "  Predictions for $($response.predictions.Count) symbols:" -ForegroundColor White
    $response.predictions | ForEach-Object {
        $conf = [math]::Round($_.confidence, 2)
        $score = [math]::Round($_.score, 4)
        Write-Host "    $($_.symbol): $($_.action) (confidence=$conf, score=$score)" -ForegroundColor White
    }
    Write-Host "  ✓ PASS`n" -ForegroundColor Green
} catch {
    Write-Host "  ✗ FAIL: $($_.Exception.Message)`n" -ForegroundColor Red
}

# Test 3: Liquidity Status
Write-Host "[3/4] Testing Liquidity Status..." -ForegroundColor Green
try {
    $response = Invoke-RestMethod -Uri "$baseUrl/liquidity" -Headers $headers -Method Get
    Write-Host "  Status: $($response.status)" -ForegroundColor White
    Write-Host "  Universe: $($response.universe_size) symbols" -ForegroundColor White
    Write-Host "  Selection: $($response.selection_size) symbols" -ForegroundColor White
    Write-Host "  Last refresh: $($response.last_refresh_time)" -ForegroundColor White
    
    if ($response.top_symbols) {
        Write-Host "`n  Top symbols:" -ForegroundColor White
        $response.top_symbols | Select-Object -First 5 | ForEach-Object {
            Write-Host "    $_" -ForegroundColor White
        }
    }
    Write-Host "  ✓ PASS`n" -ForegroundColor Green
} catch {
    Write-Host "  ✗ FAIL: $($_.Exception.Message)`n" -ForegroundColor Red
}

# Test 4: Risk Status
Write-Host "[4/4] Testing Risk Status..." -ForegroundColor Green
try {
    $response = Invoke-RestMethod -Uri "$baseUrl/risk" -Headers $headers -Method Get
    $killSwitch = if ($response.state.kill_switch_enabled) { "ENABLED" } else { "DISABLED" }
    Write-Host "  Kill-switch: $killSwitch" -ForegroundColor White
    Write-Host "  Daily loss: $([math]::Round($response.state.daily_loss, 2)) / $([math]::Round($response.state.max_daily_loss, 2))" -ForegroundColor White
    Write-Host "  Gross exposure: $([math]::Round($response.state.gross_exposure, 2)) / $([math]::Round($response.state.max_gross_exposure, 2))" -ForegroundColor White
    Write-Host "  ✓ PASS`n" -ForegroundColor Green
} catch {
    Write-Host "  ✗ FAIL: $($_.Exception.Message)`n" -ForegroundColor Red
}

Write-Host "========================================" -ForegroundColor Cyan
Write-Host "Test suite completed!" -ForegroundColor Cyan
Write-Host "========================================`n" -ForegroundColor Cyan

Write-Host "Next steps:" -ForegroundColor Yellow
Write-Host "  1. Check if AI signals show BUY/SELL (not just HOLD)" -ForegroundColor White
Write-Host "  2. Trigger execution: .\backend\trigger_execution.ps1" -ForegroundColor White
Write-Host "  3. Monitor logs for 'AI adjusted' messages" -ForegroundColor White
Write-Host "  4. Review ExecutionJournal for orders with 'AI=' in reason`n" -ForegroundColor White
