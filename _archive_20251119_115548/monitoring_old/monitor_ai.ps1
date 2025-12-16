$lastCheck = Get-Date
Write-Host "⏰ Started: $(Get-Date -Format 'HH:mm:ss')
" -ForegroundColor Green

while ($true) {
    $now = Get-Date
    
    # Check for new AI signals (last 10 seconds)
    $signals = docker logs quantum_backend --since 10s | Select-String "AI signal|confidence|Creating order"
    if ($signals) {
        Write-Host "
🔔 [$($now.ToString('HH:mm:ss'))] NY AI SIGNAL:" -ForegroundColor Cyan
        $signals | ForEach-Object {
            if ($_ -match "confidence") {
                Write-Host "  🧠 $_" -ForegroundColor Yellow
            } elseif ($_ -match "Creating order") {
                Write-Host "  📈 $_" -ForegroundColor Green
            } else {
                Write-Host "  ℹ️  $_" -ForegroundColor White
            }
        }
    }
    
    # Check for filled orders
    $fills = docker logs quantum_backend --since 10s | Select-String "filled|FILLED|ORDER.*success"
    if ($fills) {
        Write-Host "
💰 [$($now.ToString('HH:mm:ss'))] ORDER FILLED:" -ForegroundColor Green
        $fills | ForEach-Object { Write-Host "  ✅ $_" -ForegroundColor Green }
    }
    
    # Check for errors
    $errors = docker logs quantum_backend --since 10s | Select-String "ERROR|error|failed|Failed" | Select-String -NotMatch "rate limit"
    if ($errors) {
        Write-Host "
❌ [$($now.ToString('HH:mm:ss'))] ERROR:" -ForegroundColor Red
        $errors | Select-Object -First 3 | ForEach-Object { Write-Host "  ⚠️  $_" -ForegroundColor Red }
    }
    
    # Status update every 30 seconds
    if (($now - $lastCheck).TotalSeconds -ge 30) {
        Write-Host "
📊 [$($now.ToString('HH:mm:ss'))] Status Check..." -ForegroundColor Gray
        $lastCheck = $now
    }
    
    Start-Sleep 5
}
