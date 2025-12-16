Write-Host "
⏰ LIVE MONITOR STARTED - Sjekker hvert 20 sekund
" -ForegroundColor Green

while ($true) {
    $now = Get-Date -Format "HH:mm:ss"
    
    # Check auto_set_tpsl.py logs
    Write-Host "───────────────────────────────────────────────" -ForegroundColor Gray
    Write-Host "[$now] 📊 HYBRID TP/SL STATUS:" -ForegroundColor Cyan
    
    # Show if script is still running
    $pythonProc = Get-Process python -ErrorAction SilentlyContinue | Where-Object { $_.CommandLine -like "*auto_set_tpsl*" }
    if ($pythonProc) {
        Write-Host "  ✅ auto_set_tpsl.py is RUNNING" -ForegroundColor Green
    } else {
        Write-Host "  ❌ auto_set_tpsl.py is NOT RUNNING!" -ForegroundColor Red
    }
    
    # Check for recent executions in backend logs
    Write-Host "
  🔔 Checking for order fills (last 20s)..." -ForegroundColor Yellow
    $fills = docker logs quantum_backend --since 20s 2>&1 | Select-String "filled|FILLED|partial.*exit|TP.*triggered|SL.*triggered"
    if ($fills) {
        $fills | Select-Object -First 5 | ForEach-Object {
            Write-Host "    💰 $_" -ForegroundColor Green
        }
    } else {
        Write-Host "    ⏳ No fills yet..." -ForegroundColor Gray
    }
    
    # Show current P&L summary every minute
    $elapsed = [int]((Get-Date).Second)
    if ($elapsed -lt 20) {
        Write-Host "
  📈 TIP: Sjekk Binance app for live P&L updates" -ForegroundColor Cyan
        Write-Host "  🎯 Hybrid: 50% exits ved TP, 50% kjører videre!" -ForegroundColor Yellow
    }
    
    Start-Sleep 20
}
