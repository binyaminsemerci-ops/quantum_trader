# Watch for new trades in real-time
Write-Host "`n🔍 MONITORING NEW TRADES..." -ForegroundColor Cyan
Write-Host "Press Ctrl+C to stop`n" -ForegroundColor Yellow

$lastCheck = Get-Date
while ($true) {
    $now = Get-Date
    $sinceSec = ($now - $lastCheck).TotalSeconds
    
    # Check logs from last check
    $logs = docker logs quantum_backend --since "$([int]$sinceSec)s" 2>&1 | 
            Select-String -Pattern "MONEY.*Cash|Order placed|TRADE APPROVED|Failed to place|CONSENSUS FILTER|ZeroDivisionError|equity_usd=\$"
    
    if ($logs) {
        foreach ($log in $logs) {
            $timestamp = Get-Date -Format "HH:mm:ss"
            
            if ($log -match "MONEY.*Cash") {
                Write-Host "[$timestamp] 💰 " -NoNewline -ForegroundColor Green
                Write-Host $log
            }
            elseif ($log -match "Order placed") {
                Write-Host "[$timestamp] ✅ " -NoNewline -ForegroundColor Green
                Write-Host $log
            }
            elseif ($log -match "TRADE APPROVED") {
                Write-Host "[$timestamp] 🎯 " -NoNewline -ForegroundColor Cyan
                Write-Host $log
            }
            elseif ($log -match "CONSENSUS FILTER") {
                Write-Host "[$timestamp] 🔍 " -NoNewline -ForegroundColor Yellow
                Write-Host $log
            }
            elseif ($log -match "Failed to place|ZeroDivisionError") {
                Write-Host "[$timestamp] ❌ " -NoNewline -ForegroundColor Red
                Write-Host $log
            }
            else {
                Write-Host "[$timestamp] ℹ️  $log" -ForegroundColor Gray
            }
        }
    }
    
    $lastCheck = $now
    Start-Sleep -Seconds 2
}
