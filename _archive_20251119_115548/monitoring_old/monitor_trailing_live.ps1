param(
    [int]$IntervalSeconds = 30
)

Write-Host "`n🔄 Trailing Stop Live Monitor" -ForegroundColor Cyan
Write-Host "Checking every $IntervalSeconds seconds..." -ForegroundColor Gray
Write-Host "Press Ctrl+C to stop`n" -ForegroundColor Gray

while ($true) {
    Clear-Host
    Write-Host "`n" -NoNewline
    Write-Host "=" * 80 -ForegroundColor Cyan
    Write-Host "🔄 TRAILING STOP STATUS - $(Get-Date -Format 'HH:mm:ss')" -ForegroundColor Yellow
    Write-Host "=" * 80 -ForegroundColor Cyan
    
    # Get positions
    $positions = docker exec quantum_backend python /app/quick_positions.py 2>$null
    Write-Output $positions
    
    Write-Host "`n" -NoNewline
    Write-Host "=" * 80 -ForegroundColor Cyan
    Write-Host "📊 TRAILING ANALYSIS" -ForegroundColor Yellow  
    Write-Host "=" * 80 -ForegroundColor Cyan
    
    # Run trailing analysis
    docker exec quantum_backend python /app/monitor_trailing.py 2>$null
    
    Write-Host "`n" -NoNewline
    Write-Host "=" * 80 -ForegroundColor Cyan
    Write-Host "📋 RECENT TRAILING EVENTS" -ForegroundColor Yellow
    Write-Host "=" * 80 -ForegroundColor Cyan
    
    # Check logs for trailing events
    $logs = docker logs quantum_backend --since "${IntervalSeconds}s" 2>&1 | 
            Select-String "peak|trough|Trailing SL updated" | 
            Select-Object -Last 10
    
    if ($logs) {
        $logs | ForEach-Object {
            $line = $_.Line
            if ($line -match "new peak") {
                Write-Host "  📈 $line" -ForegroundColor Green
            }
            elseif ($line -match "new trough") {
                Write-Host "  📉 $line" -ForegroundColor Red
            }
            elseif ($line -match "Trailing SL updated") {
                Write-Host "  🎯 $line" -ForegroundColor Yellow
            }
            else {
                Write-Host "  ℹ️  $line" -ForegroundColor Gray
            }
        }
    }
    else {
        Write-Host "  No trailing events in last ${IntervalSeconds}s" -ForegroundColor Gray
    }
    
    Write-Host "`n" -NoNewline
    Write-Host "=" * 80 -ForegroundColor Cyan
    Write-Host "Next update in $IntervalSeconds seconds..." -ForegroundColor Gray
    
    Start-Sleep -Seconds $IntervalSeconds
}
