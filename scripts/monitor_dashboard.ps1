# Exit-Monitor Status Dashboard
# Real-time monitoring of tracked positions and exit triggers

# Run this in PowerShell to monitor the system

function Get-ExitMonitorStatus {
    $vps = "root@46.224.116.254"
    $key = "~/.ssh/hetzner_fresh"
    
    Clear-Host
    Write-Host "=" * 80 -ForegroundColor Cyan
    Write-Host "EXIT-MONITOR STATUS DASHBOARD" -ForegroundColor Yellow
    Write-Host "Updated: $(Get-Date -Format 'yyyy-MM-dd HH:mm:ss')" -ForegroundColor Gray
    Write-Host "=" * 80 -ForegroundColor Cyan
    Write-Host ""
    
    # Health Status
    Write-Host "HEALTH STATUS:" -ForegroundColor Green
    wsl ssh -i $key $vps 'curl -s http://localhost:8007/health' | ConvertFrom-Json | Format-List
    
    # Consumer Group
    Write-Host "CONSUMER GROUP:" -ForegroundColor Green
    wsl ssh -i $key $vps 'redis-cli XINFO GROUPS quantum:stream:trade.execution.res' | Select-String -Pattern "entries-read|lag|pending"
    Write-Host ""
    
    # Recent Log Events
    Write-Host "RECENT EVENTS (last 10):" -ForegroundColor Green
    wsl ssh -i $key $vps 'tail -30 /var/log/quantum/exit-monitor.log | grep -E "TRACKING|TP_HIT|SL_HIT|EXIT|BOOTSTRAP" | tail -10'
    Write-Host ""
    
    # Current Binance Positions
    Write-Host "OPEN POSITIONS ON BINANCE:" -ForegroundColor Green
    wsl ssh -i $key $vps 'python3 -c "from binance.client import Client; import os; c = Client(os.getenv(\"BINANCE_API_KEY\"), os.getenv(\"BINANCE_API_SECRET\"), testnet=True); p = c.futures_position_information(); [print(f\"{pos[\"symbol\"]}: {float(pos[\"positionAmt\"]):.2f} @ \${float(pos[\"entryPrice\"]):.4f}\") for pos in p if abs(float(pos[\"positionAmt\"])) > 0]"' 2>$null
}

# Run dashboard in loop
Write-Host "Starting Exit-Monitor Dashboard..." -ForegroundColor Yellow
Write-Host "Press Ctrl+C to stop" -ForegroundColor Gray
Write-Host ""

while ($true) {
    Get-ExitMonitorStatus
    Start-Sleep -Seconds 10
}
