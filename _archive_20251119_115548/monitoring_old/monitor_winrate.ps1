#!/usr/bin/env pwsh
# Real-time Win Rate Monitor for Quantum Trader

Write-Host "`n🎯 QUANTUM TRADER - WIN RATE MONITOR`n" -ForegroundColor Cyan

function Get-PositionStatus {
    $result = docker exec quantum_backend python -c @"
from binance.client import Client
import os

c = Client(os.getenv('BINANCE_API_KEY'), os.getenv('BINANCE_API_SECRET'))
positions = c.futures_position_information()

open_pos = [p for p in positions if float(p.get('positionAmt', 0)) != 0]

print(f'OPEN:{len(open_pos)}')
for p in open_pos:
    sym = p['symbol']
    qty = float(p['positionAmt'])
    entry = float(p['entryPrice'])
    mark = float(p['markPrice'])
    pnl = float(p['unRealizedProfit'])
    pnl_pct = (pnl / (abs(qty) * entry)) * 100 if entry > 0 else 0
    side = 'LONG' if qty > 0 else 'SHORT'
    status = 'WIN' if pnl > 0 else 'LOSS' if pnl < 0 else 'EVEN'
    print(f'{sym}|{side}|{entry:.4f}|{mark:.4f}|{pnl:.2f}|{pnl_pct:.2f}|{status}')
"@
    return $result
}

function Get-TradeHistory {
    try {
        $trades = Invoke-RestMethod -Uri "http://localhost:8000/trades?limit=50" -Method GET
        return $trades
    } catch {
        Write-Host "⚠️ Could not fetch trade history from API" -ForegroundColor Yellow
        return @()
    }
}

function Calculate-WinRate {
    param($trades)
    
    # Group trades by symbol to calculate P&L per closed position
    $closedPositions = @{}
    
    foreach ($trade in $trades) {
        $sym = $trade.symbol
        if (-not $closedPositions.ContainsKey($sym)) {
            $closedPositions[$sym] = @{
                buys = @()
                sells = @()
            }
        }
        
        if ($trade.side -eq "BUY") {
            $closedPositions[$sym].buys += @{
                qty = $trade.quantity
                price = $trade.price
                time = $trade.timestamp
            }
        } else {
            $closedPositions[$sym].sells += @{
                qty = $trade.quantity
                price = $trade.price
                time = $trade.timestamp
            }
        }
    }
    
    # Calculate realized P&L for closed positions
    $wins = 0
    $losses = 0
    $totalPnl = 0
    
    foreach ($sym in $closedPositions.Keys) {
        $pos = $closedPositions[$sym]
        
        # Simple P&L calculation (assumes FIFO)
        $totalBuyQty = ($pos.buys | Measure-Object -Property qty -Sum).Sum
        $totalSellQty = ($pos.sells | Measure-Object -Property qty -Sum).Sum
        
        if ($totalBuyQty -gt 0 -and $totalSellQty -gt 0) {
            $avgBuyPrice = ($pos.buys | ForEach-Object { $_.qty * $_.price } | Measure-Object -Sum).Sum / $totalBuyQty
            $avgSellPrice = ($pos.sells | ForEach-Object { $_.qty * $_.price } | Measure-Object -Sum).Sum / $totalSellQty
            
            $closedQty = [Math]::Min($totalBuyQty, $totalSellQty)
            $pnl = ($avgSellPrice - $avgBuyPrice) * $closedQty
            
            $totalPnl += $pnl
            
            if ($pnl -gt 0) {
                $wins++
            } elseif ($pnl -lt 0) {
                $losses++
            }
        }
    }
    
    $totalTrades = $wins + $losses
    $winRate = if ($totalTrades -gt 0) { ($wins / $totalTrades) * 100 } else { 0 }
    
    return @{
        Wins = $wins
        Losses = $losses
        Total = $totalTrades
        WinRate = $winRate
        TotalPnL = $totalPnl
    }
}

# Main monitoring loop
while ($true) {
    Clear-Host
    Write-Host "`n🎯 QUANTUM TRADER - LIVE WIN RATE MONITOR" -ForegroundColor Cyan
    Write-Host "=" * 70 -ForegroundColor Gray
    Write-Host "Timestamp: $(Get-Date -Format 'yyyy-MM-dd HH:mm:ss')`n" -ForegroundColor Gray
    
    # Get open positions
    Write-Host "📊 ÅPNE POSISJONER:" -ForegroundColor Yellow
    $posData = Get-PositionStatus
    $lines = $posData -split "`n"
    $openCount = 0
    $openWins = 0
    $openLosses = 0
    $totalUnrealizedPnl = 0
    
    foreach ($line in $lines) {
        if ($line -match '^OPEN:(\d+)') {
            $openCount = [int]$matches[1]
            Write-Host "  Aktive: $openCount/4" -ForegroundColor White
        } elseif ($line -match '^([A-Z]+)\|([A-Z]+)\|([0-9.]+)\|([0-9.]+)\|(-?[0-9.]+)\|(-?[0-9.]+)\|([A-Z]+)') {
            $sym = $matches[1]
            $side = $matches[2]
            $entry = [decimal]$matches[3]
            $mark = [decimal]$matches[4]
            $pnl = [decimal]$matches[5]
            $pnlPct = [decimal]$matches[6]
            $status = $matches[7]
            
            $totalUnrealizedPnl += $pnl
            
            $color = switch ($status) {
                "WIN" { $openWins++; "Green" }
                "LOSS" { $openLosses++; "Red" }
                default { "Gray" }
            }
            
            $statusIcon = switch ($status) {
                "WIN" { "[+]" }
                "LOSS" { "[-]" }
                default { "[=]" }
            }
            
            if ($status -eq "WIN") {
                $statusIconChar = "[+]"
            } elseif ($status -eq "LOSS") {
                $statusIconChar = "[-]"
            } else {
                $statusIconChar = "[=]"
            }
            
            Write-Host "  $statusIconChar $sym $side" -ForegroundColor $color
        }
    }
    
    Write-Host "`n  Total Unrealized: $([char]36)$($totalUnrealizedPnl.ToString('F2'))" -ForegroundColor $(if ($totalUnrealizedPnl -gt 0) { "Green" } else { "Red" })
    Write-Host "  Current: $openWins vinnere, $openLosses tapere" -ForegroundColor Gray
    
    # Get trade history and calculate win rate
    Write-Host "`n📈 HISTORISK WIN RATE:" -ForegroundColor Yellow
    $trades = Get-TradeHistory
    
    if ($trades.Count -gt 0) {
        $stats = Calculate-WinRate -trades $trades
        
        $winRateColor = if ($stats.WinRate -ge 60) { "Green" } 
                       elseif ($stats.WinRate -ge 50) { "Yellow" } 
                       else { "Red" }
        
        Write-Host "  Win Rate: " -NoNewline -ForegroundColor White
        Write-Host "$($stats.WinRate.ToString('F1'))%" -ForegroundColor $winRateColor
        Write-Host "  Vinnere: $($stats.Wins)" -ForegroundColor Green
        Write-Host "  Tapere: $($stats.Losses)" -ForegroundColor Red
        Write-Host "  Total trades: $($stats.Total)" -ForegroundColor White
        
        if ($stats.TotalPnL -ne 0) {
            $pnlColor = if ($stats.TotalPnL -gt 0) { "Green" } else { "Red" }
            Write-Host "  Realized P&L: $([char]36)$($stats.TotalPnL.ToString('F2'))" -ForegroundColor $pnlColor
        }
        
        Write-Host "`n  Siste $($trades.Count) trades fra execution journal" -ForegroundColor Gray
    } else {
        Write-Host "  Ingen trade history tilgjengelig ennå..." -ForegroundColor Gray
    }
    
    # System status
    Write-Host "`n" -NoNewline
    Write-Host "[SYSTEM]" -ForegroundColor Yellow -NoNewline
    Write-Host " Max: 4 pos | Size: ~$([char]36)147-200 | TP/SL: Hybrid AI | Mode: " -NoNewline -ForegroundColor White
    Write-Host "LIVE" -ForegroundColor Green
    
    Write-Host "`nPress Ctrl+C to exit | Refreshing in 10s..." -ForegroundColor Gray
    
    Start-Sleep -Seconds 10
}
