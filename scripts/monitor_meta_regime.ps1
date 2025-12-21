# ============================================================================
# Meta-Regime Correlator - Continuous Monitoring Script
# ============================================================================
# Real-time monitoring of regime detection and correlation
# Usage: .\monitor_meta_regime.ps1
# Press Ctrl+C to exit
# ============================================================================

param(
    [string]$VpsHost = "46.224.116.254",
    [string]$VpsUser = "qt",
    [string]$SshKey = "~/.ssh/hetzner_fresh",
    [int]$RefreshInterval = 20
)

function Get-RegimeData {
    param($VpsUser, $VpsHost, $SshKey)
    
    $data = @{}
    
    # Get preferred regime
    $data.PreferredRegime = wsl ssh -i $SshKey ${VpsUser}@${VpsHost} "docker exec quantum_redis redis-cli GET quantum:governance:preferred_regime"
    
    # Get stream length
    $data.StreamLength = wsl ssh -i $SshKey ${VpsUser}@${VpsHost} "docker exec quantum_redis redis-cli XLEN quantum:stream:meta.regime"
    
    # Get current policy
    $data.CurrentPolicy = wsl ssh -i $SshKey ${VpsUser}@${VpsHost} "docker exec quantum_redis redis-cli GET quantum:governance:policy"
    
    # Get regime stats
    $statsJson = wsl ssh -i $SshKey ${VpsUser}@${VpsHost} "docker exec quantum_redis redis-cli GET quantum:governance:regime_stats"
    if ($statsJson) {
        try {
            $data.RegimeStats = $statsJson | ConvertFrom-Json
        } catch {
            $data.RegimeStats = $null
        }
    } else {
        $data.RegimeStats = $null
    }
    
    # Get AI Engine health
    $healthResponse = wsl ssh -i $SshKey ${VpsUser}@${VpsHost} "curl -s http://localhost:8001/health"
    try {
        $health = $healthResponse | ConvertFrom-Json
        $data.MetaRegimeHealth = $health.metrics.meta_regime
    } catch {
        $data.MetaRegimeHealth = $null
    }
    
    # Get recent logs (last line with regime detection)
    $logs = wsl ssh -i $SshKey ${VpsUser}@${VpsHost} "docker logs --tail 50 quantum_meta_regime"
    $data.RecentLogs = $logs | Select-String "Regime analysis complete" | Select-Object -Last 1
    
    return $data
}

function Show-RegimeDisplay {
    param($Data, $Iteration)
    
    Clear-Host
    
    Write-Host ""
    Write-Host "═══════════════════════════════════════════════════════════════════════════" -ForegroundColor Cyan
    Write-Host "  🧠 META-REGIME CORRELATOR - REAL-TIME MONITOR" -ForegroundColor Cyan
    Write-Host "═══════════════════════════════════════════════════════════════════════════" -ForegroundColor Cyan
    Write-Host ""
    Write-Host "  VPS: $VpsHost | Iteration: $Iteration | $(Get-Date -Format 'yyyy-MM-dd HH:mm:ss')" -ForegroundColor Gray
    Write-Host ""
    
    # Preferred Regime
    Write-Host "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━" -ForegroundColor DarkGray
    Write-Host "  📊 CURRENT REGIME STATUS" -ForegroundColor Yellow
    Write-Host "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━" -ForegroundColor DarkGray
    Write-Host ""
    
    if ($Data.PreferredRegime) {
        $regimeColor = switch ($Data.PreferredRegime) {
            "BULL" { "Green" }
            "BEAR" { "Red" }
            "RANGE" { "Yellow" }
            "VOLATILE" { "Magenta" }
            default { "White" }
        }
        Write-Host "  Preferred Regime: " -NoNewline -ForegroundColor White
        Write-Host $Data.PreferredRegime -ForegroundColor $regimeColor
    } else {
        Write-Host "  Preferred Regime: " -NoNewline -ForegroundColor White
        Write-Host "UNKNOWN (warming up)" -ForegroundColor Gray
    }
    
    if ($Data.CurrentPolicy) {
        $policyColor = switch ($Data.CurrentPolicy) {
            "AGGRESSIVE" { "Green" }
            "CONSERVATIVE" { "Red" }
            "BALANCED" { "Yellow" }
            default { "White" }
        }
        Write-Host "  Current Policy:   " -NoNewline -ForegroundColor White
        Write-Host $Data.CurrentPolicy -ForegroundColor $policyColor
    }
    
    Write-Host "  Stream Samples:   " -NoNewline -ForegroundColor White
    Write-Host $Data.StreamLength -ForegroundColor Cyan
    
    # AI Engine Health
    if ($Data.MetaRegimeHealth) {
        Write-Host ""
        Write-Host "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━" -ForegroundColor DarkGray
        Write-Host "  🏥 AI ENGINE HEALTH" -ForegroundColor Yellow
        Write-Host "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━" -ForegroundColor DarkGray
        Write-Host ""
        Write-Host "  Enabled:          " -NoNewline -ForegroundColor White
        Write-Host $Data.MetaRegimeHealth.enabled -ForegroundColor $(if ($Data.MetaRegimeHealth.enabled) { "Green" } else { "Red" })
        Write-Host "  Status:           " -NoNewline -ForegroundColor White
        Write-Host $Data.MetaRegimeHealth.status -ForegroundColor $(if ($Data.MetaRegimeHealth.status -eq "active") { "Green" } else { "Yellow" })
        
        if ($Data.MetaRegimeHealth.best_regime) {
            Write-Host "  Best Regime:      " -NoNewline -ForegroundColor White
            Write-Host $Data.MetaRegimeHealth.best_regime -ForegroundColor Cyan
            Write-Host "  Best PnL:         " -NoNewline -ForegroundColor White
            Write-Host $Data.MetaRegimeHealth.best_pnl -ForegroundColor $(if ($Data.MetaRegimeHealth.best_pnl -gt 0) { "Green" } else { "Red" })
        }
        
        Write-Host "  Regimes Detected: " -NoNewline -ForegroundColor White
        Write-Host $Data.MetaRegimeHealth.regimes_detected -ForegroundColor Cyan
    }
    
    # Regime Statistics
    if ($Data.RegimeStats) {
        Write-Host ""
        Write-Host "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━" -ForegroundColor DarkGray
        Write-Host "  📈 REGIME PERFORMANCE BREAKDOWN" -ForegroundColor Yellow
        Write-Host "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━" -ForegroundColor DarkGray
        Write-Host ""
        Write-Host "  Regime      Samples   Avg PnL    Win Rate   Avg Vol" -ForegroundColor Gray
        Write-Host "  ───────────────────────────────────────────────────" -ForegroundColor DarkGray
        
        foreach ($regime in $Data.RegimeStats.PSObject.Properties) {
            $name = $regime.Name
            $stats = $regime.Value
            
            $regimeColor = switch ($name) {
                "BULL" { "Green" }
                "BEAR" { "Red" }
                "RANGE" { "Yellow" }
                "VOLATILE" { "Magenta" }
                default { "White" }
            }
            
            $pnlColor = if ($stats.avg_pnl -gt 0) { "Green" } else { "Red" }
            
            Write-Host "  $($name.PadRight(10))" -NoNewline -ForegroundColor $regimeColor
            Write-Host "  $($stats.count.ToString().PadLeft(7))" -NoNewline -ForegroundColor White
            Write-Host "  $($stats.avg_pnl.ToString("0.0000").PadLeft(8))" -NoNewline -ForegroundColor $pnlColor
            Write-Host "  $($stats.win_rate.ToString("0.00%").PadLeft(9))" -NoNewline -ForegroundColor White
            Write-Host "  $($stats.avg_volatility.ToString("0.000").PadLeft(7))" -ForegroundColor White
        }
    }
    
    # Recent Activity
    if ($Data.RecentLogs) {
        Write-Host ""
        Write-Host "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━" -ForegroundColor DarkGray
        Write-Host "  📜 RECENT ACTIVITY" -ForegroundColor Yellow
        Write-Host "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━" -ForegroundColor DarkGray
        Write-Host ""
        Write-Host "  $($Data.RecentLogs)" -ForegroundColor Gray
    }
    
    Write-Host ""
    Write-Host "═══════════════════════════════════════════════════════════════════════════" -ForegroundColor Cyan
    Write-Host "  Next refresh in $RefreshInterval seconds... (Press Ctrl+C to exit)" -ForegroundColor Gray
    Write-Host "═══════════════════════════════════════════════════════════════════════════" -ForegroundColor Cyan
    Write-Host ""
}

# Main monitoring loop
Write-Host ""
Write-Host "🚀  Starting Meta-Regime Monitor..." -ForegroundColor Cyan
Write-Host "    VPS: $VpsHost" -ForegroundColor Gray
Write-Host "    Refresh: ${RefreshInterval}s" -ForegroundColor Gray
Write-Host ""
Write-Host "Press Ctrl+C to exit" -ForegroundColor Yellow
Start-Sleep -Seconds 2

$iteration = 0

try {
    while ($true) {
        $iteration++
        
        try {
            $data = Get-RegimeData -VpsUser $VpsUser -VpsHost $VpsHost -SshKey $SshKey
            Show-RegimeDisplay -Data $data -Iteration $iteration
        } catch {
            Write-Host "❌  Error fetching data: $($_.Exception.Message)" -ForegroundColor Red
        }
        
        Start-Sleep -Seconds $RefreshInterval
    }
} catch {
    Write-Host ""
    Write-Host "👋  Monitoring stopped" -ForegroundColor Yellow
}
