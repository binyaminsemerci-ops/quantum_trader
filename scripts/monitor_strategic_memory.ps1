# Phase 4S Strategic Memory - Real-Time Monitoring Dashboard
# Shows learning progress and feedback metrics

param(
    [string]$VpsHost = "46.224.116.254",
    [string]$VpsUser = "qt",
    [string]$SshKey = "~/.ssh/hetzner_fresh",
    [int]$RefreshInterval = 20
)

function Get-MemoryData {
    param([string]$SshKey, [string]$VpsUser, [string]$VpsHost)
    
    $data = @{}
    
    try {
        # Get feedback from Redis
        $feedbackJson = wsl ssh -i $SshKey ${VpsUser}@${VpsHost} "docker exec quantum_redis redis-cli GET quantum:feedback:strategic_memory"
        if ($feedbackJson -and $feedbackJson -ne "(nil)") {
            $feedback = $feedbackJson | ConvertFrom-Json
            $data.Feedback = $feedback
        } else {
            $data.Feedback = $null
        }
        
        # Get stream lengths
        $data.MetaStreamLen = wsl ssh -i $SshKey ${VpsUser}@${VpsHost} "docker exec quantum_redis redis-cli XLEN quantum:stream:meta.regime"
        $data.PnlStreamLen = wsl ssh -i $SshKey ${VpsUser}@${VpsHost} "docker exec quantum_redis redis-cli XLEN quantum:stream:portfolio.memory"
        $data.TradeStreamLen = wsl ssh -i $SshKey ${VpsUser}@${VpsHost} "docker exec quantum_redis redis-cli XLEN quantum:stream:trade.results"
        
        # Get current policy
        $data.CurrentPolicy = wsl ssh -i $SshKey ${VpsUser}@${VpsHost} "docker exec quantum_redis redis-cli GET quantum:governance:policy"
        
        # Get AI Engine health
        $healthResponse = wsl ssh -i $SshKey ${VpsUser}@${VpsHost} "curl -s http://localhost:8001/health"
        if ($healthResponse) {
            $health = $healthResponse | ConvertFrom-Json
            $data.AIEngineStatus = $health.metrics.strategic_memory
        }
        
        # Get container stats
        $containerInfo = wsl ssh -i $SshKey ${VpsUser}@${VpsHost} "docker stats quantum_strategic_memory --no-stream --format 'json'"
        if ($containerInfo) {
            $stats = $containerInfo | ConvertFrom-Json
            $data.CpuPercent = $stats.CPUPerc
            $data.MemUsage = $stats.MemUsage
        }
        
    } catch {
        $data.Error = $_.Exception.Message
    }
    
    return $data
}

function Show-MemoryDisplay {
    param($data)
    
    Clear-Host
    
    Write-Host "═══════════════════════════════════════════════════════════════════════════" -ForegroundColor Cyan
    Write-Host "     🧠 PHASE 4S - STRATEGIC MEMORY SYNC DASHBOARD" -ForegroundColor White
    Write-Host "═══════════════════════════════════════════════════════════════════════════" -ForegroundColor Cyan
    Write-Host ""
    Write-Host "📅  $(Get-Date -Format 'yyyy-MM-dd HH:mm:ss')" -ForegroundColor Gray
    Write-Host "🖥️  VPS: $VpsHost" -ForegroundColor Gray
    Write-Host ""
    
    if ($data.Error) {
        Write-Host "❌ Error: $($data.Error)" -ForegroundColor Red
        return
    }
    
    # Container Status
    Write-Host "🐳  CONTAINER STATUS" -ForegroundColor Yellow
    Write-Host "────────────────────────────────────────────────────────────────────────────" -ForegroundColor DarkGray
    Write-Host "   Container: quantum_strategic_memory" -ForegroundColor White
    if ($data.CpuPercent) {
        Write-Host "   CPU Usage: $($data.CpuPercent)" -ForegroundColor Cyan
        Write-Host "   Memory: $($data.MemUsage)" -ForegroundColor Cyan
    }
    Write-Host ""
    
    # Data Sources
    Write-Host "📊  DATA SOURCES" -ForegroundColor Yellow
    Write-Host "────────────────────────────────────────────────────────────────────────────" -ForegroundColor DarkGray
    Write-Host "   Meta-Regime Stream:    $($data.MetaStreamLen) observations" -ForegroundColor White
    Write-Host "   PnL Memory Stream:     $($data.PnlStreamLen) entries" -ForegroundColor White
    Write-Host "   Trade Results Stream:  $($data.TradeStreamLen) trades" -ForegroundColor White
    $totalSamples = [int]$data.MetaStreamLen + [int]$data.PnlStreamLen + [int]$data.TradeStreamLen
    Write-Host "   Total Samples:         $totalSamples" -ForegroundColor Green
    Write-Host ""
    
    # Current System State
    Write-Host "🎮  CURRENT SYSTEM STATE" -ForegroundColor Yellow
    Write-Host "────────────────────────────────────────────────────────────────────────────" -ForegroundColor DarkGray
    if ($data.CurrentPolicy) {
        Write-Host "   Portfolio Policy: $($data.CurrentPolicy)" -ForegroundColor Cyan
    } else {
        Write-Host "   Portfolio Policy: Not Set" -ForegroundColor Gray
    }
    Write-Host ""
    
    # Strategic Feedback
    Write-Host "🧠  STRATEGIC FEEDBACK" -ForegroundColor Yellow
    Write-Host "────────────────────────────────────────────────────────────────────────────" -ForegroundColor DarkGray
    
    if ($data.Feedback) {
        $fb = $data.Feedback
        
        Write-Host "   Status: ACTIVE" -ForegroundColor Green
        Write-Host ""
        Write-Host "   Preferred Regime:     $($fb.preferred_regime)" -ForegroundColor Cyan
        Write-Host "   Recommended Policy:   $($fb.updated_policy)" -ForegroundColor Cyan
        Write-Host "   Confidence Boost:     $($fb.confidence_boost)" -ForegroundColor Cyan
        Write-Host "   Leverage Hint:        $($fb.leverage_hint)x" -ForegroundColor Cyan
        Write-Host ""
        
        if ($fb.regime_performance) {
            $perf = $fb.regime_performance
            Write-Host "   📈  Best Regime Performance:" -ForegroundColor Yellow
            Write-Host "      Average PnL:    $($perf.avg_pnl)" -ForegroundColor White
            Write-Host "      Win Rate:       $([math]::Round($perf.win_rate * 100, 2))%" -ForegroundColor White
            Write-Host "      Sample Count:   $($perf.sample_count)" -ForegroundColor White
        }
        
        Write-Host ""
        Write-Host "   Last Update: $($fb.timestamp)" -ForegroundColor Gray
        
    } elseif ($data.AIEngineStatus -and $data.AIEngineStatus.status -eq "warming_up") {
        Write-Host "   Status: WARMING UP" -ForegroundColor Yellow
        Write-Host "   Message: $($data.AIEngineStatus.message)" -ForegroundColor Gray
        Write-Host ""
        Write-Host "   ⏳  Waiting for analysis cycle..." -ForegroundColor Gray
        Write-Host "   📊  Need at least 3 samples to generate feedback" -ForegroundColor Gray
        
    } else {
        Write-Host "   Status: NO DATA" -ForegroundColor Red
        Write-Host "   Message: Feedback not yet generated" -ForegroundColor Gray
    }
    
    Write-Host ""
    Write-Host "═══════════════════════════════════════════════════════════════════════════" -ForegroundColor Cyan
    Write-Host "  Refreshing in $RefreshInterval seconds... (Press Ctrl+C to exit)" -ForegroundColor Gray
    Write-Host "═══════════════════════════════════════════════════════════════════════════" -ForegroundColor Cyan
}

# Main monitoring loop
Write-Host "🚀  Starting Strategic Memory Monitor..." -ForegroundColor Green
Write-Host "📊  Refresh interval: $RefreshInterval seconds" -ForegroundColor Gray
Write-Host ""

try {
    while ($true) {
        $data = Get-MemoryData -SshKey $SshKey -VpsUser $VpsUser -VpsHost $VpsHost
        Show-MemoryDisplay -data $data
        Start-Sleep -Seconds $RefreshInterval
    }
} catch {
    Write-Host ""
    Write-Host "❌  Monitor stopped: $($_.Exception.Message)" -ForegroundColor Red
}
