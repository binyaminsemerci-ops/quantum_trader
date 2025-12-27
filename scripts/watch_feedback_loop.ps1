# Phase 4S+ - Continuous Feedback Loop Monitor
# Watches strategic memory feedback in real-time

param(
    [string]$VpsHost = "46.224.116.254",
    [string]$VpsUser = "qt",
    [string]$SshKey = "~/.ssh/hetzner_fresh",
    [int]$RefreshInterval = 15
)

function Get-FeedbackLoop {
    param([string]$SshKey, [string]$VpsUser, [string]$VpsHost)
    
    $data = @{}
    
    try {
        # Get feedback
        $feedbackJson = wsl ssh -i $SshKey ${VpsUser}@${VpsHost} "docker exec quantum_redis redis-cli GET quantum:feedback:strategic_memory"
        if ($feedbackJson -and $feedbackJson -ne "(nil)") {
            $data.Feedback = $feedbackJson | ConvertFrom-Json
        }
        
        # Get current policy
        $data.CurrentPolicy = wsl ssh -i $SshKey ${VpsUser}@${VpsHost} "docker exec quantum_redis redis-cli GET quantum:governance:policy"
        
        # Get preferred regime
        $data.PreferredRegime = wsl ssh -i $SshKey ${VpsUser}@${VpsHost} "docker exec quantum_redis redis-cli GET quantum:governance:preferred_regime"
        
        # Get stream lengths
        $data.MetaStreamLen = wsl ssh -i $SshKey ${VpsUser}@${VpsHost} "docker exec quantum_redis redis-cli XLEN quantum:stream:meta.regime"
        $data.TradeStreamLen = wsl ssh -i $SshKey ${VpsUser}@${VpsHost} "docker exec quantum_redis redis-cli XLEN quantum:stream:trade.results"
        
    } catch {
        $data.Error = $_.Exception.Message
    }
    
    return $data
}

function Show-FeedbackDisplay {
    param($data, $iteration)
    
    Clear-Host
    
    Write-Host "═══════════════════════════════════════════════════════════════════════════" -ForegroundColor Cyan
    Write-Host "     🔁 PHASE 4S+ - CONTINUOUS FEEDBACK LOOP MONITOR" -ForegroundColor White
    Write-Host "═══════════════════════════════════════════════════════════════════════════" -ForegroundColor Cyan
    Write-Host ""
    Write-Host "📅  $(Get-Date -Format 'yyyy-MM-dd HH:mm:ss')" -ForegroundColor Gray
    Write-Host "🖥️  VPS: $VpsHost | Iteration: #$iteration" -ForegroundColor Gray
    Write-Host ""
    
    if ($data.Error) {
        Write-Host "❌ Error: $($data.Error)" -ForegroundColor Red
        return
    }
    
    # Current System State
    Write-Host "🎮  CURRENT SYSTEM STATE" -ForegroundColor Yellow
    Write-Host "────────────────────────────────────────────────────────────────────────────" -ForegroundColor DarkGray
    Write-Host "   Current Policy:     $(if($data.CurrentPolicy) {$data.CurrentPolicy} else {'Not Set'})" -ForegroundColor $(if($data.CurrentPolicy -eq 'AGGRESSIVE') {'Red'} elseif($data.CurrentPolicy -eq 'CONSERVATIVE') {'Green'} else {'Yellow'})
    Write-Host "   Preferred Regime:   $(if($data.PreferredRegime -and $data.PreferredRegime -ne '(nil)') {$data.PreferredRegime} else {'Not Set'})" -ForegroundColor Cyan
    Write-Host "   Meta-Regime Stream: $($data.MetaStreamLen) observations" -ForegroundColor White
    Write-Host "   Trade Stream:       $($data.TradeStreamLen) trades" -ForegroundColor White
    Write-Host ""
    
    # Feedback Loop
    Write-Host "🔁  STRATEGIC FEEDBACK LOOP" -ForegroundColor Yellow
    Write-Host "────────────────────────────────────────────────────────────────────────────" -ForegroundColor DarkGray
    
    if ($data.Feedback) {
        $fb = $data.Feedback
        
        Write-Host "   Status: ACTIVE ✅" -ForegroundColor Green
        Write-Host ""
        Write-Host "   📊  Recommendations:" -ForegroundColor Yellow
        Write-Host "      Preferred Regime:   $($fb.preferred_regime)" -ForegroundColor Cyan
        Write-Host "      Updated Policy:     $($fb.updated_policy)" -ForegroundColor $(if($fb.updated_policy -eq 'AGGRESSIVE') {'Red'} elseif($fb.updated_policy -eq 'CONSERVATIVE') {'Green'} else {'Yellow'})
        Write-Host "      Confidence Boost:   $($fb.confidence_boost)" -ForegroundColor Cyan
        Write-Host "      Leverage Hint:      $($fb.leverage_hint)x" -ForegroundColor Cyan
        Write-Host ""
        
        if ($fb.regime_performance) {
            $perf = $fb.regime_performance
            Write-Host "   📈  Best Regime Performance:" -ForegroundColor Yellow
            Write-Host "      Average PnL:    $($perf.avg_pnl)" -ForegroundColor $(if($perf.avg_pnl -gt 0) {'Green'} else {'Red'})
            Write-Host "      Win Rate:       $([math]::Round($perf.win_rate * 100, 2))%" -ForegroundColor $(if($perf.win_rate -gt 0.5) {'Green'} else {'Red'})
            Write-Host "      Sample Count:   $($perf.sample_count)" -ForegroundColor White
        }
        
        Write-Host ""
        Write-Host "   ⏱️  Last Update: $($fb.timestamp)" -ForegroundColor Gray
        
        # Policy Change Alert
        if ($data.CurrentPolicy -and $fb.updated_policy -ne $data.CurrentPolicy) {
            Write-Host ""
            Write-Host "   ⚠️  POLICY CHANGE RECOMMENDED!" -ForegroundColor Yellow
            Write-Host "      Current:    $($data.CurrentPolicy)" -ForegroundColor White
            Write-Host "      Suggested:  $($fb.updated_policy)" -ForegroundColor Cyan
            Write-Host "      Reason:     Best performance in $($fb.preferred_regime) regime" -ForegroundColor Gray
        }
        
    } else {
        Write-Host "   Status: NO DATA ⏳" -ForegroundColor Yellow
        Write-Host ""
        Write-Host "   Message: Feedback not yet generated" -ForegroundColor Gray
        Write-Host "   💡 Strategic Memory needs 3+ samples to generate feedback" -ForegroundColor Gray
    }
    
    Write-Host ""
    Write-Host "═══════════════════════════════════════════════════════════════════════════" -ForegroundColor Cyan
    Write-Host "  Refreshing in $RefreshInterval seconds... (Press Ctrl+C to exit)" -ForegroundColor Gray
    Write-Host "═══════════════════════════════════════════════════════════════════════════" -ForegroundColor Cyan
}

# Main monitoring loop
Write-Host "🚀  Starting Continuous Feedback Loop Monitor..." -ForegroundColor Green
Write-Host "📊  Refresh interval: $RefreshInterval seconds" -ForegroundColor Gray
Write-Host ""

$iteration = 0

try {
    while ($true) {
        $iteration++
        $data = Get-FeedbackLoop -SshKey $SshKey -VpsUser $VpsUser -VpsHost $VpsHost
        Show-FeedbackDisplay -data $data -iteration $iteration
        Start-Sleep -Seconds $RefreshInterval
    }
} catch {
    Write-Host ""
    Write-Host "❌  Monitor stopped: $($_.Exception.Message)" -ForegroundColor Red
}
