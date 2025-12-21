# ============================================================================
# Meta-Regime Correlator - Test Script
# ============================================================================
# Tests meta-regime functionality by injecting various regime scenarios
# Usage: .\test_meta_regime.ps1
# ============================================================================

param(
    [string]$VpsHost = "46.224.116.254",
    [string]$VpsUser = "qt",
    [string]$SshKey = "~/.ssh/hetzner_fresh"
)

function Inject-RegimeScenario {
    param(
        [string]$ScenarioName,
        [array]$Observations
    )
    
    Write-Host ""
    Write-Host "📊  Testing Scenario: $ScenarioName" -ForegroundColor Cyan
    Write-Host "    Injecting $($Observations.Count) observations..." -ForegroundColor Gray
    
    foreach ($obs in $Observations) {
        $timestamp = (Get-Date).ToUniversalTime().ToString("yyyy-MM-ddTHH:mm:ssZ")
        $cmd = "docker exec quantum_redis redis-cli XADD quantum:stream:meta.regime '*' " +
               "regime $($obs.regime) " +
               "pnl $($obs.pnl) " +
               "volatility $($obs.volatility) " +
               "trend $($obs.trend) " +
               "confidence $($obs.confidence) " +
               "timestamp '$timestamp'"
        
        wsl ssh -i $SshKey ${VpsUser}@${VpsHost} $cmd | Out-Null
        Start-Sleep -Milliseconds 200
    }
    
    Write-Host "    ✅  Injected successfully" -ForegroundColor Green
    Start-Sleep -Seconds 3
    
    # Check result
    $preferred = wsl ssh -i $SshKey ${VpsUser}@${VpsHost} "docker exec quantum_redis redis-cli GET quantum:governance:preferred_regime"
    $policy = wsl ssh -i $SshKey ${VpsUser}@${VpsHost} "docker exec quantum_redis redis-cli GET quantum:governance:policy"
    
    Write-Host "    📈  Result:" -ForegroundColor Yellow
    Write-Host "        Preferred Regime: $preferred" -ForegroundColor White
    Write-Host "        Current Policy: $policy" -ForegroundColor White
}

Write-Host ""
Write-Host "═══════════════════════════════════════════════════════════════════════════" -ForegroundColor Cyan
Write-Host "  🧪 META-REGIME CORRELATOR - TEST SUITE" -ForegroundColor Cyan
Write-Host "═══════════════════════════════════════════════════════════════════════════" -ForegroundColor Cyan
Write-Host ""

# Clear existing data
Write-Host "🧹  Clearing existing regime data..." -ForegroundColor Yellow
wsl ssh -i $SshKey ${VpsUser}@${VpsHost} "docker exec quantum_redis redis-cli DEL quantum:stream:meta.regime"
wsl ssh -i $SshKey ${VpsUser}@${VpsHost} "docker exec quantum_redis redis-cli DEL quantum:governance:preferred_regime"
wsl ssh -i $SshKey ${VpsUser}@${VpsHost} "docker exec quantum_redis redis-cli DEL quantum:governance:regime_stats"
Write-Host "✅  Cleared" -ForegroundColor Green

# Scenario 1: Strong Bull Market
$bullScenario = @(
    @{regime="BULL"; pnl=0.45; volatility=0.012; trend=0.0035; confidence=0.92}
    @{regime="BULL"; pnl=0.38; volatility=0.015; trend=0.0028; confidence=0.89}
    @{regime="BULL"; pnl=0.52; volatility=0.011; trend=0.0042; confidence=0.94}
    @{regime="BULL"; pnl=0.41; volatility=0.014; trend=0.0031; confidence=0.88}
    @{regime="BULL"; pnl=0.36; volatility=0.016; trend=0.0025; confidence=0.85}
    @{regime="RANGE"; pnl=0.12; volatility=0.008; trend=0.0002; confidence=0.78}
)
Inject-RegimeScenario -ScenarioName "Strong Bull Market" -Observations $bullScenario

# Scenario 2: Bear Market with Losses
$bearScenario = @(
    @{regime="BEAR"; pnl=-0.22; volatility=0.025; trend=-0.0038; confidence=0.81}
    @{regime="BEAR"; pnl=-0.18; volatility=0.028; trend=-0.0032; confidence=0.79}
    @{regime="BEAR"; pnl=-0.15; volatility=0.022; trend=-0.0025; confidence=0.83}
    @{regime="VOLATILE"; pnl=-0.35; volatility=0.048; trend=0.0015; confidence=0.72}
    @{regime="BEAR"; pnl=-0.12; volatility=0.021; trend=-0.0028; confidence=0.80}
)
Inject-RegimeScenario -ScenarioName "Bear Market with Losses" -Observations $bearScenario

# Scenario 3: Volatile Mixed Performance
$volatileScenario = @(
    @{regime="VOLATILE"; pnl=-0.28; volatility=0.052; trend=0.0008; confidence=0.68}
    @{regime="VOLATILE"; pnl=-0.32; volatility=0.045; trend=-0.0012; confidence=0.71}
    @{regime="VOLATILE"; pnl=0.15; volatility=0.041; trend=0.0022; confidence=0.73}
    @{regime="VOLATILE"; pnl=-0.18; volatility=0.038; trend=-0.0005; confidence=0.75}
    @{regime="RANGE"; pnl=0.08; volatility=0.009; trend=0.0001; confidence=0.82}
)
Inject-RegimeScenario -ScenarioName "Volatile Mixed Performance" -Observations $volatileScenario

# Scenario 4: Range-Bound Small Profits
$rangeScenario = @(
    @{regime="RANGE"; pnl=0.08; volatility=0.006; trend=0.0001; confidence=0.86}
    @{regime="RANGE"; pnl=0.12; volatility=0.007; trend=-0.0001; confidence=0.84}
    @{regime="RANGE"; pnl=0.09; volatility=0.005; trend=0.0002; confidence=0.88}
    @{regime="RANGE"; pnl=0.11; volatility=0.008; trend=0.0000; confidence=0.83}
    @{regime="BULL"; pnl=0.22; volatility=0.013; trend=0.0018; confidence=0.87}
)
Inject-RegimeScenario -ScenarioName "Range-Bound Small Profits" -Observations $rangeScenario

# Final Summary
Write-Host ""
Write-Host "═══════════════════════════════════════════════════════════════════════════" -ForegroundColor Cyan
Write-Host "  📊 FINAL STATISTICS" -ForegroundColor Cyan
Write-Host "═══════════════════════════════════════════════════════════════════════════" -ForegroundColor Cyan
Write-Host ""

$streamLen = wsl ssh -i $SshKey ${VpsUser}@${VpsHost} "docker exec quantum_redis redis-cli XLEN quantum:stream:meta.regime"
$preferred = wsl ssh -i $SshKey ${VpsUser}@${VpsHost} "docker exec quantum_redis redis-cli GET quantum:governance:preferred_regime"
$policy = wsl ssh -i $SshKey ${VpsUser}@${VpsHost} "docker exec quantum_redis redis-cli GET quantum:governance:policy"

Write-Host "  Total Observations: $streamLen" -ForegroundColor White
Write-Host "  Preferred Regime:   $preferred" -ForegroundColor Cyan
Write-Host "  Current Policy:     $policy" -ForegroundColor Yellow
Write-Host ""

# Get detailed stats from AI Engine
Write-Host "  Fetching detailed statistics from AI Engine..." -ForegroundColor Gray
$healthResponse = wsl ssh -i $SshKey ${VpsUser}@${VpsHost} "curl -s http://localhost:8001/health"
$health = $healthResponse | ConvertFrom-Json
$metaRegime = $health.metrics.meta_regime

if ($metaRegime) {
    Write-Host ""
    Write-Host "  🎯  Best Performing Regime: $($metaRegime.best_regime)" -ForegroundColor Green
    Write-Host "      Average PnL: $($metaRegime.best_pnl)" -ForegroundColor White
    Write-Host "      Total Regimes Detected: $($metaRegime.regimes_detected)" -ForegroundColor White
    Write-Host "      Status: $($metaRegime.status)" -ForegroundColor $(if ($metaRegime.status -eq "active") { "Green" } else { "Yellow" })
}

Write-Host ""
Write-Host "═══════════════════════════════════════════════════════════════════════════" -ForegroundColor Cyan
Write-Host "  ✅  TEST SUITE COMPLETED" -ForegroundColor Green
Write-Host "═══════════════════════════════════════════════════════════════════════════" -ForegroundColor Cyan
Write-Host ""
Write-Host "  💡  Next Steps:" -ForegroundColor Yellow
Write-Host "      • Run: .\monitor_meta_regime.ps1 (for real-time monitoring)" -ForegroundColor Gray
Write-Host "      • Check logs: docker logs -f quantum_meta_regime" -ForegroundColor Gray
Write-Host "      • Health endpoint: curl http://localhost:8001/health" -ForegroundColor Gray
Write-Host ""
