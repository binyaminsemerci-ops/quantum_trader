# Quick Status Check: Orchestrator LIVE Mode Step 1
# This script checks if the Orchestrator is active and enforcing signal filtering.

Write-Host "=" -NoNewline
Write-Host ("=" * 69)
Write-Host "🔴 ORCHESTRATOR LIVE MODE - STEP 1 STATUS CHECK"
Write-Host "=" -NoNewline
Write-Host ("=" * 69)
Write-Host ""

try {
    # Get recent logs
    $logs = docker logs quantum_backend --tail 500 2>&1 | Out-String
    
    # Check for LIVE mode initialization
    if ($logs -match "Orchestrator LIVE enforcing") {
        Write-Host "✅ LIVE MODE ACTIVE" -ForegroundColor Green
        $match = [regex]::Match($logs, "Orchestrator LIVE enforcing: (.+?)[\`"]")
        if ($match.Success) {
            Write-Host "   Enforcing: $($match.Groups[1].Value)"
        }
    } else {
        Write-Host "❌ LIVE MODE NOT DETECTED" -ForegroundColor Red
        Write-Host "   Expected: 'Orchestrator LIVE enforcing: signal_filter, confidence'"
    }
    
    Write-Host ""
    
    # Check for recent policy enforcement
    $policyEnforced = [regex]::Matches($logs, "LIVE MODE - Policy ENFORCED: (.+?)[\`"]")
    if ($policyEnforced.Count -gt 0) {
        Write-Host "✅ Policy enforced $($policyEnforced.Count) times" -ForegroundColor Green
        Write-Host "   Latest: $($policyEnforced[-1].Groups[1].Value)"
    } else {
        Write-Host "⏳ No policy enforcement logged yet (may be in cooldown)" -ForegroundColor Yellow
    }
    
    Write-Host ""
    
    # Check for blocked signals
    $blockedSignals = [regex]::Matches($logs, "BLOCKED by policy: (.+?)[\`"]")
    if ($blockedSignals.Count -gt 0) {
        Write-Host "🚫 $($blockedSignals.Count) signals blocked by policy" -ForegroundColor Yellow
        $last3 = $blockedSignals | Select-Object -Last 3
        for ($i = 0; $i -lt $last3.Count; $i++) {
            Write-Host "   $($i + 1). $($last3[$i].Groups[1].Value)"
        }
    } else {
        Write-Host "✅ No signals blocked yet (policy allowing all)" -ForegroundColor Green
    }
    
    Write-Host ""
    
    # Check for policy confidence adjustments
    $confChanges = [regex]::Matches($logs, "Using policy confidence: (\d+\.\d+) \(was (\d+\.\d+)\)")
    if ($confChanges.Count -gt 0) {
        $latest = $confChanges[-1]
        Write-Host "📊 Confidence threshold: $($latest.Groups[1].Value) (was $($latest.Groups[2].Value))" -ForegroundColor Cyan
        Write-Host "   Policy adjusted $($confChanges.Count) times"
    } else {
        Write-Host "📊 No confidence adjustments logged yet" -ForegroundColor Yellow
    }
    
    Write-Host ""
    
    # Check for recent trading cycles
    $cycles = [regex]::Matches($logs, "_check_and_execute\(\) started")
    if ($cycles.Count -gt 0) {
        Write-Host "🔄 Trading cycles active: $($cycles.Count) cycles in last 500 log lines" -ForegroundColor Green
    }
    
    Write-Host ""
    Write-Host "=" -NoNewline
    Write-Host ("=" * 69)
    Write-Host "Status checked at: $(Get-Date -Format 'yyyy-MM-dd HH:mm:ss')"
    Write-Host "=" -NoNewline
    Write-Host ("=" * 69)
    
} catch {
    Write-Host "❌ Error: $_" -ForegroundColor Red
}
