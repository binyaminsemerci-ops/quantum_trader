<#
.SYNOPSIS
    Check Quantum Trader system health

.DESCRIPTION
    Quick health check for:
    - Model Supervisor mode (ENFORCED vs OBSERVE)
    - Model bias detection (SHORT/LONG >70%)
    - Failed AI models
    - Retraining status

.PARAMETER Fix
    Automatically attempt to fix detected issues

.PARAMETER Watch
    Continuous monitoring mode

.PARAMETER Interval
    Watch interval in seconds (default: 60)

.EXAMPLE
    .\Check-SystemHealth.ps1
    .\Check-SystemHealth.ps1 -Fix
    .\Check-SystemHealth.ps1 -Watch -Interval 30
#>

param(
    [switch]$Fix,
    [switch]$Watch,
    [int]$Interval = 60,
    [string]$Url = "http://localhost:8000"
)

function Write-Banner {
    Write-Host ""
    Write-Host ("=" * 80)
    Write-Host "🏥 QUANTUM TRADER - SYSTEM HEALTH CHECK"
    Write-Host ("=" * 80)
    Write-Host "Timestamp: $(Get-Date -Format 'yyyy-MM-dd HH:mm:ss')"
    Write-Host ("=" * 80)
    Write-Host ""
}

function Get-HealthStatus {
    param([string]$BaseUrl)
    
    try {
        $response = Invoke-RestMethod -Uri "$BaseUrl/health/monitor" -Method Get -TimeoutSec 10
        return $response
    }
    catch {
        Write-Host "❌ ERROR: Cannot connect to backend" -ForegroundColor Red
        Write-Host "   URL: $BaseUrl"
        Write-Host "   Error: $($_.Exception.Message)"
        return @{ status = "error"; error = $_.Exception.Message }
    }
}

function Invoke-AutoHeal {
    param([string]$BaseUrl)
    
    try {
        $response = Invoke-RestMethod -Uri "$BaseUrl/health/monitor/auto-heal" -Method Post -TimeoutSec 30
        return $response
    }
    catch {
        Write-Host "❌ ERROR: Auto-heal failed" -ForegroundColor Red
        Write-Host "   Error: $($_.Exception.Message)"
        return @{ status = "error"; error = $_.Exception.Message }
    }
}

function Show-HealthStatus {
    param($HealthData)
    
    if ($HealthData.status -eq "disabled") {
        Write-Host "⚠️  Health Monitor: DISABLED" -ForegroundColor Yellow
        Write-Host "   Enable with: QT_HEALTH_MONITOR_ENABLED=true"
        return
    }
    
    if ($HealthData.status -eq "error") {
        Write-Host "❌ Error: $($HealthData.error)" -ForegroundColor Red
        return
    }
    
    $overall = $HealthData.overall_health
    $summary = $HealthData.summary
    $recommendations = $HealthData.recommendations
    
    # Overall status
    $statusColor = switch ($overall) {
        "HEALTHY" { "Green"; $emoji = "✅" }
        "DEGRADED" { "Yellow"; $emoji = "🟡" }
        "CRITICAL" { "Red"; $emoji = "🔴" }
        "FAILED" { "Red"; $emoji = "❌" }
        default { "Gray"; $emoji = "❓" }
    }
    
    Write-Host "$emoji OVERALL STATUS: $overall" -ForegroundColor $statusColor
    Write-Host ""
    
    # Summary stats
    $totalIssues = $summary.total_issues
    $bySeverity = $summary.issues_by_severity
    
    if ($totalIssues -eq 0) {
        Write-Host "✅ All systems healthy - no issues detected" -ForegroundColor Green
        Write-Host ""
        return
    }
    
    Write-Host "⚠️  $totalIssues issues detected:" -ForegroundColor Yellow
    if ($bySeverity.CRITICAL -gt 0) {
        Write-Host "   🔴 CRITICAL: $($bySeverity.CRITICAL)" -ForegroundColor Red
    }
    if ($bySeverity.DEGRADED -gt 0) {
        Write-Host "   🟡 DEGRADED: $($bySeverity.DEGRADED)" -ForegroundColor Yellow
    }
    if ($bySeverity.FAILED -gt 0) {
        Write-Host "   ❌ FAILED: $($bySeverity.FAILED)" -ForegroundColor Red
    }
    Write-Host ""
    
    # Expected config
    $expected = $summary.expected_config
    if ($expected) {
        Write-Host "📋 Expected Configuration:" -ForegroundColor Cyan
        Write-Host "   Model Supervisor Mode: $($expected.model_supervisor_mode)"
        Write-Host "   Bias Threshold: $([math]::Round($expected.bias_threshold * 100))%"
        Write-Host "   Min Samples: $($expected.min_samples)"
        Write-Host ""
    }
    
    # Detailed issues
    if ($recommendations -and $recommendations.Count -gt 0) {
        Write-Host "🔍 DETECTED ISSUES:" -ForegroundColor Cyan
        Write-Host ("-" * 80)
        
        $i = 1
        foreach ($rec in $recommendations) {
            $severityColor = switch ($rec.severity) {
                "CRITICAL" { "Red"; $severityEmoji = "🔴" }
                "DEGRADED" { "Yellow"; $severityEmoji = "🟡" }
                "FAILED" { "Red"; $severityEmoji = "❌" }
                default { "Gray"; $severityEmoji = "⚠️" }
            }
            
            Write-Host ""
            Write-Host "$i. [$($rec.component)] $severityEmoji $($rec.severity)" -ForegroundColor $severityColor
            Write-Host "   Problem: $($rec.problem)"
            
            if ($rec.auto_fixable) {
                Write-Host "   ✅ Auto-fixable: $($rec.fix_action)" -ForegroundColor Green
            }
            else {
                Write-Host "   ⚠️  Requires manual intervention" -ForegroundColor Yellow
                if ($rec.fix_action) {
                    Write-Host "   Action: $($rec.fix_action)"
                }
            }
            
            $i++
        }
        
        Write-Host ""
        Write-Host ("-" * 80)
    }
}

function Watch-Health {
    param(
        [string]$BaseUrl,
        [int]$CheckInterval
    )
    
    Write-Host "👁️  Watching system health (checking every $CheckInterval seconds)"
    Write-Host "Press Ctrl+C to stop"
    Write-Host ""
    
    try {
        while ($true) {
            Write-Banner
            $healthData = Get-HealthStatus -BaseUrl $BaseUrl
            Show-HealthStatus -HealthData $healthData
            
            Write-Host ""
            Write-Host "⏰ Next check in $CheckInterval seconds..."
            Start-Sleep -Seconds $CheckInterval
        }
    }
    catch {
        Write-Host ""
        Write-Host ""
        Write-Host "👋 Monitoring stopped"
    }
}

# Main execution
if ($Watch) {
    Watch-Health -BaseUrl $Url -CheckInterval $Interval
    exit
}

Write-Banner

# Check health
$healthData = Get-HealthStatus -BaseUrl $Url
Show-HealthStatus -HealthData $healthData

# Auto-fix if requested
if ($Fix -and $healthData.status -eq "ok") {
    $recommendations = $healthData.recommendations
    $fixableCount = ($recommendations | Where-Object { $_.auto_fixable }).Count
    
    if ($fixableCount -gt 0) {
        Write-Host ""
        Write-Host "🔧 Attempting to auto-fix $fixableCount issues..." -ForegroundColor Cyan
        
        $healResult = Invoke-AutoHeal -BaseUrl $Url
        
        if ($healResult.status -eq "ok") {
            $fixesApplied = $healResult.fixes_applied
            $remaining = $healResult.remaining_issues
            
            if ($fixesApplied -gt 0) {
                Write-Host "✅ Successfully fixed $fixesApplied issues" -ForegroundColor Green
            }
            else {
                Write-Host "⚠️  No auto-fixes could be applied" -ForegroundColor Yellow
            }
            
            if ($remaining -and $remaining.Count -gt 0) {
                Write-Host ""
                Write-Host "⚠️  $($remaining.Count) issues require manual intervention:" -ForegroundColor Yellow
                foreach ($issue in $remaining) {
                    Write-Host "   - [$($issue.component)] $($issue.problem)"
                }
            }
        }
        else {
            Write-Host "❌ Auto-heal failed: $($healResult.error)" -ForegroundColor Red
        }
    }
    else {
        Write-Host ""
        Write-Host "✅ No auto-fixable issues" -ForegroundColor Green
    }
}

Write-Host ""
