# ============================================================================
# Portfolio Governance Monitor (VPS)
# ============================================================================
# Real-time monitoring av Portfolio Governance Agent på VPS
# ============================================================================

param(
    [int]$RefreshInterval = 15,
    [switch]$ShowLogs = $false
)

# VPS Configuration
$VPS_HOST = "46.224.116.254"
$VPS_USER = "qt"
$SSH_KEY = "$HOME\.ssh\hetzner_fresh"

function Get-VPSCommand {
    param([string]$Command)
    try {
        $result = wsl ssh -i $SSH_KEY "$VPS_USER@$VPS_HOST" $Command 2>&1
        return $result
    } catch {
        return $null
    }
}

Write-Host ""
Write-Host "📊 =============================================" -ForegroundColor Cyan
Write-Host "   PORTFOLIO GOVERNANCE MONITOR (VPS)" -ForegroundColor Green
Write-Host "=============================================" -ForegroundColor Cyan
Write-Host ""
Write-Host "🔄 Refresh Interval: $RefreshInterval seconds" -ForegroundColor Gray
Write-Host "📡 Target: $VPS_USER@$VPS_HOST" -ForegroundColor Gray
Write-Host "Press Ctrl+C to stop" -ForegroundColor Yellow
Write-Host ""

while ($true) {
    Clear-Host
    
    $timestamp = Get-Date -Format "yyyy-MM-dd HH:mm:ss"
    Write-Host "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━" -ForegroundColor Cyan
    Write-Host "📊 PORTFOLIO GOVERNANCE STATUS - $timestamp" -ForegroundColor Green
    Write-Host "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━" -ForegroundColor Cyan
    Write-Host ""
    
    # Container Status
    Write-Host "🐳 Container Status:" -ForegroundColor Yellow
    $containerStatus = Get-VPSCommand "docker ps --format 'table {{.Names}}\t{{.Status}}\t{{.State}}' | grep portfolio_governance"
    if ($containerStatus) {
        Write-Host "   $containerStatus" -ForegroundColor Green
    } else {
        Write-Host "   ❌ Container not running!" -ForegroundColor Red
    }
    Write-Host ""
    
    # Current Policy
    Write-Host "🎯 Current Policy:" -ForegroundColor Yellow
    $policy = Get-VPSCommand "docker exec redis redis-cli GET quantum:governance:policy 2>/dev/null"
    if ($policy) {
        $policyColor = switch ($policy) {
            "CONSERVATIVE" { "Red" }
            "BALANCED" { "Yellow" }
            "AGGRESSIVE" { "Green" }
            default { "Gray" }
        }
        Write-Host "   $policy" -ForegroundColor $policyColor
    } else {
        Write-Host "   (Not set)" -ForegroundColor Gray
    }
    Write-Host ""
    
    # Portfolio Score
    Write-Host "📈 Portfolio Score:" -ForegroundColor Yellow
    $score = Get-VPSCommand "docker exec redis redis-cli GET quantum:governance:score 2>/dev/null"
    if ($score) {
        try {
            $scoreValue = [double]$score
            $scoreColor = if ($scoreValue -lt 0.3) { "Red" } elseif ($scoreValue -lt 0.7) { "Yellow" } else { "Green" }
            Write-Host "   $score" -ForegroundColor $scoreColor
        } catch {
            Write-Host "   $score" -ForegroundColor Gray
        }
    } else {
        Write-Host "   0.0" -ForegroundColor Gray
    }
    Write-Host ""
    
    # Memory Samples
    Write-Host "💾 Memory Samples:" -ForegroundColor Yellow
    $memoryLen = Get-VPSCommand "docker exec redis redis-cli XLEN quantum:stream:portfolio.memory 2>/dev/null"
    if ($memoryLen) {
        Write-Host "   $memoryLen events in stream" -ForegroundColor White
    } else {
        Write-Host "   0 events" -ForegroundColor Gray
    }
    Write-Host ""
    
    # Policy Parameters
    Write-Host "⚙️  Policy Parameters:" -ForegroundColor Yellow
    $params = Get-VPSCommand "docker exec redis redis-cli GET quantum:governance:params 2>/dev/null"
    if ($params) {
        try {
            $paramsObj = $params | ConvertFrom-Json
            Write-Host "   Max Leverage: $($paramsObj.max_leverage)" -ForegroundColor White
            Write-Host "   Position Size Limit: $($paramsObj.position_size_limit)" -ForegroundColor White
            Write-Host "   Risk Per Trade: $($paramsObj.risk_per_trade)" -ForegroundColor White
        } catch {
            Write-Host "   (Parsing error)" -ForegroundColor Gray
        }
    } else {
        Write-Host "   (Not set)" -ForegroundColor Gray
    }
    Write-Host ""
    
    # AI Engine Health
    Write-Host "🧠 AI Engine Integration:" -ForegroundColor Yellow
    $health = Get-VPSCommand "curl -s http://localhost:8001/health 2>/dev/null"
    if ($health) {
        try {
            $healthObj = $health | ConvertFrom-Json
            if ($healthObj.metrics.portfolio_governance) {
                $pgMetrics = $healthObj.metrics.portfolio_governance
                Write-Host "   ✅ Available in health endpoint" -ForegroundColor Green
                Write-Host "   Policy: $($pgMetrics.policy)" -ForegroundColor White
                Write-Host "   Score: $($pgMetrics.score)" -ForegroundColor White
                Write-Host "   Samples: $($pgMetrics.memory_samples)" -ForegroundColor White
                Write-Host "   Status: $($pgMetrics.status)" -ForegroundColor White
            } else {
                Write-Host "   ⚠️  Not yet available" -ForegroundColor Yellow
            }
        } catch {
            Write-Host "   ⚠️  Health check error" -ForegroundColor Yellow
        }
    } else {
        Write-Host "   ❌ Cannot reach AI Engine" -ForegroundColor Red
    }
    Write-Host ""
    
    # Recent Logs (if enabled)
    if ($ShowLogs) {
        Write-Host "📜 Recent Logs (last 10 lines):" -ForegroundColor Yellow
        Write-Host "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━" -ForegroundColor Gray
        $logs = Get-VPSCommand "docker logs --tail 10 quantum_portfolio_governance 2>&1"
        if ($logs) {
            $logs | ForEach-Object {
                if ($_ -match "ERROR|CRITICAL") {
                    Write-Host $_ -ForegroundColor Red
                } elseif ($_ -match "WARNING") {
                    Write-Host $_ -ForegroundColor Yellow
                } elseif ($_ -match "policy|POLICY") {
                    Write-Host $_ -ForegroundColor Cyan
                } else {
                    Write-Host $_ -ForegroundColor Gray
                }
            }
        }
        Write-Host "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━" -ForegroundColor Gray
        Write-Host ""
    }
    
    # Recent Memory Events
    Write-Host "📊 Recent Memory Events (last 5):" -ForegroundColor Yellow
    Write-Host "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━" -ForegroundColor Gray
    $events = Get-VPSCommand "docker exec redis redis-cli XREVRANGE quantum:stream:portfolio.memory + - COUNT 5 2>/dev/null"
    if ($events) {
        Write-Host $events -ForegroundColor White
    } else {
        Write-Host "   (No events yet)" -ForegroundColor Gray
    }
    Write-Host "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━" -ForegroundColor Gray
    Write-Host ""
    
    Write-Host "Next refresh in $RefreshInterval seconds..." -ForegroundColor Gray
    Write-Host "(Press Ctrl+C to stop)" -ForegroundColor Yellow
    
    Start-Sleep -Seconds $RefreshInterval
}
