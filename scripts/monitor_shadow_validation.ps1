# Shadow Validation Monitor - Phase 3.5
# Monitors ensemble performance for 24-48 hours

param(
    [int]$IntervalMinutes = 30,
    [string]$OutputFile = "shadow_validation_report.txt"
)

$VPS = "root@46.224.116.254"
$SSH_KEY = "~/.ssh/hetzner_fresh"

Write-Host "`n========================================" -ForegroundColor Cyan
Write-Host "  QUANTUM TRADER - SHADOW VALIDATION" -ForegroundColor Cyan
Write-Host "  Phase 3.5: 24-48 Hour Monitoring" -ForegroundColor Cyan
Write-Host "========================================`n" -ForegroundColor Cyan

function Get-Timestamp {
    return (Get-Date).ToString("yyyy-MM-dd HH:mm:ss")
}

function Log-Entry {
    param([string]$Message)
    $entry = "[$(Get-Timestamp)] $Message"
    Write-Host $entry
    Add-Content -Path $OutputFile -Value $entry
}

function Get-EnsembleMetrics {
    Log-Entry "`n--- ENSEMBLE PERFORMANCE ---"
    
    # Get recent predictions
    $predictions = wsl ssh -i $SSH_KEY $VPS 'docker logs quantum_ai_engine 2>&1 | tail -200 | grep -E "CHART\] ENSEMBLE" | tail -10'
    Log-Entry "Recent Predictions:"
    Log-Entry $predictions
    
    # Get model breakdown
    $breakdown = wsl ssh -i $SSH_KEY $VPS 'docker logs quantum_ai_engine 2>&1 | tail -200 | grep -E "(XGB:|LGBM:|NH:|PT:)" | tail -20'
    Log-Entry "`nModel Breakdown:"
    Log-Entry $breakdown
}

function Get-GovernanceMetrics {
    Log-Entry "`n--- GOVERNANCE METRICS ---"
    
    # Get weight adjustments
    $weights = wsl ssh -i $SSH_KEY $VPS 'docker logs quantum_ai_engine 2>&1 | tail -500 | grep "Adjusted weights" | tail -5'
    Log-Entry "Dynamic Weight Adjustments:"
    Log-Entry $weights
    
    # Get final weights per cycle
    $cycles = wsl ssh -i $SSH_KEY $VPS 'docker logs quantum_ai_engine 2>&1 | tail -500 | grep "Cycle complete" | tail -5'
    Log-Entry "`nGovernance Cycles:"
    Log-Entry $cycles
}

function Get-ConfidenceMetrics {
    Log-Entry "`n--- CONFIDENCE METRICS ---"
    
    # Get confidence distribution
    $confidence = wsl ssh -i $SSH_KEY $VPS 'docker logs quantum_ai_engine 2>&1 | tail -200 | grep -oP "confidence[=:][\s]*\K[0-9.]+(?!x)" | head -20'
    
    if ($confidence) {
        $values = $confidence -split "`n" | Where-Object { $_ -match '^\d+\.?\d*$' } | ForEach-Object { [double]$_ }
        if ($values.Count -gt 0) {
            $avg = ($values | Measure-Object -Average).Average
            $max = ($values | Measure-Object -Maximum).Maximum
            $min = ($values | Measure-Object -Minimum).Minimum
            
            Log-Entry "Confidence Stats (last 20 predictions):"
            Log-Entry "  Average: $($avg.ToString('0.00'))"
            Log-Entry "  Max: $($max.ToString('0.00'))"
            Log-Entry "  Min: $($min.ToString('0.00'))"
            Log-Entry "  Count: $($values.Count)"
        }
    }
}

function Get-ModelHealth {
    Log-Entry "`n--- MODEL HEALTH ---"
    
    # Check for errors
    $errors = wsl ssh -i $SSH_KEY $VPS 'docker logs quantum_ai_engine 2>&1 | tail -500 | grep -i "error\|failed\|exception" | tail -10'
    if ($errors) {
        Log-Entry "Recent Errors:"
        Log-Entry $errors
    } else {
        Log-Entry "No recent errors detected ✓"
    }
    
    # Check model loading status
    $models = wsl ssh -i $SSH_KEY $VPS 'docker logs quantum_ai_engine 2>&1 | grep "Found latest" | tail -4'
    Log-Entry "`nLoaded Models:"
    Log-Entry $models
}

function Get-PredictionDistribution {
    Log-Entry "`n--- PREDICTION DISTRIBUTION ---"
    
    # Count BUY/SELL/HOLD
    $actions = wsl ssh -i $SSH_KEY $VPS 'docker logs quantum_ai_engine 2>&1 | tail -500 | grep -oP "ENSEMBLE \w+: (BUY|SELL|HOLD)" | cut -d" " -f3'
    
    if ($actions) {
        $actionList = $actions -split "`n" | Where-Object { $_ }
        $buy = ($actionList | Where-Object { $_ -eq "BUY" }).Count
        $sell = ($actionList | Where-Object { $_ -eq "SELL" }).Count
        $hold = ($actionList | Where-Object { $_ -eq "HOLD" }).Count
        $total = $buy + $sell + $hold
        
        if ($total -gt 0) {
            Log-Entry "Action Distribution (last 500 logs):"
            Log-Entry "  BUY:  $buy ($(($buy/$total*100).ToString('0.0'))%)"
            Log-Entry "  SELL: $sell ($(($sell/$total*100).ToString('0.0'))%)"
            Log-Entry "  HOLD: $hold ($(($hold/$total*100).ToString('0.0'))%)"
            Log-Entry "  Total: $total predictions"
        }
    }
}

function Get-SystemStatus {
    Log-Entry "`n--- SYSTEM STATUS ---"
    
    # Container health
    $health = wsl ssh -i $SSH_KEY $VPS 'docker ps --filter name=quantum_ai_engine --format "{{.Status}}"'
    Log-Entry "AI Engine: $health"
    
    # Memory usage
    $mem = wsl ssh -i $SSH_KEY $VPS 'docker stats quantum_ai_engine --no-stream --format "{{.MemUsage}}"'
    Log-Entry "Memory: $mem"
    
    # Disk usage
    $disk = wsl ssh -i $SSH_KEY $VPS 'df -h /home/qt/quantum_trader | tail -1'
    Log-Entry "Disk: $disk"
}

# Initialize report
Log-Entry "==================================================="
Log-Entry "SHADOW VALIDATION MONITORING STARTED"
Log-Entry "Monitoring Interval: $IntervalMinutes minutes"
Log-Entry "==================================================="

# Main monitoring loop
$iteration = 1
while ($true) {
    Log-Entry "`n`n########## ITERATION $iteration ##########"
    
    try {
        Get-SystemStatus
        Get-ModelHealth
        Get-EnsembleMetrics
        Get-GovernanceMetrics
        Get-ConfidenceMetrics
        Get-PredictionDistribution
        
        Log-Entry "`nNext check in $IntervalMinutes minutes..."
        
        # Summary stats
        $fileSize = (Get-Item $OutputFile -ErrorAction SilentlyContinue).Length
        if ($fileSize) {
            Log-Entry "Report size: $([math]::Round($fileSize/1KB, 2)) KB"
        }
        
    } catch {
        Log-Entry "ERROR: $($_.Exception.Message)"
    }
    
    $iteration++
    
    # Wait for next interval
    Start-Sleep -Seconds ($IntervalMinutes * 60)
}
