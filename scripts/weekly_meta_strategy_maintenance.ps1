# Weekly Meta-Strategy Maintenance Script
# Runs monitoring and tuning for Meta-Strategy Selector
# 
# Schedule: Every Monday at 09:00 AM
# Purpose: Track Q-learning convergence and tune epsilon/alpha parameters

param(
    [switch]$DryRun = $false,
    [switch]$SkipTuning = $false
)

# Configuration
$ProjectRoot = Split-Path -Parent $PSScriptRoot
$VenvPath = "$ProjectRoot\.venv\Scripts\Activate.ps1"
$LogDir = "$ProjectRoot\logs\meta_strategy"
$Timestamp = Get-Date -Format "yyyy-MM-dd_HHmmss"
$LogFile = "$LogDir\maintenance_$Timestamp.log"

# Ensure log directory exists
if (-not (Test-Path $LogDir)) {
    New-Item -ItemType Directory -Path $LogDir -Force | Out-Null
}

# Logging function
function Write-Log {
    param([string]$Message)
    $LogMessage = "$(Get-Date -Format 'yyyy-MM-dd HH:mm:ss') - $Message"
    Write-Host $LogMessage
    Add-Content -Path $LogFile -Value $LogMessage
}

Write-Log "=" * 80
Write-Log "META-STRATEGY WEEKLY MAINTENANCE"
Write-Log "=" * 80
Write-Log ""

# Activate virtual environment
Write-Log "Activating virtual environment..."
try {
    & $VenvPath
    Write-Log "✅ Virtual environment activated"
} catch {
    Write-Log "❌ Failed to activate virtual environment: $_"
    exit 1
}

# Change to project root
Set-Location $ProjectRoot
Write-Log "Working directory: $ProjectRoot"
Write-Log ""

# ============================================================================
# STEP 1: Run Performance Monitor
# ============================================================================

Write-Log "─" * 80
Write-Log "STEP 1: Running Performance Monitor"
Write-Log "─" * 80
Write-Log ""

try {
    $MonitorOutput = & python monitor_meta_strategy_performance.py 2>&1
    Write-Log $MonitorOutput
    Write-Log ""
    Write-Log "✅ Performance monitoring completed"
    
    # Save monitor output to separate file
    $MonitorLogFile = "$LogDir\monitor_$Timestamp.txt"
    $MonitorOutput | Out-File -FilePath $MonitorLogFile -Encoding UTF8
    Write-Log "📊 Monitor report saved to: $MonitorLogFile"
} catch {
    Write-Log "❌ Performance monitoring failed: $_"
}

Write-Log ""

# ============================================================================
# STEP 2: Run Parameter Tuning
# ============================================================================

if (-not $SkipTuning) {
    Write-Log "─" * 80
    Write-Log "STEP 2: Running Parameter Tuning"
    Write-Log "─" * 80
    Write-Log ""
    
    if ($DryRun) {
        Write-Log "⚠️  DRY RUN MODE - No changes will be applied"
        Write-Log ""
    }
    
    try {
        # Run tuning script with automatic 'dry' input if DryRun mode
        if ($DryRun) {
            $TuningOutput = "dry" | & python tune_meta_strategy_parameters.py 2>&1
        } else {
            $TuningOutput = "y" | & python tune_meta_strategy_parameters.py 2>&1
        }
        
        Write-Log $TuningOutput
        Write-Log ""
        Write-Log "✅ Parameter tuning completed"
        
        # Save tuning output to separate file
        $TuningLogFile = "$LogDir\tuning_$Timestamp.txt"
        $TuningOutput | Out-File -FilePath $TuningLogFile -Encoding UTF8
        Write-Log "🔧 Tuning report saved to: $TuningLogFile"
        
        # Check if .env was updated (and not in dry run)
        if (-not $DryRun -and $TuningOutput -match "\.env file updated") {
            Write-Log ""
            Write-Log "⚠️  IMPORTANT: .env file was updated with new parameters"
            Write-Log "⚠️  Backend restart required to apply changes"
            Write-Log ""
            Write-Log "To restart backend, run:"
            Write-Log "   docker-compose --profile dev restart backend"
        }
    } catch {
        Write-Log "❌ Parameter tuning failed: $_"
    }
} else {
    Write-Log "⏭️  Skipping parameter tuning (--SkipTuning flag set)"
}

Write-Log ""

# ============================================================================
# STEP 3: Summary
# ============================================================================

Write-Log "─" * 80
Write-Log "MAINTENANCE SUMMARY"
Write-Log "─" * 80
Write-Log ""
Write-Log "📁 Log files:"
Write-Log "   Main log: $LogFile"
if (Test-Path "$LogDir\monitor_$Timestamp.txt") {
    Write-Log "   Monitor:  $LogDir\monitor_$Timestamp.txt"
}
if (Test-Path "$LogDir\tuning_$Timestamp.txt") {
    Write-Log "   Tuning:   $LogDir\tuning_$Timestamp.txt"
}
Write-Log ""

# Clean up old logs (keep last 30 days)
$OldLogs = Get-ChildItem -Path $LogDir -Filter "*.log" | 
    Where-Object { $_.LastWriteTime -lt (Get-Date).AddDays(-30) }

if ($OldLogs) {
    Write-Log "🗑️  Cleaning up old logs (> 30 days): $($OldLogs.Count) files"
    $OldLogs | Remove-Item -Force
}

Write-Log ""
Write-Log "=" * 80
Write-Log "MAINTENANCE COMPLETED"
Write-Log "=" * 80

# Return exit code based on success
exit 0
