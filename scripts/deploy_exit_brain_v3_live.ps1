<#
.SYNOPSIS
    Deploy Exit Brain V3 to LIVE Mode - Safe Activation Script
    
.DESCRIPTION
    This script guides you through the safe activation of Exit Brain V3 LIVE mode
    following the official activation runbook. It includes safety checks and rollback options.
    
.PARAMETER Mode
    The deployment mode: shadow, prelive, or live
    
.PARAMETER SkipValidation
    Skip pre-deployment validation checks (not recommended)
    
.EXAMPLE
    .\deploy_exit_brain_v3_live.ps1 -Mode shadow
    Deploy to shadow mode (observation only)
    
.EXAMPLE
    .\deploy_exit_brain_v3_live.ps1 -Mode live
    Deploy to LIVE mode (AI controls exits)
#>

[CmdletBinding()]
param(
    [Parameter(Mandatory=$true)]
    [ValidateSet('shadow', 'prelive', 'live')]
    [string]$Mode,
    
    [Parameter(Mandatory=$false)]
    [switch]$SkipValidation
)

$ErrorActionPreference = "Stop"

# Colors for output
function Write-Success { Write-Host "✅ $args" -ForegroundColor Green }
function Write-Info { Write-Host "ℹ️  $args" -ForegroundColor Cyan }
function Write-Warning-Custom { Write-Host "⚠️  $args" -ForegroundColor Yellow }
function Write-Critical { Write-Host "🔴 $args" -ForegroundColor Red }
function Write-Step { Write-Host "`n🔹 $args" -ForegroundColor Magenta }

$WORKSPACE_ROOT = Split-Path -Parent $PSScriptRoot
$ENV_FILE = Join-Path $WORKSPACE_ROOT ".env"
$BACKEND_MAIN = Join-Path $WORKSPACE_ROOT "backend\main.py"
$DIAGNOSTIC_TOOL = Join-Path $WORKSPACE_ROOT "backend\tools\print_exit_status.py"
$RUNBOOK = Join-Path $WORKSPACE_ROOT "docs\EXIT_BRAIN_V3_ACTIVATION_RUNBOOK.md"

# Banner
Write-Host "`n╔══════════════════════════════════════════════════════╗" -ForegroundColor Cyan
Write-Host "║  EXIT BRAIN V3 LIVE MODE DEPLOYMENT SCRIPT          ║" -ForegroundColor Cyan
Write-Host "║  Phase 2B - AI-Controlled Exit Orders               ║" -ForegroundColor Cyan
Write-Host "╚══════════════════════════════════════════════════════╝`n" -ForegroundColor Cyan

Write-Info "Deployment Mode: $Mode"
Write-Info "Workspace: $WORKSPACE_ROOT"
Write-Info "Date: $(Get-Date -Format 'yyyy-MM-dd HH:mm:ss')"

# Check if runbook exists
if (-not (Test-Path $RUNBOOK)) {
    Write-Critical "Activation runbook not found: $RUNBOOK"
    exit 1
}

# Pre-flight checks
Write-Step "Pre-flight Checks"

if (-not (Test-Path $ENV_FILE)) {
    Write-Critical ".env file not found: $ENV_FILE"
    exit 1
}
Write-Success ".env file found"

if (-not (Test-Path $BACKEND_MAIN)) {
    Write-Critical "Backend main.py not found: $BACKEND_MAIN"
    exit 1
}
Write-Success "Backend main.py found"

if (-not (Test-Path $DIAGNOSTIC_TOOL)) {
    Write-Warning-Custom "Diagnostic tool not found (optional): $DIAGNOSTIC_TOOL"
} else {
    Write-Success "Diagnostic tool found"
}

# Check Python environment
Write-Step "Checking Python Environment"
try {
    $pythonVersion = python --version 2>&1
    Write-Success "Python: $pythonVersion"
} catch {
    Write-Critical "Python not found in PATH"
    exit 1
}

# Determine configuration based on mode
Write-Step "Configuring for $Mode mode"

$config = @{}
switch ($Mode) {
    'shadow' {
        Write-Info "Shadow Mode: AI observes and logs decisions, NO orders placed"
        $config = @{
            EXIT_MODE = "EXIT_BRAIN_V3"
            EXIT_EXECUTOR_MODE = "SHADOW"
            EXIT_BRAIN_V3_LIVE_ROLLOUT = "DISABLED"
            EXIT_BRAIN_V3_ENABLED = "true"
        }
        Write-Success "Configuration: AI in observation mode (safe)"
    }
    'prelive' {
        Write-Warning-Custom "Pre-LIVE Mode: Testing LIVE config with kill-switch active"
        $config = @{
            EXIT_MODE = "EXIT_BRAIN_V3"
            EXIT_EXECUTOR_MODE = "LIVE"
            EXIT_BRAIN_V3_LIVE_ROLLOUT = "DISABLED"
            EXIT_BRAIN_V3_ENABLED = "true"
        }
        Write-Info "This tests kill-switch fallback (should stay in SHADOW)"
    }
    'live' {
        Write-Critical "LIVE MODE: AI WILL PLACE REAL EXIT ORDERS!"
        Write-Critical "Legacy exit modules will be BLOCKED!"
        Write-Host "`n⚠️  Are you sure you want to enable LIVE mode?" -ForegroundColor Yellow
        Write-Host "   - Shadow mode completed 24-48h validation?" -ForegroundColor Yellow
        Write-Host "   - AI decisions confirmed to improve outcomes?" -ForegroundColor Yellow
        Write-Host "   - Monitoring and rollback plan in place?" -ForegroundColor Yellow
        Write-Host ""
        
        $confirmation = Read-Host "Type 'ENABLE_LIVE' to proceed"
        if ($confirmation -ne "ENABLE_LIVE") {
            Write-Info "Deployment cancelled by user"
            exit 0
        }
        
        $config = @{
            EXIT_MODE = "EXIT_BRAIN_V3"
            EXIT_EXECUTOR_MODE = "LIVE"
            EXIT_BRAIN_V3_LIVE_ROLLOUT = "ENABLED"
            EXIT_BRAIN_V3_ENABLED = "true"
        }
        Write-Critical "🔴 LIVE MODE CONFIGURATION CONFIRMED 🔴"
    }
}

# Display configuration
Write-Step "Configuration Summary"
foreach ($key in $config.Keys | Sort-Object) {
    $value = $config[$key]
    $color = if ($value -eq "ENABLED" -or $value -eq "LIVE") { "Red" } else { "Cyan" }
    Write-Host "  $key = $value" -ForegroundColor $color
}

# Backup current .env
Write-Step "Backing up current .env"
$backupFile = "$ENV_FILE.backup.$(Get-Date -Format 'yyyyMMdd_HHmmss')"
Copy-Item $ENV_FILE $backupFile
Write-Success "Backup created: $backupFile"

# Update .env file
Write-Step "Updating .env file"
$envContent = Get-Content $ENV_FILE

# Remove old EXIT_MODE related settings
$envContent = $envContent | Where-Object { 
    $_ -notmatch '^EXIT_MODE=' -and 
    $_ -notmatch '^EXIT_EXECUTOR_MODE=' -and 
    $_ -notmatch '^EXIT_BRAIN_V3_LIVE_ROLLOUT=' -and
    $_ -notmatch '^EXIT_BRAIN_V3_ENABLED='
}

# Add new configuration at the end
$newLines = @(
    "",
    "# Exit Brain V3 Configuration (Updated: $(Get-Date -Format 'yyyy-MM-dd HH:mm:ss'))",
    "EXIT_MODE=$($config.EXIT_MODE)",
    "EXIT_EXECUTOR_MODE=$($config.EXIT_EXECUTOR_MODE)",
    "EXIT_BRAIN_V3_LIVE_ROLLOUT=$($config.EXIT_BRAIN_V3_LIVE_ROLLOUT)",
    "EXIT_BRAIN_V3_ENABLED=$($config.EXIT_BRAIN_V3_ENABLED)"
)

$envContent += $newLines
$envContent | Set-Content $ENV_FILE

Write-Success ".env file updated with new configuration"

# Set environment variables for current session
Write-Step "Setting environment variables"
foreach ($key in $config.Keys) {
    [Environment]::SetEnvironmentVariable($key, $config[$key], "Process")
    Write-Info "Set $key=$($config[$key])"
}

# Stop running backend
Write-Step "Checking for running backend"
$backendProcess = Get-Process | Where-Object { $_.CommandLine -like "*python*backend*main.py*" } -ErrorAction SilentlyContinue
if ($backendProcess) {
    Write-Warning-Custom "Found running backend process (PID: $($backendProcess.Id))"
    Write-Host "Stopping backend..." -ForegroundColor Yellow
    Stop-Process -Id $backendProcess.Id -Force
    Start-Sleep -Seconds 2
    Write-Success "Backend stopped"
} else {
    Write-Info "No running backend found"
}

# Start backend
Write-Step "Starting backend with new configuration"

$logFile = Join-Path $WORKSPACE_ROOT "backend\logs\deployment_$(Get-Date -Format 'yyyyMMdd_HHmmss').log"

# Use the backend startup script if it exists, otherwise direct python
$startupScript = Join-Path $WORKSPACE_ROOT "scripts\start-backend.ps1"
if (Test-Path $startupScript) {
    Write-Info "Using backend startup script: $startupScript"
    $process = Start-Process -FilePath "powershell" -ArgumentList "-NoProfile", "-ExecutionPolicy", "Bypass", "-File", $startupScript -PassThru
} else {
    Write-Info "Starting backend directly with virtual environment"
    
    # Create a startup script that activates venv and starts backend
    $tempScript = Join-Path $env:TEMP "deploy_backend_$(Get-Date -Format 'yyyyMMddHHmmss').ps1"
    $scriptContent = @"
Set-Location '$WORKSPACE_ROOT'
if (Test-Path '.venv\Scripts\Activate.ps1') { 
    & '.venv\Scripts\Activate.ps1'
}
`$env:EXIT_MODE='$($config.EXIT_MODE)'
`$env:EXIT_EXECUTOR_MODE='$($config.EXIT_EXECUTOR_MODE)'
`$env:EXIT_BRAIN_V3_LIVE_ROLLOUT='$($config.EXIT_BRAIN_V3_LIVE_ROLLOUT)'
`$env:EXIT_BRAIN_V3_ENABLED='$($config.EXIT_BRAIN_V3_ENABLED)'
python backend\main.py
"@
    $scriptContent | Out-File -FilePath $tempScript -Encoding UTF8
    $process = Start-Process -FilePath "powershell" -ArgumentList "-NoProfile", "-ExecutionPolicy", "Bypass", "-File", $tempScript -PassThru
}

Write-Success "Backend started (PID: $($process.Id))"
Write-Info "Waiting 10 seconds for initialization..."
Start-Sleep -Seconds 10

# Verify backend is running
if (-not $process.HasExited) {
    Write-Success "Backend process is running"
} else {
    Write-Critical "Backend process exited unexpectedly"
    Write-Info "Check logs: $logFile"
    exit 1
}

# Run diagnostics
Write-Step "Running diagnostics"

if (Test-Path $DIAGNOSTIC_TOOL) {
    Write-Info "Running: python $DIAGNOSTIC_TOOL"
    try {
        $diagnosticOutput = & python $DIAGNOSTIC_TOOL 2>&1
        Write-Host "`n$diagnosticOutput`n"
    } catch {
        Write-Warning-Custom "Diagnostic tool failed: $_"
    }
} else {
    Write-Info "Diagnostic tool not available, skipping"
}

# Health check
Write-Step "Health Check"
$maxAttempts = 5
$attempt = 0
$healthOk = $false

while ($attempt -lt $maxAttempts -and -not $healthOk) {
    $attempt++
    Write-Info "Health check attempt $attempt/$maxAttempts..."
    
    try {
        $response = Invoke-RestMethod -Uri "http://localhost:8000/health" -TimeoutSec 5
        if ($response) {
            Write-Success "Backend health check passed"
            $healthOk = $true
        }
    } catch {
        Write-Warning-Custom "Health check failed: $_"
        if ($attempt -lt $maxAttempts) {
            Write-Info "Waiting 5 seconds before retry..."
            Start-Sleep -Seconds 5
        }
    }
}

if (-not $healthOk) {
    Write-Critical "Health check failed after $maxAttempts attempts"
    Write-Info "Backend may still be starting. Check logs manually."
}

# Exit Brain status check
Write-Step "Exit Brain Status Check"
try {
    $exitBrainStatus = Invoke-RestMethod -Uri "http://localhost:8000/health/exit_brain_status" -TimeoutSec 5
    Write-Success "Exit Brain status endpoint accessible"
    
    Write-Host "`n📊 Exit Brain Status:" -ForegroundColor Cyan
    Write-Host "   Operational State: $($exitBrainStatus.operational_state)" -ForegroundColor $(
        if ($exitBrainStatus.operational_state -eq "EXIT_BRAIN_V3_LIVE") { "Red" } else { "Yellow" }
    )
    Write-Host "   EXIT_MODE: $($exitBrainStatus.config.EXIT_MODE)" -ForegroundColor Cyan
    Write-Host "   EXIT_EXECUTOR_MODE: $($exitBrainStatus.config.EXIT_EXECUTOR_MODE)" -ForegroundColor Cyan
    Write-Host "   LIVE_ROLLOUT: $($exitBrainStatus.config.EXIT_BRAIN_V3_LIVE_ROLLOUT)" -ForegroundColor Cyan
    
    if ($exitBrainStatus.executor_state) {
        Write-Host "   Executor Running: $($exitBrainStatus.executor_state.is_running)" -ForegroundColor Green
        Write-Host "   Effective Mode: $($exitBrainStatus.executor_state.effective_mode)" -ForegroundColor $(
            if ($exitBrainStatus.executor_state.effective_mode -eq "LIVE") { "Red" } else { "Yellow" }
        )
    }
} catch {
    Write-Warning-Custom "Exit Brain status check failed: $_"
    Write-Info "Check manually: curl http://localhost:8000/health/exit_brain_status"
}

# Deployment summary
Write-Host "`n╔══════════════════════════════════════════════════════╗" -ForegroundColor Green
Write-Host "║  DEPLOYMENT COMPLETE                                 ║" -ForegroundColor Green
Write-Host "╚══════════════════════════════════════════════════════╝`n" -ForegroundColor Green

Write-Success "Mode: $Mode"
Write-Success "Backend PID: $($process.Id)"
Write-Success "Logs: $logFile"
Write-Success "Backup: $backupFile"

# Mode-specific next steps
Write-Host "`n📋 Next Steps:" -ForegroundColor Cyan

switch ($Mode) {
    'shadow' {
        Write-Host "  1. Monitor shadow logs: tail -f backend/logs/quantum_trader.log | grep EXIT_BRAIN_SHADOW" -ForegroundColor White
        Write-Host "  2. Collect shadow decisions: backend/data/exit_brain_shadow.jsonl" -ForegroundColor White
        Write-Host "  3. Run for 24-48 hours to validate AI decisions" -ForegroundColor White
        Write-Host "  4. Analyze: python backend/tools/analyze_exit_brain_shadow.py" -ForegroundColor White
        Write-Host "  5. If satisfied, proceed to: .\scripts\deploy_exit_brain_v3_live.ps1 -Mode prelive" -ForegroundColor Yellow
    }
    'prelive' {
        Write-Host "  1. Verify system stayed in SHADOW mode (kill-switch worked)" -ForegroundColor White
        Write-Host "  2. Check logs for fallback message: grep 'Forcing SHADOW mode' backend/logs/*.log" -ForegroundColor White
        Write-Host "  3. If kill-switch verified, proceed to: .\scripts\deploy_exit_brain_v3_live.ps1 -Mode live" -ForegroundColor Yellow
    }
    'live' {
        Write-Critical "🔴 AI IS NOW CONTROLLING EXIT ORDERS 🔴"
        Write-Host ""
        Write-Host "  1. MONITOR CLOSELY for next 1-2 hours:" -ForegroundColor Red
        Write-Host "     tail -f backend/logs/quantum_trader.log | grep -E 'EXIT_BRAIN_LIVE|EXIT_GUARD'" -ForegroundColor White
        Write-Host ""
        Write-Host "  2. Watch for AI execution logs:" -ForegroundColor Yellow
        Write-Host "     [EXIT_BRAIN_LIVE] Executing ..." -ForegroundColor White
        Write-Host ""
        Write-Host "  3. Confirm legacy modules blocked:" -ForegroundColor Yellow
        Write-Host "     [EXIT_GUARD] 🛑 BLOCKED: Legacy module ..." -ForegroundColor White
        Write-Host ""
        Write-Host "  4. Check metrics:" -ForegroundColor Yellow
        Write-Host "     curl http://localhost:8000/health/exit_brain_status | jq '.metrics'" -ForegroundColor White
        Write-Host ""
        Write-Host "  5. EMERGENCY ROLLBACK OPTIONS:" -ForegroundColor Red
        Write-Host "     Option A (5s):  Set EXIT_BRAIN_V3_LIVE_ROLLOUT=DISABLED in .env, restart" -ForegroundColor White
        Write-Host "     Option B (10s): Set EXIT_EXECUTOR_MODE=SHADOW in .env, restart" -ForegroundColor White
        Write-Host "     Option C (30s): Set EXIT_MODE=LEGACY in .env, restart" -ForegroundColor White
    }
}

Write-Host "`n📖 Full documentation: docs/EXIT_BRAIN_V3_ACTIVATION_RUNBOOK.md" -ForegroundColor Cyan
Write-Host ""
