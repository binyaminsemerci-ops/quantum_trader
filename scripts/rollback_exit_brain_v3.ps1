<#
.SYNOPSIS
    Emergency rollback for Exit Brain V3 LIVE mode
    
.DESCRIPTION
    Quickly revert to safe mode in case of issues with LIVE deployment
    
.PARAMETER RollbackLevel
    Level of rollback: kill-switch, shadow, or legacy
    
.EXAMPLE
    .\rollback_exit_brain_v3.ps1 -RollbackLevel kill-switch
    Fastest rollback - disable LIVE rollout only (5s)
    
.EXAMPLE
    .\rollback_exit_brain_v3.ps1 -RollbackLevel legacy
    Complete rollback - return to legacy exit system (30s)
#>

[CmdletBinding()]
param(
    [Parameter(Mandatory=$true)]
    [ValidateSet('kill-switch', 'shadow', 'legacy')]
    [string]$RollbackLevel
)

$ErrorActionPreference = "Stop"

function Write-Critical { Write-Host "🔴 $args" -ForegroundColor Red }
function Write-Success { Write-Host "✅ $args" -ForegroundColor Green }
function Write-Info { Write-Host "ℹ️  $args" -ForegroundColor Cyan }

$WORKSPACE_ROOT = Split-Path -Parent $PSScriptRoot
$ENV_FILE = Join-Path $WORKSPACE_ROOT ".env"

Write-Critical "EMERGENCY ROLLBACK INITIATED"
Write-Info "Level: $RollbackLevel"
Write-Info "Time: $(Get-Date -Format 'yyyy-MM-dd HH:mm:ss')"

# Backup current state
$backupFile = "$ENV_FILE.rollback.$(Get-Date -Format 'yyyyMMdd_HHmmss')"
Copy-Item $ENV_FILE $backupFile
Write-Success "State backed up to: $backupFile"

# Determine new configuration
$config = @{}
switch ($RollbackLevel) {
    'kill-switch' {
        Write-Info "Kill-switch: Disabling LIVE rollout (fastest, ~5s)"
        $config = @{
            EXIT_MODE = "EXIT_BRAIN_V3"
            EXIT_EXECUTOR_MODE = "LIVE"
            EXIT_BRAIN_V3_LIVE_ROLLOUT = "DISABLED"  # Key change
            EXIT_BRAIN_V3_ENABLED = "true"
        }
    }
    'shadow' {
        Write-Info "Shadow mode: Full shadow mode (medium, ~10s)"
        $config = @{
            EXIT_MODE = "EXIT_BRAIN_V3"
            EXIT_EXECUTOR_MODE = "SHADOW"  # Key change
            EXIT_BRAIN_V3_LIVE_ROLLOUT = "DISABLED"
            EXIT_BRAIN_V3_ENABLED = "true"
        }
    }
    'legacy' {
        Write-Info "Legacy mode: Complete rollback to legacy system (~30s)"
        $config = @{
            EXIT_MODE = "LEGACY"  # Key change
            EXIT_EXECUTOR_MODE = "SHADOW"
            EXIT_BRAIN_V3_LIVE_ROLLOUT = "DISABLED"
            EXIT_BRAIN_V3_ENABLED = "false"
        }
    }
}

# Update .env
Write-Info "Updating .env file..."
$envContent = Get-Content $ENV_FILE

$envContent = $envContent | Where-Object { 
    $_ -notmatch '^EXIT_MODE=' -and 
    $_ -notmatch '^EXIT_EXECUTOR_MODE=' -and 
    $_ -notmatch '^EXIT_BRAIN_V3_LIVE_ROLLOUT=' -and
    $_ -notmatch '^EXIT_BRAIN_V3_ENABLED='
}

$newLines = @(
    "",
    "# Exit Brain V3 Configuration (ROLLBACK: $(Get-Date -Format 'yyyy-MM-dd HH:mm:ss'))",
    "EXIT_MODE=$($config.EXIT_MODE)",
    "EXIT_EXECUTOR_MODE=$($config.EXIT_EXECUTOR_MODE)",
    "EXIT_BRAIN_V3_LIVE_ROLLOUT=$($config.EXIT_BRAIN_V3_LIVE_ROLLOUT)",
    "EXIT_BRAIN_V3_ENABLED=$($config.EXIT_BRAIN_V3_ENABLED)"
)

$envContent += $newLines
$envContent | Set-Content $ENV_FILE
Write-Success ".env updated"

# Stop backend
Write-Info "Stopping backend..."
$backendProcess = Get-Process | Where-Object { $_.CommandLine -like "*python*backend*main.py*" } -ErrorAction SilentlyContinue
if ($backendProcess) {
    Stop-Process -Id $backendProcess.Id -Force
    Start-Sleep -Seconds 2
    Write-Success "Backend stopped"
} else {
    Write-Info "No running backend found"
}

# Set environment variables
foreach ($key in $config.Keys) {
    [Environment]::SetEnvironmentVariable($key, $config[$key], "Process")
}

# Restart backend
Write-Info "Restarting backend with rollback configuration..."
$BACKEND_MAIN = Join-Path $WORKSPACE_ROOT "backend\main.py"
$logFile = Join-Path $WORKSPACE_ROOT "backend\logs\rollback_$(Get-Date -Format 'yyyyMMdd_HHmmss').log"
$process = Start-Process -FilePath "python" -ArgumentList $BACKEND_MAIN -PassThru -NoNewWindow -RedirectStandardOutput $logFile

Write-Success "Backend restarted (PID: $($process.Id))"
Write-Info "Waiting 10 seconds..."
Start-Sleep -Seconds 10

# Verify
Write-Info "Verifying rollback..."
try {
    $status = Invoke-RestMethod -Uri "http://localhost:8000/health/exit_brain_status" -TimeoutSec 5
    Write-Success "Backend responding"
    Write-Host "`n📊 Current State:" -ForegroundColor Cyan
    Write-Host "   Operational State: $($status.operational_state)" -ForegroundColor Yellow
    Write-Host "   EXIT_MODE: $($status.config.EXIT_MODE)" -ForegroundColor Cyan
    Write-Host "   EXIT_EXECUTOR_MODE: $($status.config.EXIT_EXECUTOR_MODE)" -ForegroundColor Cyan
    Write-Host "   LIVE_ROLLOUT: $($status.config.EXIT_BRAIN_V3_LIVE_ROLLOUT)" -ForegroundColor Cyan
} catch {
    Write-Critical "Backend not responding yet: $_"
    Write-Info "Check logs: $logFile"
}

Write-Host "`n╔══════════════════════════════════════════════════════╗" -ForegroundColor Green
Write-Host "║  ROLLBACK COMPLETE                                   ║" -ForegroundColor Green
Write-Host "╚══════════════════════════════════════════════════════╝`n" -ForegroundColor Green

Write-Success "Rollback level: $RollbackLevel"
Write-Success "New mode: $($config.EXIT_MODE)"
Write-Success "Backup: $backupFile"
Write-Success "Logs: $logFile"

Write-Host "`n📋 Next Steps:" -ForegroundColor Cyan
Write-Host "  1. Verify system is operating correctly" -ForegroundColor White
Write-Host "  2. Review logs to understand what triggered rollback" -ForegroundColor White
Write-Host "  3. Address issues before re-attempting LIVE mode" -ForegroundColor White
