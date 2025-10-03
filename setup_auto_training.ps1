# setup_auto_training.ps1 - Setup automatic AI training for Quantum Trader
# Run as Administrator to install the scheduled task

param(
    [switch]$Install,
    [switch]$Uninstall,
    [switch]$Status,
    [switch]$TestRun
)

$TaskName = "Quantum Trader AI Auto Training"
$TaskPath = "\"
$BatchFile = "C:\quantum_trader\auto_training_scheduler.bat"
$XMLFile = "C:\quantum_trader\quantum_trader_auto_training.xml"

function Write-Banner {
    Write-Host "================================================" -ForegroundColor Cyan
    Write-Host "🤖 QUANTUM TRADER AI AUTO TRAINING SETUP" -ForegroundColor Yellow
    Write-Host "================================================" -ForegroundColor Cyan
}

function Test-AdminRights {
    $currentUser = [Security.Principal.WindowsIdentity]::GetCurrent()
    $principal = New-Object Security.Principal.WindowsPrincipal($currentUser)
    return $principal.IsInRole([Security.Principal.WindowsBuiltInRole]::Administrator)
}

function Install-AutoTraining {
    Write-Host "📝 Installing automatic training task..." -ForegroundColor Green
    
    if (-not (Test-Path $BatchFile)) {
        Write-Host "❌ Error: $BatchFile not found!" -ForegroundColor Red
        return $false
    }
    
    if (-not (Test-Path $XMLFile)) {
        Write-Host "❌ Error: $XMLFile not found!" -ForegroundColor Red
        return $false
    }
    
    try {
        # Import the task
        Register-ScheduledTask -TaskName $TaskName -Xml (Get-Content $XMLFile | Out-String) -Force
        Write-Host "✅ Task installed successfully!" -ForegroundColor Green
        Write-Host "🔄 AI training will now start automatically:" -ForegroundColor Yellow
        Write-Host "   - At PC startup (after 2 minute delay)" -ForegroundColor White
        Write-Host "   - Every hour while PC is active" -ForegroundColor White
        Write-Host "📂 Logs will be saved to: C:\quantum_trader\logs\auto_training\" -ForegroundColor White
        return $true
    }
    catch {
        Write-Host "❌ Failed to install task: $_" -ForegroundColor Red
        return $false
    }
}

function Uninstall-AutoTraining {
    Write-Host "🗑️ Removing automatic training task..." -ForegroundColor Yellow
    
    try {
        Unregister-ScheduledTask -TaskName $TaskName -Confirm:$false -ErrorAction SilentlyContinue
        Write-Host "✅ Task removed successfully!" -ForegroundColor Green
        return $true
    }
    catch {
        Write-Host "❌ Failed to remove task: $_" -ForegroundColor Red
        return $false
    }
}

function Get-TaskStatus {
    Write-Host "📊 Checking auto training status..." -ForegroundColor Blue
    
    $task = Get-ScheduledTask -TaskName $TaskName -ErrorAction SilentlyContinue
    
    if ($task) {
        Write-Host "✅ Auto training task is INSTALLED" -ForegroundColor Green
        Write-Host "📅 State: $($task.State)" -ForegroundColor White
        Write-Host "⏰ Next Run Time: $($task.NextRunTime)" -ForegroundColor White
        
        # Check for recent log files
        $logDir = "C:\quantum_trader\logs\auto_training"
        if (Test-Path $logDir) {
            $recentLogs = Get-ChildItem $logDir -Filter "*.log" | Sort-Object LastWriteTime -Descending | Select-Object -First 3
            if ($recentLogs) {
                Write-Host "📋 Recent training logs:" -ForegroundColor Yellow
                foreach ($log in $recentLogs) {
                    Write-Host "   - $($log.Name) ($($log.LastWriteTime))" -ForegroundColor White
                }
            }
        }
    } else {
        Write-Host "❌ Auto training task is NOT INSTALLED" -ForegroundColor Red
        Write-Host "💡 Run with -Install to set it up" -ForegroundColor Yellow
    }
}

function Test-Training {
    Write-Host "🧪 Running test training..." -ForegroundColor Blue
    
    if (Test-Path $BatchFile) {
        Write-Host "▶️ Executing: $BatchFile" -ForegroundColor White
        & $BatchFile 20
        Write-Host "✅ Test training completed!" -ForegroundColor Green
    } else {
        Write-Host "❌ Training script not found: $BatchFile" -ForegroundColor Red
    }
}

# Main execution
Write-Banner

if (-not (Test-AdminRights)) {
    Write-Host "⚠️ WARNING: This script should be run as Administrator for full functionality" -ForegroundColor Yellow
    Write-Host "💡 Right-click PowerShell and 'Run as Administrator'" -ForegroundColor White
    Write-Host ""
}

if ($Install) {
    Install-AutoTraining
} elseif ($Uninstall) {
    Uninstall-AutoTraining
} elseif ($Status) {
    Get-TaskStatus
} elseif ($TestRun) {
    Test-Training
} else {
    Write-Host "🚀 USAGE:" -ForegroundColor Green
    Write-Host "  .\setup_auto_training.ps1 -Install    # Install auto training" -ForegroundColor White
    Write-Host "  .\setup_auto_training.ps1 -Uninstall  # Remove auto training" -ForegroundColor White
    Write-Host "  .\setup_auto_training.ps1 -Status     # Check current status" -ForegroundColor White
    Write-Host "  .\setup_auto_training.ps1 -TestRun    # Test training manually" -ForegroundColor White
    Write-Host ""
    Write-Host "📋 FEATURES:" -ForegroundColor Yellow
    Write-Host "  ✅ Automatic training at PC startup" -ForegroundColor White
    Write-Host "  ✅ Hourly model updates while PC active" -ForegroundColor White
    Write-Host "  ✅ Automatic log management (24h retention)" -ForegroundColor White
    Write-Host "  ✅ Network-aware (only runs when online)" -ForegroundColor White
    Write-Host ""
}

Write-Host "================================================" -ForegroundColor Cyan