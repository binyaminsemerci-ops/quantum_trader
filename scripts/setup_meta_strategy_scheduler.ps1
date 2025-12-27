# Setup Windows Task Scheduler for Meta-Strategy Weekly Maintenance
# 
# This script creates a scheduled task that runs every Monday at 09:00 AM
# to perform Meta-Strategy monitoring and parameter tuning.
#
# Run this script once to set up the scheduled task:
#   .\scripts\setup_meta_strategy_scheduler.ps1
#
# To remove the task:
#   Unregister-ScheduledTask -TaskName "QuantumTrader-MetaStrategy-Weekly" -Confirm:$false

param(
    [string]$RunTime = "09:00",
    [ValidateSet("Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday")]
    [string]$RunDay = "Monday"
)

# Configuration
$TaskName = "QuantumTrader-MetaStrategy-Weekly"
$ProjectRoot = Split-Path -Parent $PSScriptRoot
$ScriptPath = "$ProjectRoot\scripts\weekly_meta_strategy_maintenance.ps1"

Write-Host "=" * 80
Write-Host "  SETUP META-STRATEGY WEEKLY SCHEDULER"
Write-Host "=" * 80
Write-Host ""

# Check if running as Administrator
$IsAdmin = ([Security.Principal.WindowsPrincipal] [Security.Principal.WindowsIdentity]::GetCurrent()).IsInRole([Security.Principal.WindowsBuiltInRole]::Administrator)

if (-not $IsAdmin) {
    Write-Host "❌ This script requires Administrator privileges" -ForegroundColor Red
    Write-Host ""
    Write-Host "Please run PowerShell as Administrator and try again:"
    Write-Host "  Right-click PowerShell → Run as Administrator" -ForegroundColor Yellow
    Write-Host "  cd $ProjectRoot" -ForegroundColor Yellow
    Write-Host "  .\scripts\setup_meta_strategy_scheduler.ps1" -ForegroundColor Yellow
    Write-Host ""
    exit 1
}

# Verify script exists
if (-not (Test-Path $ScriptPath)) {
    Write-Host "❌ Maintenance script not found: $ScriptPath" -ForegroundColor Red
    exit 1
}

Write-Host "Configuration:"
Write-Host "  Task Name:  $TaskName"
Write-Host "  Script:     $ScriptPath"
Write-Host "  Schedule:   Every $RunDay at $RunTime"
Write-Host ""

# Check if task already exists
$ExistingTask = Get-ScheduledTask -TaskName $TaskName -ErrorAction SilentlyContinue

if ($ExistingTask) {
    Write-Host "⚠️  Task '$TaskName' already exists" -ForegroundColor Yellow
    Write-Host ""
    $Response = Read-Host "Do you want to replace it? (y/n)"
    
    if ($Response -ne 'y') {
        Write-Host "Cancelled" -ForegroundColor Yellow
        exit 0
    }
    
    Write-Host "Removing existing task..."
    Unregister-ScheduledTask -TaskName $TaskName -Confirm:$false
    Write-Host "✅ Existing task removed"
    Write-Host ""
}

# Create scheduled task action
$Action = New-ScheduledTaskAction `
    -Execute "pwsh.exe" `
    -Argument "-NoProfile -ExecutionPolicy Bypass -File `"$ScriptPath`"" `
    -WorkingDirectory $ProjectRoot

# Create scheduled task trigger (weekly on specified day)
$Trigger = New-ScheduledTaskTrigger `
    -Weekly `
    -DaysOfWeek $RunDay `
    -At $RunTime

# Create scheduled task settings
$Settings = New-ScheduledTaskSettingsSet `
    -AllowStartIfOnBatteries `
    -DontStopIfGoingOnBatteries `
    -StartWhenAvailable `
    -RunOnlyIfNetworkAvailable `
    -ExecutionTimeLimit (New-TimeSpan -Hours 1)

# Create scheduled task principal (run as current user)
$Principal = New-ScheduledTaskPrincipal `
    -UserId $env:USERNAME `
    -LogonType S4U `
    -RunLevel Highest

# Register the scheduled task
try {
    $Task = Register-ScheduledTask `
        -TaskName $TaskName `
        -Action $Action `
        -Trigger $Trigger `
        -Settings $Settings `
        -Principal $Principal `
        -Description "Weekly Meta-Strategy Selector monitoring and parameter tuning for Quantum Trader"
    
    Write-Host "✅ Scheduled task created successfully!" -ForegroundColor Green
    Write-Host ""
    Write-Host "Task Details:"
    Write-Host "  Name:       $($Task.TaskName)"
    Write-Host "  State:      $($Task.State)"
    Write-Host "  Schedule:   Every $RunDay at $RunTime"
    Write-Host "  Next Run:   $((Get-ScheduledTask -TaskName $TaskName | Get-ScheduledTaskInfo).NextRunTime)"
    Write-Host ""
    
    Write-Host "📋 Task Management Commands:" -ForegroundColor Cyan
    Write-Host "  View task:"
    Write-Host "    Get-ScheduledTask -TaskName '$TaskName'" -ForegroundColor Yellow
    Write-Host ""
    Write-Host "  Run task immediately (for testing):"
    Write-Host "    Start-ScheduledTask -TaskName '$TaskName'" -ForegroundColor Yellow
    Write-Host ""
    Write-Host "  View task history:"
    Write-Host "    Get-ScheduledTaskInfo -TaskName '$TaskName'" -ForegroundColor Yellow
    Write-Host ""
    Write-Host "  View logs:"
    Write-Host "    Get-ChildItem -Path '$ProjectRoot\logs\meta_strategy' -Filter *.log | Sort-Object LastWriteTime -Descending | Select-Object -First 5" -ForegroundColor Yellow
    Write-Host ""
    Write-Host "  Remove task:"
    Write-Host "    Unregister-ScheduledTask -TaskName '$TaskName' -Confirm:`$false" -ForegroundColor Yellow
    Write-Host ""
    
    # Test run option
    Write-Host "─" * 80
    Write-Host "Would you like to test the task now? (y/n)"
    $TestResponse = Read-Host
    
    if ($TestResponse -eq 'y') {
        Write-Host ""
        Write-Host "🧪 Running maintenance script (DRY RUN mode)..."
        Write-Host ""
        
        & pwsh.exe -NoProfile -ExecutionPolicy Bypass -File $ScriptPath -DryRun
        
        Write-Host ""
        Write-Host "✅ Test completed - check output above"
    }
    
} catch {
    Write-Host "❌ Failed to create scheduled task: $_" -ForegroundColor Red
    exit 1
}

Write-Host ""
Write-Host "=" * 80
Write-Host "  SETUP COMPLETE"
Write-Host "=" * 80
