<#
Register the CI watcher script as a Windows Scheduled Task that runs at user logon.

Usage (run as the user who should own the task):
  pwsh .\scripts\register-ci-watch-task.ps1

Notes:
 - This creates a task named 'QuantumTrader-CI-Watcher' that runs the local
   `scripts\ci-watch.ps1` script with PowerShell. It will run at user logon.
 - To change the interval or run options, edit the -ArgumentList below.
 - Creating a scheduled task may require appropriate privileges. The script
   will attempt to create a task in the current user's context.
#>

$taskName = 'QuantumTrader-CI-Watcher'
$scriptPath = (Resolve-Path -Path "$PSScriptRoot\ci-watch.ps1" -ErrorAction SilentlyContinue)
if (-not $scriptPath) {
  Write-Error "Could not find ci-watch.ps1 in the same directory as this script."
  exit 1
}

Write-Host "Registering scheduled task '$taskName' to run: $scriptPath"

# Build argument list for pwsh
$pwshPath = (Get-Command pwsh -ErrorAction SilentlyContinue).Source
if (-not $pwshPath) { $pwshPath = 'pwsh' }
$arg = "-NoProfile -ExecutionPolicy Bypass -File `"$scriptPath`" -PollIntervalSeconds 300"

# First try using the ScheduledTask cmdlets (preferred)
try {
  $action = New-ScheduledTaskAction -Execute $pwshPath -Argument $arg
  $trigger = New-ScheduledTaskTrigger -AtLogOn
  $principal = New-ScheduledTaskPrincipal -UserId $env:USERNAME -LogonType Interactive -RunLevel Limited
  Register-ScheduledTask -TaskName $taskName -Action $action -Trigger $trigger -Principal $principal -Force -ErrorAction Stop
  Write-Host "Registered Scheduled Task '$taskName' to run at user logon (via Register-ScheduledTask)."
  exit 0
} catch {
  Write-Warning "Register-ScheduledTask failed or unsupported in this environment: $($_.Exception.Message)"
}

# Fallback: use schtasks.exe to create a task in the current user's context
try {
  # Build schtasks command with careful quoting: user, task name and action must be quoted
  $taskNameQuoted = '"' + $taskName + '"'
  $actionQuoted = '"' + $pwshPath + ' ' + $arg + '"'
  $cmd = "schtasks /Create /RU $env:USERNAME /RL LIMITED /SC ONLOGON /TN $taskNameQuoted /TR $actionQuoted /F"
  Write-Host "Falling back to schtasks.exe: $cmd"
  $out = cmd.exe /c $cmd 2>&1
  Write-Host $out
  if ($LASTEXITCODE -eq 0) {
    Write-Host "Registered Scheduled Task '$taskName' via schtasks.exe."
    exit 0
  } else {
    Write-Error "schtasks failed: $out"
    exit 2
  }
} catch {
  Write-Error "Fallback schtasks registration failed: $($_.Exception.Message)"
  exit 2
}
