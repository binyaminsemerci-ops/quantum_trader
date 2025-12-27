<#
Attempt to install the CI watcher as a Windows service.

This script does NOT automatically download NSSM. It shows two options:
  1) Use NSSM (recommended): copy nssm.exe to a known location and run nssm install
  2) Use sc.exe to create a service that runs a wrapper CMD (less flexible)

Note: Installing a service requires administrative privileges. Run this script as Administrator.
#>

Write-Host "create_service.ps1: This action requires Administrator privileges."
if (-not ([bool](whoami /groups | Select-String "S-1-5-32-544"))) {
    Write-Warning "It doesn't look like you're running as Administrator. Run this script from an elevated PowerShell to install a service."
}

$pwshPath = (Get-Command pwsh -ErrorAction SilentlyContinue).Source
if (-not $pwshPath) { $pwshPath = 'pwsh' }
$watcherPath = (Resolve-Path '.\scripts\ci-watch.ps1').ProviderPath

Write-Host "Option A: Install with NSSM (recommended). Example commands to run as Admin:"
Write-Host "  nssm install QuantumTraderCIWatcher `"$pwshPath`" -NoProfile -ExecutionPolicy Bypass -File `"$watcherPath`" -PollIntervalSeconds 300"
Write-Host "  nssm start QuantumTraderCIWatcher"

Write-Host "Option B: Create a simple service using sc.exe that runs a CMD wrapper (requires creating the wrapper)."
Write-Host "  sc create QuantumTraderCIWatcher binPath=\"C:\\Windows\\System32\\cmd.exe /c start \"\" `"$pwshPath`" -NoProfile -ExecutionPolicy Bypass -File `"$watcherPath`" -PollIntervalSeconds 300\""

Write-Host "This script will now attempt to create a wrapper CMD in C:\\quantum_trader\\scripts named service_wrapper.cmd and will show the sc.exe command you can run as Admin."

$wrapper = Join-Path (Get-Location) 'scripts\service_wrapper.cmd'
$line1 = '@echo off'
$line2 = '"' + $pwshPath + '" -NoProfile -ExecutionPolicy Bypass -File "' + $watcherPath + '" -PollIntervalSeconds 300'
Set-Content -Path $wrapper -Value ($line1 + "`r`n" + $line2 + "`r`n") -Encoding ASCII -Force
Write-Host "WROTE wrapper: $wrapper"

Write-Host 'To create the service using sc.exe (run as Admin), execute the following exactly:'
$scCmd = 'sc create QuantumTraderCIWatcher binPath= "C:\\Windows\\System32\\cmd.exe /c ""' + $wrapper + '"" start= auto'
Write-Host $scCmd

Write-Host 'Then to start the service (as Admin):'
Write-Host '  sc start QuantumTraderCIWatcher'

Write-Host 'Note: service-based installation typically requires testing to ensure the watcher behaves correctly as a service (logging, working directory, graceful shutdown).'
