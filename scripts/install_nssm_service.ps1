<#
Install the CI watcher as a Windows service using NSSM.

Usage (run as Administrator):
  1. Download nssm.exe and place it in scripts\tools\nssm.exe (or in PATH).
  2. Run: pwsh -NoProfile -ExecutionPolicy Bypass -File .\scripts\install_nssm_service.ps1

This script will:
 - detect nssm.exe under scripts\tools or in PATH
 - call nssm install QuantumTraderCIWatcher <pwsh> <args>
 - set the service to start automatically and start it

Requires Administrator privileges to install/start services.
#>

function Find-Nssm {
    $local = Join-Path (Join-Path (Get-Location) 'scripts') 'tools\nssm.exe'
    if (Test-Path $local) { return (Resolve-Path $local).ProviderPath }
    try { $p = (Get-Command nssm -ErrorAction SilentlyContinue).Source; if ($p) { return $p } } catch {}
    return $null
}

$nssm = Find-Nssm
if (-not $nssm) {
    Write-Error "Could not find nssm.exe. Please download NSSM and place nssm.exe in scripts\\tools\\ or ensure nssm is in PATH."
    exit 1
}

$pwshPath = (Get-Command pwsh -ErrorAction SilentlyContinue).Source
if (-not $pwshPath) { $pwshPath = 'pwsh' }
$watcherPath = (Resolve-Path '.\scripts\ci-watch.ps1').ProviderPath

$serviceName = 'QuantumTraderCIWatcher'
$args = '-NoProfile -ExecutionPolicy Bypass -File "' + $watcherPath + '" -PollIntervalSeconds 300'

Write-Host "Using nssm: $nssm"
Write-Host "Installing service $serviceName -> $pwshPath $args"

try {
    & $nssm install $serviceName $pwshPath $args
    & $nssm set $serviceName Start SERVICE_AUTO_START
    & $nssm start $serviceName
    Write-Host "Service $serviceName installed and started via NSSM."
} catch {
    Write-Error "NSSM service installation failed: $($_.Exception.Message)"
    exit 2
}
